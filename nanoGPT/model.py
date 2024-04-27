import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CasualSelfAttention(nn.Module):
    """CasualSelfAttention module performs causal self-attention on the input tensor.

    Args:
        config (object): Configuration object containing the following attributes:
            - n_embd (int): Embedding dimensionality.
            - n_head (int): Number of attention heads.
            - block_size (int): Size of the input block.
            - bias (bool): Whether to include bias in linear layers.
            - dropout (float): Dropout probability.

    Attributes:
        c_attn (nn.Linear): Linear layer for key, query, value projections.
        c_proj (nn.Linear): Linear layer for output projection.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
        resid_dropout (nn.Dropout): Dropout layer for residual connections.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimensionality.
        dropout (float): Dropout probability.
        flash (bool): Whether Flash Attention is available.
        bias (torch.Tensor): Casual mask to ensure attention is only applied to the left in 
        the input sequence.

    Methods:
        forward(x): Performs forward pass of the CasualSelfAttention module.

    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "Warning: Flash Attention not available, falling back to standard attention."
            )
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) module.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        c_fc (nn.Linear): Linear layer for the first fully connected operation.
        gelu (nn.GELU): GELU activation function.
        c_proj (nn.Linear): Linear layer for the second fully connected operation.
        dropout (nn.Dropout): Dropout layer for regularization.

    """

    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.

        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias) -> None:
        """Initializes a LayerNorm module.

        Args:
            ndim (int): The number of dimensions of the input tensor.
            bias (bool): Whether to include a bias term in the normalization.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """Performs forward pass of the LayerNorm module.

        Args:
            input (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized output tensor.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, config) -> None:
        """Block class represents a single block in the nanoGPT model.

        Args:
            config (object): Configuration object containing parameters for the block.

        Attributes:
            ln_1 (LayerNorm): Layer normalization module for the first layer.
            attn (CasualSelfAttention): Casual self-attention module.
            ln_2 (LayerNorm): Layer normalization module for the second layer.
            mlp (MLP): Multi-layer perceptron module.

        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        """Forward pass of the Block.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after passing through the block.

        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print(f"number of parameters: {self.get_num_params()/1e6:.2fM}")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
    
    
