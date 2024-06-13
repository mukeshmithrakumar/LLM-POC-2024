from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainingConfig:
    # No default configurations
    lr_init: float
    weight_decay: float
    max_iters: int  # how long we want to train for
    eval_interval: int  # how often we want to evaluate
    warmup_iters: int  # Number of warmup iterations
    warmup_factor: (
        float  # Warmup factor (initial learning rate is multiplied by this factor)
    )
    lr_final: float  # Minimum learning rate
    out_dir: str
    # Default configurations
    dim: int = 128  # 4096
    n_layers: int = 12  # 32
    n_heads: int = 4  # 32
    n_kv_heads: Optional[int] = 1  # None
    vocab_size: int = 32000
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000  # 500000
    batch_size: int = 24
    seq_len: int = (
        512  # 8192 but their maximum chunk size when running inference is 2048
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dropout_rate: float = 0.1
