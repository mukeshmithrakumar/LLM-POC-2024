from dataclasses import dataclass


@dataclass
class TrainingConfig:
    out_dir: str
    eval_interval: int
    log_interval: int
    eval_iters: int
    eval_only: bool  # if True, script exits right after the first eval
    always_save_checkpoint: bool  # if True, always save a checkpoint after each eval
    init_from: str  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log: bool  # disabled by default
    wandb_project: str
    wandb_run_name: str  # 'run' + str(time.time())

    # dataset
    dataset: str
    gradient_accumulation_steps: int  # used to simulate larger batch sizes
    batch_size: int  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int

    # model
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool  # do we use bias inside LayerNorm and Linear layers?

    # adamw optimizer
    learning_rate: float  # max learning rate
    max_iters: int  # total number of training iterations
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool  # whether to decay the learning rate
    warmup_iters: int  # how many steps to warm up for
    lr_decay_iters: int  # should be ~= max_iters per Chinchilla
    min_lr: float  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # DDP settings
    backend: str  # 'nccl', 'gloo', etc.

    # system
    device: str  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    compile: bool  # use PyTorch 2.0 to compile the model to be faster


@dataclass
class GenerateConfig:
    # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    init_from: str
    out_dir: str  # ignored if init_from is not 'resume'
    start: str  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int  # number of samples to draw
    max_new_tokens: int  # number of tokens generated in each sample
    temperature: float  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int
    device: str  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    compile: bool  # use PyTorch 2.0 to compile the model to be faster
