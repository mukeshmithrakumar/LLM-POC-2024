import argparse
import logging
import sys
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
import yaml
from configs.config import TrainingConfig
from model import GPT, GPTConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("logs/train.log"), logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------------------------------- #
# Configuration
def read_configurations(default_config_path: str):
    # parse default configs from yaml file
    with open(default_config_path, "r") as f:
        default_config = yaml.safe_load(f)
    logging.debug(f"default_config: {default_config}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GPT Training Configuration")
    # make the following argument optional
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args, unknown = parser.parse_known_args()

    # Read the optional configuration from file and update the default configuration
    with open(args.config, "r") as f:
        optional_config = yaml.safe_load(f)
    logging.info(f"optional_config: {optional_config}")

    for key in optional_config.keys():
        if default_config.get(key) is not None:
            default_config[key] = optional_config.get(key)
    logging.debug(f"default_config: {default_config}")

    # Add arguments to the parser for each configuration
    for key in default_config.keys():
        parser.add_argument(f"--{key}")

    # Parse command line arguments again, this time including the new arguments
    args = parser.parse_args()

    # Update configuration with command line arguments
    for key in default_config.keys():
        if getattr(args, key) is not None:
            default_config[key] = getattr(args, key)
    logging.info(f"config: {default_config}")

    config = TrainingConfig(**default_config)
    return config


config = read_configurations(default_config_path="configs/default_configs.yaml")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
# ---------------------------------------------------------------------------- #

# distributed training setup for a PyTorch model
# In a DDP environment, each process is assigned a unique rank, which is used to identify the process.
# ddp is set to True if the 'RANK' environment variable is set
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    # is a group of processes that can communicate with each other
    # backend specifies the communication protocol
    init_process_group(backend=config.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # the rank within the machine
    ddp_world_size = int(
        os.environ["WORLD_SIZE"]
    )  # the total number of processes in the process group
    # This means that each process will use a different GPU
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing, etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert (
        config.gradient_accumulation_steps % ddp_world_size == 0
    ), "gradient_accumulation_steps must be divisible by world_size"
    config.gradient_accumulation_steps //= ddp_world_size
    # Gradient accumulation is a technique used to train with larger effective batch sizes
    # than what can fit in GPU memory. By dividing the number of gradient accumulation steps
    # by the world size, the script ensures that the total number of training examples seen
    # before updating the model parameters is the same, regardless of the number of processes.
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    config.gradient_accumulation_steps
    * ddp_world_size
    * config.batch_size
    * config.block_size
)
logging.info(f"tokens per iter: {tokens_per_iter:,}")

if master_process:
    os.makedirs(config.out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = (
    "cuda" if "cuda" in config.device else "cpu"
)  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", config.dataset)


def get_batch(split):
    # we recreate np.memmap every batch to avoid a memory leark, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + config.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + config.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x, y which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(
            config.device, non_blocking=True
        )
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    block_size=config.block_size,
    bias=config.bias,
    vocab_size=None,
    dropout=config.dropout,
)  # start with model_args from command line

if config.init_from == "scratch":
    # init a new model from scratch
    logging.info("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        logging.info(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif config.init_from == "resume":
    logging.info(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint
    ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif config.init_from.startswith("gpt2"):
    logging.info(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if config.block_size < model.config.block_size:
    model.crop_block_size(config.block_size)
    model_args["block_size"] = (
        config.block_size
    )  # so that the checkpoint will have the right value
model.to(config.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type
)
if config.init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if config.compile:
    logging.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (
        config.lr_decay_iters - config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1, f"bad decay_ratio {decay_ratio}"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# logging
if config.wandb_log and master_process:
    import wandb

    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")  # featch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config.eval_interval == 0 and master_process:
        losses = estimate_loss()
        logging.info(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if config.wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or config.always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                logging.info(f"saving checkpoint to {config.out_dir}")
                torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))
    if iter_num == 0 and config.eval_only:
        break

    # forward backward update, with optimal gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == config.gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            # scale the loss to account for gradient accumulations
            loss = loss / config.gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    if config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(
                config.batch_size * config.gradient_accumulation_steps, dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        logging.info(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > config.max_iters:
        break

if ddp:
    destroy_process_group()
