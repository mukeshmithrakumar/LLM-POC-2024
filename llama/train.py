"""Train Script.
Codes from the following repositories are used in this script: 
- https://github.com/meta-llama/llama3
- https://github.com/evintunador/minLlama3
"""

import logging
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
from configs.config import TrainingConfig
from helpers import read_configurations
from model import Llama3

# ------------------------------ Set up logging ------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("logs/train.log"), logging.StreamHandler(sys.stdout)],
)

# --------------------------- Set up Configurations -------------------------- #
default_config = read_configurations(
    default_config_path="configs/default_training_configs.yaml"
)
config = TrainingConfig(**default_config)
data_dir = os.path.join("data")

os.makedirs(config.out_dir, exist_ok=True)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
logging.info(f"Meta Path: {meta_path}")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    config.vocab_size = meta["vocab_size"]
    logging.info(f"updated vocab_size = {config.vocab_size}")


# ----------------------------- Training Helpers ----------------------------- #
def get_batch(split):
    # we recreate np.memmap every batch to avoid a memory leark, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - config.seq_len, (config.batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + config.seq_len]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + config.seq_len]).astype(np.int64))
            for i in ix
        ]
    )
    if config.device == "cuda":
        # pin arrays x, y which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(
            config.device, non_blocking=True
        )
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters=5):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def lr_lambda(current_iter):
    if current_iter < config.warmup_iters:
        return (
            config.warmup_factor
            + (1 - config.warmup_factor) * current_iter / config.warmup_iters
        )
    else:
        decay_iters = config.max_iters - config.warmup_iters
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * (current_iter - config.warmup_iters) / decay_iters)
        )
        return max(cosine_decay, config.lr_final / config.lr_init)


# --------------------------------- Training --------------------------------- #
logging.info("Loading LLaMA3 Model...")
model = Llama3(config).to(config.device)
logging.info(
    f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e3, 'K parameters'}"
)
logging.debug(f"LLaMA3 Model:\n{model}")

logging.info("Starting LLaMA3 Training...")
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

best_val_loss = 1e9

start_time = time.time()
for iter_num in range(config.max_iters):
    # sample a batch of data
    xb, yb = get_batch("train")

    # train
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # update the learning rate
    scheduler.step()

    # every once in a while evaluate the loss on train and val sets
    if (iter_num % config.eval_interval == 0) or (iter_num == config.max_iters - 1):
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Iter: {iter_num}, Train Loss: {losses['train']}, Val Loss: {losses['val']}, time elapsed: {elapsed_time:.2f} s"
        )

    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        if iter_num > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            logging.info(f"saving checkpoint to {config.out_dir}")
            torch.save(checkpoint, os.path.join(config.out_dir, "ckpt.pt"))

logging.info("Training Complete!")

# --------------------------------- Inference -------------------------------- #
logging.info("Starting Inference...")
input_str = "JULIET:\nO Romeo, Romeo! wherefore art thou R"
logging.info(f"Input String:\n{input_str}")
logging.info("Generating...")
logging.info(model.generate(input_str, meta_path, max_gen_len=100))

# ------------------------------ Using KV Cache ------------------------------ #
logging.info("Generating using KV Cache...")
output = model.generate(
    input_str,
    meta_path,
    max_gen_len=config.seq_len
    - len(
        input_str
    ),  # our model doesn't have a built-in <endoftext> token so we have to specify when to stop generating
    memory_saver_div=8,  # the largest value we'll allow our query sequence length to get. makes memory consumption linear with respect to sequence length
    temperature=0.6,  # this is the default value that Llama3's official code has set
    top_p=0.9,  # this is the default value that Llama3's official code has set
    top_k=32,  # meta's code doesn't actually implement top_k selection but i've added it anyways as an alternative
)
logging.info(output)
