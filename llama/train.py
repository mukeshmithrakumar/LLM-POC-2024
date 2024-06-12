"""Train Script.
Codes from the following repositories are used in this script: 
- https://github.com/evintunador/minLlama3/tree/main
- https://github.com/meta-llama/llama3/tree/main 
"""

import sys
import os
import time
import json
import logging
import torch
from helpers import get_tokenizer, read_configurations
from configs.config import TrainingConfig
from model import Llama3
import math

from dataclasses import asdict


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

tokenizer = get_tokenizer(size=512)  # size options are 128, 256, 512 and 1024
config.vocab_size = tokenizer.vocab_len

logging.debug(f"config: {asdict(config)}")

# ---------------------------------------------------------------------------- #

# load the dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

logging.debug(f"Loaded text of length {len(text)}")
logging.debug(f"First 100 characters: {text[:100]}")

# Train and Test Splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, test_data = data[:n], data[n:]

# ---------------------------------------------------------------------------- #
model = Llama3(config, tokenizer).to(config.device)
logging.debug(
    f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e3, 'K parameters'}"
)
logging.debug(f"LLaMA3 Model:\n{model}")


# ----------------------------- Training Helpers ----------------------------- #
def get_batch(split, batch_size):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - config.max_seq_len, (batch_size,))
    x = torch.stack([data[i : i + config.max_seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + config.max_seq_len + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters=5):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
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
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay
)
scheduler = torch.OptionalType.lr_scheduler.LambdaLR(optimizer, lr_lambda)

best_val_loss = 1e9

start_time = time.time()
for iter_num in range(config.max_iters):
    # sample a batch of data
    xb, yb = get_batch("train", config.max_batch_size)

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
        losses = estimate_loss(model, batch_size=config.max_batch_size)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Iter: {iter_num}, Train Loss: {losses['train']}, Val Loss: {losses['val']}, time elapsed: {elapsed_time:.2f} s"
        )

    if losses["val"] < best_val_loss or config.always_save_checkpoint:
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
logging.info(model.generate(input_str, max_gen_len=100))

# ------------------------------ Using KV Cache ------------------------------ #
logging.info("Generating using KV Cache...")
output = model.generate(
    input_str,
    max_gen_len=config.max_seq_len
    - len(
        input_str
    ),  # our model doesn't have a built-in <endoftext> token so we have to specify when to stop generating
    memory_saver_div=8,  # the largest value we'll allow our query sequence length to get. makes memory consumption linear with respect to sequence length
    temperature=0.6,  # this is the default value that Llama3's official code has set
    top_p=0.9,  # this is the default value that Llama3's official code has set
    top_k=32,  # meta's code doesn't actually implement top_k selection but i've added it anyways as an alternative
)
logging.info(output)
