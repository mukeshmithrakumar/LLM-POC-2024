"""Generate Sample from a trained model."""

import logging
import os
import pickle
import sys
from contextlib import nullcontext

import tiktoken
import torch
from configs.config import GenerateConfig
from helpers import read_configurations
from model import GPT, GPTConfig

# ------------------------------ Set up logging ------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/generate.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

# --------------------------- Set up Configurations -------------------------- #
default_config = read_configurations(
    default_config_path="configs/default_generate_configs.yaml"
)
config = GenerateConfig(**default_config)

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = (
    "cuda" if "cuda" in config.device else "cpu"
)  # for later use in torch.autocast
ptddtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptddtype)
)

# ---------------------------------------------------------------------------- #

# model
if config.init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif config.init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(config.init_from, dict(dropout=0.0))

model.eval()
model.to(config.device)
if config.compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    (config.init_from == "resume")
    and ("config" in checkpoint)
    and (hasattr(checkpoint["config"], "dataset"))
):
    meta_path = os.path.join("data", checkpoint["config"].dataset, "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    logging.info(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO: want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]  # noqa: E731
    decode = lambda l: "".join([itos[i] for i in l])  # noqa: E731
else:
    # ok let's assume gpt-2 encodings by default
    logging.info("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa: E731
    decode = lambda l: enc.decode(l)  # noqa: E731

# encode the beginning of the prompt
if config.start.startswith("FILE:"):
    with open(config.start[5:], "r", encoding="utf-8") as f:
        start = f.read()
    start_ids = encode(start)
else:
    start_ids = encode(config.start)
x = torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(config.num_samples):
            y = model.generate(
                x,
                config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
            )
            logging.info(decode(y[0].tolist()))
            logging.info("-" * 80)
