import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import argparse
import yaml


# ---------------------------------------------------------------------------- #
# default config values designed to train a gpt2 (124M) on OpenWebText

# Parse command line arguments
parser = argparse.ArgumentParser(description='GPT Training Configuration')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
args, unknown = parser.parse_known_args()

# Read configuration from file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
print(f"config: {config}")

# Add arguments to the parser for each configuration
for key in config.keys():
    parser.add_argument(f'--{key}')

# Parse command line arguments again, this time including the new arguments
args = parser.parse_args()

# Update configuration with command line arguments
for key in config.keys():
    if getattr(args, key) is not None:
        config[key] = getattr(args, key)
print(f"config: {config}")

# Update global variables with configuration values
globals().update(config)

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# ---------------------------------------------------------------------------- #
