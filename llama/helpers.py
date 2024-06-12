import pickle
import yaml
import logging
import argparse


class SimpleTokenizer:
    def __init(self, stoi, merges):
        self.stoi = stoi
        self.merges = merges
        self.itos = {i: s for s, i in stoi.items()}  # inverse mapping for decoding
        self.vocab_len = len(stoi) + len(merges)

    def encode(self, text):
        # convert the text to a list of token IDs, using space for unknown characters
        tokens = [self.stoi.get(c, self.stoi[" "]) for c in text]

        # perform merging with the possibility of nested merges
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                # replace the current pair with its merged token
                merged_token = self.merges[pair]
                tokens[i] = merged_token
                del tokens[i + 1]
                # move back to handle possible nested merges
                if i > 0:
                    i -= 1
            else:
                i += 1
        return tokens

    def decode(self, tokens):
        def expand_token(token):
            # Base case: if the token is a direct mapping, return its character
            if token in self.itos:
                return self.itos[token]
            # Recursive case: if the token is a merged token, expand its constituents
            elif token in self.merges.values():
                pair = next(key for key, value in self.merges.items() if value == token)
                return "".join(expand_token(t) for t in pair)
            # Fallback for unknown tokens
            else:
                return ""

        return "".join(expand_token(token) for token in tokens)


def load_tokenizer_data(size: int):
    file_name = f"./tokenizers/tiny_shakespeare_tokenizer_{size}.model"
    with open(file_name, "rb") as f:
        tokenizer_data = pickle.load(f)
    return tokenizer_data


def get_tokenizer(size: int):
    tokenizer_data = load_tokenizer_data(size)
    loaded_stoi = tokenizer_data["stoi"]
    loaded_merges = tokenizer_data["merges"]
    return SimpleTokenizer(loaded_stoi, loaded_merges)


# Configuration
def read_configurations(default_config_path: str):
    # parse default configs from yaml file
    with open(default_config_path, "r") as f:
        default_config = yaml.safe_load(f)
    logging.debug(f"default_config: {default_config}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLaMA 3 Configuration")
    # make the following argument optional
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args, unknown = parser.parse_known_args()

    # Read the optional configuration from file and update the default configuration
    if args.config:
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

    return default_config
