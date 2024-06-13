import argparse
import logging

import yaml


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
