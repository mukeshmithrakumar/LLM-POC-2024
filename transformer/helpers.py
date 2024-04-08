"""Helper File with some helpers."""
from pathlib import Path
from config import Config


def latest_weights_file_path(config: Config) -> str:
    """Get the file path for the latest weights file for the given configuration.

    Args:
        config (Config): The configuration.

    Returns:
        str: The file path for the latest weights file for the given configuration.
    """
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def get_weights_file_path(config: Config, epoch: str) -> str:
    """Get the file path for the weights file for the given epoch and configuration.

    Args:
        config (Config): The configuration.
        epoch (str): _description_

    Returns:
        str: The file path for the weights file for the given epoch and configuration.
    """
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
