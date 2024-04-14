"""Helper File with some helpers."""
import pickle
import os

from typing import List

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


def load_pkl_files(datafolder_path: str, file_names: List) -> List:
    """_summary_

    Args:
        datafolder_path (str): _description_
        file_names (List): _description_

    Returns:
        List: _description_
    """
    data_objects = []
    for filename in file_names:
        with open(os.path.join(datafolder_path, filename), 'rb') as f:
            data_objects.append(pickle.load(f))
    return data_objects


def save_pkl_files(datafolder_path: str, file_names: List, files: List) -> None:
    """_summary_

    Args:
        datafolder_path (str): _description_
        file_names (List): _description_
        files (List): _description_
    """
    os.makedirs(datafolder_path, exist_ok=True)  # Create directory structure
    for filename, file in zip(file_names, files):
        with open(os.path.join(datafolder_path, filename), 'wb') as f:
            pickle.dump(file, f)
    print("Pickle files saved")
