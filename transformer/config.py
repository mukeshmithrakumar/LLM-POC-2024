"""Configurations."""

from dataclasses import dataclass


@dataclass
class Config:
    """Get the configuration for the training process.

    dropout_constant: float: Dropout Constant. default 0.1.
    num_layers: int: Number of layers. default 6.
    num_heads: int: Number of heads. default 8.
    d_ff: int: Number of linear projections. default 2048.
    batch_size: int: The batch size for the training process. default 8.
    num_epochs: int: The number of epochs for the training process. default 3.
    lr: float: The learning rate for the training process. default 10**-4.
    seq_len: int: The sequence length for the training process. default 480.
    d_model: int: The text embedding dimension. default 512.
    datasource: str: The datasource
    data_folder: str: The data folder
    lang_src: str: The source language for the translation
    lang_tgt: str: The target language for the translation
    model_folder: str: The folder where the model weights are stored
    model_basename: str: The base name for the model weights
    preload: str: The file path for the weights file to preload
    tokenizer_file: str: The file path for the tokenizer file
    experiment_name: str: The name of the experiment
    """

    dropout_constant: float = 0.1
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    batch_size: int = 8
    num_epochs: int = 3
    lr: float = 10**-4
    seq_len: int = 480
    d_model: int = 512
    datasource: str = "opus_books"
    data_folder: str = "data"
    lang_src: str = "en"
    lang_tgt: str = "fr"
    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: str = "latest"
    tokenizer_file: str = "tokenizers/tokenizer_{0}.json"
    experiment_name: str = "runs/tmodel"
