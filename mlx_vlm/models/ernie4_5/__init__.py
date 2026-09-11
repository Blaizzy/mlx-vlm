from .config import ModelConfig
from .ernie4_5 import Model

__all__ = ["Model", "ModelConfig"]

from ..base import install_auto_processor_patch
from .tokenization_ernie4_5 import Ernie45Tokenizer

install_auto_processor_patch("ernie4_5", Ernie45Tokenizer)
