from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch
from .config import ModelConfig
from .exaone4 import Model

install_auto_processor_patch("exaone4", PreTrainedTokenizerFast)

__all__ = ["Model", "ModelConfig"]
