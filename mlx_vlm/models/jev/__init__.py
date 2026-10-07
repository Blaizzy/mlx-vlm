from ..base import install_auto_processor_patch
from ..qwen3_5 import LanguageModel, TextConfig, VisionConfig, VisionModel
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig
from .jev import Model

install_auto_processor_patch("jev", Qwen3VLProcessor)

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
