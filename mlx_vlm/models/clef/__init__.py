from ..base import install_auto_processor_patch
from ..qwen3_5 import LanguageModel, TextConfig, VisionConfig, VisionModel
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .clef import Model
from .config import ModelConfig

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]

install_auto_processor_patch("clef", Qwen3VLProcessor)
