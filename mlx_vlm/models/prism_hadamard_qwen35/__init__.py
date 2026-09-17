from ..base import install_auto_processor_patch
from ..qwen3_5 import LanguageModel, TextConfig, VisionConfig, VisionModel
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig
from .prism_hadamard_qwen35 import Model

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "Qwen3VLProcessor",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]

install_auto_processor_patch("prism_hadamard_qwen35", Qwen3VLProcessor)
