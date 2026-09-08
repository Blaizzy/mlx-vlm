from ..base import install_auto_processor_patch
from ..qwen3_vl import LanguageModel, VisionModel
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig, TextConfig, VisionConfig
from .qwen3_vl_embedding import Model

install_auto_processor_patch("qwen3_vl_embedding", Qwen3VLProcessor)

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "LanguageModel",
    "VisionModel",
]
