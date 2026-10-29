from ..base import install_auto_processor_patch
from ..qwen3_5 import TextConfig, VisionConfig
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig
from .minicpm_ocr_jev import Model
from .scoring import build_prompt, score

install_auto_processor_patch("minicpm_ocr_jev", Qwen3VLProcessor)

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "build_prompt",
    "score",
]
