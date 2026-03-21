from ..base import install_auto_processor_patch

# Reuse qwen3_vl processor and install patch for qwen3_vl_moe model_type
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: F401
from .config import ModelConfig, TextConfig, VisionConfig
from .qwen3_vl_moe import LanguageModel, Model, VisionModel

install_auto_processor_patch("qwen3_vl_moe", Qwen3VLProcessor)
