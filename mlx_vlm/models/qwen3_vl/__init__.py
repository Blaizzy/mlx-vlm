import mlx_vlm.models.qwen3_vl.processing_qwen3_vl  # noqa: F401 (installs processor patch)

from .config import ModelConfig, TextConfig, VisionConfig
from .qwen3_vl import LanguageModel, Model, VisionModel
