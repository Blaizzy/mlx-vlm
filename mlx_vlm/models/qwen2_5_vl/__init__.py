import mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl  # noqa: F401 (installs processor patch)

from .config import ModelConfig, TextConfig, VisionConfig
from .qwen2_5_vl import LanguageModel, Model, VisionModel
