import mlx_vlm.models.llava_next.processing_llava_next  # noqa: F401 (installs processor patch)

from .config import ModelConfig, TextConfig, VisionConfig
from .llava_next import LanguageModel, Model, VisionModel
