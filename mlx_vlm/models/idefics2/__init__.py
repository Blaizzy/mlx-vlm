import mlx_vlm.models.idefics2.processing_idefics2  # noqa: F401 (installs processor patch)

from .config import ModelConfig, PerceiverConfig, TextConfig, VisionConfig
from .idefics2 import LanguageModel, Model, VisionModel
