import mlx_vlm.models.mage_vl.processing_mage_vl  # noqa: F401 (installs processor patch)

from .config import ModelConfig, TextConfig, VisionConfig
from .mage_vl import LanguageModel, Model, VisionModel
