import mlx_vlm.models.idefics3.processing_idefics3  # noqa: F401 (installs processor patch)

from .config import ModelConfig, TextConfig, VisionConfig
from .idefics3 import Model
from .language import LanguageModel
from .vision import VisionModel
