import mlx_vlm.models.minimax_m3.processing  # noqa: F401 (registers processor)

from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .minimax_m3 import Model
from .vision import VisionModel
