from . import processing_owlv2  # Install processor patch (side effect)
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .owlv2 import Model
from .vision import VisionModel

# Aliases for mlx-vlm framework compatibility
PerceiverConfig = ModelConfig
ProjectorConfig = ModelConfig
AudioConfig = ModelConfig
