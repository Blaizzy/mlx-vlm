from . import processing_sapiens2  # Install processor patch (side effect)
from .config import BackboneConfig, HeadConfig, ModelConfig
from .language import LanguageModel
from .sapiens2 import Model
from .vision import VisionModel

TextConfig = HeadConfig
VisionConfig = BackboneConfig
PerceiverConfig = ModelConfig
ProjectorConfig = ModelConfig
AudioConfig = ModelConfig
