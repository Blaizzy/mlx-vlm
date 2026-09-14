from . import processing_rtmdet  # install processor patch (side effect)
from .config import RTMDetConfig
from .language import LanguageModel
from .rtmdet import Model, VisionModel

# Framework compat aliases
ModelConfig = RTMDetConfig
TextConfig = RTMDetConfig
VisionConfig = RTMDetConfig
