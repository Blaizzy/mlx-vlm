# Import shared LanguageModel from deepseekocr
from ..deepseekocr.language import LanguageModel
from .config import (
    MLPConfig,
    ModelConfig,
    ProjectorConfig,
    Qwen2EncoderConfig,
    TextConfig,
    VisionConfig,
)
from .deepseekocr_2 import Model
from .processing_deepseekocr import DeepseekOCR2Processor
from .vision import Qwen2Decoder2Encoder, VisionModel
