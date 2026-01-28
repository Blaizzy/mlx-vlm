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
from .deepseekocr_2 import DeepseekOCR2Processor, Model
from .vision import Qwen2Decoder2Encoder, VisionModel
