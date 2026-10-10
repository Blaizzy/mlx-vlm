from ..base import install_auto_processor_patch
from ..qwen3_5.language import LanguageModel
from ..qwen3_5.vision import VisionModel
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .clef import Model
from .config import ModelConfig, TextConfig, VisionConfig

install_auto_processor_patch("clef", Qwen3VLProcessor)
