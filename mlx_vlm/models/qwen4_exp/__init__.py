from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig, TextConfig, VisionConfig
from .qwen4_exp import LanguageModel, Model, VisionModel

install_auto_processor_patch("qwen4_exp", Qwen3VLProcessor)
