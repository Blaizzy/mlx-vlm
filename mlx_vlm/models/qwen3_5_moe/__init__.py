from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .config import ModelConfig, TextConfig, VisionConfig
from .qwen3_5_moe import LanguageModel, Model, VisionModel

install_auto_processor_patch("qwen3_5_moe", Qwen3VLProcessor)
