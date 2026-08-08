from dataclasses import dataclass

from ..qwen3_vl.config import ModelConfig as Qwen3VLModelConfig
from ..qwen3_vl.config import TextConfig, VisionConfig

__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]


@dataclass
class ModelConfig(Qwen3VLModelConfig):
    normalize: bool = True
