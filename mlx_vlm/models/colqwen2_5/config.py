from dataclasses import dataclass

from ..qwen2_5_vl.config import ModelConfig as Qwen2_5_VLModelConfig
from ..qwen2_5_vl.config import TextConfig, VisionConfig

__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]


@dataclass
class ModelConfig(Qwen2_5_VLModelConfig):
    embedding_dim: int = 128
