from dataclasses import dataclass

from ..idefics3.config import ModelConfig as Idefics3ModelConfig
from ..idefics3.config import TextConfig, VisionConfig

__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]


@dataclass
class ModelConfig(Idefics3ModelConfig):
    embedding_dim: int = 128
    mask_non_image_embeddings: bool = False
