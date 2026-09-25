from dataclasses import dataclass

from ..lfm2_embedding.config import ModelConfig as Lfm2Config


@dataclass
class ModelConfig(Lfm2Config):
    embedding_dim: int = 128
