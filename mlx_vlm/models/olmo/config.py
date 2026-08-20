from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    embedding_size: int
    mlp_hidden_size: Optional[int] = None
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    mlp_ratio: int = 4
    weight_tying: bool = False

    def __post_init__(self):
        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = self.mlp_ratio * self.d_model
