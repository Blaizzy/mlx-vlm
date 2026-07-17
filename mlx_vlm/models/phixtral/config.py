from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "phixtral"
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32
    num_experts_per_tok: int = 2
    num_local_experts: int = 4
