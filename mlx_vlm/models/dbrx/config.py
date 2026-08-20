from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    d_model: int
    ffn_config: dict
    attn_config: dict
    n_layers: int
    n_heads: int
