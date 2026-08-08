from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    max_position_embeddings: Optional[int] = None
    norm_eps: float = 1e-5
    norm_bias: bool = False
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    attention_bias: bool = False
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    mlp_bias: bool = False
