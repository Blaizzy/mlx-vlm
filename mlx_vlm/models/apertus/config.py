from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    mlp_bias: bool
    num_attention_heads: int
    attention_bias: bool
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    post_norm: bool
    qk_norm: bool
    tie_word_embeddings: bool
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
