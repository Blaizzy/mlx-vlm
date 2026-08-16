from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    dim_model_base: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    scale_depth: float
    scale_emb: float
    max_position_embeddings: Optional[int] = None
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[str, float]]] = None
    tie_word_embeddings: bool = False
