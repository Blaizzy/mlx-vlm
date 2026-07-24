from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float
    sliding_window: int
    sliding_window_layers: List[int]
    conv_window: int
    rms_norm_eps: float
    model_type: str = "baichuan_m1"
    num_swa_attention_heads: Optional[int] = None
    num_swa_key_value_heads: Optional[int] = None
    tie_word_embeddings: bool = False
