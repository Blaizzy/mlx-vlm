from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    attention_bias: bool
    head_dim: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: str
    rope_theta: float
    tie_word_embeddings: bool
