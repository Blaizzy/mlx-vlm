from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    hidden_act: str = "gelu"
    num_labels: int = 1
