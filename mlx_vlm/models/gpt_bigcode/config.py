from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    n_embd: int
    n_layer: int
    n_inner: int
    n_head: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int = None
    multi_query: bool = True
    attention_bias: bool = True
    mlp_bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 1 if self.multi_query else self.n_head
