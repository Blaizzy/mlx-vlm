from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head
