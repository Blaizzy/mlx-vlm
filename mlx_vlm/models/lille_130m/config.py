from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "lille-130m"
    block_size: int = 2048
    layer_norm_eps: float = 1e-5
    n_embd: int = 768
    n_head: int = 12
    n_kv_heads: int = 12
    n_layer: int = 12
    rope_theta: float = 10000.0
    vocab_size: int = 50304
    tie_word_embeddings: bool = True
