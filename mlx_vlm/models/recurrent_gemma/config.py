from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    attention_bias: bool
    conv1d_width: int
    hidden_size: int
    intermediate_size: int
    logits_soft_cap: float
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    attention_window_size: int
    vocab_size: int
    embeddings_scale_by_sqrt_dim: bool = True
    block_types: Optional[List[str]] = None
    _block_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.block_types is None:
            self.block_types = self._block_types
