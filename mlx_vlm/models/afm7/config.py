from dataclasses import dataclass

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_kv_reuse_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim_scale_factor: float = 3.25
    rope_theta: float = 50000.0
    rms_norm_eps: float = 1e-5
