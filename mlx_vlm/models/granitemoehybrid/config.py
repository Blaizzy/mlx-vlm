from dataclasses import dataclass
from typing import Optional, Tuple

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    embedding_multiplier: float
    attention_multiplier: float
    logits_scaling: float
    residual_multiplier: float
    layer_types: list[str]
    rms_norm_eps: float
    rope_theta: float
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    shared_intermediate_size: Optional[int] = None
    mamba_n_heads: Optional[int] = None
    mamba_d_head: Optional[int] = None
    mamba_proj_bias: Optional[bool] = None
    mamba_d_state: Optional[int] = None
    mamba_d_conv: Optional[int] = None
    mamba_n_groups: Optional[int] = None
    mamba_conv_bias: Optional[bool] = None
    mlp_bias: bool = False
    position_embedding_type: str = "rope"
    tie_word_embeddings: bool = True
    time_step_limit: Tuple[float, float] = (0.001, 100.0)

    @property
    def use_moe(self) -> bool:
        return bool(self.num_local_experts)
