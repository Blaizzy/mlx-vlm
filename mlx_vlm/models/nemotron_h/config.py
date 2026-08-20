from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_proj_bias: bool
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    mlp_bias: bool
    layer_norm_epsilon: float
    use_bias: bool
    use_conv_bias: bool
    hybrid_override_pattern: Optional[List[str]] = None
    layers_block_type: Optional[List[str]] = None
    head_dim: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_latent_size: Optional[int] = None
    n_group: Optional[int] = None
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    topk_group: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    norm_topk_prob: Optional[bool] = None
    routed_scaling_factor: Optional[float] = None
    time_step_limit: Optional[Tuple[float, float]] = None
    time_step_min: Optional[float] = None
    time_step_max: Optional[float] = None

    _block_type_to_char = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}

    def __post_init__(self):
        if self.time_step_limit is None:
            self.time_step_limit = (0.0, float("inf"))

        if self.hybrid_override_pattern is None and self.layers_block_type is not None:
            self.hybrid_override_pattern = [
                self._block_type_to_char[t] for t in self.layers_block_type
            ]
        if self.hybrid_override_pattern is not None:
            self.num_hidden_layers = len(self.hybrid_override_pattern)
