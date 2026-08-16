from dataclasses import dataclass
from typing import Any, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    attention_method: str
    zero_expert_type: str
    hidden_size: int
    ffn_hidden_size: int
    moe_topk: int
    expert_ffn_hidden_size: int
    n_routed_experts: int
    zero_expert_num: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: Optional[int]
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    attention_bias: bool
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: Optional[dict[str, Any]] = None
