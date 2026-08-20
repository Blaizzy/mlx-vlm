from dataclasses import dataclass
from typing import List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str
    hidden_size: int = 1024
    head_dim: int = 128
    num_hidden_layers: int = 36
    intermediate_size: int = 1024
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: float = 50000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    logit_scale: float = 0.0625
    attention_bias: bool = False
    layer_norm_bias: bool = False
    sliding_window: int = 4096
    sliding_window_pattern: int = 4
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    num_shared_experts: Optional[int] = None
    moe_num_shared_experts: int = 4
    moe_gate_act: str = "sigmoid"
    expert_selection_fn: Optional[str] = None
    shared_expert_combination_strategy: str = "average"
    rms_norm_eps: Optional[float] = None
    first_k_dense_replace: int = 0
    prefix_dense_intermediate_size: Optional[int] = None
    prefix_dense_sliding_window_pattern: int = 1
    layer_types: Optional[List[str]] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.num_shared_experts is not None:
            self.moe_num_shared_experts = self.num_shared_experts
        if self.expert_selection_fn is not None:
            self.moe_gate_act = self.expert_selection_fn
        if self.prefix_dense_intermediate_size is None:
            self.prefix_dense_intermediate_size = self.intermediate_size
