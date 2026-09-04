from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import BaseModelConfig


@dataclass
class K2HorizonConfig(BaseModelConfig):
    model_type: str = "k2_horizon"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 250624
    head_dim: int = 128
    max_position_embeddings: int = 524288
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-06
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    sliding_window: Optional[int] = None
    rope_theta: float = 10000000.0
    rope_type: str = "default"
    decoder_sparse_step: int = 1
    layernorm_num_groups: int = 4
    query_key_norm: bool = False
    moe_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    mova_num_experts: int = 0
    mova_num_experts_per_tok: int = 0
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001
    router_scaling_factor: float = 1.0
    router_score_func: str = "sigmoid"
    output_router_logits: bool = False
    moe_gate_bias: bool = False
    initializer_range: float = 0.02
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: Optional[int] = None
    dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, params: Dict[str, Any]):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})
