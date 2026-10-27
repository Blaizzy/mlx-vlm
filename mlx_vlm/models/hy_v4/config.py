from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "hy_v4"
    vocab_size: int = 120832
    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.827
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    indexer_types: List[str] = field(default_factory=list)
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    eos_token_id: Optional[Union[int, List[int]]] = None
    bos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if not self.indexer_types:
            self.indexer_types = [
                "full" if index < 2 or index % 4 == 1 else "shared"
                for index in range(self.num_hidden_layers)
            ]
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError(
                "`indexer_types` must have one entry per hidden layer, "
                f"got {len(self.indexer_types)} for {self.num_hidden_layers} layers."
            )
        unsupported = set(self.indexer_types) - {"full", "shared"}
        if unsupported:
            raise ValueError(f"Unsupported HYV4 indexer types: {unsupported}")
