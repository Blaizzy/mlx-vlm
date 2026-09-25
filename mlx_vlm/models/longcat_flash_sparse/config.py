from dataclasses import dataclass
from typing import Dict, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "longcat_flash_sparse"
    attention_method: str = "LSA"  # "LSA" (sparse indexer) or "MLA" (dense)
    zero_expert_type: str = "identity"
    hidden_size: int = 3072
    ffn_hidden_size: int = 6144
    expert_ffn_hidden_size: int = 1024
    moe_topk: int = 12
    n_routed_experts: int = 256
    zero_expert_num: int = 0
    num_layers: int = 14
    vocab_size: int = 131072
    max_position_embeddings: int = 983040
    num_attention_heads: int = 32
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    routed_scaling_factor: float = 6.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    mla_scale_q_lora: bool = True
    mla_scale_kv_lora: bool = True
    attention_bias: bool = False
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: Optional[Dict] = None
    # LSA indexer (used only when attention_method == "LSA"); defaults keep dense behavior
    index_n_heads: int = 0
    index_head_dim: int = 128
    index_topk: int = 0
    index_init_tokens: int = 0  # streaming sink tokens always kept
    index_local_tokens: int = 0  # streaming local window always kept
    cli_factor: int = 1  # attention sub-blocks sharing one indexer pass
    index_k_norm_type: str = "rms"
    indexer_rope_interleave: bool = True
    # n-gram input embedding (Lite/Lite-Sparse); oe_vocab_size_ratio == 0 disables it
    oe_vocab_size_ratio: int = 0
    oe_neighbor_num: int = 4
    oe_split_num: int = 4
