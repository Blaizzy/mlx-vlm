from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "kolibri1"
    layer_types: List[str] = field(default_factory=list)
    vocab_size: int = 128000
    hidden_size: int = 2560
    num_hidden_layers: int = 50
    num_attention_heads: int = 48
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    num_experts: int = 384
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = False
    # Keys visible to a query, the query itself included (512 preceding + 1).
    # Same convention as create_causal_mask(window_size=...) and
    # RotatingKVCache(max_size=...), so it is used unchanged.
    sliding_window: int = 513
