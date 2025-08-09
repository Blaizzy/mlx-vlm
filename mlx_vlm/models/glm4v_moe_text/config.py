from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads: int
    n_group: int
    head_dim: int
    topk_group: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    num_experts_per_tok: int
    first_k_dense_replace: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict]
    use_qk_norm: bool
    tie_word_embeddings: bool
    attention_bias: bool
    partial_rotary_factor: float
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    patch_size: int
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 262144
    hidden_size: int = 2048
    pad_token_id: int = 0
    eos_token_id: Optional[List[int]] = None
