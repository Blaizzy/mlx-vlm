from dataclasses import dataclass
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig"
    vision_config: "VisionConfig"
    model_type: str
    ignore_index: int = -100
    vocab_size: int = 128259
    scale_factor: int = 2
    media_placeholder_token_id: int = 163606
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.media_placeholder_token_id


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "deepseek_v3"
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "moonvit"
    depth: int = 27
    embed_dim: int = 1152
    hidden_size: int = 1152
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    intermediate_size: int = 4304
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    merge_kernel_size: list[int, int] = None

    def __post_init__(self):
        if self.merge_kernel_size is None:
            self.merge_kernel_size = (self.spatial_merge_size, self.spatial_merge_size)
