from dataclasses import dataclass, field
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
    use_qk_norm: bool
    attention_bias: bool
    partial_rotary_factor: float
    rope_scaling: Dict = field(
        default_factory=lambda: {"type": "default", "mrope_section": [64, 32, 32]}
    )
    tie_word_embeddings: bool = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    patch_size: int
    window_size: int = 112
    image_size: int = 336
    in_channels: int = 3
    rms_norm_eps: float = 1e-05
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    out_hidden_size: int = 4096
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 151363
    image_token_id: int = 151363
    video_token_index: int = 151364
    video_token_id: int = 151364
    vision_start_token_id: int = 151339
    vision_end_token_id: int = 151340
    hidden_size: int = 2048
    pad_token_id: int = 0
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.eos_token_id is None:
            self.eos_token_id = [151329, 151336, 151338]
