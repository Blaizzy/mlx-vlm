from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int = 8192
    head_dim: int = 128
    num_hidden_layers: int = 40
    intermediate_size: int = 14336
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
    max_position_embeddings: int = 4096


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_attention_heads: int
    patch_size: int
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    image_token_index: int = 255036
    max_splits_per_img: int = 12
    downsample_factor: int = 2
    alignment_intermediate_size: int = 28672
    adapter_layer_norm_eps: float = 1e-06
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "full"
    eos_token_id: Optional[List[int]] = None
