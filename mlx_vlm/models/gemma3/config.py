from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262208
    num_key_value_heads: int = 4
    rope_global_base_freq: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_traditional: bool = False
    query_pre_attn_scalar: float = 256
    sliding_window: int = 1024
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 4096


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
