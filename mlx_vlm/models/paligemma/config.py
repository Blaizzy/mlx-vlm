import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "paligemma"
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0
    eos_token_id: Optional[List[int]] = None


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "paligemma"
    hidden_size: int = 2048
    num_hidden_layers: int = 18
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    vocab_size: int = 256000
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    rope_traditional: bool = False
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None
    max_position_embeddings: int = 4096


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    patch_size: int = 14
    projection_dim: int = 2048
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
