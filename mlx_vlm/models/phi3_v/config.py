import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "phi3_v"
    vocab_size: int = 32064

    num_hidden_layers: int = 32
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5

    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0

    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    eos_token_id: Optional[List[int]] = None


@dataclass
class TextConfig(BaseModelConfig):
    max_position_embeddings: int = 4096


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "phi3_v"
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    image_dim_out: int = (1024,)
    model_name: str = "openai/clip-vit-large-patch14-336"
    name: str = "clip_vision_model"
    num_img_tokens: int = 144
