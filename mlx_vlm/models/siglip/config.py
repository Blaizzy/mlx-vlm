from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    max_position_embeddings: int = 64
    layer_norm_eps: float = 1e-6
    projection_size: Optional[int] = None
    model_type: str = "siglip_text_model"

    def __post_init__(self):
        if self.projection_size is None:
            self.projection_size = self.hidden_size


@dataclass
class VisionConfig(BaseModelConfig):
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    model_type: str = "siglip_vision_model"


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "siglip"
