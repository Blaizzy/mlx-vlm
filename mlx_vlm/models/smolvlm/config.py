# SmolVLM uses the same configuration structure as Idefics3
# We define our own configs with defaults for SmolVLM
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "smolvlm"
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    num_key_value_heads: int = 8
    rope_theta: float = 1000000.0
    num_hidden_layers: int = 32
    rope_traditional: bool = False
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    num_attention_heads: int = 16
    patch_size: int = 14
    num_hidden_layers: int = 27
    intermediate_size: int = 4304
    image_size: int = 384
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "smolvlm"
    ignore_index: int = -100
    vocab_size: int = 49152
    scale_factor: int = 2
    image_token_id: int = 49153
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]
