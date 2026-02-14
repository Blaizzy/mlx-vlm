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
    num_attention_heads: Optional[int] = None
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_theta: float = 1000000.0
    num_hidden_layers: int = 32
    rope_traditional: bool = False
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_attention_heads is None:
            if self.head_dim is not None and self.head_dim > 0:
                self.num_attention_heads = self.hidden_size // self.head_dim
            else:
                self.num_attention_heads = 32
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    patch_size: int = 14
    num_hidden_layers: Optional[int] = None
    intermediate_size: Optional[int] = None
    image_size: int = 384
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        if self.hidden_size is None:
            self.hidden_size = 1152

        if self.num_attention_heads is None:
            if self.hidden_size % 64 == 0:
                self.num_attention_heads = self.hidden_size // 64
            else:
                self.num_attention_heads = 16

        if self.num_hidden_layers is None:
            # SmolVLM2 variants use either the 12-layer (500M) or 27-layer (2.2B) vision tower.
            self.num_hidden_layers = 12 if self.hidden_size <= 768 else 27

        if self.intermediate_size is None:
            if self.hidden_size == 768:
                self.intermediate_size = 3072
            elif self.hidden_size == 1152:
                self.intermediate_size = 4304
            else:
                self.intermediate_size = self.hidden_size * 4


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
