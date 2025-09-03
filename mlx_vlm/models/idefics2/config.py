from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float = 1e-6
    num_channels: int = 3


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class PerceiverConfig(BaseModelConfig):
    model_type: str
    num_key_value_heads: int = 4
    resampler_depth: int = 3
    resampler_head_dim: int = 96
    resampler_n_heads: int = 16
    resampler_n_latents: int = 64


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    perceiver_config: PerceiverConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 32001
    vocab_size: int = 151936
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
