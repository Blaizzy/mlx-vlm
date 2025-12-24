from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..pixtral import VisionConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    rope_theta: float = None
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None  # For Ministral3
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        # Auto-detect QK norm for Qwen3-based models if not explicitly set
        if self.use_qk_norm is None:
            self.use_qk_norm = self.model_type in ("qwen3",)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = None
    image_token_id: int = None
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    spatial_merge_size: int = 2
    multimodal_projector_bias: bool = False
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
