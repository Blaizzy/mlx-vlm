import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "granitemoehybrid"
    hidden_size: int = 2560
    num_hidden_layers: int = 40
    intermediate_size: int = 8192
    shared_intermediate_size: int = 8192
    num_attention_heads: int = 40
    rms_norm_eps: float = 1e-5
    vocab_size: int = 100353
    num_key_value_heads: int = 8
    rope_theta: float = 10000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    embedding_multiplier: float = 12.0
    attention_multiplier: float = 0.015625
    residual_multiplier: float = 0.22
    logits_scaling: float = 10.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "granite4_vision"
    image_token_index: int = 100352
    vision_feature_select_strategy: str = "full"
    image_grid_pinpoints: Optional[List[List[int]]] = None
    vocab_size: int = 100353
    ignore_index: int = -100
    projector_hidden_act: str = "gelu"
    projector_dropout: float = 0.1
    downsample_rate: str = "4/8"
    deepstack_layer_map: Optional[List[List[int]]] = None
    use_spatial_sampling: bool = True
    spatial_stride: int = 2
    spatial_vision_layer: int = -1
    spatial_target_layers: Optional[List[int]] = None
    use_image_newline_parameter: bool = True
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
