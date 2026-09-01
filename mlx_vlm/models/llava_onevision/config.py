from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "qwen2"
    hidden_size: int = 896
    num_hidden_layers: int = 24
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 2
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 32768
    # Qwen2's own default: the 0.5b checkpoints opt in explicitly, the 7b/72b do not
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    num_hidden_layers: int = 26
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    vision_use_head: bool = False  # onevision checkpoints ship no pooling head


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "llava_onevision"
    image_token_index: int = 151646
    video_token_index: int = 151647
    image_grid_pinpoints: Optional[List[List[int]]] = None
    vision_aspect_ratio: str = "anyres_max_9"
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: Union[int, List[int]] = -1
    use_image_newline_parameter: bool = True
    projector_hidden_act: str = "gelu"
    ignore_index: int = -100
    vocab_size: int = 151936
    eos_token_id: Optional[List[int]] = None
