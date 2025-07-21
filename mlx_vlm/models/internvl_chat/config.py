from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    hidden_size: int = 1024
    num_attention_heads: int = 16
    patch_size: int = 14
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    image_size: int = 448
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    drop_path_rate: float = 0.1
    qkv_bias: bool = True
    qk_normalization: bool = False
    norm_type: str = "layer_norm"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_window_layers: int
    hidden_act: str
    num_key_value_heads: Optional[int] = 8
    max_position_embeddings: Optional[int] = 40960
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    sliding_window: int = 32768
    use_sliding_window: bool = False
    use_cache: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 151667
    video_token_index: int = 151656
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    downsample_ratio: float = 0.5
    eos_token_id: Optional[List[int]] = None
