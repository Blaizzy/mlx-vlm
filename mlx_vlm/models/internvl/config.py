from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig
from ..qwen3.config import ModelConfig as TextConfig


def _dimension(value):
    return int(value[0] if isinstance(value, (list, tuple)) else value)


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    image_size: int = 448
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    layer_scale_init_value: float = 0.1
    norm_type: str = "layer_norm"
    attention_bias: bool = True
    use_qk_norm: bool = False
    use_absolute_position_embeddings: bool = True
    use_mean_pooling: bool = True
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["image_size"] = _dimension(params.get("image_size", 448))
        params["patch_size"] = _dimension(params.get("patch_size", 14))
        return super().from_dict(params)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    downsample_ratio: float = 0.5
    image_token_id: int = 151671
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "default"
    eos_token_id: Optional[List[int]] = None
