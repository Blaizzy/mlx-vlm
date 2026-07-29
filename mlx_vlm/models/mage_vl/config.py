import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


def _kw(cls, params):
    return {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "mage_vl_vision"
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    layer_norm_type: str = "layer_norm"
    rope_theta: float = 10000.0
    use_head: bool = False
    out_hidden_size: int = 2560
    spatial_merge_size: int = 2
    frame_windows_size: int = 4
    use_patch_position_encoding: bool = False
    skip_vision: bool = False


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    rope_theta: float = 5000000.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "mage_vl"
    image_token_id: int = 151655
    video_token_id: int = 151656
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vocab_size: int = 151936
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        vc = params.get("vision_config") or {}
        tc = params.get("text_config") or {}
        params["vision_config"] = (
            vc if not isinstance(vc, dict) else VisionConfig(**_kw(VisionConfig, vc))
        )
        params["text_config"] = (
            tc if not isinstance(tc, dict) else TextConfig(**_kw(TextConfig, tc))
        )
        return cls(**_kw(cls, params))
