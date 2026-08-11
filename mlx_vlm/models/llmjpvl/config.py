from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    intermediate_size: int = 4304
    patch_size: int = 16
    image_size: int = 512
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "llama"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 196608
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "llmjpvl"
    text_config: Optional[TextConfig] = None
    vision_config: Optional[VisionConfig] = None
    image_token_index: int = 14
    video_token_index: int = -1
    downsample_ratio: float = 0.5
    select_layer: int = -1
    vocab_size: int = 196608
    ignore_index: int = -100
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        # Map upstream config's img_context_token_id to image_token_index
        if (
            "img_context_token_id" in params
            and params["img_context_token_id"] is not None
        ):
            params["image_token_index"] = params["img_context_token_id"]
        import inspect as _inspect

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in _inspect.signature(cls).parameters
            }
        )
