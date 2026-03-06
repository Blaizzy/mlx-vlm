import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig

# Fields shared between ModelConfig (top-level) and TextConfig
_TEXT_FIELDS = [
    "vocab_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "hidden_size",
    "num_key_value_heads",
    "rope_theta",
    "rope_traditional",
    "partial_rotary_factor",
    "rope_scaling",
    "model_type",
]


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "phi4-siglip"
    vocab_size: int = 100352
    num_hidden_layers: int = 40
    intermediate_size: int = 17920
    num_attention_heads: int = 40
    rms_norm_eps: float = 1e-5
    hidden_size: int = 5120
    num_key_value_heads: int = 10
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    partial_rotary_factor: float = 1.0
    rope_scaling: Optional[Dict[str, Union[float, str, List[float]]]] = None


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip2_vision_model"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 512
    patch_size: int = 16
    num_patches: int = 256
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "phi4-siglip"
    max_position_embeddings: int = 32768
    original_max_position_embeddings: int = 32768
    mm_hidden_size: int = 1152
    mm_projector_type: str = "mlp2x_gelu"
    mm_vision_tower: Optional[str] = None
    mm_vision_select_layer: int = -2
    min_num_patches: int = 256
    max_num_patches: int = 3600
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    tokenizer_model_max_length: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig(**self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig(**self.vision_config)

    @classmethod
    def from_dict(cls, params):
        text_params = {k: params[k] for k in _TEXT_FIELDS if k in params}
        if text_params:
            tc = params.get("text_config", {})
            if isinstance(tc, dict):
                params = {**params, "text_config": {**tc, **text_params}}

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
