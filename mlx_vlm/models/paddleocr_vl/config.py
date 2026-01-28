import inspect
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "paddleocr_vl"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 384
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    spatial_merge_size: int = 2


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "paddleocr_vl"
    hidden_size: int = 1024
    num_hidden_layers: int = 18
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-05
    vocab_size: int = 103424
    num_key_value_heads: Optional[int] = 2
    max_position_embeddings: Optional[int] = 131072
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_cache: bool = True
    hidden_act: str = ("silu",)
    pad_token_id: int = (0,)
    bos_token_id: int = (1,)
    eos_token_id: int = (2,)
    use_bias: bool = (False,)
    head_dim: int = (128,)
    rope_parameters: Dict = None
    rope_scaling: Dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "type": "default",
            "mrope_section": [16, 24, 24],
        }
    )

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"mrope_section", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if not self.rope_scaling["type"] in ["mrope", "default"]:
                raise ValueError(f"rope_scaling type must be 'mrope' or 'default'")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "paddleocr_vl"
    ignore_index: int = -100
    image_token_id: int = 100295
    video_token_id: int = 100296
    vision_start_token_id: int = 101305
    vision_end_token_id: int = (101306,)
    eos_token_id: int = (2,)

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level
        excluded_keys = {"vision_config"}
        params["text_config"] = dict(
            filter(lambda x: x[0] not in excluded_keys, params.items())
        )

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
