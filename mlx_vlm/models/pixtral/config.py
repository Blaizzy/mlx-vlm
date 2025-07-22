import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig" = field(default_factory=lambda: TextConfig())
    vision_config: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    model_type: str = "pixtral"
    ignore_index: int = -100
    image_token_index: int = 10
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "mistral"
    hidden_size: int = 5120
    head_dim: int = 128
    num_hidden_layers: int = 40
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-06
    vocab_size: int = 131072
    num_key_value_heads: int = 8
    rope_theta: float = 1000000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    max_position_embeddings: int = 4096

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "pixtral"
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    head_dim: int = 64
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
