import inspect
from dataclasses import dataclass, field

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "moondream2"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    vocab_size: int = 51200
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    partial_rotary_factor: float = 0.5
    rms_norm_eps: float = 1e-5


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "moondream2_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    patch_size: int = 14
    crop_size: int = 378
    max_crops: int = 12
    overlap_margin: int = 4
    in_channels: int = 3
    proj_inner_dim: int = 8192
    proj_out_dim: int = 2048
    attention_bias: bool = True
    layer_norm_eps: float = 1e-5


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "moondream2"
    eos_token_id: int = 0
    bos_token_id: int = 0

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig(
                **{
                    k: v
                    for k, v in self.text_config.items()
                    if k in inspect.signature(TextConfig).parameters
                }
            )
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig(
                **{
                    k: v
                    for k, v in self.vision_config.items()
                    if k in inspect.signature(VisionConfig).parameters
                }
            )
