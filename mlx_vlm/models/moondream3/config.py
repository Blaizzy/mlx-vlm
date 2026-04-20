import inspect
from dataclasses import dataclass, field
from typing import Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "moondream3"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    vocab_size: int = 51200
    max_position_embeddings: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 64
    rope_theta: float = 1500000.0
    rope_dim: int = 32
    rms_norm_eps: float = 1e-5
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1024
    moe_start_layer: int = 4
    attention_bias: bool = True
    prefix_attn: int = 730


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "moondream3_vision"
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
    layer_norm_eps: float = 1e-6


@dataclass
class RegionConfig(BaseModelConfig):
    hidden_size: int = 2048
    coord_feat_dim: int = 256
    coord_out_dim: int = 1024
    size_feat_dim: int = 512
    size_out_dim: int = 2048


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    region_config: Optional[RegionConfig] = None
    model_type: str = "moondream3"
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
        if isinstance(self.region_config, dict):
            self.region_config = RegionConfig(
                **{
                    k: v
                    for k, v in self.region_config.items()
                    if k in inspect.signature(RegionConfig).parameters
                }
            )
