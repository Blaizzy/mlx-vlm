from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig
from ..lfm2_vl.config import VisionConfig


@dataclass
class TextConfig(BaseModelConfig):
    vocab_size: int = 65536
    hidden_size: int = 1024
    intermediate_size: int = 6656
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: Optional[List[str]] = None
    norm_eps: float = 1e-5
    conv_L_cache: int = 3
    block_ffn_dim_multiplier: float = 1.0
    block_multiple_of: int = 256
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0


@dataclass
class AudioConfig(BaseModelConfig):
    feat_in: int = 128
    n_layers: int = 17
    d_model: int = 512
    subsampling_conv_channels: int = 256
    ff_expansion_factor: int = 4
    n_heads: int = 8
    conv_kernel_size: int = 9
    residual_width: int = 512


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    model_type: str = "d1_omni"
    projector_hidden_size: int = 2048
    head_layers: int = 2
    max_length: int = 16384
    image_text_length: int = 896
    audio_text_length: int = 15360
    temperatures: Dict[str, float] = field(default_factory=dict)
    bos_token_id: int = 1
    pad_token_id: int = 0

    def __post_init__(self):
        for name, cls in (
            ("text_config", TextConfig),
            ("vision_config", VisionConfig),
            ("audio_config", AudioConfig),
        ):
            value = getattr(self, name)
            if value is None or isinstance(value, dict):
                setattr(self, name, cls.from_dict(value or {}))
        if self.text_config.layer_types is None:
            self.text_config.layer_types = ["conv"] * self.text_config.num_hidden_layers
