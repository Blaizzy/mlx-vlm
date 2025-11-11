from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "glm4v_text"
    vocab_size: int = 151552
    hidden_size: int = 4096
    eos_token_id: List[int] = field(
        default_factory=lambda: [151329, 151336, 151338, 151348]
    )
    intermediate_size: int = 13696
    max_position_embeddings: int = 65536
    num_attention_heads: int = 32
    num_hidden_layers: int = 40
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    partial_rotary_factor: float = 0.5
    rope_scaling: Dict = field(
        default_factory=lambda: {"rope_type": "default", "mrope_section": [8, 12, 12]}
    )
    pad_token_id: int = 151329
    use_cache: bool = True


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    patch_size: int
    window_size: int = 112
    image_size: int = 336
    in_channels: int = 3
    rms_norm_eps: float = 1e-05
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    out_hidden_size: int = 4096
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 151363
    image_token_id: int = 151363
    video_token_index: int = 151364
    video_token_id: int = 151364
    vision_start_token_id: int = 151339
    vision_end_token_id: int = 151340
    hidden_size: int = 2048
    pad_token_id: int = 0
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.eos_token_id is None:
            text_config = (
                asdict(self.text_config)
                if isinstance(self.text_config, TextConfig)
                else self.text_config
            )
            self.eos_token_id = text_config["eos_token_id"]
