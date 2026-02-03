from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "glm_ocr_text"
    vocab_size: int = 59392
    hidden_size: int = 1536
    eos_token_id: List[int] = field(
        default_factory=lambda: [59246, 59253]
    )
    intermediate_size: int = 4608
    max_position_embeddings: int = 131072
    num_attention_heads: int = 16
    num_hidden_layers: int = 17  # GLM-OCR has 17 layers (16 + 1 nextn)
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    partial_rotary_factor: float = 1.0
    rope_scaling: Dict = field(
        default_factory=lambda: {"rope_type": "default", "mrope_section": [16, 24, 24]}
    )
    pad_token_id: int = 59246
    use_cache: bool = True


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm4v_vision"  # Must be glm4v_vision for compatibility
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    patch_size: int = 14
    window_size: int = 112
    image_size: int = 336
    in_channels: int = 3
    rms_norm_eps: float = 1e-05
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    out_hidden_size: int = 1536
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "glm_ocr"
    vocab_size: int = 59392
    ignore_index: int = -100
    image_token_index: int = 59256
    image_token_id: int = 59280
    video_token_index: int = 59258
    video_token_id: int = 59281
    vision_start_token_id: int = 59256
    vision_end_token_id: int = 59257
    hidden_size: int = 1536
    pad_token_id: int = 59246
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.eos_token_id is None:
            text_config = (
                asdict(self.text_config)
                if isinstance(self.text_config, TextConfig)
                else self.text_config
            )
            self.eos_token_id = text_config["eos_token_id"]
