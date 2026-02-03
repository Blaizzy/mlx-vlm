from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    """Text config for GLM-OCR (0.9B params)"""
    model_type: str = "glm_ocr_text"
    vocab_size: int = 59392
    hidden_size: int = 1536
    eos_token_id: List[int] = field(default_factory=lambda: [59246, 59253])
    intermediate_size: int = 4608
    max_position_embeddings: int = 131072
    num_attention_heads: int = 16
    num_hidden_layers: int = 16
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
    dtype: str = "bfloat16"


@dataclass
class VisionConfig(BaseModelConfig):
    """Vision config for GLM-OCR"""
    model_type: str = "glm_ocr_vision"
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 14
    image_size: int = 336
    window_size: int = 112
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
    """Configuration for GLM-OCR model.
    
    GLM-OCR (0.9B) is a multimodal OCR model for complex document understanding.
    
    Reference: https://huggingface.co/zai-org/GLM-OCR
    """
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "glm_ocr"
    vocab_size: int = 59392
    ignore_index: int = -100
    image_start_token_id: int = 59256
    image_end_token_id: int = 59257
    video_start_token_id: int = 59258
    video_end_token_id: int = 59259
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
