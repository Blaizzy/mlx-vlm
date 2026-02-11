from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "glm_ocr_text"
    vocab_size: int = 59392
    hidden_size: int = 1536
    eos_token_id: List[int] = field(default_factory=lambda: [59246, 59253])
    intermediate_size: int = 4608
    max_position_embeddings: int = 131072
    num_attention_heads: int = 16
    num_hidden_layers: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-05
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rope_parameters: Dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000,
        }
    )
    pad_token_id: int = 59246
    use_cache: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1
    dtype: str = "bfloat16"

    @property
    def rope_theta(self) -> float:
        return self.rope_parameters.get("rope_theta", 10000)

    @property
    def partial_rotary_factor(self) -> float:
        return self.rope_parameters.get("partial_rotary_factor", 1.0)


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm_ocr_vision"
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    patch_size: int = 14
    image_size: int = 336
    in_channels: int = 3
    rms_norm_eps: float = 1e-05
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    hidden_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    out_hidden_size: int = 1536
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "glm_ocr"
    vocab_size: int = 59392
    ignore_index: int = -100
    image_token_id: int = 59280
    video_token_id: int = 59281
    image_start_token_id: int = 59256
    image_end_token_id: int = 59257
    video_start_token_id: int = 59258
    video_end_token_id: int = 59259
    hidden_size: int = 1536
    pad_token_id: int = 59246
    eos_token_id: Optional[List[int]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig(**self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig(**self.vision_config)

        if self.eos_token_id is None and self.text_config is not None:
            self.eos_token_id = self.text_config.eos_token_id
