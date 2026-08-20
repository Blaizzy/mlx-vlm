from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig
from ..gemma4.config import TextConfig as Gemma4TextConfig


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "gemma4_unified_audio"
    audio_samples_per_token: int = 640
    audio_embed_dim: int = 640
    hidden_size: int = 640
    output_proj_dims: int = 640
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "gemma4_unified_vision"
    patch_size: int = 16
    pooling_kernel_size: int = 3
    model_patch_size: int = 48
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    num_soft_tokens: int = 280
    rms_norm_eps: float = 1e-6
    output_proj_dims: int = 3840
    initializer_range: float = 0.02

    @property
    def hidden_size(self) -> int:
        return self.output_proj_dims


@dataclass
class TextConfig(Gemma4TextConfig):
    model_type: str = "gemma4_unified_text"
    hidden_size: int = 3840
    num_hidden_layers: int = 48
    intermediate_size: int = 15360
    num_attention_heads: int = 16
    head_dim: int = 256
    global_head_dim: int = 512
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 8
    num_global_key_value_heads: Optional[int] = 1
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    _sliding_window_pattern: int = 6
    attention_k_eq_v: bool = True
    use_double_wide_mlp: bool = False
    use_bidirectional_attention: Optional[str] = "vision"
    rope_parameters: Optional[Dict] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            }
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: Optional[VisionConfig] = field(default_factory=VisionConfig)
    audio_config: Optional[AudioConfig] = field(default_factory=AudioConfig)
    model_type: str = "gemma4_unified"
    vocab_size: int = 262144
    image_token_id: int = 258880
    audio_token_id: int = 258881
    video_token_id: Optional[int] = 258884
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    boa_token_id: int = 256000
    eoa_token_id: Optional[int] = None
    eoa_token_index: Optional[int] = 258883
    hidden_size: int = 3840
    pad_token_id: int = 0
    vision_soft_tokens_per_image: int = 280
    vision_soft_tokens_per_video_frame: int = 70
    audio_soft_tokens_per_image: int = 750
    audio_ms_per_token: int = 40
    eos_token_id: Optional[List[int]] = None
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        params = dict(params)
        if isinstance(params.get("text_config"), dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])
        if isinstance(params.get("vision_config"), dict):
            params["vision_config"] = VisionConfig.from_dict(params["vision_config"])
        if isinstance(params.get("audio_config"), dict):
            params["audio_config"] = AudioConfig.from_dict(params["audio_config"])
        if (
            params.get("eoa_token_id") is None
            and params.get("eoa_token_index") is not None
        ):
            params["eoa_token_id"] = params["eoa_token_index"]
        return super().from_dict(params)
