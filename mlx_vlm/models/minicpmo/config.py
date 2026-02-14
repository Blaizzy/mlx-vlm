import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class SliceConfig(BaseModelConfig):
    model_type: str = "minicpmv"
    patch_size: int = 14
    max_slice_nums: int = 9
    scale_resolution: int = 448


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.model_type == "siglip":
            self.model_type = "siglip_vision_model"


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int]
    head_dim: int
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: Optional[Dict[str, Union[float, str, bool, list[int]]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling is None:
            # Keep the qwen3_vl language implementation stable by always passing
            # a valid mRoPE config. For 1D text positions, all three dimensions
            # are identical so this reduces to standard rotary behavior.
            self.rope_scaling = {"type": "default", "mrope_section": [24, 20, 20]}
        elif "type" not in self.rope_scaling and "rope_type" in self.rope_scaling:
            self.rope_scaling["type"] = self.rope_scaling.pop("rope_type")


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "whisper"
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: Optional[AudioConfig] = None
    model_type: str = "minicpmo"
    query_num: int = 64
    image_size: int = 448
    patch_size: int = 14
    init_vision: bool = True
    init_audio: bool = True
    init_tts: bool = True
    batch_vision_input: bool = True
    vision_batch_size: int = 16
    audio_pool_step: int = 5
    audio_chunk_length: float = 1.0
    stream_input: bool = True
    slice_mode: bool = True
    slice_config: Optional[SliceConfig] = None
    eos_token_id: Optional[list[int]] = None

    @classmethod
    def from_dict(cls, params):
        source_params = params if isinstance(params, dict) else None
        params = dict(params)

        # MiniCPM-o config keeps most LLM fields at the root. Build text_config
        # from root values when explicit text_config is absent.
        text_params = params.pop("text_config", None)
        if not text_params:
            text_fields = {
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "rms_norm_eps",
                "vocab_size",
                "num_key_value_heads",
                "head_dim",
                "rope_theta",
                "max_position_embeddings",
                "rope_scaling",
                "tie_word_embeddings",
                "attention_bias",
                "hidden_act",
            }
            text_params = {k: v for k, v in params.items() if k in text_fields}
        if source_params is not None:
            source_params["text_config"] = dict(text_params)
        text_config = TextConfig.from_dict(text_params)

        vision_params = dict(params.pop("vision_config", {}))
        if vision_params.get("model_type") == "siglip":
            vision_params["model_type"] = "siglip_vision_model"
        if source_params is not None:
            source_params["vision_config"] = dict(vision_params)
        vision_config = VisionConfig.from_dict(vision_params)

        audio_params = dict(params.pop("audio_config", {}))
        if source_params is not None:
            source_params["audio_config"] = dict(audio_params)
        audio_config = (
            AudioConfig.from_dict(audio_params) if len(audio_params) > 0 else None
        )

        slice_params = params.pop("slice_config", None)
        slice_config = (
            SliceConfig.from_dict(slice_params)
            if isinstance(slice_params, dict)
            else slice_params
        )

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            slice_config=slice_config,
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            },
        )
