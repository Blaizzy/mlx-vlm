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
class MiniCPMTTSConfig(BaseModelConfig):
    model_type: str = "minicpmtts"
    llm_dim: int = 4096
    llm_hidden_size: Optional[int] = None
    llm_intermediate_size: int = 768
    llm_down_scale: bool = False
    llm_dim_model_base: int = 256
    projector_type: str = "mlp"
    hidden_act: str = "silu"
    aug_loss_weight: bool = False
    aug_layer_loss_weight: bool = False
    filter_tts_loss: bool = False
    tts_filter_loss_fix: bool = False
    long_weight: float = 0.1
    short_weight: float = 0.1
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 20
    num_key_value_heads: int = 12
    max_position_embeddings: int = 4096
    num_audio_tokens: int = 6562
    num_text_tokens: int = 152064
    num_mel_bins: int = 100
    num_vq: int = 1
    use_llm_hidden_state: bool = False
    audio_bos_token_id: int = 151687
    text_eos_token_id: int = 151692
    use_text: bool = True
    streaming: bool = False
    streaming_text_chunk_min: int = 3
    streaming_text_chunk_max: int = 7
    streaming_text_reserved_len: int = 300
    streaming_audio_chunk_size: int = 50
    attn_implementation: str = "sdpa"
    condition_type: str = "hidden_text_merge"
    backbone_model: str = "llama"
    backbone_vocab_size: int = 32000
    audio_tokenizer_type: str = "s3tokenizer"
    audio_tokenizer_sample_rate: int = 16000
    streaming_sliding_window: bool = False
    streaming_sliding_window_max_text_len: int = 500
    streaming_sliding_window_average_speed: int = 5
    streaming_sliding_window_fast_speed: int = 7
    streaming_sliding_window_slow_speed: int = 3
    streaming_sliding_window_audio_frame_rate: int = 50
    streaming_sliding_window_audio_init_text_length: int = 10
    streaming_sliding_window_audio_window_size: int = 300
    streaming_sliding_window_text_window_size: int = 50
    normalize_projected_hidden: bool = False
    interleaved: bool = False
    attention_type: str = "full_attention"
    recomputed_chunks: int = 1
    window_size: int = 2
    s3_stream_chunk_size: int = 25
    s3_stream_generate: bool = False
    s3_stream_n_timesteps: int = 10
    s3_stream_prelook_size: int = 3
    top_p: float = 0.85
    top_k: int = 25
    repetition_penalty: float = 1.05
    temperature: float = 0.8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.llm_hidden_size is None:
            self.llm_hidden_size = self.llm_dim


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: Optional[AudioConfig] = None
    tts_config: Optional[MiniCPMTTSConfig] = None
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

        tts_params = dict(params.pop("tts_config", {}))
        if source_params is not None:
            source_params["tts_config"] = dict(tts_params)
        tts_config = (
            MiniCPMTTSConfig.from_dict(tts_params) if len(tts_params) > 0 else None
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
            tts_config=tts_config,
            slice_config=slice_config,
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            },
        )
