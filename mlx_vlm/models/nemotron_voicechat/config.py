from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_audio.codec.models.nemotron_voicechat import (
    NemotronVoiceChatCodecConfig as CodecConfig,
)
from mlx_audio.stt.models.nemotron_asr.config import (
    ConformerArgs,
    JointArgs,
    PredictArgs,
    PreprocessArgs,
)

from ..nemotron_h.config import ModelConfig as TextConfig


@dataclass
class AudioConfig:
    preprocessor: PreprocessArgs = field(default_factory=PreprocessArgs)
    encoder: ConformerArgs = field(
        default_factory=lambda: ConformerArgs(att_context_size=[[70, 0]])
    )
    decoder: PredictArgs = field(default_factory=lambda: PredictArgs(vocab_size=1024))
    joint: JointArgs = field(default_factory=lambda: JointArgs(num_classes=1024))
    output_dim: int = 4480
    max_symbols: int = 10

    @classmethod
    def from_dict(cls, config: dict | None) -> "AudioConfig":
        values = dict(config or {})
        preprocessor = _dataclass_from_dict(
            PreprocessArgs, values.get("preprocessor", {})
        )
        encoder_values = dict(values.get("encoder", {}))
        if "att_context_size" in encoder_values:
            context = encoder_values["att_context_size"]
            if context and isinstance(context[0], int):
                encoder_values["att_context_size"] = [context]
        encoder = _dataclass_from_dict(ConformerArgs, encoder_values)
        decoder = _dataclass_from_dict(PredictArgs, values.get("decoder", {}))
        joint = _dataclass_from_dict(JointArgs, values.get("joint", {}))
        return cls(
            preprocessor=preprocessor,
            encoder=encoder,
            decoder=decoder,
            joint=joint,
            output_dim=values.get("output_dim", 4480),
            max_symbols=values.get("max_symbols", 10),
        )


@dataclass
class CharacterEncoderConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4608
    num_hidden_layers: int = 1
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256.0
    attn_logit_softcapping: float = 50.0
    rope_base: float = 10_000.0
    char_vocab_size: int = 257

    @classmethod
    def from_dict(cls, config: dict | None) -> "CharacterEncoderConfig":
        return _dataclass_from_dict(cls, config or {})


@dataclass
class MoGConfig:
    intermediate_size: int = 4608
    low_rank: int = 64
    min_log_std: float = -4.0
    num_layers: int = 3
    num_predictions: int = 1024
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, config: dict | None) -> "MoGConfig":
        return _dataclass_from_dict(cls, config or {})


@dataclass
class TTSConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4608
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    sliding_window: int = 7500
    sliding_window_pattern: int = 6
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256.0
    rope_global_base_freq: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    latent_size: int = 512
    num_quantizers: int = 31
    codebook_size: int = 1024
    num_delay_speech_tokens: int = 2
    num_iterations: int = 8
    guidance_scale: float = 0.2
    top_p: float = 0.95
    noise_scale: float = 0.001
    exponent: float = 3.0
    disable_eos_prediction: bool = True
    use_gated_fusion_for_text_audio: bool = True
    use_subword_flag_emb: bool = True
    use_bos_eos_emb: bool = True
    use_audio_prompt_frozen_projection: bool = True
    audio_prompt_duration: float = 3.0
    character_encoder: CharacterEncoderConfig = field(
        default_factory=CharacterEncoderConfig
    )
    mog_head: MoGConfig = field(default_factory=MoGConfig)

    @classmethod
    def from_dict(cls, config: dict | None) -> "TTSConfig":
        values = dict(config or {})
        values["character_encoder"] = CharacterEncoderConfig.from_dict(
            values.get("character_encoder")
        )
        values["mog_head"] = MoGConfig.from_dict(values.get("mog_head"))
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass
class ModelConfig:
    text_config: TextConfig
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    codec_config: CodecConfig = field(default_factory=CodecConfig)
    model_type: str = "nemotron_voicechat"
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 12
    silence_token_id: int = 11
    rnnt_blank_id: int = 1024
    input_sample_rate: int = 16_000
    output_sample_rate: int = 22_050
    frame_duration: float = 0.08
    function_channel_weight: float = 2.0
    default_system_prompt: str = ""
    rnnt_vocabulary: list[str] = field(default_factory=list)
    speaker: str = "Aria"
    source_revision: str | None = None
    base_tokenizer_revision: str | None = None
    quantization: dict[str, Any] | None = None
    quantization_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        text_config = config.get("text_config")
        if isinstance(text_config, TextConfig):
            text = text_config
        elif text_config:
            text = TextConfig.from_dict(text_config)
        else:
            raise ValueError("Nemotron VoiceChat requires text_config")

        codec_values = config.get("codec_config", {})
        codec = (
            codec_values
            if isinstance(codec_values, CodecConfig)
            else CodecConfig.from_dict(codec_values)
        )
        values = {
            "text_config": text,
            "audio_config": AudioConfig.from_dict(config.get("audio_config")),
            "tts_config": TTSConfig.from_dict(config.get("tts_config")),
            "codec_config": codec,
        }
        for key in cls.__dataclass_fields__:
            if key not in values and key in config:
                values[key] = config[key]
        return cls(**values)


def _dataclass_from_dict(cls, values: dict):
    known = cls.__dataclass_fields__
    return cls(**{key: value for key, value in values.items() if key in known})
