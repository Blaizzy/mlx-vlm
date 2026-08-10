import inspect
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "apertus1p5_text"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 21504
    mlp_bias: bool = False
    num_attention_heads: int = 32
    attention_bias: bool = False
    rms_norm_eps: float = 1e-5
    vocab_size: int = 266752
    output_vocab_size: Optional[int] = 131072
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    post_norm: bool = False
    qk_norm: bool = True
    tie_word_embeddings: bool = False
    rope_traditional: bool = False
    rope_parameters: Optional[Dict[str, Union[float, str]]] = field(
        default_factory=lambda: {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 4000000,
            "rope_type": "llama3",
        }
    )
    rope_theta: float = 4000000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if not self.qk_norm:
            raise ValueError("Apertus 1.5 requires qk_norm=True.")
        if self.post_norm:
            raise ValueError("Apertus 1.5 requires post_norm=False.")
        if self.attention_bias:
            raise ValueError("Apertus 1.5 requires attention_bias=False.")

        if self.output_vocab_size is not None:
            if not 1 <= self.output_vocab_size <= self.vocab_size:
                raise ValueError(
                    "output_vocab_size must be in [1, vocab_size], got "
                    f"{self.output_vocab_size} for vocab_size={self.vocab_size}"
                )
            if self.output_vocab_size == self.vocab_size:
                self.output_vocab_size = None
            elif self.tie_word_embeddings:
                raise ValueError(
                    "A pruned output vocabulary cannot be tied to the extended "
                    "input embeddings."
                )

        if self.rope_parameters is not None:
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )
            scaling = {
                key: value
                for key, value in self.rope_parameters.items()
                if key != "rope_theta"
            }
            if "rope_type" not in scaling:
                scaling["rope_type"] = scaling.pop("type", "llama3")
            self.rope_scaling = scaling


@dataclass
class VisionTokenizerConfig(BaseModelConfig):
    model_type: str = "apertus1p5_vision_tokenizer"
    codebook_size: int = 131072
    embed_dim: int = 256
    latent_channels: int = 256
    in_channels: int = 3
    base_channels: int = 256
    channel_multiplier: tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 4
    attn_resolutions: tuple[int, ...] = (16,)
    resolution: int = 256
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        for name in ("channel_multiplier", "attn_resolutions"):
            if name in params:
                params[name] = tuple(params[name])
        return super().from_dict(params)

    @property
    def spatial_scale_factor(self) -> int:
        return 2 ** (len(self.channel_multiplier) - 1)


@dataclass
class AudioTokenizerConfig(BaseModelConfig):
    model_type: str = "wavtokenizer"
    audio_channels: int = 1
    num_filters: int = 32
    upsampling_ratios: tuple[int, ...] = (6, 5, 5, 4)
    num_residual_layers: int = 1
    dilation_growth_rate: int = 2
    compress: int = 2
    use_conv_shortcut: bool = True
    use_causal_conv: bool = False
    pad_mode: str = "reflect"
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    num_lstm_layers: int = 2
    hidden_size: int = 512
    codebook_size: int = 4096
    codebook_dim: int = 512
    sampling_rate: int = 24000

    def __post_init__(self):
        # Weight normalization is fused into plain conv weights in sanitize;
        # time_group_norm checkpoints carry norm modules this port does not
        # instantiate, so their weights would silently go unused.
        if self.norm_type != "weight_norm":
            raise ValueError(
                "The Apertus 1.5 audio tokenizer port supports "
                f'norm_type="weight_norm" only, got {self.norm_type!r}.'
            )
        if self.pad_mode not in ("reflect", "constant", "zero"):
            raise ValueError(
                'The audio tokenizer pad_mode must be "reflect", "constant" '
                f'or "zero", got {self.pad_mode!r}.'
            )
        if self.codebook_dim != self.hidden_size:
            raise ValueError(
                "WavTokenizer uses no projections around the quantizer, so "
                f"codebook_dim ({self.codebook_dim}) must equal hidden_size "
                f"({self.hidden_size})."
            )

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        if "upsampling_ratios" in params:
            params["upsampling_ratios"] = tuple(params["upsampling_ratios"])
        return super().from_dict(params)

    @property
    def hop_length(self) -> int:
        hop = 1
        for ratio in self.upsampling_ratios:
            hop *= ratio
        return hop


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_tokenizer_config: VisionTokenizerConfig = field(
        default_factory=VisionTokenizerConfig
    )
    audio_tokenizer_config: Optional[AudioTokenizerConfig] = None
    model_type: str = "apertus1p5"
    image_token_id: int = 131079
    audio_token_id: int = 131085
    image_token_offset: int = 131272
    audio_token_offset: int = 262344
    tie_word_embeddings: bool = False
    eos_token_id: Optional[list[int]] = None

    def __post_init__(self):
        self._validate_token_ranges()

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        top_level_tie = bool(params.get("tie_word_embeddings", False))
        text_params = params.pop("text_config", {})
        if isinstance(text_params, TextConfig):
            if top_level_tie and not text_params.tie_word_embeddings:
                text_params = TextConfig.from_dict(
                    {**text_params.to_dict(), "tie_word_embeddings": True}
                )
            text_config = text_params
        else:
            text_params = dict(text_params or {})
            text_params["tie_word_embeddings"] = bool(
                text_params.get("tie_word_embeddings", False) or top_level_tie
            )
            # The released checkpoints store eos_token_id inside text_config,
            # but the shared stopping criteria read it from the top-level
            # config (see utils.load), so hoist it when only nested.
            if params.get("eos_token_id") is None:
                params["eos_token_id"] = text_params.get("eos_token_id")
            text_config = TextConfig.from_dict(text_params)

        # Transformers treats either the top-level or nested flag as enabling
        # tying. Keep both views synchronized because the language model reads
        # the nested text configuration.
        params["tie_word_embeddings"] = text_config.tie_word_embeddings
        if isinstance(params.get("eos_token_id"), int):
            params["eos_token_id"] = [params["eos_token_id"]]
        vision_params = params.pop("vision_tokenizer_config", {})
        vision_tokenizer_config = (
            vision_params
            if isinstance(vision_params, VisionTokenizerConfig)
            else VisionTokenizerConfig.from_dict(vision_params)
        )
        audio_params = params.pop("audio_tokenizer_config", None)
        if audio_params is None or isinstance(audio_params, AudioTokenizerConfig):
            audio_tokenizer_config = audio_params
        else:
            audio_tokenizer_config = AudioTokenizerConfig.from_dict(audio_params)

        config = cls(
            text_config=text_config,
            vision_tokenizer_config=vision_tokenizer_config,
            audio_tokenizer_config=audio_tokenizer_config,
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            },
        )
        return config

    def _validate_token_ranges(self):
        visual_end = (
            self.image_token_offset + self.vision_tokenizer_config.codebook_size
        )
        if visual_end != self.audio_token_offset:
            raise ValueError(
                "The visual token range must end at audio_token_offset, got "
                f"{visual_end} and {self.audio_token_offset}."
            )

        audio_codebook_size = 4096
        if self.audio_tokenizer_config is not None:
            audio_codebook_size = self.audio_tokenizer_config.codebook_size
        if self.audio_token_offset + audio_codebook_size > self.text_config.vocab_size:
            raise ValueError("The audio token range exceeds the input vocabulary.")
