from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "apertus1p5_text"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 21504
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    # Extended vocabulary: text + visual + audio code tokens.
    vocab_size: int = 266752
    max_position_embeddings: int = 262144
    post_norm: bool = False
    qk_norm: bool = True
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # The head is physically pruned to the text-only prefix of `vocab_size`.
    output_vocab_size: Optional[int] = None
    # Apertus 1.5 carries a single `rope_parameters` dict; the trunk reads
    # `rope_theta` / `rope_scaling`, which are derived from it below.
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 4000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            return
        self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        scaling = {k: v for k, v in self.rope_parameters.items() if k != "rope_theta"}
        # `initialize_rope` dispatches on `rope_scaling["rope_type"]`.
        scaling.setdefault("rope_type", scaling.pop("type", "llama3"))
        self.rope_scaling = scaling


@dataclass
class VisionConfig(BaseModelConfig):
    """IBQ image tokenizer (an encode-only port of the EMU3.5 vision tokenizer)."""

    model_type: str = "apertus1p5_vision_tokenizer"
    codebook_size: int = 131072
    embed_dim: int = 256
    latent_channels: int = 256
    in_channels: int = 3
    base_channels: int = 256
    channel_multiplier: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 4
    # Feature-map resolutions, relative to `resolution`, that carry attention.
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    # Reference resolution used to place the attention blocks. It does not
    # restrict the actual input size, which is dynamic.
    resolution: int = 256
    num_groups: int = 32
    norm_eps: float = 1e-6

    @property
    def spatial_scale_factor(self) -> int:
        return 2 ** (len(self.channel_multiplier) - 1)


@dataclass
class AudioConfig(BaseModelConfig):
    """WavTokenizer, encoder side only (Apertus 1.5 never synthesises audio)."""

    model_type: str = "wavtokenizer"
    audio_channels: int = 1
    hidden_size: int = 512
    num_filters: int = 32
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    num_residual_layers: int = 1
    num_lstm_layers: int = 2
    compress: int = 2
    upsampling_ratios: List[int] = field(default_factory=lambda: [6, 5, 5, 4])
    codebook_size: int = 4096
    codebook_dim: int = 512
    use_causal_conv: bool = False
    use_conv_shortcut: bool = True
    pad_mode: str = "reflect"
    norm_type: str = "weight_norm"
    sampling_rate: int = 24000

    @property
    def hop_length(self) -> int:
        hop = 1
        for ratio in self.upsampling_ratios:
            hop *= ratio
        return hop


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    # Named after the checkpoint keys rather than mlx-vlm's usual
    # `vision_config` / `audio_config`: these are code tokenizers, not towers,
    # and the names keep `update_module_configs` from overwriting them with
    # defaults built from the (absent) `vision_config` / `audio_config` keys.
    vision_tokenizer_config: VisionConfig = field(default_factory=VisionConfig)
    audio_tokenizer_config: AudioConfig = field(default_factory=AudioConfig)
    model_type: str = "apertus1p5"
    # Placeholder tokens in the prompt, one per discrete code.
    image_token_id: int = 131079
    audio_token_id: int = 131085
    # Code ids are shifted by these offsets to land in the shared vocabulary.
    image_token_offset: int = 131272
    audio_token_offset: int = 262344
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        sub_configs = {
            "text_config": TextConfig,
            "vision_tokenizer_config": VisionConfig,
            "audio_tokenizer_config": AudioConfig,
        }
        for key, config_class in sub_configs.items():
            value = params.get(key)
            if not isinstance(value, config_class):
                params[key] = config_class.from_dict(value or {})
        return super().from_dict(params)

    def __post_init__(self):
        expected_audio_offset = (
            self.image_token_offset + self.vision_tokenizer_config.codebook_size
        )
        if self.audio_token_offset != expected_audio_offset:
            raise ValueError(
                "Apertus 1.5 lays the code vocabularies out back to back, so "
                "`image_token_offset + codebook_size == audio_token_offset` must "
                f"hold, got {self.image_token_offset} + "
                f"{self.vision_tokenizer_config.codebook_size} != "
                f"{self.audio_token_offset}."
            )
        audio_vocab_end = (
            self.audio_token_offset + self.audio_tokenizer_config.codebook_size
        )
        if audio_vocab_end > self.text_config.vocab_size:
            raise ValueError(
                f"The audio codes end at {audio_vocab_end}, past the extended "
                f"vocabulary of {self.text_config.vocab_size} tokens."
            )
