# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass, field

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    post_norm_eps: float
    rope_parameters: dict
    # A 0 marks a layer without RoPE.
    layer_rope_theta: list[float]
    layer_types: list[str]
    sliding_window: int
    final_logit_softcapping: float
    output_multiplier: float
    num_local_experts: int
    num_experts_per_tok: int
    moe_hidden_size: int
    moe_intermediate_size: int
    qk_scale_factor: float = 5.4730064863875
    routed_scaling_factor: float = 1.0
    num_shared_experts: int = 0
    mlp_layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.num_shared_experts or any(
            t != "sparse" for t in self.mlp_layer_types or []
        ):
            raise NotImplementedError("Dense feed-forward layers are not supported.")
        self.rope_theta = self.rope_parameters["rope_theta"]
        if (
            len(self.layer_rope_theta) < self.num_hidden_layers
            or len(self.layer_types) < self.num_hidden_layers
        ):
            raise ValueError("Text layer schedules must cover num_hidden_layers.")
        self.layer_rope_theta = self.layer_rope_theta[: self.num_hidden_layers]
        self.layer_types = self.layer_types[: self.num_hidden_layers]
        if not 0 < self.num_experts_per_tok <= self.num_local_experts:
            raise ValueError(
                "num_experts_per_tok must be between 1 and num_local_experts."
            )


@dataclass
class VisionConfig(BaseModelConfig):
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    # The position table; images can be larger or smaller.
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    layer_norm_eps: float = 1e-5
    rope_parameters: dict = field(default_factory=dict)
    # window_attention attends inside pos_emb blocks of patches.
    layer_types: list[str] | None = None
    use_ij_pos_encoding: bool = False

    @classmethod
    def from_dict(cls, params):
        return super().from_dict(params) if params else None

    def __post_init__(self) -> None:
        if self.pos_emb_height != self.pos_emb_width:
            raise NotImplementedError("Only square position tables are supported.")
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.grid_size = self.pos_emb_height
        if self.layer_types is None:
            # As transformers: full attention in every 4th and the last layer.
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % 4 == 0 or i == self.num_hidden_layers - 1
                    else "window_attention"
                )
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "muse_spark_audio"
    patch_sizes: tuple[int, ...] = (16, 6, 5, 4)
    hidden_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    depths: tuple[int, ...] = (2, 2, 2, 2)
    num_hidden_layers: int = 4
    mlp_ratio: float = 4.0
    conv_kernel_size: int = 7
    hidden_act: str = "gelu"
    rms_norm_eps: float = 1e-6
    out_hidden_size: int = 8192

    @classmethod
    def from_dict(cls, params):
        return super().from_dict(params) if params else None

    def __post_init__(self):
        if (
            not len(self.patch_sizes)
            == len(self.hidden_sizes)
            == len(self.depths)
            == self.num_hidden_layers
        ):
            raise ValueError(
                "Audio patch_sizes, hidden_sizes and depths must have one entry per layer."
            )
        if any(p <= 0 for p in self.patch_sizes):
            raise ValueError("Audio patch sizes must be positive.")
        if self.hidden_act != "gelu":
            raise ValueError(f"Unsupported audio activation: {self.hidden_act}")

    @property
    def samples_per_token(self):
        return math.prod(self.patch_sizes)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig | None = None
    audio_config: AudioConfig | None = None
    model_type: str = "muse_spark"
    image_token_id: int = 200092
    audio_token_id: int = 200038
    video_token_id: int = 200091
    out_hidden_size: int | None = None
    projector_hidden_size: int | None = None
    eos_token_id: list[int] = field(default_factory=lambda: [200001, 200008])
    thinking_start_token: str = "to=self<|message|>"
    thinking_end_token: str = "<|eom|>"

    def __post_init__(self):
        for name, cls in (
            ("text", TextConfig),
            ("vision", VisionConfig),
            ("audio", AudioConfig),
        ):
            value = getattr(self, name + "_config")
            if isinstance(value, dict):
                setattr(self, name + "_config", cls.from_dict(value) if value else None)
        if self.text_config is None:
            raise ValueError("Muse Spark requires text_config.")
        if self.vision_config is not None:
            if self.out_hidden_size is None:
                self.out_hidden_size = (
                    self.vision_config.hidden_size * self.vision_config.merge_size**2
                )
            if self.projector_hidden_size is None:
                self.projector_hidden_size = self.out_hidden_size
        if (
            self.audio_config is not None
            and self.audio_config.out_hidden_size != self.text_config.hidden_size
        ):
            raise ValueError("Audio out_hidden_size must match the text hidden_size.")
        self.vocab_size = self.text_config.vocab_size

    @property
    def text(self):
        return self.text_config

    @property
    def vision(self):
        return self.vision_config
