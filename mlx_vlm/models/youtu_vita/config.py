"""Configuration dataclasses for Youtu-Parsing-Omni (``youtu_vita``).

Ported from tencent/Youtu-Parsing-Omni (https://huggingface.co/tencent/Youtu-Parsing-Omni).
Copyright (C) 2026 Tencent. The original code is distributed under the Youtu-Parsing
license (https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/main/LICENSE);
modified / re-implemented here for MLX.
"""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class OmniConfig(BaseModelConfig):
    """Config of the shared Qwen3-style omni encoder (``omni_config``)."""

    model_type: str = "youtu_vita_omni"
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    merger_hidden_size: int = 2560
    out_hidden_size: int = 2560
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_theta_m: float = 100.0
    rope_m_dim: int = 4
    video_omni_chunked_mthw_rope: bool = False
    video_omni_interleaved_mthw_rope: bool = False
    # per-layer 0/1 mask: 1 = fusion layer (frames + audio of a group share one attention
    # window), 0 = every frame / audio chunk is its own window. None = all fusion.
    video_fusion_layer_freq: Optional[list] = None
    video_group_attention: bool = False
    # audio front-end / merger
    num_mel_bins: int = 128
    downsample_hidden_size: int = 512
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    temporal_merge_size: int = 1


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "youtu_vita_text"
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    # NOTE: text_config.num_key_value_heads (=8) and head_dim (=64) are
    # misleading for MLA: kv_b_proj is [heads*(nope+v), kv_lora_rank].
    num_key_value_heads: int = 32
    vocab_size: int = 133632
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    rope_theta: float = 1600000.0
    rope_parameters: Optional[Dict] = None
    rope_interleave: bool = True
    rope_traditional: bool = True
    rope_scaling: Optional[Dict] = None
    # The checkpoint has an UNTIED lm_head even though text_config says tied;
    # the top-level config's tie_word_embeddings=false is authoritative.
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_dict(cls, params):
        # mlx_vlm.utils.update_module_configs re-parses the RAW text_config (tie_word_embeddings=true)
        # after ModelConfig.from_dict; the checkpoint has a separate lm_head (differs from
        # embed_tokens by up to 2.4e-4), so never tie.
        params = dict(params or {})
        params["tie_word_embeddings"] = False
        return super().from_dict(params)

    def __post_init__(self):
        if self.rope_parameters:
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            if self.rope_parameters.get("rope_type", "default") != "default":
                self.rope_scaling = self.rope_parameters
        self.rope_traditional = bool(self.rope_interleave)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    omni_config: OmniConfig = None
    model_type: str = "youtu_vita"
    # <|image_pad|>
    image_token_id: int = 128257
    video_token_id: int = 128261
    # <|audio_pad|>
    audio_token_id: int = 128265
    vocab_size: int = 133632
    tie_word_embeddings: bool = False
    # config.json carries no top-level eos; generation_config.json has [128001, 128257]
    eos_token_id: Optional[List[int]] = field(default_factory=lambda: [128001, 128257])
    # dtype of the omni encoder (+ patch embed + merger). Activations reach ~4e5:
    # fp16 overflows and bf16 shifts coordinate tokens, so keep fp32 ('same' = checkpoint dtype).
    encoder_dtype: str = "float32"
    video_omni_fusion: bool = False

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["tie_word_embeddings"] = params.get("tie_word_embeddings", False)
        text = dict(params.get("text_config") or {})
        # top-level tie_word_embeddings (false) overrides the text-config value
        text["tie_word_embeddings"] = params["tie_word_embeddings"]
        params["text_config"] = text
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        if isinstance(self.omni_config, dict):
            self.omni_config = OmniConfig.from_dict(self.omni_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
