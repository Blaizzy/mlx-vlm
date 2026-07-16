from dataclasses import dataclass
from typing import List, Optional

from ..base import BaseModelConfig

# Aliases mirroring `InklingTextConfig.attribute_map` in transformers: some
# checkpoints store config.json keys under these older/alternate names.
_TEXT_CONFIG_ALIASES = {
    "embedding_multiplier": "logits_mup_width_multiplier",
    "sliding_window": "sliding_window_size",
    "num_local_experts": "n_routed_experts",
    "sconv_kernel_size": "conv_kernel_size",
    "model_max_length": "max_position_embeddings",
}


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "inkling_text"
    vocab_size: int = 201024
    unpadded_vocab_size: Optional[int] = None
    hidden_size: int = 6144
    num_hidden_layers: int = 66
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 16
    swa_head_dim: int = 128
    sliding_window_size: int = 512
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: Optional[int] = None
    log_scaling_alpha: float = 0.1
    local_layer_ids: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    mlp_layer_types: Optional[List[str]] = None
    dense_mlp_idx: int = 0
    intermediate_size: int = 24576
    hidden_act: str = "silu"
    # MoE
    moe_intermediate_size: int = 3072
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    route_scale: float = 8.0
    logits_mup_width_multiplier: float = 24.0
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.layer_types is None:
            if self.local_layer_ids is not None:
                local_layer_ids = set(self.local_layer_ids)
            else:
                local_layer_ids = {
                    i for i in range(self.num_hidden_layers) if (i + 1) % 6
                }
            self.layer_types = [
                "hybrid_sliding" if i in local_layer_ids else "hybrid"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.dense_mlp_idx else "sparse"
                for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        params = dict(params)
        for old, new in _TEXT_CONFIG_ALIASES.items():
            if old in params and new not in params:
                params[new] = params.pop(old)
        return super().from_dict(params)


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "inkling_audio"
    n_mel_bins: int = 80
    mel_vocab_size: int = 256
    text_hidden_size: int = 6144
    rms_norm_eps: float = 1e-6


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "inkling_vision"
    text_hidden_size: int = 6144
    patch_size: int = 40
    temporal_patch_size: int = 2
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: AudioConfig
    model_type: str = "inkling_mm_model"
    image_token_id: int = 200054
    audio_token_id: int = 200053
    image_bos_token_id: int = 200005
    audio_bos_token_id: int = 200020
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None
    vocab_size: int = 201024

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.audio_config.text_hidden_size = self.text_config.hidden_size

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["text_config"] = TextConfig.from_dict(params.get("text_config") or {})
        params["vision_config"] = VisionConfig.from_dict(
            params.get("vision_config") or {}
        )
        params["audio_config"] = AudioConfig.from_dict(
            params.get("audio_config") or {}
        )
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k
                in (
                    "text_config",
                    "vision_config",
                    "audio_config",
                    "model_type",
                    "image_token_id",
                    "audio_token_id",
                    "image_bos_token_id",
                    "audio_bos_token_id",
                    "image_token_index",
                    "eos_token_id",
                    "vocab_size",
                )
            }
        )
