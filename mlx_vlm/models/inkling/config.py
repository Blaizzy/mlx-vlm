from dataclasses import dataclass
from typing import List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "inkling"
    hidden_size: int = 6144
    num_hidden_layers: int = 66
    vocab_size: int = 201024
    unpadded_vocab_size: Optional[int] = None
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_embed_norm: bool = True
    logits_mup_width_multiplier: float = 1.0
    max_position_embeddings: int = 1048576
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 16
    swa_head_dim: int = 128
    sliding_window_size: int = 512
    local_layer_ids: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: Optional[int] = None
    log_scaling_alpha: float = 0.1
    sconv_kernel_size: int = 4
    dense_mlp_idx: int = 0
    mlp_layer_types: Optional[List[str]] = None
    intermediate_size: int = 24576
    moe_intermediate_size: int = 3072
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    route_scale: float = 8.0

    def layer_is_sliding(self, i: int) -> bool:
        """Sliding-window (local) vs global full-attention layer. Real checkpoints set
        either ``layer_types`` (0.6B) or ``local_layer_ids`` (975B); the modulo is the
        reference default (every 6th layer global)."""
        if self.layer_types is not None:
            return self.layer_types[i] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return i in set(self.local_layer_ids)
        return bool((i + 1) % 6)

    def layer_is_dense(self, i: int) -> bool:
        if self.mlp_layer_types is not None:
            return self.mlp_layer_types[i] == "dense"
        return i < self.dense_mlp_idx


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "inkling_vision"
    patch_size: int = 40
    temporal_patch_size: int = 2
    num_channels: int = 3
    n_layers: int = 4  # HMLP encoder layers
    text_hidden_size: int = 6144
    rms_norm_eps: float = 1e-6


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "inkling_audio"
    n_mel_bins: int = 80
    mel_vocab_size: int = 16
    text_hidden_size: int = 6144
    rms_norm_eps: float = 1e-6


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: Union[TextConfig, dict, None] = None
    vision_config: Union[VisionConfig, dict, None] = None
    audio_config: Union[AudioConfig, dict, None] = None
    model_type: str = "inkling"
    image_token_id: int = 200054
    audio_token_id: int = 200053
    vocab_size: int = 201024
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        # Coerce dict sub-configs (from config.json) into typed dataclasses, and wire the
        # shared text hidden size into the towers so their projections land in LM space.
        if self.text_config is None:
            self.text_config = TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if self.audio_config is None:
            self.audio_config = AudioConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.audio_config.text_hidden_size = self.text_config.hidden_size
