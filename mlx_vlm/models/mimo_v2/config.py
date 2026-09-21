from dataclasses import dataclass, field
from typing import List, Optional

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "mimovl"
    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    out_hidden_size: int = 4096
    num_heads: int = 32
    num_key_value_heads: int = 8
    num_query_groups: int = 4
    in_chans: int = 3
    patch_size: int = 16
    spatial_patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    tokens_per_second: int = 2
    hidden_act: str = "silu"
    use_sink: bool = True
    window_size: int = 128
    visual_token_window_size: int = 64
    fullatt_block_indexes: List[int] = field(default_factory=lambda: [0, 9, 18, 27])
    vit_window_attn_types: List[int] = field(default_factory=list)


@dataclass
class AudioConfig(BaseModelConfig):
    out_hidden_size: int = 4096
    audio_channels: int = 20
    audio_segment_size: int = 6000
    group_size: int = 4
    input_full_attention: bool = True
    input_local_attn_heads: int = 16
    input_local_dim: int = 1024
    input_local_head_dim: int = 64
    input_local_hidden_dropout: float = 0.0
    input_local_intermediate_size: int = 4096
    input_local_layers: int = 6
    add_post_norm: bool = True
    partial_rotary_factor: float = 1.0
    projection_layers: int = 2
    rope_theta: float = 640000.0
    speech_vocab_size: int = 1280
    speech_zeroemb_idx: int = 1024

    def __post_init__(self):
        # the released configs ship these two as strings, not ints
        self.speech_vocab_size = int(self.speech_vocab_size)
        self.speech_zeroemb_idx = int(self.speech_zeroemb_idx)


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    rope_theta: float = 10000000.0
    swa_rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.334
    max_position_embeddings: int = 1048576
    layernorm_epsilon: float = 1e-06
    sliding_window_size: int = 128
    hybrid_layer_pattern: List[int] = field(default_factory=list)
    moe_layer_freq: List[int] = field(default_factory=list)
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    n_routed_experts: Optional[int] = 256
    n_shared_experts: Optional[int] = None
    num_experts_per_tok: int = 8
    routed_scaling_factor: Optional[float] = None
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"
    attention_value_scale: float = 0.707
    attention_chunk_size: int = 128
    attention_projection_layout: Optional[str] = "fused_qkv"
    moe_router_dtype: str = "bfloat16"
    num_nextn_predict_layers: int = 0


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    audio_config: AudioConfig
    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151669
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151673
    audio_end_token_id: int = 151674
    eos_token_id: Optional[int] = 151645
    skip_vision: bool = False

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["vision_config"] = VisionConfig.from_dict(params.get("vision_config"))
        params["audio_config"] = AudioConfig.from_dict(params.get("audio_config"))
        # MiMo-V2 keeps the text hyperparameters at the top level rather than
        # under a "text_config" key, so TextConfig is built from the same params.
        params["text_config"] = TextConfig.from_dict(params)
        return super().from_dict(params)
