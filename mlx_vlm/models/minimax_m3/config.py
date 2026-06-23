from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "minimax_m3_vl"
    hidden_size: int = 6144
    intermediate_size: int = 3072            # routed-expert intermediate
    dense_intermediate_size: int = 12288     # MLP width of the leading dense layers
    shared_intermediate_size: int = 3072     # shared-expert intermediate
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 200064
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rotary_dim: int = 64                      # partial rotary: only first `rotary_dim` of head_dim
    partial_rotary_factor: float = 0.5
    hidden_act: str = "swigluoai"
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    use_qk_norm: bool = True
    qk_norm_type: str = "per_head"
    use_gemma_norm: bool = True
    attention_output_gate: bool = False
    # MoE
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True
    routed_scaling_factor: float = 2.0
    moe_layer_freq: Optional[List[int]] = None       # per-layer 0=dense, 1=MoE
    # MiniMax Sparse Attention (MSA): {use_sparse_attention, sparse_block_size,
    # sparse_topk_blocks, sparse_index_dim, sparse_num_index_heads, sparse_attention_freq[...], ...}
    sparse_attention_config: Optional[Dict] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # number of leading dense (non-MoE) layers, e.g. 3 for M3
        self.first_k_dense = 0
        if self.moe_layer_freq:
            for i, v in enumerate(self.moe_layer_freq):
                if v == 1:
                    self.first_k_dense = i
                    break
        sc = self.sparse_attention_config or {}
        self.index_head_dim = sc.get("sparse_index_dim", 128)
        self.index_n_heads = sc.get("sparse_num_index_heads", 4)
        self.index_block_size = sc.get("sparse_block_size", 128)
        self.index_topk_blocks = sc.get("sparse_topk_blocks", 16)
        self.index_local_blocks = sc.get("sparse_local_block", 1)

    def is_sparse_layer(self, layer_idx: int) -> bool:
        cfg = self.sparse_attention_config or {}
        if not cfg.get("use_sparse_attention"):
            return False
        freq = cfg.get("sparse_attention_freq") or []
        return bool(layer_idx < len(freq) and freq[layer_idx])


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "clip_vision_model"
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    patch_size: int = 14
    image_size: int = 2016
    projection_dim: int = 6144
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    position_embedding_type: str = "rope"
    rope_mode: str = "3d"
    rope_theta: float = 10000.0
    spatial_merge_size: int = 2              # patch_merge compression
    temporal_patch_size: int = 2


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "minimax_m3_vl"
    image_token_index: int = 200025
    video_token_index: int = 200026
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "full"
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 6144
    multimodal_projector_bias: bool = True

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["text_config"] = TextConfig.from_dict(params.get("text_config", {}))
        params["vision_config"] = VisionConfig.from_dict(params.get("vision_config", {}))
        return super().from_dict(params)
