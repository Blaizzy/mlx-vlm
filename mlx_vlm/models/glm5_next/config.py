from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "glm5_next_text"
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    pad_token_id: Optional[int] = 154820
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    mlp_layer_types: Optional[List[str]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: Optional[List[str]] = None
    indexer_types: Optional[List[str]] = None
    swiglu_limit: float = 10.0
    linear_attn_config: Dict = field(default_factory=dict)
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: Optional[float] = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True
    index_kpool_compress: bool = True
    index_hisa_block: int = 8
    index_hisa_keep: int = 64
    index_hisa_min_pools: int = 2048
    indexer_rope_interleave: bool = True
    mla_use_nope: bool = True
    first_k_dense_replace: int = 3
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mlp_layer_types is None:
            dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * dense + ["sparse"] * (
                self.num_hidden_layers - dense
            )

        if self.layer_types is None:
            self.layer_types = [
                (
                    "linear_attention"
                    if layer_idx % 4 != 3
                    else "deepseek_sparse_attention"
                )
                for layer_idx in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            (
                "deepseek_sparse_attention"
                if layer_type == "full_attention"
                else layer_type
            )
            for layer_type in self.layer_types
        ]

        if self.indexer_types is None:
            self.indexer_types = ["full"] * self.num_hidden_layers

        if self.linear_attn_config:
            self.linear_head_dim = self.linear_attn_config.get(
                "head_dim", self.linear_head_dim
            )
            self.linear_num_heads = self.linear_attn_config.get(
                "num_heads", self.linear_num_heads
            )
            self.linear_conv_kernel_dim = self.linear_attn_config.get(
                "short_conv_kernel_size", self.linear_conv_kernel_dim
            )
            self.linear_lower_bound = self.linear_attn_config.get(
                "gate_lower_bound", self.linear_lower_bound
            )
            if (
                self.linear_attn_config.get("safe_gate", True)
                and self.linear_lower_bound is None
            ):
                self.linear_lower_bound = -5.0

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("`layer_types` must have one entry per hidden layer.")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("`mlp_layer_types` must have one entry per hidden layer.")
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError("`indexer_types` must have one entry per hidden layer.")
        if self.qk_rope_head_dim != 0 or not self.mla_use_nope:
            raise ValueError("GLM-5-Next requires NoPE sparse attention.")
        if self.index_kpool < 1 or self.index_topk % self.index_kpool:
            raise ValueError("`index_topk` must be divisible by `index_kpool`.")
        if self.index_hisa_block < 0 or self.index_hisa_keep < 0:
            raise ValueError("HISA block and keep sizes cannot be negative.")
        if bool(self.index_hisa_block) != bool(self.index_hisa_keep):
            raise ValueError("HISA block and keep sizes must be enabled together.")
        if self.index_hisa_min_pools < 0:
            raise ValueError("`index_hisa_min_pools` cannot be negative.")
        if self.index_hisa_block and (
            self.index_hisa_block * self.index_hisa_keep
            < self.index_topk // self.index_kpool
        ):
            raise ValueError("HISA candidate capacity must cover the final top-k.")


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm5_next_vision"
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    intermediate_size: int = 4096
    projection_intermediate_size: int = 10240
    swiglu_limit: float = 10.0


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: Optional[TextConfig] = None
    vision_config: Optional[VisionConfig] = None
    model_type: str = "glm5_next"
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False
    vocab_size: int = 154880
    hidden_size: int = 4096
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        elif self.text_config is None:
            self.text_config = TextConfig()
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VisionConfig()

        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.text_config.pad_token_id
