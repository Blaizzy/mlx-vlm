from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    intermediate_size: int = 33792
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    linear_attn_config: Dict[str, Any] = field(default_factory=dict)
    hidden_act: str = "situ"
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None
    attn_res_block_size: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    num_experts: Optional[int] = None
    num_experts_per_token: int = 16
    num_shared_experts: int = 0
    moe_intermediate_size: Optional[int] = None
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.hidden_act != "situ":
            raise ValueError(f"Unsupported activation '{self.hidden_act}'")
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError(
                f"Unsupported MoE router activation '{self.moe_router_activation_func}'"
            )


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "moonvit3d"
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    pos_emb_type: str = "divided_fixed"
    pos_emb_interpolation_mode: str = "bilinear"
    vt_num_attention_heads: int = 12
    vt_num_hidden_layers: int = 27
    vt_hidden_size: int = 1024
    vt_intermediate_size: int = 4096
    qkv_hidden_size: int = 1536
    norm_type: str = "rmsnorm"
    mlp_type: str = "mlp2"
    activation_func: str = "gelu_pytorch_tanh"
    attn_bias: bool = False
    linear_bias: bool = False
    patch_embed_proj_bias: bool = False
    merge_kernel_size: list = None
    merge_type: str = "sd2_tpool"
    mm_projector_type: str = "patchmergerv2"
    mm_hidden_size: int = 1024
    projector_hidden_act: str = "gelu"
    projector_ln_eps: float = 1e-5
    text_hidden_size: int = 7168

    def __post_init__(self):
        if self.merge_kernel_size is None:
            self.merge_kernel_size = (2, 2)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: "TextConfig"
    vision_config: "VisionConfig"
    model_type: str = "kimi_k3"
    ignore_index: int = -100
    vocab_size: int = 163840
    media_placeholder_token_id: int = 163605
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.media_placeholder_token_id
