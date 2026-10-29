from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "raon1_mini"
    hidden_size: int = 2560
    vocab_size: int = 259840
    num_hidden_layers: int = 25
    num_attention_heads: int = 40
    head_dim: int = 128
    intermediate_size: int = 12032
    rms_norm_eps: float = 1e-5
    mla_latent_norm_eps: float = 1e-6
    raon_mhc_selector_eps: float = 1e-5
    raon_mhc_streams: int = 4
    raon_mhc_post_factor: float = 2.0
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    num_experts: Optional[int] = 128
    num_experts_per_token: int = 6
    moe_intermediate_size: int = 1088
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    use_grouped_topk: bool = False
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    q_lora_rank: int = 576
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    rda_config: Optional[dict] = None
    rda_decay_rank: int = 128
    num_nextn_predict_layers: int = 0
    attn_res_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rda_config is None:
            self.rda_config = {
                "rda_layers": [
                    1,
                    2,
                    3,
                    5,
                    6,
                    7,
                    9,
                    10,
                    11,
                    13,
                    14,
                    15,
                    17,
                    18,
                    19,
                    21,
                    22,
                    23,
                ],
                "mla_layers": [4, 8, 12, 16, 20, 24, 25],
            }
        if self.hidden_act != "situ" or self.moe_router_activation_func != "sigmoid":
            raise ValueError("Raon-1 Mini requires SiTU and sigmoid expert routing")
        if (
            self.use_grouped_topk
            or self.routed_expert_hidden_size is not None
            or self.latent_moe_use_norm
        ):
            raise ValueError(
                "Raon-1 Mini does not support grouped routing or latent MoE"
            )
        if not self.mla_use_nope or self.attn_res_block_size is not None:
            raise ValueError("Raon-1 Mini requires NoPE attention and mHC residuals")
        if self.raon_mhc_streams < 2 or self.moe_layer_freq < 1:
            raise ValueError("Invalid residual stream count or expert layer frequency")
        if self.num_experts and not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("Active expert count must be between one and num_experts")
        if not self.rda_config.get("full_rank_gate", True):
            raise ValueError("Raon-1 Mini requires full-rank RDA output gates")
        layers = self.rda_config.get("rda_layers", []) + self.rda_config.get(
            "mla_layers", []
        )
        if sorted(layers) != list(range(1, self.num_hidden_layers + 1)):
            raise ValueError(
                "rda_layers and mla_layers must partition the decoder layers"
            )
        if self.rda_config.get("conv_kernel_size", 4) < 1:
            raise ValueError("RDA convolution kernel size must be positive")
