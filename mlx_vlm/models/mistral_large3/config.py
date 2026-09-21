import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig
from ..pixtral import VisionConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "mistral_large3"
    vocab_size: int = 131072
    hidden_size: int = 7168
    intermediate_size: int = 16384
    moe_intermediate_size: int = 4096
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    # MLA (Multi-Latent Attention) parameters
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    attention_bias: bool = False
    # MoE parameters
    n_routed_experts: Optional[int] = 128
    n_shared_experts: Optional[int] = 1
    num_experts_per_tok: int = 4
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    routed_scaling_factor: float = 1.0
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    max_position_embeddings: int = 294912
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "mistral_large3"
    image_token_id: int = 10
    image_token_index: Optional[int] = None
    vision_feature_layer: int = -1
    spatial_merge_size: int = 2
    multimodal_projector_bias: bool = False
    vocab_size: int = 131072
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig(
                **{
                    k: v
                    for k, v in self.text_config.items()
                    if k in inspect.signature(TextConfig).parameters
                }
            )
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig(
                **{
                    k: v
                    for k, v in self.vision_config.items()
                    if k in inspect.signature(VisionConfig).parameters
                }
            )
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


def config_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a Mistral-native ``params.json`` into an mlx-vlm config dict.

    Mistral Large 3 ships no ``config.json`` (only ``params.json`` + ``tekken.json``),
    so the loader has nothing to key on until this mapping produces one.
    """
    ve = params["vision_encoder"]
    moe = params["moe"]
    text = dict(
        model_type="mistral_large3",
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        intermediate_size=params["hidden_dim"],
        moe_intermediate_size=moe["expert_hidden_dim"],
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params["n_kv_heads"],
        n_routed_experts=moe["num_experts"],
        n_shared_experts=moe["num_shared_experts"],
        num_experts_per_tok=moe["num_experts_per_tok"],
        first_k_dense_replace=moe["first_k_dense_replace"],
        q_lora_rank=params["q_lora_rank"],
        kv_lora_rank=params["kv_lora_rank"],
        qk_nope_head_dim=params["qk_nope_head_dim"],
        qk_rope_head_dim=params["qk_rope_head_dim"],
        v_head_dim=params["v_head_dim"],
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        max_position_embeddings=params["max_position_embeddings"],
    )
    vision = dict(
        model_type="pixtral",
        hidden_size=ve["hidden_size"],
        num_hidden_layers=ve["num_hidden_layers"],
        num_attention_heads=ve["num_attention_heads"],
        head_dim=ve["hidden_size"] // ve["num_attention_heads"],
        intermediate_size=ve["intermediate_size"],
        image_size=ve["image_size"],
        patch_size=ve["patch_size"],
        rope_theta=ve["rope_theta"],
    )
    return dict(
        model_type="mistral_large3",
        text_config=text,
        vision_config=vision,
        image_token_id=ve["image_token_id"],
        spatial_merge_size=ve["spatial_merge_size"],
        multimodal_projector_bias=ve["adapter_bias"],
        vocab_size=params["vocab_size"],
    )
