import inspect
from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    """GLM-4.7-Flash (``glm4_moe_lite``) fields the MTP drafter depends on.

    mlx-vlm has no ``glm4_moe_lite`` backbone in its model zoo yet, so this
    carries the subset of the source config needed to shape the drafter and
    validate target compatibility instead of delegating to a backbone config.
    """

    model_type: str = "glm4_moe_lite"
    hidden_size: int = 2048
    vocab_size: int = 154880
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    first_k_dense_replace: int = 1
    kv_lora_rank: int = 512
    q_lora_rank: int = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    v_head_dim: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False


@dataclass
class Glm4MoeLiteMTPConfig(BaseModelConfig):
    model_type: str = "glm4_moe_lite_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 2
    runtime_block_size: Optional[int] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.runtime_block_size is None and self.text_config is not None:
            nextn_depth = getattr(self.text_config, "num_nextn_predict_layers", 1)
            self.runtime_block_size = min(self.block_size, int(nextn_depth) + 1)

    @classmethod
    def from_dict(cls, params: dict) -> "Glm4MoeLiteMTPConfig":
        flat = dict(params)
        text_config = flat.get("text_config") or {}
        nextn_depth = text_config.get("num_nextn_predict_layers", 1)
        flat.setdefault("block_size", int(nextn_depth) + 1)
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
