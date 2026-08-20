import inspect
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ....models.base import BaseModelConfig


@dataclass
class DFlashConfig(BaseModelConfig):
    architectures: List[str] = field(default_factory=list)
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    rope_scaling: Optional[dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 8, 15, 22, 29])
    num_target_layers: int = 32
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    runtime_block_size: int | None = None
    draft_window_size: int | None = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    is_causal: bool | None = None

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        rope = flat.pop("rope_parameters", None)
        if rope is not None:
            flat.setdefault("rope_scaling", rope)
            flat.setdefault("rope_theta", rope.get("rope_theta", 10000.0))
        nested_fields = (
            "mask_token_id",
            "runtime_block_size",
            "draft_window_size",
            "input_embedding_scale",
            "output_multiplier",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "final_logit_softcapping",
        )
        for name in nested_fields:
            if name in dflash_cfg:
                flat[name] = dflash_cfg[name]
        if "block_size" in dflash_cfg:
            flat["block_size"] = dflash_cfg["block_size"]
        if "target_layer_ids" in dflash_cfg:
            flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
