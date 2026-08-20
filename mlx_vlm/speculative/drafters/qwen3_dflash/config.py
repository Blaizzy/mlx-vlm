import inspect
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ....models.base import BaseModelConfig


@dataclass
class DFlashConfig(BaseModelConfig):
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
    # DFlash 2 extras. All default to 0, which reproduces DFlash v1 exactly, so
    # v1 checkpoints keep loading unchanged. The v2 modules are only built when
    # the drafter's own dflash_config asks for them:
    #   conv_*     -> GroupedDynamicCausalConv in every decoder layer
    #   selector_* -> CandidateSelector (top-k per position + path search)
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        if "mask_token_id" in dflash_cfg:
            flat["mask_token_id"] = dflash_cfg["mask_token_id"]
        if "target_layer_ids" in dflash_cfg:
            flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
        if "runtime_block_size" in dflash_cfg:
            flat["runtime_block_size"] = dflash_cfg["runtime_block_size"]
        if "draft_window_size" in dflash_cfg:
            flat["draft_window_size"] = dflash_cfg["draft_window_size"]
        # DFlash 2 moved block_size into dflash_config; v1 kept it top-level.
        if "block_size" in dflash_cfg:
            flat["block_size"] = dflash_cfg["block_size"]
        for key in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            if key in dflash_cfg:
                flat[key] = int(dflash_cfg[key])
        # transformers 5.x nests RoPE settings under rope_parameters. Without
        # this, rope_theta silently falls back to the class default.
        rope_params = flat.get("rope_parameters")
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            flat.setdefault("rope_theta", rope_params["rope_theta"])
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
