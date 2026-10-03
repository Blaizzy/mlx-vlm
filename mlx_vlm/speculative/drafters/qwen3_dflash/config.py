import inspect
from collections.abc import Mapping
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
    is_causal: bool = False
    attention_sink_bias: bool = False

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        if "mask_token_id" in dflash_cfg:
            flat["mask_token_id"] = dflash_cfg["mask_token_id"]
        if "target_layer_ids" in dflash_cfg:
            flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
        if "num_target_layers" in dflash_cfg:
            flat["num_target_layers"] = dflash_cfg["num_target_layers"]
        if "runtime_block_size" in dflash_cfg:
            flat["runtime_block_size"] = dflash_cfg["runtime_block_size"]
        if "draft_window_size" in dflash_cfg:
            flat["draft_window_size"] = dflash_cfg["draft_window_size"]
        if "causal" in dflash_cfg:
            flat["is_causal"] = bool(dflash_cfg["causal"])
        elif "dflash_query_causal" in flat:
            flat["is_causal"] = bool(flat.pop("dflash_query_causal"))
        if "attention_sink_bias" in dflash_cfg:
            flat["attention_sink_bias"] = bool(dflash_cfg["attention_sink_bias"])
        if "num_target_layers" not in flat and flat.get("target_layer_ids"):
            flat["num_target_layers"] = max(flat["target_layer_ids"]) + 1
        rope_parameters = flat.pop("rope_parameters", None)
        if isinstance(rope_parameters, Mapping):
            rope_parameters = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                flat["rope_theta"] = rope_parameters.pop("rope_theta")
            flat["rope_scaling"] = rope_parameters
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
