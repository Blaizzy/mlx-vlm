import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ....models.base import BaseModelConfig

_REQUIRED_CHECKPOINT_FIELDS = {
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "draft_vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "layer_types",
    "sliding_windows",
    "sliding_window",
    "gating",
    "eagle_aux_hidden_state_layer_ids",
}
_REQUIRED_DFLASH_FIELDS = {
    "block_size",
    "mask_token_id",
    "target_layer_ids",
    "num_target_layers",
    "causal",
}


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _target_layer_count(config: Any) -> Optional[int]:
    for key in ("num_hidden_layers", "num_layers", "n_layers"):
        value = _value(config, key)
        if value is not None:
            return int(value)
    return None


def _target_vocab_size(config: Any) -> Optional[int]:
    for key in ("vocab_size", "padded_vocab_size"):
        value = _value(config, key)
        if value is not None:
            return int(value)
    return None


def _is_strictly_increasing_ints(values: list[int]) -> bool:
    return all(isinstance(value, int) for value in values) and values == sorted(
        set(values)
    )


def validate_laguna_dflash_target(
    config: "DFlashConfig",
    *,
    target_model_config: Any = None,
    target_model_layer_count: Optional[int] = None,
    target_tokenizer_length: Optional[int] = None,
) -> None:
    """Validate target-model identity before binding shared target tensors."""
    if target_model_layer_count is None and target_model_config is not None:
        target_model_layer_count = _target_layer_count(target_model_config)
    if target_model_layer_count is None:
        raise ValueError("Laguna DFlash binding requires the target model layer count.")
    if target_model_layer_count != config.num_target_layers:
        raise ValueError(
            "Laguna DFlash target layer count mismatch: "
            f"draft expects {config.num_target_layers}, target has "
            f"{target_model_layer_count}."
        )

    if target_tokenizer_length is None and target_model_config is not None:
        target_tokenizer_length = _target_vocab_size(target_model_config)
    if target_tokenizer_length is None:
        raise ValueError("Laguna DFlash binding requires the target tokenizer length.")
    if target_tokenizer_length != config.vocab_size:
        raise ValueError(
            "Laguna DFlash tokenizer vocabulary mismatch: "
            f"draft has {config.vocab_size}, target tokenizer has "
            f"{target_tokenizer_length}."
        )


@dataclass
class DFlashConfig(BaseModelConfig):
    """Validated DFlash checkpoint metadata for a Laguna target."""

    model_type: str = ""
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    draft_vocab_size: int = 0
    max_position_embeddings: int = 0
    rope_theta: float = 0.0
    rope_parameters: Optional[dict[str, Any]] = None
    layer_types: list[str] = field(default_factory=list)
    sliding_windows: list[int] = field(default_factory=list)
    sliding_window: int = 0
    gating: str = ""
    block_size: int = 0
    mask_token_id: int = -1
    target_layer_ids: list[int] = field(default_factory=list)
    num_target_layers: int = 0
    aux_hidden_state_layer_ids: list[int] = field(default_factory=list)
    causal: bool = False
    attention_bias: bool = False
    qkv_bias: bool = False
    tie_word_embeddings: bool = True

    def validate(self) -> None:
        if self.model_type != "laguna":
            raise ValueError(
                f"Laguna DFlash requires model_type='laguna', got {self.model_type!r}."
            )
        positive_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "draft_vocab_size",
            "max_position_embeddings",
            "sliding_window",
            "block_size",
            "num_target_layers",
        )
        for key in positive_fields:
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"Laguna DFlash requires a positive integer {key}.")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(
                "Laguna DFlash mask_token_id must be inside the draft vocabulary."
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "Laguna DFlash num_key_value_heads cannot exceed num_attention_heads."
            )
        if self.gating != "per-head":
            raise ValueError("Laguna DFlash currently supports per-head gating only.")
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type != "sliding_attention" for layer_type in self.layer_types
        ):
            raise ValueError(
                "Laguna DFlash requires one sliding_attention layer type per draft layer."
            )
        if len(self.sliding_windows) != self.num_hidden_layers or any(
            window != self.sliding_window for window in self.sliding_windows
        ):
            raise ValueError(
                "Laguna DFlash requires one matching sliding window per draft layer."
            )
        if (
            len(self.target_layer_ids) != self.num_hidden_layers
            or not _is_strictly_increasing_ints(self.target_layer_ids)
            or any(
                layer_id < 0 or layer_id >= self.num_target_layers
                for layer_id in self.target_layer_ids
            )
        ):
            raise ValueError(
                "Laguna DFlash target_layer_ids must be unique, increasing target-layer "
                "indices with one capture per draft layer."
            )
        if (
            len(self.aux_hidden_state_layer_ids) != self.num_hidden_layers
            or not _is_strictly_increasing_ints(self.aux_hidden_state_layer_ids)
            or any(
                layer_id < 0 or layer_id > self.num_target_layers
                for layer_id in self.aux_hidden_state_layer_ids
            )
        ):
            raise ValueError(
                "Laguna DFlash auxiliary layer IDs must be unique, increasing, and "
                "compatible with the target-layer count."
            )
        if not self.causal:
            raise ValueError("Laguna S 2.1 DFlash requires causal block attention.")
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError(
                "Laguna DFlash requires draft_vocab_size == vocab_size because "
                "the target tokenizer is shared."
            )

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        raw = dict(params)
        dflash = raw.pop("dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise ValueError("Laguna DFlash config requires a dflash_config object.")
        missing = sorted(_REQUIRED_CHECKPOINT_FIELDS - set(raw))
        if missing:
            raise ValueError(
                "Laguna DFlash config is missing checkpoint fields: "
                f"{', '.join(missing)}."
            )
        missing = sorted(_REQUIRED_DFLASH_FIELDS - set(dflash))
        if missing:
            raise ValueError(
                "Laguna DFlash config is missing dflash_config fields: "
                f"{', '.join(missing)}."
            )

        merged = dict(raw)
        for key in (
            "block_size",
            "mask_token_id",
            "target_layer_ids",
            "num_target_layers",
            "causal",
        ):
            if key in dflash:
                merged[key] = dflash[key]
        if "eagle_aux_hidden_state_layer_ids" in raw:
            merged["aux_hidden_state_layer_ids"] = raw[
                "eagle_aux_hidden_state_layer_ids"
            ]

        signature = inspect.signature(cls).parameters
        config = cls(**{k: v for k, v in merged.items() if k in signature})
        config.validate()
        return config


def expected_laguna_dflash_weight_shapes(
    config: DFlashConfig,
) -> dict[str, tuple[int, ...]]:
    config.validate()
    qkv_size = (
        config.num_attention_heads + 2 * config.num_key_value_heads
    ) * config.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (
            config.hidden_size,
            len(config.target_layer_ids) * config.hidden_size,
        ),
        "hidden_norm.weight": (config.hidden_size,),
        "norm.weight": (config.hidden_size,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"aux_hidden_norms.{index}.weight": (config.hidden_size,),
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                f"{prefix}.self_attn.qkv_proj.weight": (qkv_size, config.hidden_size),
                f"{prefix}.self_attn.o_proj.weight": (
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ),
                f"{prefix}.self_attn.g_proj.weight": (
                    config.num_attention_heads,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
            }
        )
    return shapes


def validate_laguna_dflash_weights(
    weights: Mapping[str, Any], config: DFlashConfig
) -> None:
    expected = expected_laguna_dflash_weight_shapes(config)
    actual_keys = set(weights)
    expected_keys = set(expected)
    quantized_weight_keys = {
        key
        for key in expected_keys
        if key.endswith(".weight")
        and key.removesuffix(".weight") + ".scales" in actual_keys
    }
    quantized_aux_keys = {
        key.removesuffix(".weight") + suffix
        for key in quantized_weight_keys
        for suffix in (".scales", ".biases")
    }
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys - quantized_aux_keys)
    if missing or unexpected:
        raise ValueError(
            "Laguna DFlash weight keys do not match the published checkpoint: "
            f"missing={missing}, unexpected={unexpected}."
        )
    bad_shapes = {
        key: (tuple(weights[key].shape), shape)
        for key, shape in expected.items()
        if key not in quantized_weight_keys and tuple(weights[key].shape) != shape
    }
    if bad_shapes:
        raise ValueError(
            "Laguna DFlash weight shapes do not match the published checkpoint: "
            f"{bad_shapes}."
        )


__all__ = [
    "DFlashConfig",
    "expected_laguna_dflash_weight_shapes",
    "validate_laguna_dflash_target",
    "validate_laguna_dflash_weights",
]
