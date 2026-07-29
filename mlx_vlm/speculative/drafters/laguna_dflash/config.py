import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ....models.base import BaseModelConfig

LAGUNA_DFLASH_HIDDEN_SIZE = 3072
LAGUNA_DFLASH_INTERMEDIATE_SIZE = 12288
LAGUNA_DFLASH_NUM_LAYERS = 6
LAGUNA_DFLASH_NUM_ATTENTION_HEADS = 72
LAGUNA_DFLASH_NUM_KV_HEADS = 8
LAGUNA_DFLASH_HEAD_DIM = 128
LAGUNA_DFLASH_SLIDING_WINDOW = 512
LAGUNA_DFLASH_VOCAB_SIZE = 100352
LAGUNA_DFLASH_BLOCK_SIZE = 16
LAGUNA_DFLASH_MASK_TOKEN_ID = 12
LAGUNA_DFLASH_TARGET_LAYERS = [1, 10, 19, 29, 38, 47]
LAGUNA_DFLASH_TARGET_LAYER_COUNT = 48
LAGUNA_DFLASH_AUX_LAYER_IDS = [2, 11, 20, 30, 39, 48]


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
    """Exact configuration contract for Poolside Laguna S 2.1 DFlash."""

    model_type: str = "laguna"
    hidden_size: int = LAGUNA_DFLASH_HIDDEN_SIZE
    intermediate_size: int = LAGUNA_DFLASH_INTERMEDIATE_SIZE
    num_hidden_layers: int = LAGUNA_DFLASH_NUM_LAYERS
    num_attention_heads: int = LAGUNA_DFLASH_NUM_ATTENTION_HEADS
    num_key_value_heads: int = LAGUNA_DFLASH_NUM_KV_HEADS
    head_dim: int = LAGUNA_DFLASH_HEAD_DIM
    rms_norm_eps: float = 1e-6
    vocab_size: int = LAGUNA_DFLASH_VOCAB_SIZE
    draft_vocab_size: int = LAGUNA_DFLASH_VOCAB_SIZE
    max_position_embeddings: int = 1048576
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict[str, Any]] = None
    layer_types: list[str] = field(
        default_factory=lambda: ["sliding_attention"] * LAGUNA_DFLASH_NUM_LAYERS
    )
    sliding_windows: list[int] = field(
        default_factory=lambda: [LAGUNA_DFLASH_SLIDING_WINDOW]
        * LAGUNA_DFLASH_NUM_LAYERS
    )
    sliding_window: int = LAGUNA_DFLASH_SLIDING_WINDOW
    gating: str = "per-head"
    block_size: int = LAGUNA_DFLASH_BLOCK_SIZE
    mask_token_id: int = LAGUNA_DFLASH_MASK_TOKEN_ID
    target_layer_ids: list[int] = field(
        default_factory=lambda: list(LAGUNA_DFLASH_TARGET_LAYERS)
    )
    num_target_layers: int = LAGUNA_DFLASH_TARGET_LAYER_COUNT
    aux_hidden_state_layer_ids: list[int] = field(
        default_factory=lambda: list(LAGUNA_DFLASH_AUX_LAYER_IDS)
    )
    causal: bool = True
    attention_bias: bool = False
    qkv_bias: bool = False
    tie_word_embeddings: bool = True

    def validate(self) -> None:
        expected = {
            "hidden_size": LAGUNA_DFLASH_HIDDEN_SIZE,
            "intermediate_size": LAGUNA_DFLASH_INTERMEDIATE_SIZE,
            "num_hidden_layers": LAGUNA_DFLASH_NUM_LAYERS,
            "num_attention_heads": LAGUNA_DFLASH_NUM_ATTENTION_HEADS,
            "num_key_value_heads": LAGUNA_DFLASH_NUM_KV_HEADS,
            "head_dim": LAGUNA_DFLASH_HEAD_DIM,
            "sliding_window": LAGUNA_DFLASH_SLIDING_WINDOW,
            "gating": "per-head",
            "block_size": LAGUNA_DFLASH_BLOCK_SIZE,
            "mask_token_id": LAGUNA_DFLASH_MASK_TOKEN_ID,
            "num_target_layers": LAGUNA_DFLASH_TARGET_LAYER_COUNT,
            "vocab_size": LAGUNA_DFLASH_VOCAB_SIZE,
            "draft_vocab_size": LAGUNA_DFLASH_VOCAB_SIZE,
        }
        for key, expected_value in expected.items():
            actual = getattr(self, key)
            if actual != expected_value:
                raise ValueError(
                    f"Laguna S 2.1 DFlash {key} mismatch: expected "
                    f"{expected_value!r}, got {actual!r}."
                )

        if self.model_type != "laguna":
            raise ValueError(
                f"Laguna S 2.1 DFlash requires model_type='laguna', got {self.model_type!r}."
            )
        if self.layer_types != ["sliding_attention"] * LAGUNA_DFLASH_NUM_LAYERS:
            raise ValueError(
                "Laguna S 2.1 DFlash requires all six draft layers to use "
                "sliding_attention."
            )
        if (
            self.sliding_windows
            != [LAGUNA_DFLASH_SLIDING_WINDOW] * LAGUNA_DFLASH_NUM_LAYERS
        ):
            raise ValueError(
                "Laguna S 2.1 DFlash requires a 512 sliding window for every draft layer."
            )
        if self.target_layer_ids != LAGUNA_DFLASH_TARGET_LAYERS:
            raise ValueError(
                "Laguna S 2.1 DFlash target_layer_ids mismatch: expected "
                f"{LAGUNA_DFLASH_TARGET_LAYERS}, got {self.target_layer_ids}."
            )
        if self.aux_hidden_state_layer_ids != LAGUNA_DFLASH_AUX_LAYER_IDS:
            raise ValueError(
                "Laguna S 2.1 DFlash auxiliary layer IDs mismatch: expected "
                f"{LAGUNA_DFLASH_AUX_LAYER_IDS}, got {self.aux_hidden_state_layer_ids}."
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
