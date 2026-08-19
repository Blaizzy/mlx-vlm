import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ....models.base import BaseModelConfig


def _is_strictly_increasing_ints(values) -> bool:
    return all(
        isinstance(value, int) and (index == 0 or values[index - 1] < value)
        for index, value in enumerate(values)
    )


@dataclass
class MuseGlimmerAssistantConfig(BaseModelConfig):
    """Published Muse Glimmer 30B DFlash assistant configuration."""

    model_type: str = "muse_glimmer_assistant"
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 131072
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int = 2048
    block_size: int = 16
    mask_token_id: int = 201818
    target_layer_ids: list[int] = field(default_factory=list)
    num_target_layers: int = 52
    vocab_size: int = 202048
    bos_token_id: int = 200000
    eos_token_id: int = 200001
    pad_token_id: int = 200018
    runtime_block_size: int | None = None
    draft_window_size: int | None = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_theta": 500000.0,
                "rope_type": "default",
            }
        if not self.layer_types:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        if not self.target_layer_ids:
            self.target_layer_ids = [1, 13, 25, 37, 49]
        self.validate()

    @property
    def rope_theta(self) -> float:
        return float(self.rope_parameters.get("rope_theta", 500000.0))

    def validate(self) -> None:
        if self.model_type != "muse_glimmer_assistant":
            raise ValueError(
                "Muse Glimmer DFlash requires "
                f"model_type='muse_glimmer_assistant', got {self.model_type!r}."
            )

        positive_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "sliding_window",
            "block_size",
            "num_target_layers",
            "vocab_size",
        )
        for key in positive_fields:
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Muse Glimmer DFlash requires a positive integer {key}."
                )

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "Muse Glimmer DFlash attention heads must be divisible by KV heads."
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(
                "Muse Glimmer DFlash mask_token_id must be inside the target vocabulary."
            )
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type != "sliding_attention" for layer_type in self.layer_types
        ):
            raise ValueError(
                "Muse Glimmer DFlash requires one sliding_attention type per "
                "assistant layer."
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
                "Muse Glimmer DFlash target_layer_ids must be unique, increasing "
                "target-layer indices with one capture per assistant layer."
            )

    @classmethod
    def from_dict(cls, params: dict) -> "MuseGlimmerAssistantConfig":
        flat = dict(params)
        dflash = flat.pop("dflash_config", None)
        if isinstance(dflash, Mapping):
            for key in (
                "block_size",
                "mask_token_id",
                "target_layer_ids",
                "runtime_block_size",
                "draft_window_size",
            ):
                if key in dflash:
                    flat[key] = dflash[key]

        signature = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict


def validate_muse_glimmer_target(
    config: MuseGlimmerAssistantConfig,
    *,
    target_model_config: Any,
    target_model_layer_count: int | None = None,
) -> None:
    """Reject target models that cannot supply this checkpoint's features."""

    model_type = getattr(target_model_config, "model_type", None)
    if model_type != "muse_glimmer_text":
        raise ValueError(
            "Muse Glimmer DFlash requires a Muse Glimmer text target, got "
            f"model_type={model_type!r}."
        )

    target_hidden_size = getattr(target_model_config, "hidden_size", None)
    if target_hidden_size != config.hidden_size:
        raise ValueError(
            "Muse Glimmer DFlash target hidden size mismatch: "
            f"assistant={config.hidden_size}, target={target_hidden_size}."
        )

    configured_layers = getattr(target_model_config, "num_hidden_layers", None)
    layer_count = target_model_layer_count or configured_layers
    if layer_count != config.num_target_layers:
        raise ValueError(
            "Muse Glimmer DFlash target layer count mismatch: "
            f"assistant={config.num_target_layers}, target={layer_count}."
        )

    target_vocab_size = getattr(target_model_config, "vocab_size", None)
    if target_vocab_size != config.vocab_size:
        raise ValueError(
            "Muse Glimmer DFlash target vocabulary mismatch: "
            f"assistant={config.vocab_size}, target={target_vocab_size}."
        )


def expected_muse_glimmer_assistant_weight_shapes(
    config: MuseGlimmerAssistantConfig,
) -> dict[str, tuple[int, ...]]:
    config.validate()
    shapes: dict[str, tuple[int, ...]] = {
        "encoder.fc.weight": (
            config.hidden_size,
            len(config.target_layer_ids) * config.hidden_size,
        ),
        "encoder.output_norm_enc.weight": (config.hidden_size,),
        "norm.weight": (config.hidden_size,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                f"{prefix}.self_attn.q_proj.weight": (
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.k_proj.weight": (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
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


def validate_muse_glimmer_assistant_weights(
    weights: Mapping[str, Any], config: MuseGlimmerAssistantConfig
) -> None:
    expected = expected_muse_glimmer_assistant_weight_shapes(config)
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
            "Muse Glimmer DFlash weight keys do not match the published checkpoint: "
            f"missing={missing}, unexpected={unexpected}."
        )

    bad_shapes = {
        key: (tuple(weights[key].shape), shape)
        for key, shape in expected.items()
        if key not in quantized_weight_keys and tuple(weights[key].shape) != shape
    }
    if bad_shapes:
        raise ValueError(
            "Muse Glimmer DFlash weight shapes do not match the published checkpoint: "
            f"{bad_shapes}."
        )


__all__ = [
    "MuseGlimmerAssistantConfig",
    "expected_muse_glimmer_assistant_weight_shapes",
    "validate_muse_glimmer_assistant_weights",
    "validate_muse_glimmer_target",
]
