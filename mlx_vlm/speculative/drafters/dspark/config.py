import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ....models.base import BaseModelConfig


def _is_strictly_increasing_ints(values: list[int]) -> bool:
    return all(
        isinstance(value, int) and (index == 0 or values[index - 1] < value)
        for index, value in enumerate(values)
    )


@dataclass
class DSparkConfig(BaseModelConfig):
    """Model-agnostic configuration for a DSpark drafter.

    Checkpoints publish ``block_size`` as the number of proposals (gamma),
    while mlx-vlm's DFlash loop uses the verification width (anchor plus
    proposals). ``from_dict`` normalizes that boundary once.
    """

    model_type: str = "dspark"
    backbone_model_type: str = "qwen3"
    architectures: list[str] = field(default_factory=list)
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128000
    max_position_embeddings: int = 128000
    rope_theta: float = 10000000.0
    rope_scaling: dict[str, Any] | None = None
    rope_is_neox_style: bool = True
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    final_logit_softcapping: float | None = None
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int | None = None

    # mlx-vlm verification width (anchor + proposals).
    block_size: int = 0
    # Checkpoint proposal count (called gamma upstream).
    proposal_length: int = 0
    mask_token_id: int = -1
    target_layer_ids: list[int] = field(default_factory=list)
    num_target_layers: int = 0

    runtime_block_size: int | None = None
    draft_window_size: int | None = None
    block_size_policy: str = "fixed"
    dflash_initial_block_size: int | None = None

    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.validate()

    def validate(self) -> None:
        if self.model_type != "dspark":
            raise ValueError(
                f"DSpark requires normalized model_type='dspark', got "
                f"{self.model_type!r}."
            )
        positive_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "block_size",
            "proposal_length",
            "num_target_layers",
            "markov_rank",
        )
        for key in positive_fields:
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"DSpark requires a positive integer {key}.")
        if self.block_size != self.proposal_length + 1:
            raise ValueError(
                "DSpark verification block_size must equal proposal_length + 1."
            )
        if self.runtime_block_size is not None and (
            not isinstance(self.runtime_block_size, int)
            or not 1 <= self.runtime_block_size <= self.block_size
        ):
            raise ValueError(
                "DSpark runtime_block_size must be between 1 and block_size."
            )
        if self.block_size_policy not in {"fixed", "adaptive"}:
            raise ValueError("DSpark block_size_policy must be 'fixed' or 'adaptive'.")
        if self.dflash_initial_block_size is not None and (
            not isinstance(self.dflash_initial_block_size, int)
            or not 2 <= self.dflash_initial_block_size <= self.block_size
        ):
            raise ValueError(
                "DSpark dflash_initial_block_size must be between 2 and block_size."
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "DSpark hidden_size must equal attention heads * head_dim."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DSpark attention heads must be divisible by KV heads.")
        if self.hidden_act != "silu":
            raise ValueError("DSpark currently supports only hidden_act='silu'.")
        if self.markov_head_type != "vanilla":
            raise ValueError(
                "DSpark currently supports only markov_head_type='vanilla'."
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("DSpark mask_token_id must be inside the vocabulary.")
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type != "full_attention" for layer_type in self.layer_types
        ):
            raise ValueError("DSpark requires one full_attention type per draft layer.")
        if (
            not self.target_layer_ids
            or not _is_strictly_increasing_ints(self.target_layer_ids)
            or any(
                layer_id < 0 or layer_id >= self.num_target_layers
                for layer_id in self.target_layer_ids
            )
        ):
            raise ValueError(
                "DSpark target_layer_ids must be unique, increasing indices "
                "inside the target layer range."
            )

    @classmethod
    def from_dict(cls, params: dict) -> "DSparkConfig":
        flat = dict(params)
        dflash = flat.pop("dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise TypeError("DSpark requires a dflash_config object.")

        projector_type = dflash.get("projector_type")
        if projector_type not in (None, "dspark"):
            raise ValueError(
                "DSpark requires dflash_config.projector_type='dspark', got "
                f"{projector_type!r}."
            )

        flat["backbone_model_type"] = str(flat.pop("model_type", "qwen3"))
        flat["model_type"] = "dspark"

        for key in (
            "mask_token_id",
            "target_layer_ids",
            "num_target_layers",
            "runtime_block_size",
            "draft_window_size",
            "block_size_policy",
            "dflash_initial_block_size",
            "markov_rank",
            "markov_head_type",
            "enable_confidence_head",
            "confidence_head_with_markov",
        ):
            if key in dflash:
                flat[key] = dflash[key]

        rope_parameters = flat.pop("rope_parameters", None)
        if isinstance(rope_parameters, Mapping):
            rope_parameters = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                flat["rope_theta"] = rope_parameters.pop("rope_theta")
            flat["rope_scaling"] = rope_parameters

        raw_block_size = dflash.get("block_size", flat.get("block_size"))
        if raw_block_size is None:
            raise ValueError("DSpark requires a checkpoint block_size.")
        flat["proposal_length"] = int(raw_block_size)
        flat["block_size"] = int(raw_block_size) + 1

        if "runtime_block_size" not in flat:
            flat["runtime_block_size"] = min(8, flat["block_size"])

        # Canonical DSpark configs declare their projector explicitly and use
        # the generic adaptive controller. Older configs predate that field and
        # retain their fixed-width behavior unless they opt in explicitly.
        legacy_preference = flat.pop("prefer_requested_block_size", None)
        if "prefer_requested_block_size" in dflash:
            legacy_preference = dflash["prefer_requested_block_size"]
        if "block_size_policy" not in flat:
            if legacy_preference is not None:
                flat["block_size_policy"] = (
                    "fixed" if bool(legacy_preference) else "adaptive"
                )
            else:
                flat["block_size_policy"] = (
                    "adaptive" if projector_type == "dspark" else "fixed"
                )

        if (
            flat["block_size_policy"] == "adaptive"
            and "dflash_initial_block_size" not in flat
        ):
            runtime_width = flat.get("runtime_block_size") or flat["block_size"]
            flat["dflash_initial_block_size"] = min(4, int(runtime_width))

        signature = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict


__all__ = ["DSparkConfig"]
