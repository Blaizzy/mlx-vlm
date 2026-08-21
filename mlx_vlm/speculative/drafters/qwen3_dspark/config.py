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
    """Configuration for the Qwen3-backbone DSpark LFM2 drafter.

    The published ``block_size`` is DSpark's proposal count (``gamma``).
    mlx-vlm's DFlash loop uses the total verification width, which also
    includes the anchor token, so ``block_size`` is normalized to
    ``proposal_length + 1`` while loading the checkpoint.
    """

    model_type: str = "qwen3"
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
    rope_is_neox_style: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    final_logit_softcapping: float | None = None
    layer_types: list[str] = field(default_factory=list)
    sliding_window: int | None = None

    # mlx-vlm verification width (anchor + proposals).
    block_size: int = 10
    # DSpark checkpoint proposal count (called gamma upstream).
    proposal_length: int = 9
    mask_token_id: int = 125017
    target_layer_ids: list[int] = field(default_factory=list)
    num_target_layers: int = 30
    runtime_block_size: int | None = None
    draft_window_size: int | None = None

    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if not self.target_layer_ids:
            self.target_layer_ids = [2, 9, 17, 21, 27]
        self.validate()

    def validate(self) -> None:
        if self.model_type != "qwen3":
            raise ValueError(
                f"LFM2 DSpark requires model_type='qwen3', got {self.model_type!r}."
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
                raise ValueError(f"LFM2 DSpark requires a positive integer {key}.")
        if self.block_size != self.proposal_length + 1:
            raise ValueError(
                "LFM2 DSpark verification block_size must equal " "proposal_length + 1."
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "LFM2 DSpark hidden_size must equal attention heads * head_dim."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "LFM2 DSpark attention heads must be divisible by KV heads."
            )
        if self.hidden_act != "silu":
            raise ValueError("LFM2 DSpark currently supports only hidden_act='silu'.")
        if self.markov_head_type != "vanilla":
            raise ValueError(
                "LFM2 DSpark currently supports only markov_head_type='vanilla'."
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(
                "LFM2 DSpark mask_token_id must be inside the target vocabulary."
            )
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type != "full_attention" for layer_type in self.layer_types
        ):
            raise ValueError(
                "LFM2 DSpark requires one full_attention type per draft layer."
            )
        if (
            not self.target_layer_ids
            or not _is_strictly_increasing_ints(self.target_layer_ids)
            or any(
                layer_id < 0 or layer_id >= self.num_target_layers
                for layer_id in self.target_layer_ids
            )
        ):
            raise ValueError(
                "LFM2 DSpark target_layer_ids must be unique, increasing indices "
                "inside the target layer range."
            )

    @classmethod
    def from_dict(cls, params: dict) -> "DSparkConfig":
        flat = dict(params)
        dflash = flat.pop("dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise TypeError("LFM2 DSpark requires a dflash_config object.")

        for key in (
            "mask_token_id",
            "target_layer_ids",
            "num_target_layers",
            "runtime_block_size",
            "draft_window_size",
        ):
            if key in dflash:
                flat[key] = dflash[key]

        raw_block_size = dflash.get("block_size", flat.get("block_size"))
        if raw_block_size is None:
            raise ValueError("LFM2 DSpark requires a checkpoint block_size.")
        flat["proposal_length"] = int(raw_block_size)
        flat["block_size"] = int(raw_block_size) + 1

        signature = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict
