import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..qwen3_dflash.config import DFlashConfig


def _is_strictly_increasing_ints(values: list[int]) -> bool:
    return all(
        isinstance(value, int) and (index == 0 or values[index - 1] < value)
        for index, value in enumerate(values)
    )


@dataclass
class DFlash2Config(DFlashConfig):
    model_type: str = "dflash2"
    backbone_model_type: str = "qwen3"
    architectures: list[str] = field(default_factory=list)
    hidden_act: str = "silu"
    rope_is_neox_style: bool = True
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    is_causal: bool = False

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.validate()

    def validate(self) -> None:
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
            "num_target_layers",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        )
        for key in positive_fields:
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"DFlash2 requires a positive integer {key}.")
        if self.model_type != "dflash2":
            raise ValueError(
                f"DFlash2 requires normalized model_type='dflash2', got "
                f"{self.model_type!r}."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DFlash2 attention heads must be divisible by KV heads.")
        if self.hidden_size % self.conv_group_size:
            raise ValueError(
                "DFlash2 hidden_size must be divisible by conv_group_size."
            )
        if self.hidden_act != "silu":
            raise ValueError("DFlash2 currently supports only hidden_act='silu'.")
        if not 0 < self.selector_top_k <= self.vocab_size:
            raise ValueError("DFlash2 selector_top_k must fit inside the vocabulary.")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("DFlash2 mask_token_id must be inside the vocabulary.")
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type not in {"full_attention", "sliding_attention"}
            for layer_type in self.layer_types
        ):
            raise ValueError(
                "DFlash2 requires one supported attention type per draft layer."
            )
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError(
                "DFlash2 requires sliding_window for sliding_attention layers."
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
                "DFlash2 target_layer_ids must be unique, increasing indices "
                "inside the target layer range."
            )
        if self.runtime_block_size is not None and (
            not isinstance(self.runtime_block_size, int)
            or not 2 <= self.runtime_block_size <= self.block_size
        ):
            raise ValueError(
                "DFlash2 runtime_block_size must be between 2 and block_size."
            )
        if self.is_causal:
            raise ValueError("DFlash2 causal draft blocks are not currently supported.")

    @classmethod
    def from_dict(cls, params: dict) -> "DFlash2Config":
        flat = dict(params)
        dflash = flat.pop("dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise TypeError("DFlash2 requires a dflash_config object.")

        flat["backbone_model_type"] = str(flat.pop("model_type", "qwen3"))
        flat["model_type"] = "dflash2"
        for key in (
            "block_size",
            "mask_token_id",
            "target_layer_ids",
            "runtime_block_size",
            "draft_window_size",
            "final_logit_softcapping",
            "input_embedding_scale",
            "output_multiplier",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            if key in dflash:
                flat[key] = dflash[key]

        rope_parameters = flat.pop("rope_parameters", None)
        if isinstance(rope_parameters, Mapping):
            rope_parameters = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                flat["rope_theta"] = rope_parameters.pop("rope_theta")
            flat["rope_scaling"] = rope_parameters

        if "runtime_block_size" not in flat:
            flat["runtime_block_size"] = min(5, int(flat["block_size"]))

        signature = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict


__all__ = ["DFlash2Config"]
