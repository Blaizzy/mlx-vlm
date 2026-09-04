from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..dspark.config import DSparkConfig, _is_strictly_increasing_ints


@dataclass
class Gemma4DsparkConfig(DSparkConfig):
    """DSpark drafter over a Gemma 4 draft backbone.

    Published checkpoints declare the generic ``gemma4_text`` model type and
    identify themselves through the ``Gemma4DSparkModel`` architecture tag. They
    also publish the DSpark fields flat rather than under ``dflash_config``, so
    ``from_dict`` reads either layout. The remaining fields mirror the Gemma 4
    text config that the reused ``models.gemma4.language`` layers read.
    """

    model_type: str = "gemma4_dspark"
    backbone_model_type: str = "gemma4"
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int | None = 512
    num_global_key_value_heads: int | None = 1
    attention_k_eq_v: bool = True
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    final_logit_softcapping: float | None = 30.0
    tie_word_embeddings: bool = False
    sliding_window: int | None = 1024
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    rope_traditional: bool = False
    num_anchors: int = 512

    def validate(self) -> None:
        """Validate the model-agnostic DSpark invariants.

        The base checks additionally assert Qwen3 backbone properties that
        Gemma 4 does not share: ``hidden_size`` is independent of
        ``num_attention_heads * head_dim`` because full-attention layers project
        to ``global_head_dim``, and the activation is GeGLU rather than SiLU.
        """
        if self.model_type != "gemma4_dspark":
            raise ValueError(
                "Gemma 4 DSpark requires normalized model_type='gemma4_dspark', "
                f"got {self.model_type!r}."
            )
        for key in (
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
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"DSpark requires a positive integer {key}.")
        if self.block_size != self.proposal_length + 1:
            raise ValueError(
                "DSpark verification block_size must equal proposal_length + 1."
            )
        if self.markov_head_type != "vanilla":
            raise ValueError(
                "DSpark currently supports only markov_head_type='vanilla'."
            )
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("DSpark mask_token_id must be inside the vocabulary.")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("DSpark requires one layer type per draft layer.")
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
    def from_dict(cls, params: dict) -> "Gemma4DsparkConfig":
        flat = dict(params)
        nested = flat.pop("dflash_config", None)
        source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else flat

        flat["backbone_model_type"] = str(flat.get("model_type", "gemma4"))
        flat["model_type"] = "gemma4_dspark"

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
            if key in source:
                flat[key] = source[key]

        raw_block_size = source.get("block_size", flat.get("block_size"))
        if raw_block_size is None:
            raise ValueError("DSpark requires a checkpoint block_size.")
        flat["proposal_length"] = int(raw_block_size)
        flat["block_size"] = int(raw_block_size) + 1
        flat.setdefault("runtime_block_size", min(8, flat["block_size"]))
        flat.setdefault("block_size_policy", "fixed")

        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in flat.items() if k in known})
