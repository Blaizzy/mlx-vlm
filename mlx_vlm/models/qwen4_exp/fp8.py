"""Conversion of the official Qwen4-Exp FP8 checkpoint layout."""

import re
from collections import defaultdict

import mlx.core as mx

from ...fp8 import _dequantize_fp8_weight

_EXPERT_WEIGHT_RE = re.compile(
    r"^(.*\.layers\.\d+\.mlp)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_SHARD_MARKER = ".ple.ple_embedding.ngram_embedding.shard_"


def convert_qwen4_exp_fp8_weights(weights: dict[str, mx.array]):
    """Restore FP8 experts/PLE and pack experts into mlx-vlm's model layout."""
    has_fp8_experts = any(
        _EXPERT_WEIGHT_RE.match(key)
        and (
            f"{key}_scale_inv" in weights
            or f"{key[: -len('.weight')]}.scales" in weights
        )
        for key in weights
    )
    has_fp8_ple = any(key.endswith(_PLE_SCALE_SUFFIX) for key in weights)
    if not (has_fp8_experts or has_fp8_ple):
        return weights

    converted = dict(weights)
    expert_groups = defaultdict(lambda: defaultdict(dict))
    for key in list(converted):
        match = _EXPERT_WEIGHT_RE.match(key)
        if match is None:
            continue
        prefix, expert, projection = match.groups()
        scale_inv_key = f"{key}_scale_inv"
        parameter_prefix = key[: -len(".weight")]
        scale_key = f"{parameter_prefix}.scales"
        if scale_inv_key in converted:
            expert_groups[prefix][projection][int(expert)] = {
                "dense": _dequantize_fp8_weight(
                    converted.pop(key), converted.pop(scale_inv_key)
                )
            }
        elif scale_key in converted:
            parameters = {
                "weight": converted.pop(key),
                "scales": converted.pop(scale_key),
            }
            bias_key = f"{parameter_prefix}.biases"
            if bias_key in converted:
                parameters["biases"] = converted.pop(bias_key)
            expert_groups[prefix][projection][int(expert)] = parameters
        else:
            raise ValueError(f"Missing FP8 scale for expert tensor {key!r}.")

    for prefix, projections in expert_groups.items():
        if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"Incomplete FP8 expert projections under {prefix!r}.")
        expert_ids = sorted(projections["gate_proj"])
        if expert_ids != list(range(len(expert_ids))) or any(
            sorted(projections[name]) != expert_ids for name in projections
        ):
            raise ValueError(f"FP8 expert IDs must be contiguous under {prefix!r}.")

        parameter_names = set(projections["gate_proj"][expert_ids[0]])
        if any(
            set(projections[name][expert]) != parameter_names
            for name in projections
            for expert in expert_ids
        ):
            raise ValueError(f"Inconsistent FP8 expert layout under {prefix!r}.")

        stacked = {
            name: {
                parameter: mx.stack(
                    [projections[name][expert][parameter] for expert in expert_ids]
                )
                for parameter in parameter_names
            }
            for name in projections
        }
        if parameter_names == {"dense"}:
            converted[f"{prefix}.experts.gate_up_proj"] = mx.concatenate(
                [stacked["gate_proj"]["dense"], stacked["up_proj"]["dense"]],
                axis=-2,
            )
            converted[f"{prefix}.experts.down_proj"] = stacked["down_proj"]["dense"]
        else:
            for parameter in parameter_names:
                converted[f"{prefix}.experts.gate_up_proj.{parameter}"] = (
                    mx.concatenate(
                        [
                            stacked["gate_proj"][parameter],
                            stacked["up_proj"][parameter],
                        ],
                        axis=-2,
                    )
                )
                converted[f"{prefix}.experts.down_proj.{parameter}"] = stacked[
                    "down_proj"
                ][parameter]

    ple_scale_keys = [key for key in converted if key.endswith(_PLE_SCALE_SUFFIX)]
    for scale_key in ple_scale_keys:
        prefix = scale_key[: -len("weight_scale")]
        scale = converted.pop(scale_key)
        shard_keys = sorted(
            (key for key in converted if key.startswith(prefix + "shard_")),
            key=lambda key: int(key.split(".shard_", 1)[1].split(".", 1)[0]),
        )
        if not shard_keys:
            raise ValueError(f"Missing FP8 PLE shards for scale {scale_key!r}.")
        for shard_key in shard_keys:
            weight = converted[shard_key]
            if weight.dtype != mx.uint8:
                raise ValueError(f"FP8 PLE shard must load as uint8: {shard_key!r}.")
            converted[shard_key] = mx.from_fp8(
                weight, dtype=mx.bfloat16
            ) * scale.reshape(())

    return converted
