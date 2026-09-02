"""Conversion of the official Qwen4-Exp FP8 checkpoint layout."""

import re
from collections import defaultdict

import mlx.core as mx

from ..qwen3_5.fp8 import _dequantize_qwen_fp8_weight

_EXPERT_WEIGHT_RE = re.compile(
    r"^(.*\.layers\.\d+\.mlp)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_SHARD_MARKER = ".ple.ple_embedding.ngram_embedding.shard_"


def convert_qwen4_exp_fp8_weights(weights: dict[str, mx.array]):
    """Restore FP8 experts/PLE and pack experts into mlx-vlm's model layout."""
    has_fp8_experts = any(
        _EXPERT_WEIGHT_RE.match(key) and f"{key}_scale_inv" in weights
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
        scale_key = f"{key}_scale_inv"
        if scale_key not in converted:
            raise ValueError(f"Missing FP8 scale for expert tensor {key!r}.")
        expert_groups[prefix][projection][int(expert)] = (
            converted.pop(key),
            converted.pop(scale_key),
        )

    for prefix, projections in expert_groups.items():
        if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"Incomplete FP8 expert projections under {prefix!r}.")
        expert_ids = sorted(projections["gate_proj"])
        if expert_ids != list(range(len(expert_ids))) or any(
            sorted(projections[name]) != expert_ids for name in projections
        ):
            raise ValueError(f"FP8 expert IDs must be contiguous under {prefix!r}.")

        restored = {
            name: mx.stack(
                [
                    _dequantize_qwen_fp8_weight(*projections[name][expert])
                    for expert in expert_ids
                ]
            )
            for name in projections
        }
        converted[f"{prefix}.experts.gate_up_proj"] = mx.concatenate(
            [restored["gate_proj"], restored["up_proj"]], axis=1
        )
        converted[f"{prefix}.experts.down_proj"] = restored["down_proj"]

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
