#!/usr/bin/env python3
"""Spike one-storage Q4 gate+up gather at GLM-5.3 production dimensions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_vlm.models.switch_layers import (
    FusedSwitchGLU,
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
)
from mlx_vlm.models.fast_ops import exact_affine_switch_gate_up


def quantized(input_dims, output_dims, experts):
    layer = QuantizedSwitchLinear(
        input_dims,
        output_dims,
        experts,
        bias=False,
        group_size=64,
        bits=4,
        mode="affine",
    )
    layer.scales = layer.scales.astype(mx.bfloat16)
    layer.biases = layer.biases.astype(mx.bfloat16)
    return layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    mx.random.seed(913)
    model = SwitchGLU(1, 1, 1, bias=False)
    model.up_proj = quantized(4096, 2048, args.experts)
    model.gate_proj = quantized(4096, 2048, args.experts)
    model.down_proj = quantized(2048, 4096, args.experts)
    model.eval()

    pair = QuantizedSwitchLinear(64, 64, 1, bias=False, group_size=64, bits=4)
    pair.weight = mx.contiguous(
        mx.concatenate([model.gate_proj.weight, model.up_proj.weight], axis=1)
    )
    pair.scales = mx.contiguous(
        mx.concatenate([model.gate_proj.scales, model.up_proj.scales], axis=1)
    )
    pair.biases = mx.contiguous(
        mx.concatenate([model.gate_proj.biases, model.up_proj.biases], axis=1)
    )
    pair.group_size = 64
    pair.bits = 4
    pair.mode = "affine"
    fused_model = FusedSwitchGLU(1, 1, 1, activation=model.activation, bias=False)
    fused_model.gate_up_proj = pair
    fused_model.down_proj = model.down_proj
    fused_model.eval()
    mx.eval(model.parameters(), fused_model.parameters())

    x = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    indices = mx.array(
        [[[value % args.experts for value in (0, 17, 41, 88, 131, 177, 233, 287)]]],
        dtype=mx.int32,
    )
    weights = mx.softmax(mx.random.normal((1, 1, 8)), axis=-1)
    shared = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)

    def fused():
        return fused_model(x, indices, weights, shared)

    for _ in range(4):
        mx.eval(model(x, indices, weights, shared), fused())

    pairs = []
    for index in range(args.runs):
        order = (
            ("stock", lambda: model(x, indices, weights, shared)),
            ("fused", fused),
        )
        if index % 2:
            order = tuple(reversed(order))
        timings = {}
        outputs = {}
        for label, operation in order:
            started = time.perf_counter()
            output = operation()
            mx.eval(output)
            timings[label] = time.perf_counter() - started
            outputs[label] = output
        pairs.append(
            {
                "stock_ms": timings["stock"] * 1000,
                "fused_ms": timings["fused"] * 1000,
                "speedup": timings["stock"] / timings["fused"],
                "bit_exact": bool(mx.array_equal(outputs["stock"], outputs["fused"])),
                "max_abs_diff": float(
                    mx.max(mx.abs(outputs["stock"] - outputs["fused"]))
                ),
            }
        )

    report = {
        "schema_version": 1,
        "shape": {
            "tokens": 1,
            "hidden_size": 4096,
            "intermediate_size": 2048,
            "experts": args.experts,
            "experts_per_token": 8,
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "median_stock_ms": statistics.median(pair["stock_ms"] for pair in pairs),
        "median_fused_ms": statistics.median(pair["fused_ms"] for pair in pairs),
        "median_paired_speedup": statistics.median(pair["speedup"] for pair in pairs),
        "all_bit_exact": all(pair["bit_exact"] for pair in pairs),
        "max_abs_diff": max(pair["max_abs_diff"] for pair in pairs),
        "pairs": pairs,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    projection_checks = {}
    for length in (1, 2, 4, 8, 16, 64):
        block_x = mx.random.normal((1, length, 4096)).astype(mx.bfloat16)
        block_indices = (
            mx.arange(length * 8, dtype=mx.int32).reshape(1, length, 8) % args.experts
        )
        projected = mx.expand_dims(block_x, (-2, -3))
        selected = block_indices
        sorted_indices = block_indices.size >= 64
        if sorted_indices:
            projected, selected, _ = _gather_sort(projected, block_indices)
        stock_up = model.up_proj(projected, selected, sorted_indices=sorted_indices)
        stock_gate = model.gate_proj(projected, selected, sorted_indices=sorted_indices)
        fused_pair = pair(projected, selected, sorted_indices=sorted_indices)
        fused_gate, fused_up = mx.split(fused_pair, 2, axis=-1)
        mx.eval(stock_up, stock_gate, fused_up, fused_gate)
        up_diff = float(mx.max(mx.abs(stock_up - fused_up)))
        gate_diff = float(mx.max(mx.abs(stock_gate - fused_gate)))
        projection_checks[str(length)] = {
            "sorted": sorted_indices,
            "up_bit_exact": bool(mx.array_equal(stock_up, fused_up)),
            "gate_bit_exact": bool(mx.array_equal(stock_gate, fused_gate)),
            "max_abs_diff": max(up_diff, gate_diff),
        }
        if 1 < length <= 8:
            exact = exact_affine_switch_gate_up(model, block_x, block_indices)
            direct_pair = pair(
                mx.expand_dims(block_x, (-2, -3)),
                block_indices,
                sorted_indices=False,
            )
            direct_gate, direct_up = mx.split(direct_pair, 2, axis=-1)
            direct_up = direct_up.squeeze(-2)
            direct_gate = direct_gate.squeeze(-2)
            mx.eval(*exact, direct_up, direct_gate)
            projection_checks[str(length)]["short_block_exact"] = bool(
                mx.array_equal(exact[0], direct_up)
                and mx.array_equal(exact[1], direct_gate)
            )
            projection_checks[str(length)]["short_block_max_abs_diff"] = max(
                float(mx.max(mx.abs(exact[0] - direct_up))),
                float(mx.max(mx.abs(exact[1] - direct_gate))),
            )
    report["projection_checks"] = projection_checks
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "pairs"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
