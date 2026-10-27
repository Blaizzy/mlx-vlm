"""JSON model contracts, checkpoint loading, sanitization, and document layout."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import textwrap
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from fnmatch import fnmatchcase
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm import embedding_loader
from mlx_vlm.models.base import InputEmbeddingsFeatures, create_attention_mask
from mlx_vlm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_vlm.models.lfm2_encoder import Model as Lfm2Encoder
from mlx_vlm.utils import (
    _drop_modules_without_weights,
    _load_safetensors,
    get_model_and_args,
    get_model_path,
    load,
    load_config,
    load_model,
)


class TestMiniMaxM31Flash(unittest.TestCase):
    def test_router_dtype_sanitization_is_idempotent(self):
        from mlx_vlm.models.minimax_m3_vl import Model, ModelConfig

        config = ModelConfig.from_dict(
            {
                "text_config": {
                    "hidden_size": 64,
                    "intermediate_size": 64,
                    "shared_intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 32,
                    "vocab_size": 128,
                    "num_local_experts": 2,
                    "num_experts_per_tok": 1,
                    "moe_layer_freq": [1],
                    "nvfp4_experts": True,
                },
                "vision_config": {
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 0,
                    "image_size": 8,
                    "patch_size": 2,
                },
                "projector_hidden_size": 64,
            }
        )
        model = Model(config)
        prefix = "language_model.model.layers.0.block_sparse_moe"
        gate_key = f"{prefix}.gate.weight"
        bias_key = f"{prefix}.e_score_correction_bias"
        scale_key = f"{prefix}.switch_mlp.gate_proj.weight_global_scale"
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            with self.subTest(dtype=dtype):
                gate = mx.arange(128).reshape(2, 64) / 129.0
                bias = mx.array([0.137, -0.271])
                source = {
                    "language_model.model.embed_tokens.weight": mx.zeros(
                        (128, 64), dtype
                    ),
                    gate_key: gate,
                    bias_key: bias,
                    scale_key: mx.array([17.3, 15123.7]),
                }
                first = model.sanitize(dict(source))
                second = model.sanitize(dict(first))
                for key in source:
                    expected = (
                        source[key].astype(dtype)
                        if key in (gate_key, bias_key)
                        else source[key]
                    )
                    self.assertEqual(first[key].dtype, expected.dtype)
                    np.testing.assert_array_equal(
                        np.array(first[key].astype(mx.float32)),
                        np.array(expected.astype(mx.float32)),
                    )
                    np.testing.assert_array_equal(
                        np.array(second[key].astype(mx.float32)),
                        np.array(first[key].astype(mx.float32)),
                    )
                self.assertTrue(model.cast_predicate(gate_key))
                self.assertTrue(model.cast_predicate(bias_key))
                self.assertFalse(model.cast_predicate(scale_key))

                # A previously quantized router must retain its packed bits.
                source[gate_key] = mx.full((2, 16), 0x12345678, dtype=mx.uint32)
                sanitized = model.sanitize(dict(source))
                self.assertEqual(sanitized[gate_key].dtype, mx.uint32)
                np.testing.assert_array_equal(
                    np.array(sanitized[gate_key]), np.array(source[gate_key])
                )

    def test_nvfp4_global_scales_match_explicit_dequantization(self):
        from mlx_vlm.models.switch_layers import QuantizedSwitchLinear
        from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear

        mx.random.seed(31)

        layer = ScaledQuantizedSwitchLinear(
            QuantizedSwitchLinear(
                64, 32, 3, bias=False, group_size=16, bits=4, mode="nvfp4"
            ),
            mx.array([0.5, 17.3, 15123.7]),
            "weight_global_scale",
        )
        layer.weight, layer.scales = mx.quantize(
            mx.random.normal((3, 32, 64)) * 1000,
            group_size=16,
            bits=4,
            mode="nvfp4",
        )
        # Decode the source E2M1 nibbles and E4M3 bytes independently of MLX.
        packed = np.array(layer.weight).view(np.uint8)
        codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(3, 32, 64)
        fp4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
        values = fp4[codes & 7] * np.where(codes & 8, -1.0, 1.0)
        scales = np.array(layer.scales)
        exponent = (scales >> 3) & 15
        mantissa = scales & 7
        decoded_scales = np.where(
            exponent == 0,
            mantissa / 512.0,
            (1 + mantissa / 8.0) * np.exp2(exponent.astype(np.int32) - 7),
        ) * np.where(scales & 128, -1.0, 1.0)
        weights = values * np.repeat(decoded_scales, 16, axis=-1)
        weights /= np.array(layer.weight_global_scale)[:, None, None]

        for sorted_indices in (False, True):
            indices = mx.array([0, 1, 2] if sorted_indices else [2, 0, 1])
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                with self.subTest(sorted_indices=sorted_indices, dtype=dtype):
                    x = mx.random.normal((3, 1, 64)).astype(dtype)
                    actual = layer(x, indices, sorted_indices=sorted_indices)
                    expected = np.array(x.astype(mx.float32)) @ weights[
                        np.array(indices)
                    ].transpose(0, 2, 1)
                    np.testing.assert_allclose(
                        np.array(actual.astype(mx.float32)),
                        expected,
                        rtol=0.02 if dtype == mx.bfloat16 else 0.002,
                        atol=2.0 if dtype != mx.float32 else 0.02,
                    )
                    self.assertEqual(actual.dtype, dtype)

    def test_mixed_checkpoint_load_forward_cache_and_roundtrip(self):
        import copy
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.minimax_m3_vl import Model, ModelConfig
        from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear
        from mlx_vlm.utils import load_model, save_weights

        config = {
            "model_type": "minimax_m3_vl",
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 64,
                "dense_intermediate_size": 128,
                "shared_intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "rotary_dim": 16,
                "rope_theta": 10000000,
                "vocab_size": 128,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": [0, 1],
                "pack_shared_expert": False,
                "sparse_attention_config": {
                    "use_sparse_attention": True,
                    "sparse_attention_freq": [1, 1],
                    "sparse_index_dim": 32,
                    "sparse_num_index_heads": 1,
                    "sparse_block_size": 4,
                    "sparse_topk_blocks": 2,
                },
            },
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 0,
                "image_size": 8,
                "patch_size": 2,
            },
            "projector_hidden_size": 64,
            "vocab_size": 128,
        }
        mx.random.seed(31)
        reference = Model(ModelConfig.from_dict(copy.deepcopy(config)))
        source = {}
        reference_weights = {}
        for key, value in tree_flatten(reference.parameters()):
            if ".switch_mlp." in key:
                prefix, projection = key.split(".switch_mlp.")
                source_name = {
                    "gate_proj.weight": "w1",
                    "down_proj.weight": "w2",
                    "up_proj.weight": "w3",
                }[projection]
                dequantized = []
                for expert in range(2):
                    global_scale = mx.array(0.7 if expert == 0 else 15123.7)
                    packed, scales = mx.quantize(
                        value[expert] * global_scale,
                        group_size=16,
                        bits=4,
                        mode="nvfp4",
                    )
                    target = f"{prefix}.experts.{expert}.{source_name}"
                    source[f"{target}.weight_packed"] = packed.view(mx.uint8)
                    source[f"{target}.weight_scale"] = scales
                    source[f"{target}.weight_global_scale"] = global_scale
                    source[f"{target}.input_global_scale"] = mx.array(1.0)
                    dequantized.append(
                        mx.dequantize(
                            packed, scales, group_size=16, bits=4, mode="nvfp4"
                        )
                        / global_scale
                    )
                reference_weights[key] = mx.stack(dequantized)
            elif key.startswith("language_model.") and key.endswith("_proj.weight"):
                packed, scales = mx.quantize(value, 32, 8, mode="mxfp8")
                source[key] = packed.view(mx.uint8)
                source[key.replace(".weight", ".weight_scale_inv")] = scales
                reference_weights[key] = mx.dequantize(
                    packed, scales, group_size=32, bits=8, mode="mxfp8"
                )
            else:
                source[key] = value
                reference_weights[key] = value
        reference.load_weights(list(reference_weights.items()))
        reference.eval()
        config["quantization_config"] = {
            "quant_method": "mxfp8",
            "weight_block_size": [1, 32],
            "group_size": 16,
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config))
            mx.save_safetensors(str(path / "model.safetensors"), source)
            model = load_model(path)
            moe = model.layers[1].block_sparse_moe
            self.assertIsInstance(moe.switch_mlp.gate_proj, ScaledQuantizedSwitchLinear)
            self.assertIsInstance(moe.shared_experts.gate_proj, nn.QuantizedLinear)
            self.assertEqual(moe.shared_experts.gate_proj.mode, "mxfp8")
            self.assertFalse(moe.pack_shared_expert)
            self.assertTrue(
                all(layer.self_attn.has_sparse_index for layer in model.layers)
            )
            self.assertFalse(
                model.cast_predicate(
                    "language_model.model.layers.1.block_sparse_moe.switch_mlp.gate_proj.weight_global_scale"
                )
            )

            tokens = mx.array([[1, 2, 3, 4, 5]])
            expected = reference(tokens).logits
            actual = model(tokens).logits
            np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-4)
            cache = model.make_cache()
            model(tokens[:, :-1], cache=cache)
            decoded = model(tokens[:, -1:], cache=cache).logits
            np.testing.assert_allclose(
                np.array(decoded), np.array(actual[:, -1:]), atol=2e-4
            )

            converted = path / "converted"
            save_weights(converted, model)
            (converted / "config.json").write_text(json.dumps(config))
            reloaded = load_model(converted)
            np.testing.assert_allclose(
                np.array(reloaded(tokens).logits), np.array(actual), atol=2e-4
            )


class TestMiniMaxSparseAccuracy(unittest.TestCase):
    @staticmethod
    def round_bf16(value):
        bits = np.asarray(value, dtype=np.float32).view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

    def test_swiglu_preserves_bf16_operation_boundaries(self):
        from mlx_vlm.models.minimax_m3_vl.language import _swiglu_oai

        rng = np.random.default_rng(876)
        gate = self.round_bf16(rng.normal(size=(4, 257)) * 4)
        up = self.round_bf16(rng.normal(size=gate.shape) * 4)
        clipped_gate = np.minimum(gate, 7)
        clipped_up = np.clip(up, -7, 7)
        scaled = self.round_bf16(clipped_gate * np.float32(1.702))
        sigmoid = self.round_bf16(1 / (1 + np.exp(-scaled.astype(np.float64))))
        glu = self.round_bf16(clipped_gate * sigmoid)
        linear = self.round_bf16(clipped_up + np.float32(1))
        expected = self.round_bf16(linear * glu)
        actual = _swiglu_oai(
            mx.array(up).astype(mx.bfloat16), mx.array(gate).astype(mx.bfloat16)
        )
        self.assertEqual(actual.dtype, mx.bfloat16)
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected)

    def test_expert_sum_is_independent_of_topk_output_order(self):
        from mlx_vlm.models.minimax_m3_vl.config import TextConfig
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxSparseMoeBlock

        layer = MiniMaxSparseMoeBlock(
            TextConfig(
                hidden_size=32,
                intermediate_size=32,
                num_local_experts=4,
                num_experts_per_tok=4,
                n_shared_experts=0,
                pack_shared_expert=False,
                routed_scaling_factor=1.0,
            )
        )
        # Equal probabilities, different selection orders. The BF16 reference
        # sums 256, 1, -256, 1 in expert-ID order, producing 1 rather than 2.
        layer.gate = lambda x: mx.zeros((*x.shape[:-1], 4))
        values = mx.broadcast_to(
            mx.array([1024, 4, -1024, 4], mx.bfloat16)[:, None], (4, 32)
        )
        layer.switch_mlp = lambda x, indices: values[indices]
        x = mx.zeros((1, 1, 32), mx.bfloat16)
        for bias in ([1, 2, 4, 3], [4, 3, 2, 1], [2, 4, 1, 3]):
            with self.subTest(bias=bias):
                layer.e_score_correction_bias = mx.array(bias, mx.float32)
                np.testing.assert_array_equal(
                    np.array(layer(x).astype(mx.float32)), np.ones((1, 1, 32))
                )

    @staticmethod
    def attention():
        from mlx_vlm.models.minimax_m3_vl.config import TextConfig
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxAttention

        return MiniMaxAttention(
            TextConfig(
                hidden_size=64,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=32,
                rotary_dim=16,
                num_hidden_layers=1,
                sparse_attention_config={
                    "use_sparse_attention": True,
                    "sparse_attention_freq": [1],
                    "sparse_index_dim": 32,
                    "sparse_num_index_heads": 4,
                    "sparse_block_size": 4,
                    "sparse_topk_blocks": 2,
                    "sparse_local_block": 1,
                },
            ),
            0,
        )

    @staticmethod
    def attention_reference(q, k, v, mask):
        # Independent FP64 attention over explicitly expanded KV heads.
        q, k, v = (
            np.asarray(a.astype(mx.float32), dtype=np.float64) for a in (q, k, v)
        )
        k = np.repeat(k, q.shape[1] // k.shape[1], axis=1)
        v = np.repeat(v, q.shape[1] // v.shape[1], axis=1)
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
        scores = (
            np.where(mask, scores, -np.inf) if mask.dtype == bool else scores + mask
        )
        peak = np.max(scores, axis=-1, keepdims=True)
        scores = np.exp(scores - np.where(np.isfinite(peak), peak, 0))
        return (scores / np.maximum(scores.sum(-1, keepdims=True), 1e-30)) @ v

    def test_independent_index_heads_and_padding(self):
        attention = self.attention()
        rng = np.random.default_rng(31)
        for batch in (1, 4):
            queries = rng.normal(size=(batch, 4, 3, 4)).astype(np.float32)
            keys = rng.normal(size=(batch, 1, 23, 4)).astype(np.float32)
            positions = np.broadcast_to([0, 17, 22], (batch, 3)).copy()
            for score_type, masked, initial in (
                ("max", False, 0),
                ("max", False, 1),
                ("max", True, 0),
                ("lse", True, 0),
            ):
                with self.subTest(
                    batch=batch, score_type=score_type, masked=masked, initial=initial
                ):
                    attention.indexer.score_type = score_type
                    attention.indexer.init_blocks = initial
                    valid = np.ones((batch, 1, 3, 23), dtype=bool)
                    if masked:
                        valid[..., 4:8] = False  # Exclude an entire earlier block.
                        valid[..., 21:] = False  # Partially padded final block.
                    scores = (queries @ keys.swapaxes(-1, -2)) * attention.scale
                    causal = (
                        np.arange(23)[None, None, None, :]
                        <= positions[:, None, :, None]
                    )
                    scores = np.where(causal & valid, scores, -np.inf)
                    scores = np.pad(
                        scores,
                        ((0, 0), (0, 0), (0, 0), (0, 1)),
                        constant_values=-np.inf,
                    )
                    blocks = scores.reshape(batch, 4, 3, 6, 4)
                    pooled = (
                        blocks.max(-1)
                        if score_type == "max"
                        else np.logaddexp.reduce(blocks, axis=-1)
                    )
                    allowed = np.isfinite(pooled)
                    if initial:
                        pooled[..., :initial] = np.where(
                            allowed[..., :initial], 1e30, -np.inf
                        )
                    for b in range(batch):
                        for t in range(3):
                            local = positions[b, t] // 4
                            pooled[b, :, t, local] = np.where(
                                allowed[b, :, t, local], 1e29, -np.inf
                            )
                    selected = np.argsort(-pooled, axis=-1)[..., :2]
                    selected = np.where(
                        np.take_along_axis(allowed, selected, -1), selected, 6
                    )
                    selected = np.sort(selected, axis=-1)
                    selected[selected == 6] = -1
                    actual = attention.indexer.select_blocks(
                        mx.array(queries),
                        mx.array(keys),
                        0,
                        mx.array(valid) if masked else None,
                        mx.array(positions),
                    )
                    np.testing.assert_array_equal(np.array(actual), selected)
                    if not initial:
                        self.assertFalse(np.array_equal(selected[:, 0], selected[:, 1]))
                    expected_mask = np.any(
                        selected[..., None] == np.arange(23) // 4, axis=-2
                    )
                    expected_mask = np.repeat(expected_mask & causal & valid, 2, axis=1)
                    actual_mask = attention.indexer.build_block_mask(
                        actual,
                        mx.array(valid) if masked else None,
                        23,
                        mx.float32,
                        mx.array(positions),
                    )
                    np.testing.assert_array_equal(np.array(actual_mask), expected_mask)
                    unsorted = attention.indexer.select_blocks(
                        mx.array(queries),
                        mx.array(keys),
                        0,
                        mx.array(valid) if masked else None,
                        mx.array(positions),
                        sort_blocks=False,
                    )
                    unsorted_mask = attention.indexer.build_block_mask(
                        unsorted,
                        mx.array(valid) if masked else None,
                        23,
                        mx.float32,
                        mx.array(positions),
                    )
                    np.testing.assert_array_equal(
                        np.array(unsorted_mask), expected_mask
                    )
                    legacy = attention._build_sparse_mask(
                        mx.array(queries),
                        mx.array(keys),
                        0,
                        mx.array(valid) if masked else None,
                        q_positions=mx.array(positions),
                    )
                    np.testing.assert_array_equal(np.array(legacy), expected_mask)

    def test_sparse_block_mask_invalid_duplicate_and_partial_blocks(self):
        attention = self.attention()
        indices = np.array(
            [
                [
                    [[5, 1, 1, -1], [-1, -1, 6, 99], [0, 5, -2, 2]],
                    [[0, 2, 4, 4], [1, -1, 3, 5], [-1, -1, -1, -1]],
                ]
            ],
            dtype=np.int32,
        )
        positions = np.array([[10, 17, 22]])
        expected = np.any(indices[..., None] == np.arange(23) // 4, axis=-2)
        expected &= np.arange(23)[None, None, None, :] <= positions[:, None, :, None]
        expected = np.repeat(expected, 4, axis=1)
        allowed = np.ones((1, 8, 3, 23), dtype=bool)
        allowed[:, ::2, :, 4:9] = False
        for kind in ("none", "bool", "additive"):
            with self.subTest(mask=kind):
                mask = None
                ref = expected
                if kind == "bool":
                    mask = mx.array(allowed)
                    ref = expected & allowed
                elif kind == "additive":
                    bias = np.where(allowed, -0.25, -np.inf).astype(np.float32)
                    mask = mx.array(bias)
                    ref = np.where(expected, 0.0, -np.inf) + bias
                actual = attention.indexer.build_block_mask(
                    mx.array(indices), mask, 23, mx.float32, mx.array(positions)
                )
                np.testing.assert_array_equal(np.array(actual), ref)

    def test_sparse_attention_full_prefill_matches_cached_chunks(self):
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache

        mx.random.seed(31)
        attention = self.attention()
        for batch in (1, 4):
            for length in (23, 513):
                with self.subTest(batch=batch, length=length):
                    x = mx.random.normal((batch, length, 64))
                    expected = attention(x, mask="causal")
                    cache = MiniMaxM3KVCache()
                    chunks = []
                    # The long case crosses the dense/sparse decode boundary.
                    stops = (
                        (0, 7, 14, 22, 23) if length == 23 else (0, 500, 511, 512, 513)
                    )
                    for start, stop in zip(stops, stops[1:]):
                        chunk = attention(x[:, start:stop], mask="causal", cache=cache)
                        mx.eval(chunk)
                        chunks.append(chunk)
                    np.testing.assert_allclose(
                        np.array(mx.concatenate(chunks, axis=1)),
                        np.array(expected),
                        rtol=2e-5,
                        atol=2e-6,
                    )

    @unittest.skipUnless(mx.metal.is_available(), "Requires Metal sparse attention")
    def test_sparse_metal_kernel_per_group(self):
        from mlx_vlm.models.minimax_m3_vl.language import (
            _minimax_m3_sparse_prefill_attention,
        )

        mx.random.seed(31)
        for batch, heads in ((1, 1), (1, 4), (4, 2), (4, 4)):
            for dtype in (mx.float16, mx.bfloat16):
                with self.subTest(batch=batch, index_heads=heads, dtype=dtype):
                    q = mx.random.normal((batch, 8, 3, 32)).astype(dtype)
                    k = mx.random.normal((batch, 4, 137, 32)).astype(dtype)
                    v = mx.random.normal(k.shape).astype(dtype)
                    indices = np.empty((batch, heads, 3, 2), dtype=np.int32)
                    for b in range(batch):
                        for h in range(heads):
                            indices[b, h, :, 0] = 1 + (b + h) % 4
                            indices[b, h, :, 1] = 34
                    indices[..., 0, 1] = -1
                    positions = np.broadcast_to([20, 130, 136], (batch, 3)).copy()
                    mask = np.any(indices[..., None] == np.arange(137) // 4, axis=-2)
                    mask &= (
                        np.arange(137)[None, None, None, :]
                        <= positions[:, None, :, None]
                    )
                    mask = np.repeat(mask, 8 // heads, axis=1)
                    actual = _minimax_m3_sparse_prefill_attention(
                        q,
                        k,
                        v,
                        mx.array(indices),
                        mx.array(positions),
                        32**-0.5,
                        4,
                    )
                    self.assertIsNotNone(actual)
                    expected = self.attention_reference(q, k, v, mask)
                    np.testing.assert_allclose(
                        np.array(actual.astype(mx.float32)),
                        expected,
                        rtol=0.008 if dtype == mx.bfloat16 else 0.002,
                        atol=0.004 if dtype == mx.bfloat16 else 0.0005,
                    )

    def test_sparse_gather_decode_per_group(self):
        attention = self.attention()
        mx.random.seed(31)
        for heads in (1, 2, 4):
            for mask_type in ("none", "bool", "additive"):
                with self.subTest(index_heads=heads, mask_type=mask_type):
                    batch, length = 4, 1027
                    q = mx.random.normal((batch, 8, 1, 32))
                    k = mx.random.normal((batch, 4, length, 32))
                    v = mx.random.normal(k.shape)
                    indices = np.empty((batch, heads, 1, 2), dtype=np.int32)
                    for b in range(batch):
                        for h in range(heads):
                            indices[b, h, 0] = [1 + b + h, 256]
                    indices[0, 0, 0, 1] = -1
                    positions = np.array([[1024], [1025], [1026], [1026]])
                    mask = np.any(indices[..., None] == np.arange(length) // 4, axis=-2)
                    mask &= (
                        np.arange(length)[None, None, None, :]
                        <= positions[:, None, :, None]
                    )
                    mask = np.repeat(mask, 8 // heads, axis=1)
                    explicit = None
                    if mask_type != "none":
                        allowed = np.ones((batch, 8, 1, length), dtype=bool)
                        allowed[:, ::2, :, ::3] = False
                        explicit = (
                            allowed
                            if mask_type == "bool"
                            else np.where(allowed, -0.7, -np.inf).astype(np.float32)
                        )
                        mask = (
                            mask & explicit
                            if mask_type == "bool"
                            else np.where(mask, explicit, -np.inf)
                        )
                    actual = attention._sparse_decode_attention(
                        q,
                        k,
                        v,
                        mx.array(indices),
                        None if explicit is None else mx.array(explicit),
                        mx.array(positions),
                    )
                    self.assertIsNotNone(actual)
                    np.testing.assert_allclose(
                        np.array(actual),
                        self.attention_reference(q, k, v, mask),
                        rtol=2e-5,
                        atol=2e-6,
                    )

    def test_router_projection_rounds_before_fp32_scoring(self):
        from mlx_vlm.models.minimax_m3_vl.config import TextConfig
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxSparseMoeBlock

        mx.random.seed(31)
        layer = MiniMaxSparseMoeBlock(
            TextConfig(
                hidden_size=32,
                intermediate_size=32,
                num_local_experts=4,
                num_experts_per_tok=2,
                n_shared_experts=0,
                pack_shared_expert=False,
            )
        )
        layer.gate.weight = mx.random.normal((4, 32)).astype(mx.bfloat16)
        x = mx.random.normal((2, 3, 32)).astype(mx.bfloat16)
        values = (mx.random.normal((4, 32)) * 100).astype(mx.bfloat16)
        layer.switch_mlp = lambda x, inds: values[inds]
        logits = (
            np.array(x.astype(mx.float32)).astype(np.float64)
            @ np.array(layer.gate.weight.astype(mx.float32)).astype(np.float64).T
        )
        rounded = np.array(
            mx.array(logits.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
        ).astype(np.float64)
        probabilities = 1 / (1 + np.exp(-rounded))
        indices = np.argsort(-probabilities, axis=-1)[..., :2]
        scores = np.take_along_axis(probabilities, indices, axis=-1)
        scores = scores / scores.sum(-1, keepdims=True)
        order = np.argsort(indices, axis=-1)
        indices = np.take_along_axis(indices, order, -1)
        scores = np.take_along_axis(scores, order, -1).astype(np.float32)
        weighted = self.round_bf16(
            np.array(values.astype(mx.float32))[indices] * scores[..., None]
        )
        expected = self.round_bf16(weighted[..., 0, :] + weighted[..., 1, :])
        expected = self.round_bf16(expected * layer.routed_scaling_factor)
        actual = layer(x)
        self.assertEqual(actual.dtype, mx.bfloat16)
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)),
            expected,
        )

    def test_expert_contributions_round_in_expert_order(self):
        from mlx_vlm.models.minimax_m3_vl.config import TextConfig
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxSparseMoeBlock

        mx.random.seed(31)
        for biased in (False, True):
            for packed in (False, True):
                with self.subTest(biased=biased, packed=packed):
                    layer = MiniMaxSparseMoeBlock(
                        TextConfig(
                            hidden_size=32,
                            intermediate_size=32,
                            shared_intermediate_size=32,
                            num_local_experts=4,
                            num_experts_per_tok=2,
                            pack_shared_expert=packed,
                            n_shared_experts=int(packed),
                            use_routing_bias=biased,
                            routed_scaling_factor=1.7,
                        )
                    )
                    logits = mx.array([[-1.25, 0.17, 1.13, 0.61]])[None]
                    bias = mx.array([0.0, 0.5, 0.0, 0.0]) if biased else mx.zeros(4)
                    layer.gate = lambda x: logits
                    if biased:
                        layer.e_score_correction_bias = bias
                    expert_outputs = (mx.random.normal((5, 32)) * 10).astype(
                        mx.bfloat16
                    )
                    layer.switch_mlp = lambda x, inds: expert_outputs[inds]
                    x = mx.zeros((1, 1, 32), dtype=mx.bfloat16)
                    probability = 1 / (1 + np.exp(-np.array(logits).astype(np.float64)))
                    indices = np.argsort(-(probability + np.array(bias)), axis=-1)[
                        ..., :2
                    ]
                    weights = np.take_along_axis(probability, indices, -1)
                    weights = weights / weights.sum(-1, keepdims=True)
                    order = np.argsort(indices, axis=-1)
                    indices = np.take_along_axis(indices, order, -1)
                    weights = np.take_along_axis(weights, order, -1)
                    values = np.array(expert_outputs.astype(mx.float32)).astype(
                        np.float64
                    )
                    weighted = self.round_bf16(values[indices] * weights[..., None])
                    expected = self.round_bf16(
                        weighted[..., 0, :] + weighted[..., 1, :]
                    )
                    expected = self.round_bf16(expected * np.float32(1.7))
                    if packed:
                        expected = self.round_bf16(expected + values[4])
                    actual = layer(x)
                    self.assertEqual(actual.dtype, mx.bfloat16)
                    np.testing.assert_array_equal(
                        np.array(actual.astype(mx.float32)),
                        expected,
                    )

    def test_gemma_norm_preserves_small_weight_offsets(self):
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxRMSNorm

        mx.random.seed(31)
        norm = MiniMaxRMSNorm(128)
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            with self.subTest(dtype=dtype):
                norm.weight = (mx.random.normal((128,)) * 0.003).astype(dtype)
                x = mx.random.normal((2, 3, 128)).astype(dtype)
                xf = np.array(x.astype(mx.float32)).astype(np.float64)
                weight = np.array(norm.weight.astype(mx.float32)).astype(np.float64)
                expected = xf / np.sqrt(
                    np.mean(xf * xf, axis=-1, keepdims=True) + norm.eps
                )
                expected = mx.array(
                    (expected * (1 + weight)).astype(np.float32)
                ).astype(dtype)
                actual = norm(x)
                self.assertEqual(actual.dtype, dtype)
                np.testing.assert_allclose(
                    np.array(actual.astype(mx.float32)),
                    np.array(expected.astype(mx.float32)),
                    rtol=2e-6,
                    atol=2e-6,
                )

    @unittest.skipUnless(mx.metal.is_available(), "Requires Metal")
    def test_gemma_norm_matches_live_weights_and_strided_inputs(self):
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxRMSNorm

        mx.random.seed(41)
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            for dims in (7, 128, 6144):
                norm = MiniMaxRMSNorm(dims)
                norm.eval()
                x = mx.random.normal((2, 3, dims * 2)).astype(dtype)[..., ::2]
                norm.weight = (mx.random.normal((dims,)) * 0.003).astype(dtype)
                for change in (False, True):
                    with self.subTest(dtype=dtype, dims=dims, changed=change):
                        if change:
                            norm.weight[::2] = mx.array(0.25, dtype)
                        expected = mx.fast.rms_norm(
                            x, norm.weight.astype(mx.float32) + 1, norm.eps
                        ).astype(dtype)
                        self.assertTrue(mx.array_equal(norm(x), expected))


# Shared model contracts


def capture_positions(
    language_model, hidden_size, inputs, *, idx, offset, fa_idx=None, **kwargs
):
    class Recorder:
        embed_tokens = NS(as_linear=lambda x: x)

        def __call__(self, inputs, position_ids=None, **kwargs):
            self.positions = position_ids
            return mx.zeros((*inputs.shape, hidden_size))

    recorder = Recorder()
    if fa_idx is not None:
        recorder.fa_idx = fa_idx
    language_model.model = recorder
    language_model.lm_head = lambda x: x
    language_model(inputs, cache=[NS(_idx=idx, offset=mx.array(offset))], **kwargs)
    return recorder.positions


class ModelChecks:
    """Reusable assertions; each JSON case constructs fresh configs and models."""

    def causal_visual_prefix(self, language, config):
        vision = mx.random.normal((1, 3, config.d_model))
        prompt = mx.array([[2, 0, 7]])
        embeddings = mx.concatenate([vision, language.embed(prompt)], axis=1)
        full = language(inputs_embeds=embeddings).logits
        # A later visual token cannot change earlier visual states.
        changed = mx.concatenate(
            [vision[:, :2], vision[:, 2:] + 10, language.embed(prompt)], axis=1
        )
        np.testing.assert_allclose(
            np.asarray(full[:, :2]),
            np.asarray(language(inputs_embeds=changed).logits[:, :2]),
            atol=1e-5,
        )
        cache = language.make_cache()
        first = language(inputs_embeds=embeddings[:, :4], cache=cache).logits
        rest = language(inputs_embeds=embeddings[:, 4:], cache=cache).logits
        np.testing.assert_allclose(
            np.asarray(full),
            np.asarray(mx.concatenate([first, rest], axis=1)),
            atol=1e-5,
        )
        assert cache[0].offset == 6
        assert cache[0].keys.shape[1] == config.self_attention_kv_heads

    def forward_cache(self, model, vocab_size, *, chunk_sizes=()):
        model.eval()
        mx.eval(model.parameters())
        ids = mx.array([[1, 5, 9, 13, 2, 7, 11, 3]])
        assert model(ids).logits.shape == (1, 8, vocab_size)
        if chunk_sizes:
            for dtype in (mx.float32, mx.float16):
                model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
                self.prefill_parity(model, ids, {}, chunk_sizes)
            return
        cache = model.language_model.make_cache()
        model(ids[:, :-1], cache=cache)
        assert model(ids[:, -1:], cache=cache).logits.shape == (1, 1, vocab_size)

    def masked_lm(self, model, config):
        mask = mx.array([[1, 1, 1, 0]])
        first = model(mx.array([[1, 2, 3, 0]]), attention_mask=mask).logits
        second = model(mx.array([[1, 2, 3, 9]]), attention_mask=mask).logits
        assert first.shape == (1, 4, config.vocab_size)
        assert mx.allclose(first[:, :3], second[:, :3])

    def token_embeddings(self, model, config):
        output = model(mx.array([[1, 2, 3]]))
        assert output.last_hidden_state.shape == (1, 3, config.hidden_size)
        assert output.text_embeds.shape == (1, 3, config.embedding_dim)
        assert mx.allclose(
            mx.linalg.norm(output.text_embeds, axis=-1), mx.array(1.0), atol=1e-5
        )

    def sentence_embeddings(self, model, config):
        mask = mx.array([[1, 1, 1, 0]])
        output = model(mx.array([[1, 2, 3, 0]]), attention_mask=mask)
        changed_padding = model(mx.array([[1, 2, 3, 9]]), attention_mask=mask)
        width = config.text_config.embedding_dim
        assert output.last_hidden_state.shape == (1, 4, width)
        assert output.text_embeds.shape == (1, width)
        assert output.text_embeds.dtype == mx.float32
        self.assert_close(output.text_embeds, changed_padding.text_embeds)
        self.assert_close(mx.linalg.norm(output.text_embeds, axis=-1), mx.ones((1,)))

    def assert_close(self, actual, expected, *, logits=False):
        assert actual.shape == expected.shape
        tolerance = 5e-3 if actual.dtype == mx.float16 else 1e-4
        # One-token attention uses a different Metal kernel from full prefill.
        if logits:
            tolerance = max(tolerance, 2e-3)
        actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
        mx.eval(actual, expected)
        actual, expected = np.array(actual), np.array(expected)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual,
            expected,
            atol=tolerance,
            rtol=tolerance,
        )

    def prefill_parity(self, model, ids, media, chunk_sizes, *, features=None):
        """Compare full, cached and chunked prefill, then decode one new token."""
        language = model.language_model
        if features is None:
            features = model.get_input_embeddings(ids, **media)
        features = features.to_dict()
        embeds = features.pop("inputs_embeds")
        token = mx.full((ids.shape[0], 1), 7, mx.int32)
        full = model(
            mx.concatenate([ids, token], axis=1),
            cache=make_prompt_cache(language),
            **media,
        ).logits
        # One causal full pass supplies both the prefill and decode references.
        reference, extended = full[:, :-1], full[:, -1:]
        if ids.shape[0] > 1:
            solo = []
            for row in range(ids.shape[0]):
                row_media = {
                    key: value[row * count : (row + 1) * count]
                    for key, value in media.items()
                    for count in [value.shape[0] // ids.shape[0]]
                }
                solo.append(
                    model(
                        ids[row : row + 1],
                        cache=make_prompt_cache(language),
                        **row_media,
                    ).logits
                )
            self.assert_close(reference, mx.concatenate(solo), logits=True)
        for size in (ids.shape[1] - 1, *chunk_sizes):
            cache, chunks = make_prompt_cache(language), []
            for start in range(0, ids.shape[1], size):
                stop = min(start + size, ids.shape[1])
                chunks.append(
                    language(
                        ids[:, start:stop],
                        inputs_embeds=embeds[:, start:stop],
                        cache=cache,
                        **features,
                    ).logits
                )
            self.assert_close(mx.concatenate(chunks, axis=1), reference, logits=True)
            decode = {k: v for k, v in features.items() if k != "position_ids"}
            self.assert_close(
                language(token, cache=cache, **decode).logits, extended, logits=True
            )

    def multimodal(
        self,
        model,
        config,
        *,
        image_grid,
        video_grid,
        chunk_sizes,
        batch_sizes=(1, 2),
        layouts=(("image",), ("video",), ("image", "video"), ("video", "image")),
    ):
        """Image/video fusion and DeepStack contracts across compatible models."""
        core = getattr(model, "thinker", model)
        config = getattr(config, "thinker_config", config)
        vision, vc = core.vision_tower, config.vision_config
        layers, width = len(vc.deepstack_visual_indexes), vc.out_hidden_size
        assert layers > 0, "The multimodal case must enable DeepStack layers"
        patch_width = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        tokens = {"image": config.image_token_id, "video": config.video_token_id}
        grids = {"image": image_grid, "video": video_grid}
        model.eval()
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            for layout in (tuple(layout) for layout in layouts):
                for batch in batch_sizes:
                    rows, media, encoded = [], {}, {}
                    for row in range(batch):
                        order = layout if row == 0 else layout[::-1]
                        ids = [1, 2]
                        for kind in order:
                            count = math.prod(grids[kind]) // vc.spatial_merge_size**2
                            ids += [config.vision_start_token_id] + [
                                tokens[kind]
                            ] * count
                            ids += [config.vision_end_token_id]
                        rows.append(ids + [3, 4, 5])
                    ids = mx.array(rows, mx.int32)
                    flat_ids = [token for row in rows for token in row]
                    for kind in layout:
                        pixels = mx.random.normal(
                            (batch * math.prod(grids[kind]), patch_width)
                        ).astype(dtype)
                        grid = mx.array([grids[kind]] * batch)
                        media[
                            "pixel_values" if kind == "image" else "pixel_values_videos"
                        ] = pixels
                        media[kind + "_grid_thw"] = grid
                        encoded[kind] = vision(pixels, grid)
                    features = model.get_input_embeddings(ids, **media)
                    assert isinstance(features, InputEmbeddingsFeatures)
                    mask = (ids == tokens["image"]) | (ids == tokens["video"])
                    assert mx.array_equal(features.visual_pos_masks, mask).item()
                    residuals = features.deepstack_visual_embeds
                    if isinstance(residuals, (list, tuple)):
                        residuals = mx.stack(residuals)
                    assert residuals is not None and residuals.dtype == dtype
                    expected = mx.zeros((*ids.shape, layers, width), dtype)
                    expected_embeds = model.language_model.model.embed_tokens(ids)
                    for kind in layout:
                        positions = mx.array(
                            [i for i, t in enumerate(flat_ids) if t == tokens[kind]],
                            mx.uint32,
                        )
                        embeddings, layer_features = encoded[kind]
                        assert len(layer_features) == layers
                        flat = expected.reshape(-1, layers, width)
                        flat[positions] = mx.stack(list(layer_features), axis=1)
                        expected = flat.reshape(expected.shape)
                        flat = expected_embeds.reshape(-1, width)
                        flat[positions] = embeddings
                        expected_embeds = flat.reshape(expected_embeds.shape)
                    self.assert_close(features.inputs_embeds, expected_embeds)
                    if residuals.ndim == 4:
                        self.assert_close(residuals, expected)
                    else:
                        positions = mx.array(
                            [i for i, t in enumerate(flat_ids) if t in tokens.values()],
                            mx.uint32,
                        )
                        self.assert_close(
                            residuals,
                            expected.reshape(-1, layers, width)[positions].transpose(
                                1, 0, 2
                            ),
                        )
                    for layer in range(layers):
                        value = (
                            residuals[:, :, layer]
                            if residuals.ndim == 4
                            else residuals[layer]
                        )
                        injected = model.language_model.model._deepstack_process(
                            mx.zeros_like(features.inputs_embeds), mask, value
                        )
                        self.assert_close(injected, expected[:, :, layer])
                    self.prefill_parity(
                        model, ids, media, chunk_sizes, features=features
                    )

    def input_embeddings(self, model, model_name):
        result = model.get_input_embeddings(input_ids=mx.array([[1, 2, 3, 4, 5]]))
        assert isinstance(result, InputEmbeddingsFeatures), model_name
        assert result.inputs_embeds is not None

    def request_positions(self, model):
        language = model.language_model
        positions, deltas = mx.array([[[9, 9, 9]]], mx.int32), mx.array(
            [[99]], mx.int32
        )
        language._position_ids, language._rope_deltas = positions, deltas
        result = model.get_input_embeddings(input_ids=mx.array([[1, 2, 3]], mx.int32))
        assert result.position_ids is not None and result.rope_deltas is not None
        assert result.position_ids.shape == (1, 3)
        assert result.position_ids.tolist() == [[0, 1, 2]]
        assert result.rope_deltas.tolist() == [[0]]
        assert mx.array_equal(language._position_ids, positions).item()
        assert mx.array_equal(language._rope_deltas, deltas).item()

    def chunked_positions(self, model):
        language = model.language_model
        width = language.args.hidden_size
        positions = mx.arange(15, dtype=mx.int32).reshape(3, 1, 5)
        actual = capture_positions(
            language,
            width,
            mx.array([[7, 8]], dtype=mx.int32),
            idx=2,
            offset=2,
            fa_idx=0,
            inputs_embeds=mx.zeros((1, 2, width), dtype=mx.float32),
            position_ids=positions,
        )
        assert actual.shape == (3, 1, 2)
        assert actual.tolist() == positions[:, :, 2:4].tolist()

    def language(self, model, config, *, num_layers=None):
        if num_layers is None:
            num_layers = config.num_hidden_layers
        assert model.model_type == config.model_type
        assert len(model.layers) == num_layers
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            logits = model(mx.array([[0, 1]])).logits
            assert logits.shape == (1, 2, config.vocab_size) and logits.dtype == dtype
            logits = model(
                mx.argmax(logits[0, -1:, :], keepdims=True), cache=None
            ).logits
            assert logits.shape == (1, 1, config.vocab_size) and logits.dtype == dtype

    def language_hidden_states(self, model):
        ids = mx.array([[1, 2, 3]])
        embeds = model.get_input_embeddings(ids)
        captured = []

        def record_input(forward):
            def call(layer, hidden, *args, **kwargs):
                captured.append(hidden)
                return forward(layer, hidden, *args, **kwargs)

            return call

        reference_cache = model.make_cache()
        with ExitStack() as stack:
            for layer_type in {type(layer) for layer in model.layers}:
                stack.enter_context(
                    patch.object(
                        layer_type, "__call__", record_input(layer_type.__call__)
                    )
                )
            reference = model(ids, cache=reference_cache)
        assert reference.hidden_states is None
        next_ids = mx.array([[4]])
        next_logits = model(next_ids, cache=reference_cache).logits
        for inputs in (
            {"inputs": ids},
            {"input_ids": ids},
            {"inputs_embeds": embeds},
        ):
            cache = model.make_cache()
            output = model(**inputs, cache=cache, output_hidden_states=True)
            assert len(output.hidden_states) == len(model.layers) + 1
            for actual, expected in zip(output.hidden_states[:-1], captured):
                self.assert_close(actual, expected)
            self.assert_close(output.logits, reference.logits)
            self.assert_close(model.lm_head(output.hidden_states[-1]), output.logits)
            self.assert_close(model(next_ids, cache=cache).logits, next_logits)

    def projector(
        self, model, input_width, output_width, *, grid_hw=None, downsample_ratio=1
    ):
        batch = math.prod(grid_hw) if grid_hw else 1
        tokens = (
            math.prod(math.ceil(n / downsample_ratio) for n in grid_hw)
            if grid_hw
            else 1
        )
        kwargs = dict(zip(("n_h", "n_w"), grid_hw)) if grid_hw else {}
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            inputs = mx.random.uniform(shape=(batch, input_width), dtype=dtype)
            output = model(mx.array(inputs), **kwargs)
            assert output.shape == (tokens, output_width) and output.dtype == dtype

    def vision(
        self,
        model,
        model_type,
        width,
        channels,
        image_size,
        vision_feature_layer=-2,
        channel_first=False,
        **kwargs,
    ):
        if model_type == "llama4_vision_model":
            width = kwargs.pop("projector_output_dim", width)
        batch = kwargs.pop("batch_size", 1)
        flat = (
            "qwen2_5_vl qwen3_5 qwen3_5_moe qwen4_exp "
            "glm4v_moe glm4v hunyuan_vl siglip2_vision_model mimovl siglip2_navit"
        ).split()
        shape = (
            image_size
            if len(image_size) > 2 or model_type in flat
            else (
                (batch, channels, *image_size)
                if channel_first
                else (batch, *image_size, channels)
            )
        )
        parameters = inspect.signature(model.__call__).parameters
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            if model_type is not None:
                assert model.model_type == model_type
            inputs = mx.random.uniform(shape=shape)
            if "image_masks" in parameters:
                inputs = inputs.transpose(0, 3, 1, 2)
                kwargs["image_masks"] = mx.ones((batch, channels, image_size[0]))
            options = (
                {"output_hidden_states": True}
                if "output_hidden_states" in parameters
                else {}
            )
            hidden = model(inputs.astype(dtype), **options, **kwargs)
            if vision_feature_layer is not None:
                hidden = hidden[vision_feature_layer]
            assert hidden.shape[1 if channel_first else -1] == width
            assert hidden.dtype == dtype

    def _assert_audio_features(self, features, shape, dtype):
        mx.eval(features)
        assert features.shape == shape
        assert features.dtype == dtype
        assert mx.all(mx.isfinite(features)).item()

    def audio(self, model, config, model_name, *, frames=32, lengths=None):
        if model_name not in {"inkling", "gemma3n", "gemma4", "gemma4_unified"}:
            raise ValueError(f"Unsupported audio model: {model_name}")
        audio_config = config.audio_config
        text_width = config.text_config.hidden_size
        lengths = [frames, frames // 2] if lengths is None else lengths
        assert lengths and all((0 <= n <= frames for n in lengths))
        batch = len(lengths)
        valid_mask = mx.arange(frames)[None, :] < mx.array(lengths)[:, None]
        components = [
            getattr(model, name, None) for name in ("audio_tower", "embed_audio")
        ]
        for dtype in (mx.float32, mx.float16):
            for component in components:
                if component is not None:
                    component.eval()
                    component.update(
                        tree_map(lambda p: p.astype(dtype), component.parameters())
                    )

            if model_name == "inkling":
                bins = audio_config.n_mel_bins
                inputs = mx.arange(batch * frames * bins, dtype=mx.int32)
                inputs = (
                    inputs.reshape(batch, frames, bins) % audio_config.mel_vocab_size
                )
                features = model.audio_tower(inputs)
                shape, output_dtype = (batch, frames, text_width), dtype
            elif model_name == "gemma4_unified":
                inputs = mx.random.normal(
                    (batch, frames, audio_config.output_proj_dims)
                ).astype(dtype)
                projected = model.get_audio_features(inputs)
                self._assert_audio_features(
                    projected, (batch * frames, text_width), dtype
                )
                features = model.get_audio_features(inputs, valid_mask)
                shape, output_dtype = (sum(lengths), text_width), dtype
            else:
                is_gemma3n = model_name == "gemma3n"
                # Gemma 4's subsampling projection requires 128 input mel bins.
                bins = audio_config.input_feat_size if is_gemma3n else 128
                inputs = mx.random.normal((batch, frames, bins)).astype(dtype)
                encoded, mask = model.audio_tower(inputs, ~valid_mask)
                stride = (
                    math.prod(s[0] for s in audio_config.sscp_conv_stride_size)
                    * max(1, audio_config.conf_reduction_factor)
                    if is_gemma3n
                    else 4
                )
                steps = (frames + stride - 1) // stride
                width = (
                    getattr(audio_config, "output_proj_dims", None)
                    or audio_config.hidden_size
                )
                # Both Gemma encoders currently promote float16 inputs to float32.
                output_dtype = mx.float32
                self._assert_audio_features(
                    encoded, (batch, steps, width), output_dtype
                )
                assert mask.shape == (batch, steps)
                assert mask.dtype == mx.bool_
                assert mx.array_equal(mask, ~valid_mask[:, ::stride]).item()
                assert mx.all(mx.where(mask[..., None], encoded == 0, True)).item()
                features = (
                    model.embed_audio(inputs_embeds=encoded)
                    if is_gemma3n
                    else model.embed_audio(encoded)
                )
                shape = (batch, steps, text_width)
            self._assert_audio_features(features, shape, output_dtype)

    def mrope_cache_index(self, language_model, hidden_size):
        """Use the Python cache index (10), avoiding a GPU sync on offset (3)."""
        language_model._rope_deltas = mx.array([[0]])
        language_model._position_ids = None
        positions = capture_positions(
            language_model, hidden_size, mx.array([[5]]), idx=10, offset=3
        )
        assert positions is not None
        assert tuple(positions.shape) in {(1, 1), (3, 1, 1)}
        assert positions.reshape(-1)[0].item() == 10

    def mrope_deltas(self, language_model, hidden_size):
        """The request's delta (5) must override stale model state (99)."""
        language_model._rope_deltas = mx.array([[99]])
        language_model._position_ids = None
        positions = capture_positions(
            language_model,
            hidden_size,
            mx.array([[7]]),
            idx=10,
            offset=3,
            fa_idx=0,
            rope_deltas=mx.array([[5]]),
        )
        assert positions is not None
        assert tuple(positions.shape) == (3, 1, 1)
        assert positions[0, 0, 0].item() == 15


CONFIG_TYPES = {
    "text_config": "TextConfig",
    "vision_config": "VisionConfig",
    "projector_config": "ProjectorConfig",
    "perceiver_config": "PerceiverConfig",
    "audio_config": "AudioConfig",
    "thinker_config": "ThinkerConfig",
    "talker_config": "TalkerConfig",
    "code_predictor_config": "CodePredictorConfig",
    "code2wav_config": "Code2WavConfig",
    "vit_config": "config.VitConfig",
    "adapter_config": "config.AdapterConfig",
}
DATA = json.loads(
    Path(__file__).with_name("model_cases.json").read_text(encoding="utf-8")
)
if DATA["version"] != 2:
    raise ValueError(f"Unsupported model_cases.json version: {DATA['version']}")

TINY_DEFAULTS = DATA["tiny_defaults"]
TINY_MODELS = DATA["shared_configs"]


def model_case(name):
    """Return fresh case data for shared model and checkpoint tests."""
    return copy.deepcopy(next(case for case in DATA["cases"] if case["id"] == name))


def build_config(module, values, config_type="ModelConfig"):
    """Construct the model family's config classes from ordinary nested data."""
    fields = copy.deepcopy(values)
    if (
        module.__name__
        in ("mlx_vlm.models.minicpmv4_7", "mlx_vlm.models.nemotron_h_nano_omni")
        and config_type == "ModelConfig"
    ):
        # Preserve checkpoint aliases and per-layer text-config dispatch.
        return module.ModelConfig.from_dict(fields)
    for name, value in fields.items():
        if name in CONFIG_TYPES and isinstance(value, dict):
            fields[name] = build_config(module, value, CONFIG_TYPES[name])
    return attrgetter(config_type)(module)(**fields)


def first_attribute(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


def check_arguments(kind, case, model, config):
    """Keep shared component selection and dimension wiring in Python."""
    name = case["module"]
    # Phi3-V keeps language dimensions on the outer config.
    core_config = getattr(config, "thinker_config", config)
    text = (
        config if name == "phi3_v" else getattr(core_config, "text_config", core_config)
    )
    if kind == "forward_cache":
        return (model, text.vocab_size), case.get("forward_cache", {})
    if kind == "causal_visual_prefix":
        return (model.language_model, text), {}
    if kind in {"masked_lm", "token_embeddings", "sentence_embeddings"}:
        return (model, config), {}
    if kind == "multimodal":
        return (model, config), case["multimodal"]
    if kind == "input_embeddings":
        return (model, name), {}
    if kind == "audio":
        return (model, config, name), case.get("audio", {})
    if kind in {"request_positions", "chunked_positions"}:
        return (model,), {}
    if kind == "language_hidden_states":
        return (model.language_model,), {}
    if kind in {"language", "mrope_cache_index", "mrope_deltas"}:
        if kind == "language":
            options = (
                {}
                if hasattr(text, "num_hidden_layers")
                else {"num_layers": text.n_layers}
            )
            return (model.language_model, text), options
        return (model.language_model, text.hidden_size), {}
    if kind == "projector":
        projector = attrgetter(case.get("projector_path", "multi_modal_projector"))(
            model
        )
        if name in ("deepseek_v4", "deepseek_v41"):
            return (
                projector,
                first_attribute(config, "vision_dim", "vision_hidden_size"),
                config.hidden_size,
            ), {
                "grid_hw": case["vision"]["grid_hw"],
                "downsample_ratio": config.vision_downsample_ratio,
            }
        return (projector, config.vision_config.hidden_size, text.hidden_size), {}
    if kind == "vision":
        vision = attrgetter(case.get("vision_path", "vision_tower"))(model)
        options = case.get("vision", {})
        if name in ("deepseek_v4", "deepseek_v41"):
            return (
                vision,
                None,
                first_attribute(config, "vision_dim", "vision_hidden_size"),
                3,
                tuple(options["input_shape"]),
            ), {
                "vision_feature_layer": options["feature_layer"],
                "n_h": options["grid_hw"][0],
                "n_w": options["grid_hw"][1],
            }
        vc = config.vision_config
        image_size = options.get("input_shape")
        if image_size is None:
            image_size = (vc.image_size, vc.image_size)
        width = first_attribute(
            vc, "out_hidden_size", "hidden_size", "d_model", "width", "text_hidden_size"
        )
        # Molmo's hidden_size describes its projector; use d_model for vision.
        if name == "molmo":
            width = vc.d_model
        channels = first_attribute(vc, "num_channels", "in_channels")
        kwargs = {}
        if "feature_layer" in options:
            kwargs["vision_feature_layer"] = options["feature_layer"]
        if "channel_first" in options:
            kwargs["channel_first"] = options["channel_first"]
        if "grid_thw" in options:
            kwargs["grid_thw"] = mx.array(
                options["grid_thw"],
                dtype=getattr(mx, options.get("grid_dtype", "int64")),
            )
        if vc.model_type == "llama4_vision_model":
            kwargs["projector_output_dim"] = vc.projector_output_dim
        # MiniMax's vision wrapper does not expose model_type.
        model_type = None if name == "minimax_m3_vl" else vc.model_type
        return (vision, model_type, width, channels, tuple(image_size)), kwargs
    raise ValueError(f"Unknown model check: {kind}")


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"])
def test_model_contract(case):
    if {"multimodal", "causal_visual_prefix"}.intersection(case["checks"]):
        mx.random.seed(17)
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    config = build_config(module, case["config"])
    model = module.Model(config)
    checks = ModelChecks()
    for kind in case["checks"]:
        args, kwargs = check_arguments(kind, case, model, config)
        getattr(checks, kind)(*args, **kwargs)


def test_nemotron_parse_source_architecture_and_tied_head_mapping():
    config = {
        "model_type": "nemotron_parse",
        "architectures": ["RadioMBartPrefixForCausalLM"],
        "decoder": {"vocab_size": 72512, "self_attention_kv_heads": 8},
    }
    module, model_type = get_model_and_args(config)
    assert model_type == "nemotron_parse_v3_5"
    assert module.ModelConfig.from_dict(config).text_config.vocab_size == 72512
    weights = module.Model.sanitize(
        {
            "decoder.embed_tokens.weight": mx.zeros((4, 2)),
            "decoder.extra_heads.0.weight": mx.ones((2, 2)),
            "lm_head.weight": mx.zeros((4, 2)),
        }
    )
    assert list(weights) == ["language_model.embed_tokens.weight"]


@pytest.fixture
def flint_model():
    from mlx_vlm.models import flint

    mx.random.seed(17)
    model = flint.Model(build_config(flint, model_case("flint")["config"]))
    model.eval()
    mx.eval(model.parameters())
    return model


def _flint_numpy_reference(model, ids):
    """Independent dense attention and explicit expert oracle for tiny Flint."""
    config = model.config
    weights = {
        k.removeprefix("language_model."): np.array(v.astype(mx.float32)).astype(
            np.float64
        )
        for k, v in tree_flatten(model.parameters())
    }

    def linear(x, name):
        return x @ weights[name + ".weight"].T

    def norm(x, name):
        return (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + config.rms_norm_eps)
            * weights[name + ".weight"]
        )

    def mlp(x, prefix):
        gate = linear(x, prefix + ".gate_proj")
        return linear(
            gate / (1 + np.exp(-gate)) * linear(x, prefix + ".up_proj"),
            prefix + ".down_proj",
        )

    x = weights["model.embed_tokens.weight"][ids]
    batch, length, _ = x.shape
    positions = np.arange(length)
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        a = prefix + ".self_attn"
        hidden = norm(x, prefix + ".input_layernorm")
        q = norm(
            linear(hidden, a + ".q_proj").reshape(
                batch, length, config.num_attention_heads, config.head_dim
            ),
            a + ".q_norm",
        )
        k = norm(
            linear(hidden, a + ".k_proj").reshape(
                batch, length, config.num_key_value_heads, config.head_dim
            ),
            a + ".k_norm",
        )
        v = linear(hidden, a + ".v_proj").reshape(
            batch, length, config.num_key_value_heads, config.head_dim
        )
        if i not in config.no_rope_layer_indices:
            angles = positions[:, None] / config.rope_theta ** (
                np.arange(0, config.head_dim, 2) / config.head_dim
            )
            angles = np.concatenate([angles, angles], -1)[None, :, None, :]

            def rotate(t):
                left, right = np.split(t, 2, axis=-1)
                return t * np.cos(angles) + np.concatenate([-right, left], -1) * np.sin(
                    angles
                )

            q, k = rotate(q), rotate(k)
        repeat = config.num_attention_heads // config.num_key_value_heads
        q = q.transpose(0, 2, 1, 3)
        k, v = (np.repeat(t, repeat, axis=2).transpose(0, 2, 1, 3) for t in (k, v))
        scores = (q @ k.swapaxes(-2, -1)) / np.sqrt(config.head_dim)
        allowed = positions[:, None] >= positions[None, :]
        if config.layer_types[i] == "sliding_attention" and config.sliding_window:
            allowed &= positions[:, None] - positions[None, :] < config.sliding_window
        scores = np.where(allowed, scores, -np.inf)
        sink = np.broadcast_to(
            weights[a + ".softmax_offset"][None, :, None, None], (*scores.shape[:-1], 1)
        )
        scores = np.concatenate([scores, sink], axis=-1)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        attention = (
            (probs[..., :-1] @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        )
        x = x + linear(attention, a + ".o_proj")
        hidden = norm(x, prefix + ".post_attention_layernorm")
        p = prefix + ".mlp"
        if i in config.mlp_only_layers:
            x = x + mlp(hidden, p)
            continue
        scores = 1 / (1 + np.exp(-linear(hidden, p + ".gate")))
        indices = np.argsort(scores + weights[p + ".expert_bias"], axis=-1)[
            ..., -config.num_experts_per_tok :
        ]
        route_weights = np.take_along_axis(scores, indices, axis=-1)
        route_weights /= route_weights.sum(axis=-1, keepdims=True) + 1e-20
        routed = np.zeros_like(hidden)
        for b in range(batch):
            for t in range(length):
                for e, score in zip(indices[b, t], route_weights[b, t]):
                    gate = (
                        hidden[b, t] @ weights[p + ".switch_mlp.gate_proj.weight"][e].T
                    )
                    up = hidden[b, t] @ weights[p + ".switch_mlp.up_proj.weight"][e].T
                    routed[b, t] += score * (
                        (gate / (1 + np.exp(-gate)) * up)
                        @ weights[p + ".switch_mlp.down_proj.weight"][e].T
                    )
        shared = (
            mlp(hidden, p + ".shared_expert")
            if config.shared_expert_intermediate_size
            else 0
        )
        x = x + routed + shared
    return linear(norm(x, "model.norm"), "lm_head")


def test_flint_matches_independent_attention_and_moe_reference(flint_model):
    for layer in flint_model.layers:
        layer.self_attn.softmax_offset = mx.array([-2.0, 0.0, 1.0, 3.0])
        if hasattr(layer.mlp, "expert_bias"):
            layer.mlp.expert_bias = mx.array([0.5, -0.4, 0.1, 0.3])
    ids = np.array([[1, 4, 7, 2, 9, 3, 8, 5, 6], [7, 3, 8, 5, 6, 1, 2, 9, 4]])
    actual = flint_model(mx.array(ids)).logits
    expected = _flint_numpy_reference(flint_model, ids)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("window", [1, 4, None])
def test_flint_hybrid_cache_matches_full_forward(flint_model, window):
    flint_model.layers[1].self_attn.window_size = window
    ids = (mx.arange(41)[None] * 7 + 3) % flint_model.config.vocab_size
    expected = flint_model(ids).logits
    for schedule in ([1] * 41, [17, 7, 8, 8, 1], [1] * 19 + [5, 8, 8, 1]):
        cache = flint_model.make_cache()
        out, start = [], 0
        for size in schedule:
            logits = flint_model(ids[:, start : start + size], cache=cache).logits
            mx.eval(logits)
            out.append(logits)
            start += size
        np.testing.assert_allclose(
            np.array(mx.concatenate(out, axis=1)),
            np.array(expected),
            atol=2e-5,
            rtol=2e-5,
        )
        assert isinstance(cache[0], KVCache)
        assert isinstance(cache[1], KVCache if window is None else RotatingKVCache)
        assert all(c.offset == 41 for c in cache)


def test_flint_padded_batch_matches_individual_decoding(flint_model):
    from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache

    ids = mx.array([[0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]])
    cache = [
        (
            BatchRotatingKVCache(layer.self_attn.window_size, [2, 0])
            if layer.self_attn.window_size
            else BatchKVCache([2, 0])
        )
        for layer in flint_model.layers
    ]
    actual = flint_model(ids, cache=cache).logits[:, -1:]
    for row, pad in enumerate([2, 0]):
        expected = flint_model(ids[row : row + 1, pad:]).logits[:, -1:]
        np.testing.assert_allclose(
            np.array(actual[row : row + 1]), np.array(expected), atol=2e-5, rtol=2e-5
        )
    token = mx.array([[8], [9]])
    actual = flint_model(token, cache=cache).logits
    for row, pad in enumerate([2, 0]):
        extended = mx.concatenate(
            [ids[row : row + 1, pad:], token[row : row + 1]], axis=1
        )
        expected = flint_model(extended).logits[:, -1:]
        np.testing.assert_allclose(
            np.array(actual[row : row + 1]), np.array(expected), atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_flint_checkpoint_strict_loading_and_roundtrip(
    flint_model, tmp_path, with_bias, quantized
):
    model = flint_model
    if quantized:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda path, m: hasattr(m, "to_quantized")
            and "mlp.gate" not in path
            and m.weight.shape[-1] % 32 == 0,
        )
    expected = dict(tree_flatten(model.parameters()))
    checkpoint = {}
    for key, value in expected.items():
        key = key.removeprefix("language_model.")
        if key.endswith(".mlp.expert_bias"):
            if with_bias:
                checkpoint[
                    key.replace(".mlp.expert_bias", ".mlp.gate.e_score_correction_bias")
                ] = value
        elif ".mlp.switch_mlp." in key:
            prefix, suffix = key.split(".mlp.switch_mlp.")
            for e in range(model.config.num_experts):
                checkpoint[f"{prefix}.mlp.experts.{e}.{suffix}"] = value[e]
        else:
            checkpoint[key] = value
    sanitized = model.sanitize(checkpoint)
    assert sanitized.keys() == expected.keys()
    assert model.sanitize(sanitized).keys() == expected.keys()
    for key, value in expected.items():
        assert mx.array_equal(sanitized[key], value).item(), key
    model.load_weights(list(sanitized.items()), strict=True)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), sanitized)
    config = model.config.to_dict()
    if quantized:
        config["quantization"] = {"group_size": 32, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    reloaded = load_model(tmp_path)
    ids = mx.array([[1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(
        np.array(reloaded(ids).logits), np.array(model(ids).logits)
    )


@pytest.fixture
def nemotron_case(request):
    case = model_case(getattr(request, "param", "nemotron_super_vl"))
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return module, case["config"]


@pytest.mark.parametrize("nemotron_case", ["nemotron_bolt"], indirect=True)
def test_nemotron_puzzle_layer_configs(nemotron_case):
    arch, config = nemotron_case
    # The block list takes precedence, as in NVIDIA's Puzzle config.
    config["llm_config"]["layers_block_type"] = ["attention"]
    _, family = get_model_and_args({"model_type": "nemotron_h_puzzle"})
    assert family == "nemotron_h"
    model = arch.Model(build_config(arch, config))
    assert [layer.block_type for layer in model.language_model.layers] == [
        "M",
        "E",
        "*",
        "E",
    ]
    assert model.config.text_config.time_step_limit == (0, float("inf"))
    for idx, experts, width, topk in [(1, 2, 32, 1), (3, 3, 64, 2)]:
        moe = model.language_model.layers[idx].mixer
        assert moe.switch_mlp.fc1.weight.shape == (experts, width, 32)
        assert moe.switch_mlp.fc2.weight.shape == (experts, 32, width)
        assert moe.gate.weight.shape == (experts, 32)
        assert moe.gate.top_k == topk


@pytest.mark.parametrize("vision_norm,mtp_depth", [(True, 0), (False, 1), (None, 1)])
def test_nemotron_explicit_vision_norm_overrides_mtp(
    nemotron_case, vision_norm, mtp_depth
):
    arch, config = nemotron_case
    config["vision_final_layernorm"] = vision_norm
    config["llm_config"]["num_nextn_predict_layers"] = mtp_depth
    model = arch.Model(build_config(arch, config))
    expected = vision_norm if vision_norm is not None else mtp_depth > 0
    assert isinstance(model.vision_final_layernorm, nn.LayerNorm) == expected
    assert model.config.to_dict().get("vision_final_layernorm") == vision_norm


@pytest.mark.parametrize(
    "nemotron_case", ["nemotron_super_vl", "nemotron_bolt"], indirect=True
)
@pytest.mark.parametrize("text_key", ["llm_config", "text_config"])
def test_nemotron_checkpoint_loading(tmp_path, nemotron_case, text_key):
    arch, config = nemotron_case
    config[text_key] = config.pop("llm_config")
    _, family = get_model_and_args(config)
    assert family == arch.__name__.rsplit(".", 1)[-1]
    model = arch.Model(build_config(arch, config))
    expected = dict(tree_flatten(model.parameters()))
    weights = {}
    for key, value in expected.items():
        if ".switch_mlp." in key:
            prefix, projection = key.split(".switch_mlp.")
            projection = projection.replace("fc1", "up_proj").replace(
                "fc2", "down_proj"
            )
            for expert, weight in enumerate(value):
                weights[f"{prefix}.experts.{expert}.{projection}"] = weight
        else:
            if "conv1d.weight" in key:
                value = value.moveaxis(1, 2)
            key = key.replace("mlp1.layers.", "mlp1.")
            if key.startswith("vision_final_layernorm."):
                key = "vision_projector." + key
            weights[key] = value
    weights["vision_model.summary_idxs"] = mx.array([0, 1])
    weights["language_model.mtp.unused.weight"] = mx.ones((2, 2))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded = load_model(tmp_path)
    assert asdict(loaded.config) == asdict(model.config)
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert mx.array_equal(actual[key], value), key


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, [True] * 6),
        (0.5, [True, True, False, True, False, False]),
        (1.0, [True, True, False, False, False, False]),
    ],
)
def test_nemotron_video_retention_order_and_first_tubelet(
    nemotron_case, rate, expected
):
    arch, _ = nemotron_case
    vision = inspect.getmodule(arch.Model)
    # The second token changes most in frame 2; equal scores preserve order.
    features = mx.array(
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]]
    )
    assert vision.compute_retention_mask(features, rate).tolist() == expected
    assert vision.compute_retention_mask(mx.zeros((1, 2, 4)), rate).tolist() == [
        True,
        True,
    ]


def test_nemotron_vision_normalization_and_video_pruning(nemotron_case):
    arch, config = nemotron_case
    vision = inspect.getmodule(arch.Model)
    model = arch.Model(build_config(arch, config))
    features = mx.random.normal((1, 4, 16))
    # The projector must normalize BEFORE spatially merging the four patches.
    expected = model.mlp1(model.vision_final_layernorm(features).reshape(1, 1, 64))
    np.testing.assert_allclose(
        model._project_vision_features(features, 2, 2), expected, rtol=1e-5, atol=1e-5
    )
    # Five rectangular frames pad to three tubelets, two tokens per tubelet.
    pixels = mx.random.normal((5, 3, 32, 64))
    ids = mx.array([[1, 18, 18, 2, 18, 18, 3, 18, 18, 4]])
    full = model.extract_video_feature(pixels)
    assert full.shape == (3, 2, 32)
    keep = np.flatnonzero(vision.compute_retention_mask(full, 0.7))
    actual = model.get_input_embeddings(ids, pixel_values_videos=pixels).inputs_embeds
    assert actual.shape == (1, 6, 32)
    unpruned = model.get_input_embeddings(
        ids, pixel_values_videos=pixels, video_pruning_rate=0
    ).inputs_embeds
    assert unpruned.shape == (1, 10, 32)
    assert model.video_pruning_rate == 0.7
    with pytest.raises(ValueError, match="video_pruning_rate"):
        model.get_input_embeddings(
            ids, pixel_values_videos=pixels, video_pruning_rate=-1
        )
    np.testing.assert_allclose(actual[0, 1:3], full.reshape(-1, 32)[mx.array(keep)])
    expected_text = model.language_model.get_input_embeddings(mx.array([[1, 2, 3, 4]]))
    np.testing.assert_allclose(actual[:, [0, 3, 4, 5]], expected_text)
    images = [mx.zeros((3, 32, 32)), mx.zeros((3, 32, 64))]
    assert model.get_input_embeddings(
        mx.array([[1, 18, 18, 18, 2]]), images
    ).inputs_embeds.shape == (1, 5, 32)


def test_lfm2_encoder_sanitize_and_dispatch():
    case = next(case for case in DATA["cases"] if case["module"] == "lfm2_encoder")
    module = importlib.import_module("mlx_vlm.models.lfm2_encoder")
    model = module.Model(build_config(module, case["config"]))
    weights = {
        "lfm2.embed_tokens.weight": mx.zeros((32, 16)),
        "lfm2.layers.0.conv.conv.weight": mx.zeros((16, 1, 3)),
        "lm_head.weight": mx.zeros((32, 16)),
    }
    sanitized = model.sanitize(weights)
    assert "model.embed_tokens.weight" in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "lm_head.weight" not in sanitized

    resolved, model_type = get_model_and_args(
        {
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirectionalForMaskedLM"],
        }
    )
    assert resolved.Model is Lfm2Encoder
    assert model_type == "lfm2_encoder"


def test_lfm2_colbert_sanitize_and_loader(tmp_path, monkeypatch):
    case = next(case for case in DATA["cases"] if case["module"] == "lfm2_colbert")
    module = importlib.import_module("mlx_vlm.models.lfm2_colbert")
    model = module.Model(build_config(module, case["config"]))
    weights = {
        "embed_tokens.weight": mx.zeros((32, 16)),
        "layers.0.conv.conv.weight": mx.zeros((16, 1, 3)),
        "1_Dense.linear.weight": mx.zeros((8, 16)),
    }
    sanitized = model.sanitize(weights)
    assert "model.embed_tokens.weight" in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "projection.weight" in sanitized

    dense_dir = tmp_path / "1_Dense"
    dense_dir.mkdir()
    (dense_dir / "config.json").write_text(json.dumps({"out_features": 128}))
    captured = {}

    def fake_load(model_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embedding_loader, "load_encoder_model", fake_load)
    embedding_loader.load_embedding_model(tmp_path)
    assert captured["model_remapping"]["lfm2"] == "lfm2_colbert"
    assert captured["config_overrides"]["embedding_dim"] == 128


@pytest.fixture
def embedding_gemma2_case():
    case = next(c for c in DATA["cases"] if c["module"] == "embedding_gemma2")
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return module, build_config(module, case["config"])


@pytest.mark.parametrize("left_padding", [False, True])
def test_embedding_gemma2_padded_batch_matches_individual(
    embedding_gemma2_case, left_padding
):
    module, config = embedding_gemma2_case
    mx.random.seed(42)
    model = module.Model(config)
    ids = [1, 2, 3]
    solo = model(mx.array([ids])).text_embeds
    padded = [0, 0, *ids] if left_padding else [*ids, 0, 0]
    valid = [0, 0, 1, 1, 1] if left_padding else [1, 1, 1, 0, 0]
    output = model(
        mx.array([padded, [4, 5, 6, 7, 8]]),
        attention_mask=mx.array([valid, [1, 1, 1, 1, 1]]),
    )
    np.testing.assert_allclose(output.text_embeds[:1], solo, atol=1e-5, rtol=1e-5)


def test_embedding_gemma2_attention_is_bidirectional(embedding_gemma2_case):
    module, config = embedding_gemma2_case
    model = module.Model(config)
    first = model(mx.array([[1, 2, 3]])).last_hidden_state
    second = model(mx.array([[1, 2, 4]])).last_hidden_state
    assert not mx.allclose(first[:, 0], second[:, 0])


def test_embedding_gemma2_sliding_attention_includes_window_boundary(
    embedding_gemma2_case,
):
    module, config = embedding_gemma2_case
    mx.random.seed(42)
    model = module.Model(config)
    # Isolate the first (sliding) layer so the full layer cannot mix distant tokens.
    model.language_model.layers = model.language_model.layers[:1]
    baseline = model(mx.array([[1, 2, 3, 4, 5]])).last_hidden_state[:, 0]
    boundary = model(mx.array([[1, 2, 3, 6, 5]])).last_hidden_state[:, 0]
    outside = model(mx.array([[1, 2, 3, 4, 6]])).last_hidden_state[:, 0]
    assert not mx.allclose(baseline, boundary)
    np.testing.assert_allclose(baseline, outside, atol=1e-6)


def test_embedding_gemma2_explicit_position_ids(embedding_gemma2_case):
    module, config = embedding_gemma2_case
    mx.random.seed(42)
    model = module.Model(config)
    ids = mx.array([[1, 2, 3]])
    default = model(ids).last_hidden_state
    explicit = model(ids, position_ids=mx.array([[0, 1, 2]])).last_hidden_state
    spaced = model(ids, position_ids=mx.array([[0, 2, 4]])).last_hidden_state
    np.testing.assert_array_equal(default, explicit)
    assert not mx.allclose(default, spaced)


def test_embedding_gemma2_full_attention_config_defaults_and_index_normalization(
    embedding_gemma2_case,
):
    module, config = embedding_gemma2_case
    defaults = module.TextConfig(
        num_hidden_layers=7,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        hidden_size_per_layer_input=8,
    )
    model = module.Model(module.ModelConfig(text_config=defaults))
    assert defaults.layer_types[-1] == "full_attention"
    assert model.layers[5].self_attn.head_dim == 512
    assert model.layers[5].self_attn.num_kv_heads == 1
    config.text_config.per_layer_config = {1: {"head_dim": 16}}
    config.text_config.__post_init__()
    assert module.Model(config).layers[1].self_attn.head_dim == 16


def test_embedding_gemma2_media_scatter_preserves_batch_order_and_validates_counts(
    embedding_gemma2_case,
):
    module, _ = embedding_gemma2_case
    ids = mx.array([[1, 60, 60], [60, 2, 0]])
    embeddings = mx.zeros((2, 3, 4))
    features = mx.arange(12).reshape(3, 4)
    output = module.Model._scatter(embeddings, ids, 60, features)
    np.testing.assert_array_equal(output[0, 1:], features[:2])
    np.testing.assert_array_equal(output[1, 0], features[2])
    np.testing.assert_array_equal(output[1, 1:], mx.zeros((2, 4)))
    with pytest.raises(ValueError, match="token count"):
        module.Model._scatter(embeddings, ids, 60, features[:2])


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_embedding_gemma2_disabled_towers_reject_media(embedding_gemma2_case, modality):
    module, config = embedding_gemma2_case
    model = module.Model(config)
    with pytest.raises(ValueError, match="require.*config"):
        getattr(model, f"get_{modality}_features")(mx.zeros((1, 2, 3)), None)


def test_embedding_gemma2_sanitize_and_converted_checkpoint_roundtrip(
    embedding_gemma2_case, tmp_path
):
    module, config = embedding_gemma2_case

    config.audio_config = module.AudioConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=(4, 2),
        output_proj_dims=8,
    )
    model = module.Model(config)
    weights = dict(tree_flatten(model.parameters()))
    original = {}
    for key, value in weights.items():
        if key.endswith("conv.weight"):
            value = value.transpose(0, 3, 1, 2)
        elif key.endswith("depthwise_conv1d.weight"):
            value = value.transpose(0, 2, 1)
        original["model." + key] = value
    sanitized = model.sanitize(original)
    twice = model.sanitize(sanitized)
    assert weights.keys() == sanitized.keys() == twice.keys()
    for key in weights:
        np.testing.assert_array_equal(sanitized[key], weights[key])
        np.testing.assert_array_equal(twice[key], weights[key])

    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), sanitized)
    loaded = embedding_loader.load_embedding_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    np.testing.assert_allclose(
        loaded(ids).text_embeds, model(ids).text_embeds, atol=1e-6
    )

    config.audio_config = None
    reduced = embedding_loader.load_embedding_model(tmp_path, config=asdict(config))
    assert reduced.audio_tower is None
    np.testing.assert_allclose(
        reduced(ids).text_embeds, model(ids).text_embeds, atol=1e-6
    )


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_embedding_gemma2_quantized_checkpoint_roundtrip(
    embedding_gemma2_case, tmp_path, bits
):
    from mlx_vlm.quant_utils import quantize_model

    module, config = embedding_gemma2_case
    config.text_config.hidden_size = 64
    config.text_config.intermediate_size = 128
    config.text_config.hidden_size_per_layer_input = 64
    config.text_config.head_dim = 32
    config.text_config.per_layer_config = {"01": {"head_dim": 64}}
    model, saved_config = quantize_model(module.Model(config), asdict(config), 64, bits)
    assert model.language_model.embed_tokens.bits == bits
    (tmp_path / "config.json").write_text(json.dumps(saved_config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    reloaded = embedding_loader.load_embedding_model(tmp_path)
    ids = mx.array([[1, 2, 3], [4, 5, 6]])
    expected = model(ids).text_embeds
    actual = reloaded(ids).text_embeds
    assert reloaded.language_model.embed_tokens.bits == bits
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(mx.linalg.norm(actual, axis=-1), 1, atol=1e-6)


@pytest.fixture(params=[c["id"] for c in DATA["cases"] if c["module"] == "minicpmv4_7"])
def minicpmv47_case(request):
    case = next(c for c in DATA["cases"] if c["id"] == request.param)
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return module, build_config(module, case["config"])


@pytest.mark.parametrize("mode,count", [("16x", 4), ("4x", 16)])
def test_minicpmv47_vision_and_chunked_prefill(minicpmv47_case, mode, count):
    module, config = minicpmv47_case

    model = module.Model(config)
    ids = mx.array([[1, 50] + [48] * count + [51, 2]])
    media = {
        "pixel_values": [[mx.random.normal((3, 2, 128))]],
        "tgt_sizes": [np.array([[8, 8]])],
        "downsample_mode": mode,
    }
    features = model.get_input_embeddings(ids, **media)
    assert features.inputs_embeds.shape == (*ids.shape, 16)
    assert features.position_ids.shape == (3, 1, ids.shape[-1])
    assert features.rope_deltas.item() < 0
    ModelChecks().prefill_parity(model, ids, media, chunk_sizes=(3,))

    # The outer model must preserve the canvas delta when decoding without images.
    cache = model.language_model.make_cache()
    model(ids, cache=cache, **media)
    next_logits = model(mx.array([[3]]), cache=cache).logits
    extended = mx.concatenate([ids, mx.array([[3]])], axis=1)
    expected = model(extended, **media).logits[:, -1:]
    assert mx.allclose(next_logits, expected, atol=2e-5)


def test_minicpmv47_canvas_and_padding(minicpmv47_case):
    module, config = minicpmv47_case
    rope_module = importlib.import_module(module.__name__ + ".rope")

    rope = rope_module.CanvasRoPE(config)
    ids = np.array([[0, 0, 1, 50, 48, 48, 48, 48, 51, 2]])
    positions, deltas = rope.get_rope_index(
        ids,
        (ids == 48).astype(np.int32),
        np.array([[8, 8]]),
        attention_mask=ids != 0,
    )
    expected = [
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 0, 0, 1, 1, 2, 2, 3, 4],
        [0, 0, 0, 0, 1, 2, 1, 2, 3, 4],
    ]
    np.testing.assert_array_equal(positions[:, 0], expected)
    assert deltas.item() == -3


def test_minicpmv47_grid_validation(minicpmv47_case):
    module, config = minicpmv47_case

    model = module.Model(config)
    ids = mx.array([[50, 48, 51]])
    with pytest.raises(ValueError, match="crop grids"):
        model.get_input_embeddings(ids)
    with pytest.raises(ValueError, match="token count"):
        model.get_input_embeddings(ids, tgt_sizes=[[[8, 8]]])
    with pytest.raises(ValueError, match="crop grids"):
        model.language_model.get_rope_index(ids)


def test_minicpmv47_slices_share_the_thumbnail_canvas(minicpmv47_case):
    module, config = minicpmv47_case
    rope_module = importlib.import_module(module.__name__ + ".rope")

    ids = np.array([[1, 50, 48, 51, 52, 48, 53, 52, 48, 53, 2]])
    positions, _ = rope_module.CanvasRoPE(config).get_rope_index(
        ids, (ids == 48).astype(np.int32), np.array([[4, 4]] * 3)
    )
    expected = [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 3, 1, 1, 1, 2, 2, 2, 4],
    ]
    np.testing.assert_array_equal(positions[:, 0], expected)


@pytest.mark.parametrize("minicpmv47_case", ["minicpmv4_7"], indirect=True)
def test_minicpmv47_sanitize_is_idempotent(minicpmv47_case):
    module, config = minicpmv47_case

    model = module.Model(config)
    weights = {
        "model.language_model.layers.0.input_layernorm.weight": mx.zeros((16,)),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.ones((80, 1, 3)),
        "model.vision_tower.embeddings.patch_embedding.weight": mx.zeros((16, 3, 2, 2)),
    }
    first = model.sanitize(weights)
    second = model.sanitize(first)
    assert first.keys() == second.keys()
    for key in first:
        assert mx.array_equal(first[key], second[key])
    assert mx.array_equal(
        first["language_model.model.layers.0.input_layernorm.weight"],
        mx.ones((16,)),
    )
    assert first["vision_tower.embeddings.patch_embedding.weight"].shape == (
        16,
        2,
        2,
        3,
    )


@pytest.mark.parametrize("minicpmv47_case", ["minicpmv4_7_moe"], indirect=True)
def test_minicpmv47_native_expert_checkpoint_and_reload(minicpmv47_case, tmp_path):
    module, config = minicpmv47_case
    qwen = importlib.import_module("mlx_vlm.models.qwen3_5.qwen3_5")
    moe = importlib.import_module("mlx_vlm.models.qwen3_5_moe.language")

    model = module.Model(config)
    assert isinstance(model.layers[0].mlp, moe.Qwen3_5MoeSparseMoeBlock)
    expected = dict(tree_flatten(model.parameters()))
    raw = {}
    for key, value in expected.items():
        if ".switch_mlp." in key:
            continue
        if key.startswith("language_model.model."):
            key = key.replace("language_model.model.", "model.language_model.", 1)
            if key.endswith(qwen.NORM_WEIGHT_SUFFIXES):
                value = value - 1
            if "conv1d.weight" in key:
                value = value.swapaxes(1, 2)
        elif key.startswith("language_model.lm_head."):
            key = key.replace("language_model.", "", 1)
        else:
            key = "model." + key.replace("vit_merger.", "vision_tower.vit_merger.", 1)
            if key.endswith("patch_embedding.weight"):
                value = value.transpose(0, 3, 1, 2)
        raw[key] = value
    for index, layer in enumerate(model.layers):
        prefix = f"model.language_model.layers.{index}.mlp.experts"
        experts = layer.mlp.switch_mlp
        raw[f"{prefix}.gate_up_proj"] = mx.concatenate(
            [experts.gate_proj.weight, experts.up_proj.weight], axis=-2
        )
        raw[f"{prefix}.down_proj"] = experts.down_proj.weight

    sanitized = model.sanitize(raw)
    assert sanitized.keys() == expected.keys()
    for key, value in expected.items():
        assert mx.array_equal(sanitized[key], value), key
    reloaded = model.sanitize(sanitized)
    for key, value in sanitized.items():
        assert mx.array_equal(reloaded[key], value), key
    model.load_weights(list(reloaded.items()), strict=True)
    mx.eval(model(mx.array([[1, 2, 3]])).logits)
    (tmp_path / "config.json").write_text(json.dumps(asdict(model.config)))
    for weights in (raw, sanitized):
        mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
        loaded = load_model(tmp_path)
        assert isinstance(loaded.layers[0].mlp, moe.Qwen3_5MoeSparseMoeBlock)
        for key, value in tree_flatten(loaded.parameters()):
            assert mx.array_equal(value, expected[key]), key


@pytest.mark.parametrize("minicpmv47_case", ["minicpmv4_7_moe"], indirect=True)
def test_minicpmv47_router_quantization_and_state_dtype(minicpmv47_case):
    module, config = minicpmv47_case

    model = module.Model(config)
    prefix = "language_model.model.layers.0"
    assert model.quant_predicate(f"{prefix}.mlp.gate", None) == {
        "group_size": 64,
        "bits": 8,
    }
    assert model.quant_predicate(f"{prefix}.mlp.switch_mlp.up_proj", None)
    assert not model.quant_predicate(f"{prefix}.linear_attn.in_proj_qkv", None)
    assert not model.cast_predicate(f"{prefix}.linear_attn.A_log")
    assert model.cast_predicate(f"{prefix}.mlp.switch_mlp.up_proj.weight")


@pytest.mark.parametrize("name", DATA["dense"])
def test_dense_model(name):
    module = importlib.import_module("mlx_vlm.models." + name)
    config = DATA["dense"][name]
    model = module.Model(module.ModelConfig.from_dict(copy.deepcopy(config)))
    ModelChecks().forward_cache(model, config["vocab_size"])


def tiny_config(family, profile=None, **overrides):
    """Build a fresh tiny config, optionally selecting a named test profile."""
    case = TINY_MODELS[family]
    fields = TINY_DEFAULTS | case["config"] | case.get("profiles", {}).get(profile, {})
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return build_config(module, fields | overrides, case["config_type"])


def _prepare_minimax_distributed(model, dtype):
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear

    for layer in model.layers:
        if not layer.is_moe_layer:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.block_sparse_moe.switch_mlp, name)
            scale = mx.arange(1, projection.num_experts + 1, dtype=mx.float32) * 1344
            setattr(
                layer.block_sparse_moe.switch_mlp,
                name,
                ScaledQuantizedSwitchLinear(projection, scale, "weight_global_scale"),
            )


DISTRIBUTED_MODEL_SETUPS = {"minimax_m3_nvfp4": _prepare_minimax_distributed}


@pytest.mark.parametrize(
    "family", ["minimax_m3", "minimax_m3_nvfp4", "minimax_m3_packed"]
)
def test_minimax_shard_validation_and_partitioning(family):
    mx.random.seed(19)
    model = _make_distributed_test_model(family, mx.bfloat16)
    original = dict(tree_flatten(model.parameters()))
    model.shard(SimpleNamespace(size=lambda: 1, rank=lambda: 0))
    assert all(p is original[k] for k, p in tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="Expert count"):
        model.shard(SimpleNamespace(size=lambda: 3, rank=lambda: 0))
    assert all(p is original[k] for k, p in tree_flatten(model.parameters()))

    model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 1))
    for key, value in tree_flatten(model.parameters()):
        old = original[key]
        if ".switch_mlp." in key:
            routed = model.args.num_local_experts
            expected = old[routed // 2 : routed]
            if model.args.pack_shared_expert:
                expected = mx.concatenate([expected, old[-1:]], axis=0)
            assert mx.array_equal(value, expected).item(), key
        elif key.startswith("lm_head."):
            assert mx.array_equal(value, old[old.shape[0] // 2 :]).item(), key
        else:
            assert value is old, key
    with pytest.raises(ValueError, match="already sharded"):
        model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 1))


def test_minimax_shard_rejects_invalid_scales_without_mutating_weights():
    model = _make_distributed_test_model("minimax_m3_nvfp4", mx.bfloat16)
    projection = model.layers[-1].block_sparse_moe.switch_mlp.down_proj
    projection.weight_global_scale = mx.ones(3)
    original = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError):
        model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 0))
    assert all(p is original[k] for k, p in tree_flatten(model.parameters()))


@pytest.mark.parametrize("tied,vocab_size", [(True, 128), (False, 129)])
def test_minimax_shard_keeps_unsplittable_vocabulary_replicated(tied, vocab_size):
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel

    model = LanguageModel(
        tiny_config("minimax_m3", tie_word_embeddings=tied, vocab_size=vocab_size)
    )
    head = model.model.embed_tokens if tied else model.lm_head
    weight = head.weight
    model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 0))
    assert head.weight is weight
    assert not model._sharded_lm_head


@pytest.mark.parametrize("per_expert", [False, True])
def test_minimax_shard_preserves_quantization_scale_layouts(per_expert):
    model = _make_distributed_test_model("minimax_m3_nvfp4", mx.bfloat16)
    projection = model.layers[-1].block_sparse_moe.switch_mlp.down_proj
    projection.weight_global_scale = (
        mx.arange(4 * 128, dtype=mx.float32).reshape(4, 128) + 1
        if per_expert
        else mx.array(2688.0)
    )
    projection.pre_quant_scale = mx.arange(128, dtype=mx.float32) + 1
    if per_expert:
        projection.pre_quant_scale = mx.broadcast_to(
            projection.pre_quant_scale, (4, 128)
        )
    scale, smoothing = projection.weight_global_scale, projection.pre_quant_scale
    model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 1))
    if per_expert:
        assert mx.array_equal(projection.weight_global_scale, scale[2:]).item()
        assert mx.array_equal(projection.pre_quant_scale, smoothing[2:]).item()
    else:
        assert projection.weight_global_scale is scale
        assert projection.pre_quant_scale is smoothing


def test_minimax_shard_preserves_head_output_scale_broadcasting():
    from mlx_vlm.quantization.nvfp4 import ScaledQuantizedLinear

    model = _make_distributed_test_model("minimax_m3", mx.bfloat16)
    model.lm_head = ScaledQuantizedLinear(
        model.lm_head.to_quantized(group_size=16, bits=4, mode="nvfp4"),
        (mx.arange(128, dtype=mx.float32) + 1) * 128,
        "weight_global_scale",
    )
    hidden = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    expected = model.lm_head(hidden)[..., 64:]
    mx.eval(expected)
    model.shard(SimpleNamespace(size=lambda: 2, rank=lambda: 1))
    actual = model.lm_head(hidden)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected).item()


def test_minimax_multimodal_model_exposes_sharding():
    from mlx_vlm.models import minimax_m3_vl

    model = minimax_m3_vl.Model(
        build_config(minimax_m3_vl, model_case("minimax_m3_vl")["config"])
    )
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    with patch.object(model.language_model, "shard") as shard:
        model.shard(group)
    shard.assert_called_once_with(group)


# Distributed language-model contracts


def _make_distributed_test_model(family, dtype):
    case = TINY_MODELS[family]
    config = tiny_config(family, "distributed")
    module = importlib.import_module("mlx_vlm.models." + case["module"] + ".language")
    model = module.LanguageModel(config)
    model.update(
        tree_map(
            lambda p: p.astype(dtype) if mx.issubdtype(p.dtype, mx.floating) else p,
            model.parameters(),
        )
    )
    if setup := DISTRIBUTED_MODEL_SETUPS.get(family):
        setup(model, dtype)
    if quantization := case["distributed"].get("quantization"):
        quantization = copy.deepcopy(quantization)
        if patterns := quantization.pop("modules", None):

            def predicate(path, module):
                if hasattr(module, "to_quantized"):
                    for pattern, overrides in patterns.items():
                        if fnmatchcase(path, pattern):
                            return overrides
                return False

            quantization["class_predicate"] = predicate
        nn.quantize(model, **quantization)
    model.eval()
    return model


def _distributed_model_worker(family, dtype_name):
    group = mx.distributed.init(strict=True, backend="ring")
    options = TINY_MODELS[family]["distributed"]
    dtype = getattr(mx, dtype_name)
    mx.random.seed(19)
    reference = _make_distributed_test_model(family, dtype)
    mx.eval(reference.parameters())
    weights = dict(tree_flatten(reference.parameters()))
    sharded = _make_distributed_test_model(family, dtype)
    sharded.load_weights(list(weights.items()), strict=True)
    sharded.shard(group)
    if error := options.get("reshard_error"):
        with pytest.raises(ValueError, match=error):
            sharded.shard(group)
    mx.eval(sharded.parameters())
    local_weights = dict(tree_flatten(sharded.parameters()))
    for pattern in options["sharded_weights"]:
        names = [name for name in weights if fnmatchcase(name, pattern)]
        assert names, f"No weights match {pattern!r}"
        for name in names:
            assert local_weights[name].size * group.size() == weights[name].size, name

    records = []
    # Keep model-specific reference tolerances in the fixture. Every rank must
    # return identical logits; MiniMax also requires exact BF16 reference parity.
    tolerance = options["tolerances"][dtype_name]
    vocab_size = reference.args.vocab_size
    for batch in (1, 4):
        actual_cache = make_prompt_cache(sharded)
        expected_cache = make_prompt_cache(reference)
        for step, tokens in enumerate(
            ([1, 7, 9, 3, 11, 15, 2, 6, 8, 10, 12, 14], [13], [17], [19], [2, 4, 6])
        ):
            ids = mx.array(
                [[(t + row) % vocab_size for t in tokens] for row in range(batch)]
            )
            expected = reference(ids, cache=expected_cache).logits
            mx.eval(expected)
            # Finish independent reference work on every rank before queuing
            # distributed work on the shared GPU. Keep test-only collectives
            # on the CPU so a faster rank cannot block another rank's GPU work.
            mx.distributed.all_sum(mx.array(0), group=group, stream=mx.cpu).item()
            actual = sharded(ids, cache=actual_cache).logits
            mx.eval(actual)
            gathered = mx.distributed.all_gather(actual, group=group, stream=mx.cpu)
            mx.eval(gathered)
            ranks_agree = all(
                mx.array_equal(
                    gathered[:batch], gathered[rank * batch : (rank + 1) * batch]
                ).item()
                for rank in range(1, group.size())
            )
            records.append(
                dict(
                    batch=batch,
                    step=step,
                    max_error=mx.max(mx.abs(actual - expected)).item(),
                    ranks_agree=ranks_agree,
                    passed=mx.allclose(
                        actual, expected, atol=tolerance, rtol=tolerance
                    ).item(),
                )
            )
    passed = mx.distributed.all_sum(
        mx.array(int(all(r["passed"] and r["ranks_agree"] for r in records))),
        group=group,
        stream=mx.cpu,
    ).item()
    print(
        json.dumps(dict(rank=group.rank(), passed_ranks=passed, checks=records)),
        flush=True,
    )
    assert passed == group.size(), records


def _run_distributed_model_test(family, dtype, world_size):
    repo = Path(__file__).resolve().parents[2]
    # The ring launcher needs one consecutive port per rank. Probe the entire
    # range while holding the sockets so a four-rank case cannot reuse a port.
    for _ in range(100):
        with ExitStack() as stack:
            sockets = [stack.enter_context(socket.socket()) for _ in range(world_size)]
            sockets[0].bind(("127.0.0.1", 0))
            port = sockets[0].getsockname()[1]
            if port + world_size > 65536:
                continue
            try:
                for rank, sock in enumerate(sockets[1:], 1):
                    sock.bind(("127.0.0.1", port + rank))
            except OSError:
                continue
            break
    else:
        pytest.fail(f"Could not find {world_size} consecutive ring ports")
    command = [
        sys.executable,
        "-c",
        "from mlx._distributed_utils.launch import main; main()",
        "--backend",
        "ring",
        "-n",
        str(world_size),
        "--starting-port",
        str(port),
        "--env",
        f"PYTHONPATH={repo}",
        "--",
        sys.executable,
        "-c",
        "import sys; from mlx_vlm.tests.test_models import _distributed_model_worker; "
        "_distributed_model_worker(*sys.argv[1:])",
        family,
        dtype,
    ]
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=dict(os.environ, PYTHONPATH=str(repo)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(
            f"Distributed test timed out: {family}/{dtype}/{world_size}\n{output}"
        )
    assert process.returncode == 0, output
    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert len(records) == world_size, output
    assert {r["rank"] for r in records if r["passed_ranks"] == world_size} == set(
        range(world_size)
    ), output


@pytest.mark.skipif(not mx.distributed.is_available("ring"), reason="ring required")
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "family", [name for name, case in TINY_MODELS.items() if "distributed" in case]
)
def test_distributed_model(family, dtype, world_size):
    _run_distributed_model_test(family, dtype, world_size)


def test_gemma4_modelopt_awq_strict_load_and_forward(tmp_path):
    module = importlib.import_module("mlx_vlm.models.gemma4")
    fields = model_case("gemma4")["config"]
    fields["audio_config"] = None
    fields["text_config"].update(
        num_hidden_layers=2,
        head_dim=16,
        global_head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        use_double_wide_mlp=False,
    )
    mx.random.seed(661)
    reference = module.Model(build_config(module, fields))
    expected_weights = dict(tree_flatten(reference.parameters()))
    source = {}
    for key, value in expected_weights.copy().items():
        source_key = "model." + key.replace("language_model.model.", "language_model.")
        if (
            key.endswith(".weight")
            and any(
                f".{projection}." in key
                for projection in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )
            )
            and key.startswith("language_model.")
        ):
            prefix = source_key.removesuffix(".weight")
            factor = mx.array(0.137, mx.float32)
            packed, scales = mx.quantize(value / factor, mode="nvfp4")
            source[source_key] = packed.view(mx.uint8)
            source[prefix + ".weight_scale"] = scales
            source[prefix + ".weight_scale_2"] = factor
            source[prefix + ".input_scale"] = mx.array(0.25, mx.float32)
            decoded = mx.dequantize(packed, scales, mode="nvfp4") * factor
            if key.endswith(("down_proj.weight", "o_proj.weight")):
                pre_scale = mx.linspace(0.25, 2.0, value.shape[-1]).astype(mx.bfloat16)
                source[prefix + ".pre_quant_scale"] = pre_scale
                decoded *= pre_scale.astype(mx.float32)
            expected_weights[key] = decoded
        else:
            source[source_key] = value

    config = asdict(reference.config)
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4_AWQ",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), source)
    loaded = load_model(tmp_path, strict=True)
    reference.load_weights(list(expected_weights.items()))
    tokens = mx.array([[1, 5, 9]])
    expected = reference(tokens).logits
    actual = loaded(tokens).logits
    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-4).item()
    cache = loaded.language_model.make_cache()
    loaded(tokens[:, :2], cache=cache)
    decoded = loaded(tokens[:, 2:], cache=cache).logits
    assert mx.allclose(decoded, expected[:, 2:], atol=2e-5, rtol=2e-4).item()


def test_gemma3_can_preserve_caller_supplied_embedding_scale():
    case = next(case for case in DATA["cases"] if case["module"] == "gemma3")
    module = importlib.import_module(f"mlx_vlm.models.{case['module']}.language")
    config = build_config(
        module,
        case["config"]["text_config"]
        | TINY_DEFAULTS
        | dict(intermediate_size=32, head_dim=8, sliding_window_pattern=1),
        "TextConfig",
    )
    model = module.Gemma3Model(config, scale_inputs_embeds=False)
    capture = MagicMock(side_effect=lambda inputs, *_: inputs)
    model.layers, model.norm = [capture], nn.Identity()
    expected = mx.ones((1, 1, config.hidden_size))
    output = model(None, inputs_embeds=mx.array(expected))

    assert mx.array_equal(capture.call_args.args[0], expected)
    assert mx.array_equal(output, expected)


def rotating_language_model(case):
    """Reuse model contracts and tiny defaults for cache-specific model checks."""
    name = case["module"]
    module = importlib.import_module(f"mlx_vlm.models.{name}")
    language = importlib.import_module(f"mlx_vlm.models.{name}.language")
    outer, text = None, {}
    if "case" in case:
        outer = build_config(module, model_case(case["case"])["config"])
        outer = getattr(outer, "thinker_config", outer)
        text = asdict(outer.text_config)
    defaults = {} if name == "florence2" else TINY_DEFAULTS
    config = build_config(
        language,
        text | defaults | case["config"],
        case.get("config_type", "TextConfig"),
    )
    if name == "florence2":
        return language.Florence2Decoder(config), config
    if name in ("qwen2", "qwen3", "mllama", "moondream3", "minimax_m3_vl", "gemma3n"):
        return language.LanguageModel(config), config
    if outer is None:
        outer = module.ModelConfig(
            text_config=config, vision_config=module.VisionConfig()
        )
    else:
        outer.text_config = config
    model = (
        language.LanguageModel(outer)
        if name == "hunyuan_vl"
        else language.LanguageModel(config, outer)
    )
    return model, config


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "chunk_parity" in c["checks"]],
    ids=lambda c: c["module"],
)
@pytest.mark.parametrize("keep", [0, 2])
def test_rotating_cache_random_model_chunk_parity(case, keep):
    mx.random.seed(17)
    model, config = rotating_language_model(case)
    model.eval()
    mx.eval(model.parameters())
    ids = (mx.arange(41)[None] * 7 + 3) % config.vocab_size

    def forward(schedule):
        for attr in ("_rope_deltas", "_position_ids"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        caches = [RotatingKVCache(8, keep=keep) for _ in model.layers]
        result, start = [], 0
        for n in schedule:
            logits = model(ids[:, start : start + n], cache=caches).logits
            mx.eval(logits, [c.state for c in caches])
            result.append(logits)
            start += n
        assert start == ids.shape[1]
        return np.array(mx.concatenate(result, axis=1))

    expected = forward([1] * 41)
    # Include a chunk larger than the cache and multi-token continuation after
    # single-token decoding has wrapped the storage cursor.
    for schedule in ([3] * 13 + [1, 1], [17, 7, 8, 8, 1], [1] * 19 + [5, 8, 8, 1]):
        np.testing.assert_allclose(forward(schedule), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "prompt_metadata" in c["checks"]],
    ids=lambda c: c["module"],
)
def test_rotated_model_slices_full_prompt_metadata(case):
    name = case["module"]
    mx.random.seed(19)
    model, config = rotating_language_model(case)
    model.eval()
    ids = (mx.arange(24)[None] * 7 + 3) % config.vocab_size
    positions = mx.broadcast_to(mx.arange(24)[None, None, :], (3, 1, 24))
    positions = positions + mx.array([3, 7, 11])[:, None, None]
    if name == "falcon_ocr":
        positions = positions[0]
    metadata = {"position_ids": positions}
    if name in ("qwen3_vl", "qwen3_vl_moe", "qwen3_omni_moe"):
        metadata["visual_pos_masks"] = mx.ones((1, 24), mx.bool_)
        metadata["deepstack_visual_embeds"] = (
            mx.random.normal((1, 24, 2, config.hidden_size))
            if name == "qwen3_omni_moe"
            else [mx.random.normal((24, config.hidden_size)) for _ in range(2)]
        )

    def forward(full_metadata):
        for attr in ("_rope_deltas", "_position_ids"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        cache = [RotatingKVCache(8, keep=2) for _ in model.layers]
        for start in range(21):
            out = model(ids[:, start : start + 1], cache=cache)
            mx.eval(out.logits)
        assert cache[0].offset == 21 and cache[0]._idx != 21
        kwargs = dict(metadata)
        if not full_metadata:
            kwargs["position_ids"] = positions[..., 21:24]
            if "visual_pos_masks" in kwargs:
                kwargs["visual_pos_masks"] = kwargs["visual_pos_masks"][:, 21:24]
                embeds = kwargs["deepstack_visual_embeds"]
                kwargs["deepstack_visual_embeds"] = (
                    [e[21:24] for e in embeds]
                    if isinstance(embeds, list)
                    else embeds[:, 21:24]
                )
        if name == "falcon_ocr":
            # Falcon's full-prompt metadata is passed with precomputed embeddings.
            kwargs["inputs_embeds"] = model.model.embed_tokens(ids[:, 21:24])
            if not full_metadata:
                # This wrapper accepts full position IDs only; compare the
                # internally selected positions against a direct model call.
                mask = create_attention_mask(kwargs["inputs_embeds"], cache)
                return np.array(
                    model.lm_head(
                        model.model(
                            ids[:, 21:24],
                            inputs_embeds=kwargs["inputs_embeds"],
                            position_ids=positions[..., 21:24],
                            mask=mask,
                            cache=cache,
                        )
                    )
                )
        return np.array(model(ids[:, 21:24], cache=cache, **kwargs).logits)

    np.testing.assert_allclose(forward(True), forward(False), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "per_layer_inputs" in c["checks"]],
    ids=lambda c: c["module"],
)
def test_rotated_per_layer_inputs(case):
    mx.random.seed(23)
    model, config = rotating_language_model(case)
    # The constructor initializes this gate to zero, which would hide a bad
    # per-layer-input slice in an otherwise randomly initialized model.
    for layer in model.layers:
        layer.altup.correct_output_scale = mx.random.uniform(
            low=0.5, high=1.5, shape=(config.hidden_size,)
        )
    model.eval()
    ids = (mx.arange(41)[None] * 7 + 3) % config.vocab_size
    per_layer_inputs = model.model.get_per_layer_inputs(ids)

    def forward(step, full_metadata):
        cache, outputs = model.make_cache(), []
        for start in range(0, 41, step):
            metadata = (
                per_layer_inputs
                if full_metadata
                else per_layer_inputs[:, start : start + step]
            )
            out = model(
                ids[:, start : start + step], cache=cache, per_layer_inputs=metadata
            )
            mx.eval(out.logits)
            outputs.append(out.logits)
        return np.array(mx.concatenate(outputs, axis=1))

    expected = forward(1, False)
    np.testing.assert_allclose(forward(3, True), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "decoder_positions" in c["checks"]],
    ids=lambda c: c["module"],
)
def test_rotated_decoder_positions_and_mask(case):
    mx.random.seed(29)
    decoder, config = rotating_language_model(case)
    decoder.eval()
    encoder_states = mx.random.normal((1, 4, config.d_model))
    embeddings = mx.random.normal((1, 41, config.d_model))

    def forward(step):
        cache = [[RotatingKVCache(8, keep=2), KVCache()] for _ in decoder.layers]
        outputs = []
        for start in range(0, 41, step):
            out = decoder(
                inputs_embeds=embeddings[:, start : start + step],
                encoder_hidden_states=encoder_states,
                cache=cache,
            )
            mx.eval(out)
            outputs.append(out)
        return np.array(mx.concatenate(outputs, axis=1))

    np.testing.assert_allclose(forward(3), forward(1), atol=1e-4, rtol=1e-4)


# Document layout


class TestPPDocLayoutV3(unittest.TestCase):
    def test_pp_doclayout_v3_config_routes(self):
        from mlx_vlm.models import pp_doclayout_v3
        from mlx_vlm.utils import get_model_and_args

        model_class, model_type = get_model_and_args(
            config={"model_type": "pp_doclayout_v3"}
        )
        assert model_class is pp_doclayout_v3
        assert model_type == "pp_doclayout_v3"
        assert pp_doclayout_v3.ModelConfig().num_labels == 25

        cfg = pp_doclayout_v3.ModelConfig.from_dict(
            {
                "model_type": "pp_doclayout_v3",
                "num_labels": 37,
                "id2label": {"0": "Question", "1": "Paragraph"},
                "backbone_config": {"model_type": "hgnet_v2"},
            }
        )
        assert cfg.id2label == {0: "Question", 1: "Paragraph"}
        assert cfg.num_queries == 300
        assert cfg.decoder_layers == 6

    def test_pp_doclayout_v3_decode_order(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import decode_order

        # Chain 0 -> 1 -> 2 with strong pairwise scores.
        scores = mx.array([[-1e4, 5.0, 5.0], [-1e4, -1e4, 5.0], [-1e4, -1e4, -1e4]])
        assert decode_order(scores).tolist() == [0, 1, 2]
        # Reference formula cross-check on random input.
        rng_scores = mx.random.normal((7, 7))
        mx.eval(rng_scores)
        got = decode_order(rng_scores).tolist()
        import numpy as np

        arr = np.array(rng_scores.tolist())
        s = 1.0 / (1.0 + np.exp(-arr))
        votes = np.triu(s, 1).sum(0) + np.tril(1.0 - s.T, -1).sum(0)
        assert got == np.argsort(votes).tolist()

    def test_pp_doclayout_v3_mask_to_box(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import mask_to_box_coordinate

        mask = mx.zeros((1, 2, 8, 10))
        mask[0, 0, 2:5, 3:7] = 1.0
        mx.eval(mask)
        boxes = mask_to_box_coordinate(mask)
        mx.eval(boxes)
        # x in [3,7), y in [2,5) over W=10,H=8 -> cxcywh
        self.assertAlmostEqual(float(boxes[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 1]), 0.4375, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 2]), 0.4, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 3]), 0.375, places=5)
        # Empty mask -> zeros.
        assert float(boxes[0, 1].sum()) == 0.0

    def test_pp_doclayout_v3_bilinear_upsample(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.encoder import upsample_bilinear2x

        x = mx.array([[[[0.0], [1.0]], [[2.0], [3.0]]]])
        mx.eval(x)
        y = upsample_bilinear2x(x)
        mx.eval(y)
        # align_corners=False exact values: edges replicate, interior lerps.
        assert tuple(y.shape) == (1, 4, 4, 1)
        self.assertAlmostEqual(float(y[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(y[0, 0, 1, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(y[0, 1, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(y[0, 1, 1, 0]), 0.75, places=5)
        self.assertAlmostEqual(float(y[0, 3, 3, 0]), 3.0, places=5)

    def test_pp_doclayout_conversion_without_torch(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.convert import convert

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "source"
                src.mkdir()
                config = {"model_type": "pp_doclayout_v3"}
                (src / "config.json").write_text(json.dumps(config))
                weight = mx.arange(48).reshape(2, 3, 2, 4).astype(dtype)
                mx.save_safetensors(
                    str(src / "model.safetensors"),
                    {
                        "model.backbone.model.embedder.stem1.convolution.weight": weight,
                        "model.denoising_class_embed.weight": mx.ones((2, 2)),
                        "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": mx.array(
                            0
                        ),
                    },
                )
                with (
                    patch.dict(
                        "sys.modules", {"torch": None, "safetensors.torch": None}
                    ),
                    patch("mlx_vlm.models.pp_doclayout_v3.convert._verify") as verify,
                ):
                    out = convert(str(src), str(Path(tmp) / "converted"))
                    again = convert(
                        str(out), str(Path(tmp) / "reconverted"), "bfloat16"
                    )
                assert verify.call_count == 2
                key = "backbone.embedder.stem1.conv.weight"
                expected = weight.transpose(0, 2, 3, 1)
                converted = mx.load(str(out / "model.safetensors"))
                reconverted = mx.load(str(again / "model.safetensors"))
                assert set(converted) == {key}
                assert set(reconverted) == {key}
                assert converted[key].dtype == mx.float32
                assert reconverted[key].dtype == mx.bfloat16
                assert converted[key].tolist() == expected.tolist()
                assert reconverted[key].tolist() == expected.tolist()
                assert json.loads((out / "config.json").read_text()) == config


# Loading and utility contracts


def test_load_config_applies_generation_config_sampling_defaults(tmp_path):
    generation_config = {
        "eos_token_id": [2, 3],
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 4096,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "demo", "eos_token_id": 1}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = load_config(tmp_path)

    assert config["generation_config"] == generation_config
    assert config["eos_token_id"] == [2, 3]
    assert config["do_sample"] is True
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["top_k"] == 64
    assert "max_new_tokens" not in config


def test_get_model_path_downloads_jsonl_tokenizers(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("mlx_vlm.utils.snapshot_download", fake_snapshot_download)

    assert get_model_path("org/model") == tmp_path
    assert "*.jsonl" in captured["allow_patterns"]


def test_load_passes_revision():
    model_mock = MagicMock()
    model_mock.config = MagicMock(eos_token_id=None)
    processor_mock = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path") as mock_get_model_path,
        patch("mlx_vlm.utils.load_model", return_value=model_mock),
        patch("mlx_vlm.utils.load_processor", return_value=processor_mock),
        patch("mlx_vlm.utils.load_image_processor", return_value=None),
    ):
        mock_get_model_path.return_value = Path("/tmp/model")

        model, processor = load("repo", revision="abc")

        assert model is model_mock
        assert processor is processor_mock
        mock_get_model_path.assert_called_with(
            "repo", revision="abc", force_download=False
        )


def test_get_model_and_args_rejects_unknown_text_configs():
    with pytest.raises(ValueError):
        get_model_and_args({"model_type": "unknown_text_arch"})


class TestDropModulesWithoutWeights:
    class ParameterlessHelper(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config
            self.language_model = nn.Linear(2, 2, bias=False)
            self.vision_tower = nn.Linear(2, 2, bias=True)
            self.parameterless_helper = (
                TestDropModulesWithoutWeights.ParameterlessHelper()
            )

    def test_keeps_module_declared_in_manifest_but_not_loaded(self):
        # Manifest declares vision weights but none loaded -> keep, strict fails (#1963).
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight", "vision_tower.weight", "vision_tower.bias"}

        _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is not None
        with pytest.raises(ValueError, match="Missing"):
            model.load_weights(list(weights.items()), strict=True)

    def test_drops_module_absent_from_manifest(self, caplog):
        # The manifest also omits vision -> an intentional text-only conversion.
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight"}

        with caplog.at_level(logging.WARNING):
            _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is None
        assert "vision_tower" in caplog.text


def test_load_safetensors_reinterprets_f8_e8m0_header(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"weight": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00")

    loaded = {"weight": mx.array([1], dtype=mx.uint8)}

    def fake_mx_load(file_path):
        current = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
        if current["weight"]["dtype"] == "F8_E8M0":
            raise RuntimeError("unsupported dtype F8_E8M0")
        assert current["weight"]["dtype"] == "U8"
        return loaded

    with patch("mlx_vlm.utils.mx.load", side_effect=fake_mx_load):
        assert _load_safetensors(str(path)) is loaded

    restored = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
    assert restored["weight"]["dtype"] == "F8_E8M0"


class _CheckpointConfig:
    @classmethod
    def from_dict(cls, config):
        return cls()


class _CheckpointModel(nn.Module):
    def __init__(self, config, **modules):
        super().__init__()
        self.config = config
        for name, module in modules.items():
            setattr(self, name, module)

    def load_weights(self, weights, strict=True):
        self.loaded_weights, self.loaded_strict = weights, strict


@contextmanager
def _checkpoint_loading(config, model_class, weights, *, side_effect=None):
    with (
        patch("mlx_vlm.utils.load_config", return_value=config),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value=weights),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(
                SimpleNamespace(ModelConfig=_CheckpointConfig, Model=model_class),
                config["model_type"],
            ),
        ),
        patch("mlx_vlm.utils.nn.quantize", side_effect=side_effect) as quantize,
    ):
        yield quantize


def test_load_model_uses_deepseek_v4_fp8_quantization_config():

    quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.weight": {"group_size": 64, "bits": 8, "mode": "affine"},
    }
    with (
        patch(
            "mlx_vlm.models.deepseek_v4.language.make_quantization_config",
            return_value=quantization,
        ) as make_quantization_config,
        _checkpoint_loading(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            },
            lambda config: _CheckpointModel(
                config, language_model=nn.Linear(2, 2, bias=False)
            ),
            {},
        ) as quantize,
    ):
        model = load_model(Path("/tmp/model"), lazy=True)
    make_quantization_config.assert_called_once_with(model)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 64
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "affine"


def test_load_model_transforms_fine_grained_fp8_by_format():

    source_config = {
        "model_type": "future_compatible_model",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    with _checkpoint_loading(
        source_config,
        lambda config: _CheckpointModel(config, proj=nn.Linear(128, 128, bias=False)),
        {
            "proj.weight": mx.zeros((128, 128), dtype=mx.uint8),
            "proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.bfloat16),
        },
    ) as quantize:
        model = load_model(Path("/tmp/model"), lazy=True)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 32
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "mxfp8"
    loaded = dict(model.loaded_weights)
    assert loaded["proj.weight"].dtype == mx.uint32
    assert loaded["proj.scales"].dtype == mx.uint8
    assert "proj.weight_scale_inv" not in loaded


def test_load_model_quantizes_projector_with_scales_when_skip_vision():

    def model(config):
        projector = nn.Module()
        projector.linear_1 = nn.Linear(64, 64, bias=False)
        return _CheckpointModel(
            config,
            vision_tower=nn.Linear(64, 64, bias=False),
            multi_modal_projector=projector,
            language_model=nn.Linear(64, 64, bias=False),
        )

    weights = {
        "language_model.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "language_model.scales": mx.zeros((64, 1), dtype=mx.float16),
        "multi_modal_projector.linear_1.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "multi_modal_projector.linear_1.scales": mx.zeros((64, 1), dtype=mx.float16),
        "vision_tower.weight": mx.zeros((64, 64), dtype=mx.float16),
    }
    selected = {}

    def fake_quantize(model, *args, **kwargs):
        predicate = kwargs["class_predicate"]
        selected["language"] = predicate("language_model", model.language_model)
        selected["projector"] = predicate(
            "multi_modal_projector.linear_1", model.multi_modal_projector.linear_1
        )
        selected["vision"] = predicate("vision_tower", model.vision_tower)

    with _checkpoint_loading(
        {
            "model_type": "kimi_vl",
            "quantization": {"group_size": 64, "bits": 8},
            "vision_config": {"skip_vision": True},
        },
        model,
        weights,
        side_effect=fake_quantize,
    ):
        load_model(Path("/tmp/model"), lazy=True)
    assert selected == {"language": True, "projector": True, "vision": False}


# Local Python model files

MODEL_PY = textwrap.dedent("""
    import mlx.core as mx
    import mlx.nn as nn


    class ModelConfig:
        # Deliberately minimal: exposing text_config/vision_config attributes
        # opts in to update_module_configs, which then requires TextConfig /
        # VisionConfig classes in this module. A model_file module controls
        # both sides of that contract.
        def __init__(self, model_type="custom"):
            self.model_type = model_type

        @classmethod
        def from_dict(cls, params):
            return cls(model_type=params.get("model_type", "custom"))


    class Model(nn.Module):
        loaded_via_model_file = True

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(4, 4, bias=False)

        def __call__(self, x):
            return self.proj(x)
    """)


def _write_checkpoint(path, config_extra=None):
    config = {"model_type": "does-not-exist-in-registry", "model_file": "model.py"}
    config.update(config_extra or {})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.py").write_text(MODEL_PY, encoding="utf-8")
    mx.save_safetensors(
        str(path / "model.safetensors"),
        {"proj.weight": mx.zeros((4, 4))},
        metadata={"format": "mlx"},
    )


def test_load_model_uses_checkpoint_model_file(tmp_path):
    _write_checkpoint(tmp_path)

    model = load_model(tmp_path)

    # the class must come from the checkpoint's model.py, not the registry
    # (the registry would have raised: model_type does not exist there)
    assert getattr(model, "loaded_via_model_file", False) is True
    assert model.proj.weight.shape == (4, 4)


def test_missing_model_file_raises_clearly(tmp_path):
    _write_checkpoint(tmp_path)
    (tmp_path / "model.py").unlink()

    with pytest.raises(FileNotFoundError, match="model_file"):
        load_model(tmp_path)


def test_qwen3_omni_audio_batch_matches_repeated_input():
    from mlx_vlm.models.qwen3_omni_moe.audio import AudioModel
    from mlx_vlm.models.qwen3_omni_moe.config import AudioConfig

    config = AudioConfig(
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        num_mel_bins=8,
        output_dim=8,
        downsample_hidden_size=4,
        conv_chunksize=4,
        max_source_positions=64,
    )
    model = AudioModel(config)
    sample = mx.random.normal((8, 170))
    output = model(mx.concatenate([sample, sample], axis=1), mx.array([170, 170]))

    assert output.shape == (44, 8)
    assert mx.allclose(output[:22], output[22:])


def test_qwen3_omni_rope_delta_ignores_batch_padding():
    from mlx_vlm.models.qwen3_omni_moe.language import LanguageModel

    model = LanguageModel.__new__(LanguageModel)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
    )
    input_ids = mx.array([[0, 0, 1, 2], [1, 2, 3, 4]])
    attention_mask = mx.array([[0, 0, 1, 1], [1, 1, 1, 1]])

    _, rope_deltas = model.get_rope_index(input_ids, attention_mask=attention_mask)

    assert rope_deltas.tolist() == [[0], [0]]


# Patch embedding layouts


QWEN_PATCH_EMBED_KEY = "model.visual.patch_embed.proj.weight"


QWEN_SANITIZED_KEY = "vision_tower.patch_embed.proj.weight"


def _qwen_patch_model(
    in_channels=3, temporal_patch_size=2, patch_size=4, hidden_size=8
):
    qwen = importlib.import_module("mlx_vlm.models.qwen3_5")
    text_config = qwen.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    vision_config = qwen.VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=hidden_size,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    config = qwen.ModelConfig(
        text_config=text_config, vision_config=vision_config, model_type="qwen3_5"
    )
    return qwen.Model(config), vision_config


def test_patch_embed_is_transposed_from_ncdhw_to_ndhwc():
    """Qwen3.5 stores the Conv3d patch embed as NCDHW; MLX expects NDHWC."""
    model, vision_config = _qwen_patch_model()
    expected = model.vision_tower.patch_embed.proj.weight.shape

    ncdhw = mx.zeros(
        (
            vision_config.hidden_size,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        ),
        dtype=mx.bfloat16,
    )
    sanitized = model.sanitize({QWEN_PATCH_EMBED_KEY: ncdhw})

    assert sanitized[QWEN_SANITIZED_KEY].shape == expected


def test_glm_quantized_head_sanitization_loads_strictly():
    module = importlib.import_module("mlx_vlm.models.glm5_next")
    model = module.Model(
        module.ModelConfig(
            text_config=tiny_config("glm", hidden_size=32),
            vision_config=module.VisionConfig(
                depth=1,
                hidden_size=16,
                intermediate_size=32,
                out_hidden_size=32,
                projection_intermediate_size=32,
                num_heads=2,
                image_size=8,
                patch_size=2,
            ),
        )
    )
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, _: path.endswith("language_model.lm_head"),
    )
    checkpoint = dict(tree_flatten(model.parameters()))
    head = {
        f"lm_head.{suffix}": checkpoint.pop(f"language_model.lm_head.{suffix}")
        for suffix in ("weight", "scales", "biases")
    }
    sanitized = model.sanitize(head)
    assert sanitized.keys() == {
        f"language_model.lm_head.{suffix}" for suffix in ("weight", "scales", "biases")
    }
    again = model.sanitize(sanitized)
    assert again.keys() == sanitized.keys()
    for key, value in sanitized.items():
        assert mx.array_equal(again[key], value).item()
    model.load_weights(list(model.sanitize(checkpoint | head).items()), strict=True)


def test_moondream2_sanitize_remaps_checkpoint_layout():
    from mlx_vlm.models.moondream2 import Model

    source = {
        "model.text.wte": "text.model.embed_tokens.weight",
        "model.text.blocks.0.attn.qkv.weight": "text.model.layers.0.attn.qkv.weight",
        "model.text.post_ln.weight": "text.model.post_ln.weight",
        "model.text.lm_head.weight": "text.lm_head.weight",
        "model.vision.patch_emb.weight": "vision.encoder.patch_emb.weight",
        "model.vision.blocks.0.ln1.weight": "vision.encoder.blocks.0.ln1.weight",
        "model.vision.proj_mlp.fc1.weight": "vision.proj_mlp.fc1.weight",
    }
    weights = {key: mx.zeros((1,)) for key in source}
    weights["model.region.coord_decoder.fc1.weight"] = mx.zeros((1,))

    assert set(Model.sanitize(None, weights)) == set(source.values())


def test_moondream3_sanitize_remaps_raw_and_preserves_converted_keys():
    from mlx_vlm.models.moondream3 import Model

    raw = {
        "model.text.blocks.0.attn.qkv.weight": mx.zeros((1,)),
        "model.vision.blocks.0.ln1.weight": mx.zeros((1,)),
    }
    converted = Model.sanitize(None, raw)
    assert set(converted) == {
        "text.model.blocks.0.attn.qkv.weight",
        "vision.encoder.blocks.0.ln1.weight",
    }
    converted["text.lm_head.weight"] = mx.zeros((1,))
    converted["vision.proj_mlp.fc1.weight"] = mx.zeros((1,))
    assert Model.sanitize(None, converted).keys() == converted.keys()


class TestKolibri1Sanitize(unittest.TestCase):
    def _model(self):
        from mlx_vlm.models.kolibri1 import ModelConfig
        from mlx_vlm.models.kolibri1.language import LanguageModel

        case = next(case for case in DATA["cases"] if case["module"] == "kolibri1")
        config = copy.deepcopy(case["config"])
        config["num_hidden_layers"] = 1
        config["layer_types"] = ["full_attention"]
        return LanguageModel(ModelConfig.from_dict(config))

    def _checkpoint(self, model, layout):
        checkpoint = {}
        for key, value in tree_flatten(model.parameters()):
            if ".mlp.gate.e_score_correction_bias" in key:
                if layout == "per_expert":
                    key = key.replace(
                        ".mlp.gate.e_score_correction_bias",
                        ".moe.router.expert_bias",
                    )
                elif layout == "stacked":
                    key = key.replace(
                        ".mlp.gate.e_score_correction_bias", ".mlp.expert_bias"
                    )
            if ".mlp.switch_mlp." in key:
                prefix, projection = key.split(".mlp.switch_mlp.")
                if layout == "stacked":
                    checkpoint[f"{prefix}.mlp.experts.{projection}"] = value
                    continue
                if layout == "per_expert":
                    for expert_idx, expert_value in enumerate(value):
                        checkpoint[
                            f"{prefix}.mlp.experts.{expert_idx}.{projection}"
                        ] = expert_value
                    continue
            checkpoint[key] = value
        return checkpoint

    def test_checkpoint_layouts_sanitize_to_the_model_exactly(self):
        model = self._model()
        expected = dict(tree_flatten(model.parameters()))
        for layout in ("per_expert", "stacked", "canonical"):
            with self.subTest(layout=layout):
                sanitized = model.sanitize(self._checkpoint(model, layout))
                self.assertEqual(sanitized.keys(), expected.keys())
                for key, value in expected.items():
                    self.assertTrue(mx.array_equal(sanitized[key], value).item(), key)
                model.load_weights(list(sanitized.items()), strict=True)


class TestQwen3_5MoeText(unittest.TestCase):
    """Decoder-only Qwen3.5 MoE checkpoints (model_type qwen3_5_moe_text)."""

    def _model(self):
        from mlx_vlm.models.qwen3_5_moe_text import Model, ModelConfig

        case = next(
            case for case in DATA["cases"] if case["module"] == "qwen3_5_moe_text"
        )
        model = Model(ModelConfig.from_dict(copy.deepcopy(case["config"])))
        model.update(
            tree_map(
                lambda p: (mx.random.randint(-8, 8, p.shape) / 4).astype(p.dtype),
                model.parameters(),
            )
        )
        return model

    def _raw_checkpoint(self, model, prefix, fused):
        """Rebuild a published checkpoint from the model's own parameters."""
        from mlx_vlm.models.qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES

        raw = {}
        for key, value in tree_flatten(model.parameters()):
            if ".switch_mlp." in key:
                continue
            if key.startswith("language_model.model."):
                raw_key = prefix + key[len("language_model.model.") :]
            else:
                raw_key = key.replace("language_model.lm_head", "lm_head", 1)
            if "conv1d.weight" in key:
                value = value.swapaxes(1, 2)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                value = value - 1.0
            raw[raw_key] = value
        for layer_idx, layer in enumerate(model.layers):
            experts = f"{prefix}layers.{layer_idx}.mlp.experts"
            switch = layer.mlp.switch_mlp
            if fused:
                raw[f"{experts}.gate_up_proj"] = mx.concatenate(
                    [switch.gate_proj.weight, switch.up_proj.weight], axis=-2
                )
                raw[f"{experts}.down_proj"] = switch.down_proj.weight
            else:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    weight = getattr(switch, name).weight
                    for e in range(weight.shape[0]):
                        raw[f"{experts}.{e}.{name}.weight"] = weight[e]
        return raw

    def test_published_layouts_sanitize_to_the_model_exactly(self):
        model = self._model()
        expected = dict(tree_flatten(model.parameters()))
        for prefix in ("model.language_model.", "model."):
            for fused in (True, False):
                with self.subTest(prefix=prefix, fused=fused):
                    raw = self._raw_checkpoint(model, prefix, fused)
                    raw[f"{prefix}layers.0.mlp.gate.input_global_scale"] = mx.ones(1)
                    sanitized = model.sanitize(raw)
                    self.assertEqual(sanitized.keys(), expected.keys())
                    for key, value in expected.items():
                        self.assertTrue(
                            mx.array_equal(sanitized[key], value).item(), key
                        )
                    model.load_weights(list(sanitized.items()), strict=True)

    def test_ragged_expert_tensors_fail_clearly(self):
        model = self._model()
        raw = self._raw_checkpoint(model, "model.", fused=True)
        raw["model.layers.0.mlp.experts.gate_up_proj"] = mx.zeros((123,))
        with self.assertRaisesRegex(ValueError, "expected \\[num_experts"):
            model.sanitize(raw)

    def test_sanitize_is_idempotent_on_converted_weights(self):
        model = self._model()
        converted = dict(tree_flatten(model.parameters()))
        again = model.sanitize(dict(converted))
        self.assertEqual(again.keys(), converted.keys())
        for key, value in converted.items():
            self.assertTrue(mx.array_equal(again[key], value).item(), key)

    def test_missing_mrope_section_is_plain_rope(self):
        from mlx_vlm.models.qwen3_5_moe_text import Model, ModelConfig

        model = self._model()
        weights = list(tree_flatten(model.parameters()))
        ids = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])

        def logits(rope_parameters):
            config = vars(model.config) | {"rope_parameters": rope_parameters}
            other = Model(ModelConfig.from_dict(config))
            other.load_weights(weights, strict=True)
            return other(ids).logits

        base = {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        }
        missing = logits(dict(base))
        for section in ([2, 1, 1], [1, 1, 2], [4, 0, 0]):
            with self.subTest(section=section):
                explicit = logits(dict(base, mrope_section=section))
                self.assertTrue(mx.allclose(missing, explicit, atol=1e-5).item())


# DeepSeek-V4.1 regressions beyond the shared model contracts


class TestDeepseekV41EndToEnd(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def test_index_rotary_matches_reference_at_long_positions(self):
        from mlx_vlm.models.deepseek_v41.language import (
            _apply_index_rotary,
            _index_cos_sin,
        )

        dim, length, theta = 64, 32769, 160000
        factor, beta_fast, beta_slow, original_length = 16, 32, 1, 65536

        def correction(rotations):
            return (
                dim
                * np.log(original_length / (rotations * 2 * np.pi))
                / (2 * np.log(theta))
            )

        low = int(np.floor(correction(beta_fast)))
        high = int(np.ceil(correction(beta_slow)))
        ramp = (np.arange(dim // 2, dtype=np.float32) - low) / (high - low)
        smooth = 1 - np.clip(ramp, 0, 1)
        powers = np.power(np.float64(theta), np.arange(0, dim, 2) / dim)
        inv = 1 / powers.astype(np.float32)
        inv = inv / factor * (1 - smooth) + inv * smooth
        phase = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
        cos, sin = _index_cos_sin(
            length, dim, theta, (factor, beta_fast, beta_slow, original_length)
        )
        phase = phase.astype(np.float64)
        np.testing.assert_array_equal(np.array(cos), np.cos(phase).astype(np.float32))
        np.testing.assert_array_equal(np.array(sin), np.sin(phase).astype(np.float32))

        for batch, count in ((1, 1), (2, 17)):
            with self.subTest(batch=batch, length=count):
                x = mx.random.normal((batch, count, 4, 80)).astype(mx.bfloat16)
                c, s = cos[None, -count:, None], sin[None, -count:, None]
                data = np.array(x.astype(mx.float32))
                rotated = data[..., -dim:].copy().view(np.complex64)
                frequencies = np.array(c) + np.complex64(1j) * np.array(s)
                rotated = (rotated * frequencies).view(np.float32)
                expected = np.concatenate([data[..., :-dim], rotated], axis=-1)
                actual = _apply_index_rotary(x, c, s, dim)
                self.assertEqual(actual.dtype, x.dtype)
                self.assertTrue(
                    mx.array_equal(actual, mx.array(expected).astype(x.dtype)).item()
                )

    def test_hc_expansion_preserves_reference_rounding(self):
        from mlx_vlm.models.deepseek_v41.language import hc_expand

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for batch, length in ((1, 1), (2, 5), (1, 256)):
                with self.subTest(dtype=dtype, batch=batch, length=length):
                    x = mx.random.normal((batch, length, 128)).astype(dtype)
                    residual = mx.random.normal((batch, length, 4, 128)).astype(dtype)
                    post = mx.random.uniform(low=0, high=2, shape=(batch, length, 4))
                    comb = mx.softmax(mx.random.normal((batch, length, 4, 4)), axis=-1)
                    xx, rr, pp, cc = [
                        np.array(a.astype(mx.float32))
                        for a in (x, residual, post, comb)
                    ]
                    # Independent CPU evaluation of the source's explicit
                    # products and reduction; matmul changes the rounding.
                    expected = pp[..., None] * xx[:, :, None, :] + np.sum(
                        cc[..., None] * rr[..., None, :], axis=2, dtype=np.float32
                    )
                    actual = hc_expand(x, residual, post, comb)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(
                        mx.array_equal(actual, mx.array(expected).astype(dtype)).item()
                    )

    def test_activation_fp8_rounding_matches_reference(self):
        from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0

        # Exercise every E4M3 rounding tie and its immediate FP32 neighbors.
        # The 448 anchor fixes the block scale at one.
        levels = np.array(mx.from_fp8(mx.arange(127, dtype=mx.uint8), mx.float32))
        midpoints = (levels[:-1] + levels[1:]) / 2
        values = np.concatenate(
            [
                np.nextafter(midpoints, -np.inf),
                midpoints,
                np.nextafter(midpoints, np.inf),
            ]
        )
        ties = np.zeros((2 * len(values), 32), dtype=np.float32)
        ties[:, 0] = np.concatenate([values, -values])
        ties[:, 1] = 448
        samples = [
            mx.array(ties),
            mx.random.normal((4, 3, 512)).transpose(1, 0, 2),
            mx.random.normal((2, 64)) * 1e-8,
            mx.zeros((1, 32)),
        ]
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for sample in samples:
                with self.subTest(dtype=dtype, shape=sample.shape):
                    x = sample.astype(dtype)
                    blocks = np.array(x.astype(mx.float32)).reshape(-1, 32)
                    amax = np.maximum(np.abs(blocks).max(-1, keepdims=True), 1e-4)
                    fraction, exponent = np.frexp(amax * np.float32(1 / 448))
                    scales = np.ldexp(np.ones_like(amax), exponent - (fraction == 0.5))
                    encoded = mx.to_fp8(mx.array(np.clip(blocks / scales, -448, 448)))
                    expected = (
                        (mx.from_fp8(encoded, mx.float32) * mx.array(scales))
                        .reshape(x.shape)
                        .astype(dtype)
                    )
                    actual = fake_quant_fp8_ue8m0(x)
                    self.assertTrue(mx.array_equal(actual, expected).item())

    def test_config_accepts_unused_prediction_layers(self):
        from mlx_vlm.models import deepseek_v41

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        source = copy.deepcopy(case["config"])
        source["compress_ratios"] += [0, 0, 0]
        source["num_nextn_predict_layers"] = 3
        config = deepseek_v41.ModelConfig.from_dict(source)
        self.assertEqual(config.compress_ratios, case["config"]["compress_ratios"])

    def test_head_and_router_load_bf16_checkpoint_weights_as_fp32(self):
        from mlx_vlm.models import deepseek_v41

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        model = deepseek_v41.Model(build_config(deepseek_v41, case["config"]))
        head = model.language_model.head
        gate = model.layers[0].ffn.gate
        weight = mx.random.normal(head.weight.shape).astype(mx.bfloat16)
        gate_weight = mx.random.normal(gate.weight.shape).astype(mx.bfloat16)
        x = mx.random.normal((1, 1, model.config.hidden_size)).astype(mx.bfloat16)
        gate.weight = gate_weight
        expected_indices, expected_weights = gate(x)
        mx.eval(expected_indices, expected_weights)
        sanitized = model.sanitize(
            {"head.weight": weight, "layers.0.ffn.gate.weight": gate_weight}
        )
        model.load_weights(list(sanitized.items()), strict=False)
        self.assertEqual(head.weight.dtype, mx.float32)
        self.assertEqual(gate.weight.dtype, mx.float32)
        expected = x.astype(mx.float32) @ weight.T
        self.assertTrue(mx.array_equal(head(x), expected).item())
        indices, weights = gate(x)
        self.assertTrue(mx.array_equal(indices, expected_indices).item())
        self.assertTrue(mx.array_equal(weights, expected_weights).item())

    def test_sparse_attention_preserves_precision_and_masks(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41Attention

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        attention = DeepseekV41Attention(config, 1)
        for dtype in (mx.float32, mx.bfloat16):
            for length in (1, 3):
                with self.subTest(dtype=dtype, length=length):
                    q = mx.random.normal(
                        (2, config.num_attention_heads, length, config.head_dim)
                    ).astype(dtype)
                    window = mx.random.normal((2, 7, config.head_dim)).astype(dtype)
                    pool = mx.random.normal((2, 9, config.head_dim)).astype(dtype)
                    indices = mx.broadcast_to(
                        mx.array([[[2, -1, 0, 7]]]), (2, length, 4)
                    )
                    mask = (
                        mx.arange(7)[None, None, None]
                        <= mx.arange(length)[None, None, :, None] + 3
                    )
                    attention.attn_sink = mx.random.normal(
                        (config.num_attention_heads,)
                    )
                    actual = attention._sparse_attention(q, window, pool, indices, mask)
                    expected = np.zeros(q.shape, dtype=np.float32)
                    query, local, pooled = [
                        np.array(a.astype(mx.float32)) for a in (q, window, pool)
                    ]
                    sinks = np.array(attention.attn_sink.astype(mx.float32))
                    # Select only valid rows in an independent dense FP32 oracle.
                    for batch in range(2):
                        for pos in range(length):
                            kv = np.concatenate(
                                [local[batch, : pos + 4], pooled[batch, [2, 0, 7]]]
                            )
                            scores = (query[batch, :, pos] @ kv.T) * attention.scale
                            maximum = np.maximum(scores.max(-1), sinks)[:, None]
                            probabilities = np.exp(scores - maximum)
                            denominator = probabilities.sum(-1, keepdims=True) + np.exp(
                                sinks[:, None] - maximum
                            )
                            expected[batch, :, pos] = (probabilities / denominator) @ kv
                    # BF16 output rounds the FP32 result to its nearest value.
                    tolerance = 1e-5 if dtype == mx.float32 else 2**-8
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(
                        mx.allclose(
                            actual.astype(mx.float32),
                            mx.array(expected),
                            atol=1e-5,
                            rtol=tolerance,
                        ).item()
                    )

    def test_engram_offloading_matches_resident_rows(self):
        import tempfile

        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.engram import (
            OffloadedEngramEmbedding,
            QuantizedEngramEmbedding,
        )

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        prefix = "language_model.layers.1.engram.embed."
        ids = mx.array([[0, 127, 3], [3, 1, 0]])
        for bits, mode, sharded in (
            (None, "affine", False),
            (4, "affine", False),
            (8, "affine", True),
            (4, "mxfp4", False),
            (8, "mxfp8", True),
        ):
            with (
                self.subTest(bits=bits, mode=mode, sharded=sharded),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                config = copy.deepcopy(case["config"])
                model = deepseek_v41.Model(build_config(deepseek_v41, config))
                table = model.layers[1].engram.embed
                table.weight = table.weight.astype(mx.bfloat16)
                if bits is not None:
                    config["quantization"] = dict(group_size=32, bits=bits, mode=mode)
                    packed = QuantizedEngramEmbedding(128, 64, **config["quantization"])
                    packed.weight, packed.scales, *biases = mx.quantize(
                        table.weight, **config["quantization"]
                    )
                    packed.biases = biases[0] if biases else None
                    model.layers[1].engram.embed = packed
                    config["quantization"][prefix.rstrip(".")] = dict(
                        group_size=32, bits=bits, mode=mode
                    )
                expected = model.layers[1].engram.embed(ids)
                mx.eval(expected)
                self.assertEqual(expected.dtype, mx.bfloat16)
                weights = dict(tree_flatten(model.parameters()))
                if mode.startswith("mxfp"):
                    native_prefix = prefix.removeprefix("language_model.")
                    weights[native_prefix + "weight"] = weights.pop(
                        prefix + "weight"
                    ).view(mx.uint8)
                    weights[native_prefix + "scale"] = weights.pop(prefix + "scales")
                (path / "config.json").write_text(json.dumps(config))
                if sharded:
                    tables = {
                        key: weights.pop(key)
                        for key in list(weights)
                        if ".engram.embed." in key
                    }
                    mx.save_safetensors(str(path / "engram.safetensors"), tables)
                    (path / "model.safetensors.index.json").write_text(
                        json.dumps(
                            {
                                "weight_map": {
                                    **dict.fromkeys(weights, "model.safetensors"),
                                    **dict.fromkeys(tables, "engram.safetensors"),
                                }
                            }
                        )
                    )
                mx.save_safetensors(str(path / "model.safetensors"), weights)

                with patch("mlx_vlm.utils.mx.eval", wraps=mx.eval) as evaluate:
                    loaded = load_model(path)
                # Exportable tensors must be exposed after eager loading.
                self.assertNotIn(
                    prefix + "weight", dict(tree_flatten(evaluate.call_args.args[0]))
                )
                self.assertEqual(loaded.config.model_path, str(path))
                self.assertIsInstance(
                    loaded.layers[1].engram.embed, OffloadedEngramEmbedding
                )
                self.assertEqual(loaded.layers[1].engram.embed(ids).dtype, mx.bfloat16)
                self.assertTrue(
                    mx.array_equal(loaded.layers[1].engram.embed(ids), expected).item()
                )
                self.assertIn(
                    prefix + "weight", dict(tree_flatten(loaded.parameters()))
                )

                # Conversion uses lazy loading and must keep exportable table weights.
                lazy = load_model(path, lazy=True)
                self.assertEqual(lazy.config.model_path, str(path))
                self.assertIn(prefix + "weight", dict(tree_flatten(lazy.parameters())))
                self.assertTrue(
                    mx.array_equal(lazy.layers[1].engram.embed(ids), expected).item()
                )
                with patch.object(QuantizedEngramEmbedding, "_quantize_chunk_rows", 41):
                    quantized = lazy.layers[1].engram.embed.to_quantized(
                        group_size=32, bits=4
                    )
                resident = model.layers[1].engram.embed.to_quantized(
                    group_size=32, bits=4
                )
                for name in ("weight", "scales", "biases"):
                    self.assertTrue(
                        mx.array_equal(quantized[name], resident[name]).item()
                    )
                self.assertTrue(mx.array_equal(quantized(ids), resident(ids)).item())

                exported = str(path / "export.safetensors")
                lazy.save_weights(exported)
                self.assertIn(prefix + "weight", mx.load(exported))

    def test_fp8_scale_layouts_decode_exactly(self):
        from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight

        raw = mx.full((64, 64), 56, dtype=mx.uint8)  # E4M3 encoding of 1.0.
        for rowwise in (False, True):
            with self.subTest(rowwise=rowwise):
                scale_rows = 64 if rowwise else 2
                scales = (
                    mx.arange(scale_rows * 2).reshape(scale_rows, 2) % 4 + 125
                ).astype(mx.uint8)
                packed, expanded, mode = _pack_source_weight(raw, scales)
                decoded = mx.dequantize(
                    packed, expanded, group_size=32, bits=8, mode=mode
                )
                # Keep the reference exact, independent of GPU pow rounding.
                expected = mx.array([0.25, 0.5, 1.0, 2.0])[
                    scales.astype(mx.int32) - 125
                ]
                expected = mx.repeat(expected, 32, axis=-1)
                if not rowwise:
                    expected = mx.repeat(expected, 32, axis=0)
                self.assertTrue(mx.array_equal(decoded, expected).item())

    def test_engram_chunked_requantization(self):
        from mlx_vlm.models.deepseek_v41.engram import QuantizedEngramEmbedding

        source = QuantizedEngramEmbedding(
            7, 256, group_size=32, bits=8, mode="mxfp8", scale_dtype=mx.uint8
        )
        source.weight, source.scales = mx.quantize(
            mx.random.normal((7, 256)).astype(mx.bfloat16),
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        expected = mx.quantize(
            mx.dequantize(
                source.weight, source.scales, group_size=32, bits=8, mode="mxfp8"
            ),
            group_size=64,
            bits=4,
        )
        with patch.object(QuantizedEngramEmbedding, "_quantize_chunk_rows", 3):
            converted = source.to_quantized(group_size=64, bits=4)
        for actual, reference in zip(
            (converted.weight, converted.scales, converted.biases), expected
        ):
            self.assertTrue(mx.array_equal(actual, reference).item())
        self.assertIs(converted.to_quantized(group_size=64, bits=4), converted)

    def test_indexer_uses_unrotated_latents_and_adjacent_pair_rope(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41 import language
        from mlx_vlm.models.deepseek_v41.fakequant import (
            fake_quant_fp4_e4m3,
            fake_quant_fp4_ue8m0,
            fake_quant_fp8_ue8m0,
        )

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])

        def rotate(value, positions):
            # DeepSeek's reference apply_rotary_emb views adjacent pairs as complex.
            dtype = value.dtype
            value = np.array(value.astype(mx.float32))
            rd = config.qk_rope_head_dim
            frequencies = config.compress_rope_theta ** (
                -np.arange(0, rd, 2, dtype=np.float32) / rd
            )
            angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies
            phases = np.exp(1j * angles).astype(np.complex64)
            phases = phases.reshape(1, len(positions), *((1,) * (value.ndim - 3)), -1)
            pairs = np.ascontiguousarray(value[..., -rd:]).view(np.complex64)
            rotated = np.ascontiguousarray(pairs * phases).view(np.float32)
            return mx.array(
                np.concatenate([value[..., :-rd], rotated], axis=-1)
            ).astype(dtype)

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype):
                attn = language.DeepseekV41Attention(config, 1)
                attn.update(tree_map(lambda p: p.astype(dtype), attn.parameters()))
                # Ratio-2 pooling projections remain FP32 in the reference.
                attn.compressor.wkv.weight = attn.compressor.wkv.weight.astype(
                    mx.float32
                )
                attn.compressor.wgate.weight = attn.compressor.wgate.weight.astype(
                    mx.float32
                )
                x = mx.random.normal((2, 16, config.hidden_size)).astype(dtype)
                qr = attn.q_norm(attn.wq_a(fake_quant_fp8_ue8m0(x)))
                cache = language.DeepseekV41Cache(config.num_hidden_layers)
                latent = attn.compressor(x, 0, cache)
                positions = range(0, x.shape[1], attn.compress_ratio)
                expected_keys = fake_quant_fp4_ue8m0(
                    rotate(attn.indexer.k_norm(attn.indexer.wk(latent)), positions)
                )
                q = attn.indexer.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(
                    *x.shape[:2], config.index_n_heads, config.index_head_dim
                )
                expected_queries = fake_quant_fp4_ue8m0(rotate(q, range(x.shape[1])))
                expected_pool = fake_quant_fp4_e4m3(rotate(latent, positions))
                with patch.object(
                    language, "_index_scores", wraps=language._index_scores
                ) as score:
                    pool, indices = attn._compress_part(x, qr, 0, cache)
                for actual, expected in (
                    (cache.index_k, expected_keys),
                    (score.call_args.args[0], expected_queries),
                    (pool, expected_pool),
                ):
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(mx.allclose(actual, expected, atol=1e-6).item())

                # Match the reference's BF16 rounding after the dot product,
                # head weighting, and head reduction, including non-tied top-k.
                weights = attn.indexer.weights_proj(x) * (
                    config.index_head_dim**-0.5 * config.index_n_heads**-0.5
                )
                dots = mx.einsum(
                    "bshd,btd->bsht",
                    expected_queries.astype(mx.float32),
                    expected_keys.astype(mx.float32),
                ).astype(dtype)
                weighted = (
                    mx.maximum(dots, 0).astype(mx.float32)
                    * weights.astype(mx.float32)[..., None]
                ).astype(dtype)
                expected_scores = weighted.astype(mx.float32).sum(2).astype(dtype)
                lengths = (mx.arange(1, x.shape[1] + 1) // attn.compress_ratio)[:, None]
                expected_scores = mx.where(
                    mx.arange(expected_keys.shape[1]) < lengths,
                    expected_scores,
                    -mx.inf,
                )
                scores = language._index_scores(*score.call_args.args)
                self.assertEqual(scores.dtype, dtype)
                self.assertTrue(mx.array_equal(scores, expected_scores).item())
                selected = mx.take_along_axis(
                    expected_scores, mx.maximum(indices, 0), -1
                )
                selected = mx.where(indices < 0, -mx.inf, selected)
                best = mx.sort(expected_scores, axis=-1)[..., -indices.shape[-1] :]
                self.assertTrue(mx.array_equal(mx.sort(selected), best).item())

    def test_moe_matches_reference_activation_and_routing_precision(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41MoE

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        # MLX's sorted affine GEMM can read past a partial 64-element K tile.
        # Use aligned projections while retaining decode and sorted dispatch.
        config.hidden_size = config.moe_intermediate_size = 64

        def project(layer, x, expert=None):
            # Use MLX's native MXFP8 codec as an independent rounding oracle.
            q, scales = mx.quantize(x, group_size=32, bits=8, mode="mxfp8")
            x = mx.dequantize(
                q, scales, group_size=32, bits=8, mode="mxfp8", dtype=x.dtype
            )
            weight = layer.weight if expert is None else layer.weight[expert]
            if hasattr(layer, "scales"):
                # Match MLX's decode/prefill GEMM rounding, keeping the expert
                # loop, activation codec and routing order independent.
                if expert is not None and x.shape[1] == 1:
                    return mx.gather_qmm(
                        x[..., None, :],
                        layer.weight,
                        layer.scales,
                        layer.biases,
                        rhs_indices=mx.full(x.shape[:-1], expert, dtype=mx.int32),
                        transpose=True,
                        group_size=layer.group_size,
                        bits=layer.bits,
                    ).squeeze(-2)
                return mx.quantized_matmul(
                    x,
                    weight,
                    layer.scales if expert is None else layer.scales[expert],
                    layer.biases if expert is None else layer.biases[expert],
                    transpose=True,
                    group_size=layer.group_size,
                    bits=layer.bits,
                )
            return x @ weight.T

        def expert(module, x, weight=None, index=None):
            gate = project(module.gate_proj, x, index).astype(mx.float32)
            up = project(module.up_proj, x, index).astype(mx.float32)
            gate = mx.minimum(gate, config.swiglu_limit)
            up = mx.clip(up, -config.swiglu_limit, config.swiglu_limit)
            activated = (gate / (1 + mx.exp(-gate))) * up
            if weight is not None:
                activated = activated * weight[..., None]
            return project(module.down_proj, activated.astype(x.dtype), index)

        for bits in (None, 4, 8):
            for length in (1, 17):  # Unsorted decode and sorted expert dispatch.
                with self.subTest(bits=bits, length=length):
                    model = DeepseekV41MoE(config)
                    model.gate.weight = mx.random.normal(model.gate.weight.shape) * 0.1
                    model.update(
                        tree_map(lambda p: p.astype(mx.bfloat16), model.parameters())
                    )
                    if bits is not None:
                        nn.quantize(
                            model,
                            group_size=32,
                            bits=bits,
                            class_predicate=lambda path, module: hasattr(
                                module, "to_quantized"
                            )
                            and path.startswith(("switch_mlp.", "shared_experts.")),
                        )
                    x = (mx.random.normal((2, length, config.hidden_size)) * 8).astype(
                        mx.bfloat16
                    )
                    indices, weights = model.gate(x)
                    expected = mx.zeros(x.shape, dtype=mx.float32)
                    for index in range(config.n_routed_experts):
                        weight = mx.where(indices == index, weights, 0).sum(-1)
                        expected = expected + expert(
                            model.switch_mlp, x, weight, index
                        ).astype(mx.float32)
                    expected = (
                        expected + expert(model.shared_experts, x).astype(mx.float32)
                    ).astype(x.dtype)
                    actual = model(x)
                    self.assertEqual(actual.dtype, x.dtype)
                    self.assertTrue(
                        mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
                    )

    def test_image_tokens_use_visual_routing_and_break_engram_history(self):
        """The generation path supplies token ids, including during chunked prefill.

        Match the reference's explicit image mask and masked n-gram hashes,
        even when a prefill boundary falls inside an image span.
        """
        from mlx_vlm.generate.common import _chunked_prefill_enabled
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.engram import NgramHashState
        from mlx_vlm.models.deepseek_v41.language import LanguageModel

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        model = LanguageModel(config)
        self.assertTrue(_chunked_prefill_enabled(model))
        token_map = [i % 7 for i in range(config.vocab_size)]
        token_map[0] = 6
        model.engram_hash = NgramHashState(config, model.layout, token_map=token_map)
        model.head.weight = mx.random.normal(model.head.weight.shape) * 0.05
        for layer in model.layers:
            layer.ffn.gate.bias = mx.array([10.0, 9.0, 0.0, 0.0])
            layer.ffn.gate.bias_vl = mx.array([0.0, 0.0, 10.0, 9.0])

        ids = mx.array([[3, 7, 5, 5, 5, 11, 15, 19]])
        image_mask = ids == config.image_token_id
        embeds = model.embed_tokens(ids)
        embeds = mx.where(image_mask[..., None], mx.random.normal(embeds.shape), embeds)
        reference_cache = model.make_cache()
        hashes = model.engram_hash(ids, 0, reference_cache[0], token_mask=~image_mask)
        self.assertLess(mx.max(hashes).item(), config.engram_num_embeddings[0])
        output = model(
            ids,
            inputs_embeds=embeds,
            cache=reference_cache,
            image_mask=image_mask,
            engram_hashes=hashes,
        )
        self.assertIsNone(output.hidden_states)
        expected = output.logits
        comparisons = []
        for chunks in ((8,), (3, 2, 3), (1,) * 8):
            cache = model.make_cache()
            outputs, start = [], 0
            for length in chunks:
                stop = start + length
                outputs.append(
                    model(
                        inputs=ids[:, start:stop],
                        inputs_embeds=embeds[:, start:stop],
                        cache=cache,
                        n_to_process=length,
                    ).logits
                )
                start = stop
            comparisons.append(
                (
                    chunks,
                    (
                        mx.array_equal(cache[0].engram, reference_cache[0].engram),
                        mx.all(cache[0].engram[:, 2:5] == -1),
                        mx.allclose(
                            mx.concatenate(outputs, axis=1), expected, atol=1e-4
                        ),
                    ),
                )
            )
        mx.eval([checks for _, checks in comparisons])
        for chunks, checks in comparisons:
            with self.subTest(chunks=chunks):
                for check in checks:
                    self.assertTrue(check.item())


class TestMoERouterStopGradient:
    def test_gather_indices_are_stop_gradiented(self):
        import re

        import mlx_vlm

        offenders = []
        for path in sorted((Path(mlx_vlm.__file__).parent / "models").rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            if "take_along_axis" not in src or "argpartition" not in src:
                continue
            from_argsort = set(
                re.findall(r"(\w+)\s*=\s*[^\n]*arg(?:partition|sort)", src)
            )
            for m in re.finditer(r"(\w+)\s*=\s*(\w+)\[", src):
                if m.group(2) in from_argsort:
                    from_argsort.add(m.group(1))
            stopped = set(re.findall(r"(\w+)\s*=\s*mx\.stop_gradient", src))
            for m in re.finditer(
                r"take_along_axis\(\s*[^,]+?\s*,\s*(.+?)\s*,\s*axis", src
            ):
                idx = m.group(1).strip()
                if "stop_gradient" in idx:
                    continue
                var = re.match(r"[A-Za-z_]\w*", idx)
                var = var.group(0) if var else ""
                if var and var not in stopped and var in from_argsort:
                    offenders.append(
                        f"{path.parent.name}/{path.name}: take_along_axis(..., {idx})"
                    )
        assert (
            not offenders
        ), "gather indices need mx.stop_gradient (MLX >= 0.32.1):\n" + "\n".join(
            offenders
        )

    @pytest.mark.parametrize(
        "module,cls,extra",
        [
            (
                "qwen3_5_moe",
                "Qwen3_5MoeSparseMoeBlock",
                {"shared_expert_intermediate_size": 8, "norm_topk_prob": True},
            ),
            ("qwen3_moe", "Qwen3MoeSparseMoeBlock", {"norm_topk_prob": True}),
        ],
    )
    def test_moe_router_backpropagates_through_dispatch(self, module, cls, extra):
        import importlib
        import types

        block_cls = getattr(
            importlib.import_module(f"mlx_vlm.models.{module}.language"), cls
        )
        args = types.SimpleNamespace(
            hidden_size=16,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            **extra,
        )
        block = block_cls(args)
        x = mx.random.normal((1, 3, 16))
        _, grads = nn.value_and_grad(block, lambda m, inp: m(inp).sum())(block, x)
        total = sum(float(mx.sum(mx.abs(g))) for _, g in tree_flatten(grads))
        assert total > 0 and bool(mx.isfinite(mx.array(total)))


class TestPhiMoE:
    def test_sparsemixer(self):
        from mlx_vlm.models.phimoe.language import sparsemixer

        gates = mx.array([[3.0, 1.0, 0.0, -1.0], [3.0, 2.99, 0.0, -1.0]])
        inds, scores = sparsemixer(gates, top_k=2, jitter_eps=0.01)

        assert inds.tolist() == [[0, 1], [0, 1]]
        assert mx.allclose(scores, mx.array([[1.0, 1.0], [0.5025, 1.0]]))
        mx.eval(
            mx.grad(lambda g: sparsemixer(g, top_k=2, jitter_eps=0.01)[1].sum())(gates)
        )

    def test_plain_rope_without_scaling(self):
        from mlx_vlm.models.phimoe.config import ModelConfig
        from mlx_vlm.models.phimoe.language import Attention

        config = ModelConfig(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            rope_scaling=None,
        )

        assert isinstance(Attention(config).rope, nn.RoPE)


class TestUnlimitedOCR2Dash:
    """Unlimited-OCR-2 DASH attention: gathered decode, cached summaries, pages."""

    PAGE, IMAGE = 7, 9

    def params(self, **overrides):
        from mlx_vlm.models.unlimited_ocr.dash import DashParams

        sparse = {
            "training_stage": "dash",
            "use_sparse_attention": True,
            "dash_stage0": "head_cls",
            "sparse_block_size": 4,
            "sparse_local_block": 2,
            "sparse_index_dim": 8,
            "dash_topk_blocks": 2,
            "image_token_id": self.IMAGE,
            "page_token_id": self.PAGE,
            "per_page_image_tokens": 6,
            "sparse_main_rope_mode": "page2d",
            "sparse_rope_page_pairs": 1,
            "sparse_main_rope_tok_freq_mode": "resample",
            "sparse_rope_page_mod": 2,
            **overrides,
        }
        return sparse, DashParams.from_config(sparse, 2, 8)

    def language_model(self):
        from mlx_vlm.models.unlimited_ocr.config import TextConfig
        from mlx_vlm.models.unlimited_ocr.language import LanguageModel

        sparse, _ = self.params()
        config = TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            sparse_attention_config=sparse,
        )
        mx.random.seed(3)
        model = LanguageModel(config)
        model.update(
            tree_map(lambda w: mx.random.normal(w.shape) * 0.5, model.parameters())
        )
        return model

    def prompt(self):
        page = [self.PAGE] + [self.IMAGE] * 6
        return [1, 2, 3] + page * 3 + [4, 5, 6, 2, 3]

    def test_decode_gather_matches_dense_mask(self):
        from mlx_vlm.models.unlimited_ocr.config import TextConfig
        from mlx_vlm.models.unlimited_ocr.dash import (
            DashAttention,
            _decode_select,
            block_capacity,
            block_scores,
            route_select,
        )

        _, p = self.params(dash_topk_blocks=3, sparse_local_block=1)
        attn = DashAttention(
            TextConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=2),
            0,
            p,
        )
        rng = np.random.default_rng(5)
        for pos in (2, 9, 21, 22):  # no remote block / partial blocks
            nblk = -(-(pos + 1) // p.block_size)
            cap = block_capacity(nblk)
            q = mx.array(rng.normal(size=(1, 2, 1, 8)).astype(np.float32))
            keys = mx.array(rng.normal(size=(1, 2, pos + 1, 8)).astype(np.float32))
            values = mx.array(rng.normal(size=(1, 2, pos + 1, 8)).astype(np.float32))
            idx_q = mx.array(rng.normal(size=(1, 8)).astype(np.float32))
            n_sum = (pos + 1) // p.block_size
            summ = rng.normal(size=(n_sum, 8)).astype(np.float32)
            for allow in (np.ones(nblk, bool), np.arange(nblk) == 1):
                scores = block_scores(
                    idx_q, mx.array(summ) if n_sum else None, mx.array([pos]), nblk, p
                )
                topi, bias, valid = route_select(
                    scores, mx.array(allow[None]), attn.idx_scale, attn.floor, p
                )
                dense = attn._prefill(q, keys, values, topi, bias, valid, pos, nblk)
                padded = np.zeros((cap, 8), np.float32)
                padded[:n_sum] = summ
                cand = np.zeros((1, cap), bool)
                cand[0, :nblk] = allow
                idx, mask = _decode_select(
                    idx_q,
                    mx.array(padded),
                    mx.array(cand),
                    mx.array(pos),
                    *attn._decode_consts,
                    p,
                )
                gathered = attn._decode(q, keys, values, idx, mask)
                assert np.allclose(np.array(dense), np.array(gathered), atol=1e-5)

    def test_entmax15_is_a_sparse_distribution(self):
        from mlx_vlm.models.unlimited_ocr.dash import entmax15

        x = mx.array([[3.0, 2.9, 0.5, -1e4], [1.0, 1.0, 1.0, 1.0]])
        prior = np.array(entmax15(x))
        assert np.allclose(prior.sum(-1), 1.0, atol=1e-5)
        assert prior[0, 3] == 0 and prior[0, 0] > prior[0, 1] > prior[0, 2]
        assert np.allclose(prior[1], 0.25, atol=1e-5)

    def test_prompt_pages_coordinates_and_candidates(self):
        from mlx_vlm.models.unlimited_ocr.dash import PromptPages

        _, p = self.params()
        ids = np.array(self.prompt())
        pages = PromptPages(ids, p)
        tok, page = pages.coords
        # BOS + instruction bucket is page 0; each <PAGE> opens the next page.
        # The instruction after the last image goes back to that bucket.
        assert page.tolist() == [0] * 3 + [1] * 7 + [2] * 7 + [3] * 7 + [0] * 5
        assert tok[:4].tolist() == [0, 1, 2, 0] and tok[-1] == 7
        assert pages.n_pages == 4 and np.array(pages.pf_cnt).tolist() == [3, 7, 7, 7]
        cand = np.array(pages.prefill_candidates(np.arange(len(ids)), 8))
        # A page-2 query (position 12) only sees blocks holding page-2 images.
        assert cand[12].tolist() == [
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
        ]
        # The answer has no page yet: it falls back to every block.
        assert cand[-1].all()

    def test_chunked_prefill_and_decode_match_full_prefill(self):
        model = self.language_model()
        ids = mx.array([self.prompt()])
        n = ids.shape[1]

        full = model(ids, cache=model.make_cache()).logits
        cache = model.make_cache()
        model._prompt_ids = np.array(ids[0])
        chunks = []
        for c0, c1 in ((0, 9), (9, 22), (22, n)):
            chunks.append(model(ids[:, c0:c1], cache=cache).logits)
        assert np.allclose(
            np.array(mx.concatenate(chunks, axis=1)), np.array(full), atol=1e-4
        )

        # Decode keeps one step's cost bounded and stays finite over block edges.
        token = mx.argmax(chunks[-1][:, -1], axis=-1, keepdims=True)
        for _ in range(9):
            logits = model(token, cache=cache).logits
            assert logits.shape == (1, 1, 32)
            assert np.isfinite(np.array(logits)).all()
            token = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
        assert cache[0].offset == n + 9
        assert cache[0].num_summaries == (n + 9) // 4


class TestUnlimitedOCR2Guards:
    """Unlimited-OCR-2 native generation guards: page-count guard and no-repeat n-gram."""

    PAGE, EOS, VOCAB = 5, 1, 16

    def guard(self, **kwargs):
        from mlx_vlm.models.unlimited_ocr.guards import UnlimitedOCRGuards

        return UnlimitedOCRGuards(self.PAGE, self.EOS, **kwargs)

    def rows(self, guard, prompt, generated):
        """Logits rows seen by the sampler, fed the way ``generate_step`` feeds processors."""
        tokens = mx.array(prompt)
        out = [np.array(guard(tokens, mx.zeros((1, self.VOCAB)))[0])]
        for t in generated:
            tokens = mx.concatenate([tokens, mx.array([t])])
            out.append(np.array(guard(tokens, mx.zeros((1, self.VOCAB)))[0]))
        return out

    @staticmethod
    def native_banned(gen, n, window, whitelist=()):
        """modeling_unlimitedocr._uocr_no_repeat_ngram, verbatim semantics."""
        if n <= 0 or len(gen) < n:
            return set()
        prefix = tuple(gen[-(n - 1) :]) if n > 1 else tuple()
        banned = set()
        for i in range(max(0, len(gen) - window), len(gen) - n + 1):
            ng = tuple(gen[i : i + n])
            if ng[:-1] == prefix:
                banned.add(ng[-1])
        return banned - set(whitelist)

    @staticmethod
    def banned(row):
        return set(np.flatnonzero(np.isneginf(row)).tolist())

    def test_ngram_matches_native_reference(self):
        rng = np.random.default_rng(0)
        for n, window in [(1, 5), (2, 7), (3, 16), (5, 64), (4, 3)]:
            for whitelist in [(), (2,)]:
                g = self.guard(
                    page_count_guard=False,
                    ngram_size=n,
                    ngram_window=window,
                    ngram_whitelist=whitelist,
                )
                # tiny vocab: constant repeats
                gen = [int(t) for t in rng.integers(0, 4, 200)]
                rows = self.rows(g, [9], gen)
                for k, row in enumerate(rows):
                    expected = self.native_banned(gen[:k], n, window, whitelist)
                    assert self.banned(row) == expected, (n, window, k)

    def test_ngram_bans_the_next_token_of_a_loop(self):
        g = self.guard(page_count_guard=False, ngram_size=4, ngram_window=100)
        rows = self.rows(g, [9], [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5])
        assert self.banned(rows[-1]) == {6}
        assert self.banned(rows[3]) == set()  # nothing repeated yet

    def test_page_count_guard(self):
        g = self.guard(max_page_tokens=3, expected_pages=2)
        prompt = [self.PAGE, 9, self.PAGE, 9]  # two pages
        rows = self.rows(g, prompt, [self.PAGE, 7, 7, 7, self.PAGE, 7, 7, 7, 8])
        # EOS is banned until both pages are answered
        assert self.banned(rows[0]) == {self.EOS} == self.banned(rows[1])
        # a page longer than max_page_tokens is cut by forcing the next <PAGE> ...
        assert set(range(self.VOCAB)) - self.banned(rows[4]) == {self.PAGE}
        # ... and the last one by forcing EOS
        assert set(range(self.VOCAB)) - self.banned(rows[8]) == {self.EOS}
        # a third <PAGE> marker means the document is over
        g = self.guard()
        rows = self.rows(g, prompt, [self.PAGE, 7, self.PAGE, 7, self.PAGE])
        assert set(range(self.VOCAB)) - self.banned(rows[-1]) == {self.EOS}
        assert self.banned(rows[3]) == set()  # both pages answered: EOS allowed again

    def test_forced_token_survives_ngram_ban(self):
        g = self.guard(max_page_tokens=2, ngram_size=1, ngram_window=100)
        rows = self.rows(g, [self.PAGE, self.PAGE], [self.PAGE, 7, 8])
        assert set(range(self.VOCAB)) - self.banned(rows[-1]) == {self.PAGE}

    def test_guards_never_read_the_host_while_decoding(self):
        """The decode loop must stay asynchronous: no ``.item()`` / ``tolist`` per step."""
        g = self.guard(expected_pages=1, ngram_size=3, ngram_window=16)
        prompt = mx.array([self.PAGE, 9])
        g(prompt, mx.zeros((1, self.VOCAB)))
        tokens = prompt
        with patch.object(mx.array, "item", side_effect=AssertionError("host read")):
            for t in [4, 4, 4, 4]:
                tokens = mx.concatenate([tokens, mx.array([t])])
                g(tokens, mx.zeros((1, self.VOCAB)))

    def test_make_guards_defaults_and_options(self):
        from mlx_vlm.models.unlimited_ocr.guards import make_guards

        pages = np.array([[self.PAGE, 9, self.PAGE, 9]])
        kwargs = {"max_page_tokens": 50, "temperature": 0.0}
        (g,) = make_guards(pages, self.PAGE, [self.EOS], kwargs)
        assert (g.ngram_size, g.ngram_window, g.max_page_tokens) == (35, 1024, 50)
        assert g.eos_token_id == self.EOS and g.page_count_guard
        assert g.expected_pages == 2
        assert kwargs == {"temperature": 0.0}  # only guard options are consumed

        (g,) = make_guards(pages[:, :2], self.PAGE, self.EOS, {})
        assert g.ngram_window == 128  # a single page uses the narrow window
        (g,) = make_guards(pages, self.PAGE, self.EOS, {"no_repeat_ngram_size": 0})
        assert g.ngram_size == 0 and g.page_count_guard
        off = {"page_count_guard": False, "no_repeat_ngram_size": 0}
        assert make_guards(pages, self.PAGE, self.EOS, off) == []  # nothing to apply
        assert make_guards(pages, None, self.EOS, {}) == []  # no <PAGE> protocol (v1)
