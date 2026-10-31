# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

"""The LFM3 decode kernel must reproduce the plain ops bit for bit.

LFM3 decode at batch 1 is dispatch-bound: the q and k head norms and the two
rotary embeddings cost more per layer than the matmuls they sit between, because
``RotaryEmbedding`` is about eight elementwise dispatches. The kernel fuses them,
and the only reason that is safe is that every output is bitwise equal to the
unfused path, which is what these tests assert: ``mx.array_equal`` on the raw
values, not a tolerance.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_map

# Skip before importing the kernels: the modules build their Metal kernels at
# import time, and a Metal-less backend (the Linux CPU CI legs) has nothing to
# compile them against.
_METAL = getattr(mx, "metal", None)
if _METAL is None or not _METAL.is_available():
    pytest.skip("LFM3 fused decode kernels are Metal-only", allow_module_level=True)

from mlx_vlm import kernels
from mlx_vlm.models.lfm3 import language as lfm3
from mlx_vlm.models.lfm3.config import ModelConfig

DTYPES = [mx.float16, mx.bfloat16]

# Every layer type, sandwich norms on, a window short enough that a prefill
# shorter than the window and longer than it both reach the rotating cache.
TINY = {
    "model_type": "lfm3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 128,
    "layer_types": ["conv", "full_attention", "sliding_attention"],
    "sliding_window": 8,
    "conv_L_cache": 4,
    "use_sandwich_norm": True,
    "tie_word_embeddings": False,
}


def _cast(module, dtype):
    """Every parameter in the model dtype. Linear layers already start random;
    the norm weights start at exactly one and get perturbed below so the
    kernels cannot pass by accident on a degenerate scale."""
    module.update(tree_map(lambda a: a.astype(dtype), module.parameters()))
    return module


def _perturb_norms(module, dtype):
    def walk(tree):
        if isinstance(tree, dict):
            for key, value in tree.items():
                if key == "weight" and value.ndim == 1:
                    tree[key] = mx.random.uniform(0.5, 1.5, value.shape).astype(dtype)
                else:
                    walk(value)
        elif isinstance(tree, list):
            for item in tree:
                walk(item)

    walk(module.parameters())
    return module


def _attention(dtype, layer_type="full_attention"):
    attn = _cast(lfm3.Attention(ModelConfig(**TINY), layer_type), dtype)
    for norm in (attn.q_norm, attn.k_norm):
        norm.weight = mx.random.uniform(0.5, 1.5, norm.weight.shape).astype(dtype)
    return attn


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize("batch,length", [(1, 1), (1, 5), (2, 3)])
def test_qk_norm_rope_matches_the_unfused_path(dtype, layer_type, batch, length):
    """q/k head norms and both rotations, fused into one dispatch."""
    attn = _attention(dtype, layer_type)
    n_heads, n_kv_heads, head_dim = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = mx.random.normal((batch, length, n_heads, head_dim)).astype(dtype)
    k = mx.random.normal((batch, length, n_kv_heads, head_dim)).astype(dtype)
    offset = 7

    q_ref = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset)
    k_ref = attn.rope(attn.k_norm(k).transpose(0, 2, 1, 3), offset)

    q_got, k_got = kernels.lfm3_qk_norm_rope(
        q,
        k,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.rope.inv_freq(),
        eps=attn.q_norm.eps,
        offset=offset,
    )
    assert mx.array_equal(q_got, q_ref)
    assert mx.array_equal(k_got, k_ref)


@pytest.mark.parametrize("dtype", DTYPES)
def test_qk_norm_rope_matches_with_per_row_offsets_and_positions(dtype):
    """Batched rows decode at different offsets, and packed prompts pass
    explicit positions; both have to land on the same values."""
    attn = _attention(dtype)
    batch, length = 2, 4
    head_dim, n_heads = attn.head_dim, attn.n_heads
    q = mx.random.normal((batch, length, n_heads, head_dim)).astype(dtype)
    k = mx.random.normal((batch, length, attn.n_kv_heads, head_dim)).astype(dtype)
    offsets = mx.array([5, 11], dtype=mx.int32)
    positions = mx.array([[0, 1, 2, 3], [40, 41, 42, 43]], dtype=mx.int32)

    for off, pos in ((offsets, None), (0, positions)):
        q_ref = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), off, pos)
        k_ref = attn.rope(attn.k_norm(k).transpose(0, 2, 1, 3), off, pos)
        q_got, k_got = kernels.lfm3_qk_norm_rope(
            q,
            k,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rope.inv_freq(),
            eps=attn.q_norm.eps,
            position_ids=pos,
            offset=off,
        )
        assert mx.array_equal(q_got, q_ref)
        assert mx.array_equal(k_got, k_ref)


def _tiny_model(dtype):
    return _perturb_norms(_cast(lfm3.LanguageModel(ModelConfig(**TINY)), dtype), dtype)


def _logits(model, ids, cache):
    return model(mx.array([ids]), cache=cache).logits


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [1, 6, 13])
def test_the_fused_layers_generate_the_same_logits(dtype, length):
    """Whole model, prefill and decode, against the unfused path.

    Lengths 1, 6 and 13 cover a single token, a prompt shorter than the sliding
    window, and one longer than it, so the rotating cache is exercised too.
    """
    ids = [3, 7, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59][:length]
    model = _tiny_model(dtype)

    lfm3.set_fused(False)
    try:
        cache = model.make_cache()
        expected = _logits(model, ids, cache)
        expected_step = _logits(model, [97], cache)
    finally:
        lfm3.set_fused(True)

    cache = model.make_cache()
    got = _logits(model, ids, cache)
    got_step = _logits(model, [97], cache)

    assert mx.array_equal(got, expected)
    assert mx.array_equal(got_step, expected_step)


@pytest.mark.parametrize("dtype", DTYPES)
def test_decode_steps_match_the_unfused_path(dtype):
    """A decode step reads a populated cache, so the offsets must agree too."""
    model = _tiny_model(dtype)
    prompt = [3, 7, 11, 17, 23]

    lfm3.set_fused(False)
    try:
        cache = model.make_cache()
        _logits(model, prompt, cache)
        expected = [_logits(model, [token], cache) for token in (31, 37, 41)]
    finally:
        lfm3.set_fused(True)

    cache = model.make_cache()
    _logits(model, prompt, cache)
    got = [_logits(model, [token], cache) for token in (31, 37, 41)]

    for a, b in zip(got, expected):
        assert mx.array_equal(a, b)


def test_float32_keeps_the_plain_path(monkeypatch):
    """The kernels take fp16/bf16 only; anything else must fall back silently."""
    calls = []
    real = kernels.lfm3_qk_norm_rope

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(lfm3, "lfm3_qk_norm_rope", spy)
    model = _tiny_model(mx.float32)
    _logits(model, [3, 7, 11], model.make_cache())
    assert calls == []


def test_turning_the_fusion_off_keeps_the_plain_path(monkeypatch):
    calls = []
    real = kernels.lfm3_qk_norm_rope

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(lfm3, "lfm3_qk_norm_rope", spy)
    model = _tiny_model(mx.float16)
    lfm3.set_fused(False)
    try:
        _logits(model, [3, 7, 11], model.make_cache())
    finally:
        lfm3.set_fused(True)
    assert calls == []
