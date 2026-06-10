"""Tests for CoPE (Clipped RoPE) soft-clipping in rope_utils.

CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs
(arXiv:2602.05258).
"""

import math

import mlx.core as mx
import pytest

from mlx_vlm.models.rope_utils import (
    MRoPERotaryEmbedding,
    apply_cope_clip,
    compute_inv_freq,
)

# Qwen3.5/3.6-family rope geometry: theta=10M, 64 rotary dims, native 262144.
DIM, BASE, NATIVE = 64, 10_000_000, 262_144


def test_auto_clip_sizing():
    inv_freq = compute_inv_freq(DIM, BASE)
    clipped = apply_cope_clip(inv_freq, original_max_position_embeddings=NATIVE)

    # Components with period 2*pi/inv_freq > 262144: i >= 22, i.e. 10 of 32.
    periods = 2 * math.pi / inv_freq
    expected_n = int((periods > NATIVE).sum())
    assert expected_n == 10

    # Unclipped head is bit-identical; boundary component preserved (mask=1).
    assert mx.allclose(clipped[:22], inv_freq[:22])
    assert mx.allclose(clipped[22], inv_freq[22], rtol=1e-5)
    # Lowest-frequency component is frozen entirely.
    assert clipped[-1] == 0.0
    # Taper is monotone non-increasing across the clipped range.
    tail = clipped[22:]
    assert mx.all(tail[:-1] >= tail[1:])


def test_explicit_clip_n_and_noop():
    inv_freq = compute_inv_freq(DIM, BASE)

    clipped = apply_cope_clip(inv_freq, clip_n=4)
    assert mx.allclose(clipped[:28], inv_freq[:28])
    assert clipped[-1] == 0.0

    # clip_n=0 is a no-op.
    assert mx.allclose(apply_cope_clip(inv_freq, clip_n=0), inv_freq)

    # clip_n is bounded by the number of components.
    clipped = apply_cope_clip(inv_freq, clip_n=999)
    assert clipped.shape == inv_freq.shape


def test_requires_sizing_information():
    inv_freq = compute_inv_freq(DIM, BASE)
    with pytest.raises(ValueError):
        apply_cope_clip(inv_freq)


def test_mrope_embedding_integration():
    rope = MRoPERotaryEmbedding(
        DIM,
        max_position_embeddings=NATIVE,
        base=BASE,
        rope_parameters={
            "rope_type": "cope",
            "original_max_position_embeddings": NATIVE,
            "mrope_section": [11, 11, 10],
        },
    )
    raw = compute_inv_freq(DIM, BASE)
    assert rope.inv_freq[-1] == 0.0
    assert mx.allclose(rope.inv_freq[:22], raw[:22])

    # Without rope_type=cope the frequencies are untouched.
    rope = MRoPERotaryEmbedding(
        DIM,
        max_position_embeddings=NATIVE,
        base=BASE,
        rope_parameters={"mrope_section": [11, 11, 10]},
    )
    assert mx.allclose(rope.inv_freq, raw)


def test_qwen3_5_rotary_embedding_passthrough():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

    rope = Qwen3_5RotaryEmbedding(
        DIM,
        max_position_embeddings=NATIVE,
        base=BASE,
        mrope_section=[11, 11, 10],
        rope_parameters={
            "rope_type": "cope",
            "original_max_position_embeddings": NATIVE,
            "rope_theta": BASE,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    )
    assert rope.inv_freq[-1] == 0.0
    assert rope.inv_freq.shape[0] == DIM // 2
