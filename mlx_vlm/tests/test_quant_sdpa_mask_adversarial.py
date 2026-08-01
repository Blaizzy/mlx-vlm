"""Adversarial geometry tests for multi-row quantized SDPA masks (#1567).

Tries to break align_attention_mask_to_scores / quant SDPA with shapes that
show up (or could show up) in continuous batching: GQA/MQA/MHA, decode vs
prefill, chunked offset, windowed causal, left+right pad, additive masks,
and awkward batch sizes.
"""

from __future__ import annotations

import itertools

import mlx.core as mx
import pytest

from mlx_vlm.models.base import (
    align_attention_mask_to_scores,
    quantized_scaled_dot_product_attention,
)
from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache, create_causal_mask

GROUP = 64
BITS = 8


def _quant_kv(B, n_kv, L, D, dtype=mx.float16):
    keys = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    values = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    return (
        mx.quantize(keys, group_size=GROUP, bits=BITS),
        mx.quantize(values, group_size=GROUP, bits=BITS),
    )


def _run_sdpa(B, n_q, n_kv, L, K_cache, mask, D=GROUP):
    """K_cache is total key length (offset + L); keys filled to K_cache."""
    queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
    q_keys, q_values = _quant_kv(B, n_kv, K_cache, D)
    out = quantized_scaled_dot_product_attention(
        queries,
        q_keys,
        q_values,
        scale=D**-0.5,
        mask=mask,
        group_size=GROUP,
        bits=BITS,
    )
    mx.eval(out)
    assert out.shape == (B, n_q, L, D)
    assert mx.isfinite(out).all()
    return out


# ---------------------------------------------------------------------------
# Parametric score/mask geometries
# ---------------------------------------------------------------------------

# (B, n_q, n_kv) layouts seen or plausible in MLX VLMs
HEAD_LAYOUTS = [
    (2, 16, 8),  # Qwen3-0.6B-like GQA (server repro family)
    (2, 32, 8),  # stronger GQA
    (2, 16, 2),  # wider repeat
    (2, 16, 1),  # MQA
    (2, 8, 8),  # MHA n_repeats=1
    (3, 16, 8),  # odd batch
    (4, 16, 8),  # larger batch
    (8, 16, 8),  # B == n_kv (latent mis-align case pre-fix)
    (1, 16, 8),  # single row control
]


@pytest.mark.parametrize("B,n_q,n_kv", HEAD_LAYOUTS)
@pytest.mark.parametrize(
    "L,offset",
    [
        (1, 0),  # pure decode, empty-ish cache
        (1, 128),  # decode after long prefill
        (4, 0),  # short prefill
        (18, 0),  # server curl prompt length class
        (64, 0),  # medium prefill
        (32, 96),  # chunked prefill: L != K
        (128, 512),  # long context chunk
    ],
)
def test_left_pad_mask_all_head_and_length_combos(B, n_q, n_kv, L, offset):
    K = offset + L
    # Varied per-row left pad (capped so pad < K)
    pads = [(i * 3) % max(K, 1) for i in range(B)]
    if K > 1:
        pads = [min(p, K - 1) for p in pads]
    mask = create_causal_mask(L, offset=offset, left_padding=mx.array(pads))
    _run_sdpa(B, n_q, n_kv, L, K, mask)


@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (3, 16, 8), (2, 16, 1)])
def test_left_and_right_padding_together(B, n_q, n_kv):
    L, offset = 16, 0
    left = mx.array([2, 0] + [0] * (B - 2))
    right = mx.array([0, 3] + [0] * (B - 2))
    mask = create_causal_mask(
        L, offset=offset, left_padding=left[:B], right_padding=right[:B]
    )
    _run_sdpa(B, n_q, n_kv, L, offset + L, mask)


@pytest.mark.parametrize("window", [4, 8, 32])
@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (2, 8, 8)])
def test_sliding_window_causal_with_left_pad(window, B, n_q, n_kv):
    L, offset = 24, 40
    pads = mx.array([1, 0] if B == 2 else [1, 0] + [0] * (B - 2))
    mask = create_causal_mask(
        L, offset=offset, window_size=window, left_padding=pads[:B]
    )
    _run_sdpa(B, n_q, n_kv, L, offset + L, mask)


@pytest.mark.parametrize("B,n_q,n_kv", [(2, 16, 8), (4, 32, 8)])
def test_additive_float_mask(B, n_q, n_kv):
    L = 12
    causal = create_causal_mask(L, left_padding=mx.array([i % 2 for i in range(B)]))
    mask = mx.where(
        causal,
        mx.array(0.0, dtype=mx.float16),
        mx.array(-1e4, dtype=mx.float16),
    )
    _run_sdpa(B, n_q, n_kv, L, L, mask)


def test_string_causal_mask_gqa_batch():
    B, n_q, n_kv, L = 2, 16, 8, 8
    _run_sdpa(B, n_q, n_kv, L, L, mask="causal")


def test_none_mask_gqa_batch():
    B, n_q, n_kv, L = 2, 16, 8, 8
    _run_sdpa(B, n_q, n_kv, L, L, mask=None)


# ---------------------------------------------------------------------------
# align helper contract
# ---------------------------------------------------------------------------


def test_aligned_standard_batch_mask_is_b_1_1_l_k():
    scores = mx.zeros((2, 8, 2, 18, 18))
    mask = create_causal_mask(18, left_padding=mx.array([1, 0]))
    aligned = align_attention_mask_to_scores(mask, scores)
    assert aligned.shape == (2, 1, 1, 18, 18)


def test_aligned_decode_mask_shape():
    scores = mx.zeros((2, 8, 2, 1, 128))
    mask = create_causal_mask(1, offset=127, left_padding=mx.array([0, 5]))
    aligned = align_attention_mask_to_scores(mask, scores)
    assert aligned.shape[0] == 2
    assert aligned.shape[-2:] == (1, 128)
    mx.eval(mx.where(aligned, scores, mx.array(0.0)))


def test_2d_window_mask_to_5d():
    scores = mx.zeros((2, 4, 4, 16, 16))
    mask = create_causal_mask(16, window_size=8)  # 2D
    aligned = align_attention_mask_to_scores(mask, scores)
    mx.eval(mx.where(aligned, scores, mx.array(0.0)))


# ---------------------------------------------------------------------------
# Cache make_mask → quant SDPA (integration of the two layers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("past", [0, 8, 64])
@pytest.mark.parametrize("N", [1, 8, 32])
@pytest.mark.parametrize("left_padding", [[0, 0], [3, 0], [0, 5, 1], [7, 2, 0, 4]])
def test_batch_quantized_make_mask_into_sdpa(past, N, left_padding):
    """Mirror real attention order: make_mask(N) at current offset, then update N tokens."""
    B = len(left_padding)
    n_q, n_kv = 16, 8
    cache = BatchQuantizedKVCache(left_padding, group_size=GROUP, bits=BITS)
    if past > 0:
        k0 = mx.random.normal((B, n_kv, past, GROUP)).astype(mx.float16)
        v0 = mx.random.normal((B, n_kv, past, GROUP)).astype(mx.float16)
        cache.update_and_fetch(k0, v0)

    # Same order as Qwen3 Attention: mask from pre-update offset, then append K/V.
    mask = cache.make_mask(N)
    k = mx.random.normal((B, n_kv, N, GROUP)).astype(mx.float16)
    v = mx.random.normal((B, n_kv, N, GROUP)).astype(mx.float16)
    q_keys, q_values = cache.update_and_fetch(k, v)
    queries = mx.random.normal((B, n_q, N, GROUP)).astype(mx.float16)
    out = quantized_scaled_dot_product_attention(
        queries,
        q_keys,
        q_values,
        scale=GROUP**-0.5,
        mask=mask,
        group_size=GROUP,
        bits=BITS,
    )
    mx.eval(out)
    assert out.shape == (B, n_q, N, GROUP)


# ---------------------------------------------------------------------------
# prepare/finalize stress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "right_padding",
    [
        [1, 0],
        [0, 5],
        [3, 3],
        [7, 0, 2],
        [0, 0, 0, 9],
    ],
)
def test_prepare_finalize_extreme_right_pad(right_padding):
    B = len(right_padding)
    quant = BatchQuantizedKVCache([0] * B, group_size=GROUP, bits=BITS)
    ref = BatchKVCache([0] * B)
    quant.prepare(right_padding=right_padding)
    ref.prepare(right_padding=right_padding)
    k = mx.random.normal((B, 4, 16, GROUP))
    v = mx.random.normal((B, 4, 16, GROUP))
    quant.update_and_fetch(k, v)
    ref.update_and_fetch(k, v)
    quant.finalize()
    ref.finalize()
    assert quant.left_padding.tolist() == ref.left_padding.tolist()
    assert quant.offset.tolist() == ref.offset.tolist()


def test_prepare_left_padding_on_empty_only():
    cache = BatchQuantizedKVCache([0, 0], group_size=GROUP, bits=BITS)
    cache.prepare(left_padding=[2, 1])
    assert cache.left_padding.tolist() == [2, 1]
    k = mx.random.normal((2, 2, 4, GROUP))
    v = mx.random.normal((2, 2, 4, GROUP))
    cache.update_and_fetch(k, v)
    with pytest.raises(ValueError, match="empty"):
        cache.prepare(left_padding=[1, 0])


# ---------------------------------------------------------------------------
# Brute force small grid (catch "weird" combos)
# ---------------------------------------------------------------------------


def test_brute_small_grid():
    failures = []
    for B, n_kv, n_rep, L, offset in itertools.product(
        [1, 2, 3, 5],
        [1, 2, 4, 8],
        [1, 2, 4],
        [1, 7, 16],
        [0, 3, 33],
    ):
        n_q = n_kv * n_rep
        K = offset + L
        pads = [(i * 2) % max(1, K) for i in range(B)]
        try:
            mask = create_causal_mask(L, offset=offset, left_padding=mx.array(pads))
            _run_sdpa(B, n_q, n_kv, L, K, mask)
        except Exception as e:
            failures.append(
                f"B={B} n_q={n_q} n_kv={n_kv} L={L} off={offset}: {type(e).__name__}: {e}"
            )
    assert not failures, "Brute grid failures:\n" + "\n".join(failures[:20])
