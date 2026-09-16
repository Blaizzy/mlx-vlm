"""Adversarial geometry tests for multi-row quantized SDPA masks (#1567).

Tries to break align_attention_mask_to_scores / quant SDPA with shapes that
show up (or could show up) in continuous batching: GQA/MQA/MHA, decode vs
prefill, chunked offset, windowed causal, left+right pad, additive masks,
and awkward batch sizes.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.models.base import quantized_scaled_dot_product_attention
from mlx_vlm.models.cache import BatchQuantizedKVCache, create_causal_mask

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
        queries, q_keys, q_values, scale=D**-0.5, mask=mask, group_size=GROUP, bits=BITS
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
        causal, mx.array(0.0, dtype=mx.float16), mx.array(-1e4, dtype=mx.float16)
    )
    _run_sdpa(B, n_q, n_kv, L, L, mask)


# ---------------------------------------------------------------------------
# align helper contract
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cache make_mask → quant SDPA (integration of the two layers)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# prepare/finalize stress
# ---------------------------------------------------------------------------


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
