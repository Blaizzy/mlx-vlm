"""Bidirectional attention-mask helpers for the Gemma 4 MTP drafter.

The drafter attends from L query positions (placed just past the end of the
target's KV cache) over the target's last-layer K/V — bidirectionally for
both full-attention and sliding-window layers. For the unbatched MVP
(B=1, no padding), the masks degenerate to either ``None`` (full attention,
SDPA handles it) or a small additive bias for SWA layers when the KV is
longer than the sliding window.
"""

from typing import Optional, Tuple

import mlx.core as mx


def bidirectional_full_mask(
    query_len: int,
    kv_len: int,
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """No-op for unbatched, no-padding case — SDPA treats ``None`` as fully
    attending."""
    del query_len, kv_len, dtype
    return None


def bidirectional_swa_mask(
    query_len: int,
    query_offset: int,
    kv_len: int,
    window: int,
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Bidirectional sliding-window mask.

    For each query position ``q ∈ [query_offset, query_offset + query_len)``,
    allow attention to KV positions ``k ∈ (q - window, q + window)``.
    Returns ``None`` when no masking is needed (the entire KV fits in the
    bidirectional window of every query).
    """
    if kv_len <= window and query_offset + query_len <= kv_len + window:
        return None

    q_idx = mx.arange(query_offset, query_offset + query_len)[:, None]
    k_idx = mx.arange(kv_len)[None, :]
    dist = q_idx - k_idx
    inside = (dist > -window) & (dist < window)
    # SDPA expects an additive mask broadcastable to (B, H, L, S).
    bias = mx.where(inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
    return bias[None, None, :, :]


def make_drafter_masks(
    shared_kv_states: dict,
    query_len: int,
    query_offset: int,
    sliding_window: int,
    dtype: mx.Dtype = mx.float32,
) -> dict:
    """Build masks per layer-type for the drafter forward."""
    masks = {}
    for layer_type, kv in shared_kv_states.items():
        kv_len = _kv_len(kv)
        if layer_type == "sliding_attention":
            masks[layer_type] = bidirectional_swa_mask(
                query_len, query_offset, kv_len, sliding_window, dtype
            )
        else:
            masks[layer_type] = bidirectional_full_mask(query_len, kv_len, dtype)
    return masks


def _kv_len(kv: Tuple[mx.array, mx.array]) -> int:
    K, _ = kv
    return int(K.shape[-2])
