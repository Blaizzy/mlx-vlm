"""Bidirectional attention-mask helpers for the Gemma 4 MTP drafter.

The drafter attends from L query positions (placed just past the end of the
target's KV cache) over the target's last-layer K/V — bidirectionally for
both full-attention and sliding-window layers. For the unbatched MVP
(B=1, no padding), the masks degenerate to either ``None`` (full attention,
SDPA handles it) or a small additive bias for SWA layers when the KV is
longer than the sliding window.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx


def bidirectional_full_mask(
    query_len: int,
    kv_len: int,
    kv_valid_len: Optional[Union[int, mx.array]] = None,
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Full-attention mask.

    In the unbatched, no-padding case this is a no-op. In batched MTP, rows
    can have different prompt lengths while the K/V tensor is padded to a
    common length. The target's prefill *left-pads* short rows — so the
    valid keys live at the *end* of the buffer (positions
    ``[kv_len - kv_valid_len, kv_len)``), not the start.
    """
    del query_len
    if kv_valid_len is None or isinstance(kv_valid_len, int):
        if kv_valid_len is None or kv_valid_len >= kv_len:
            return None
        k_idx = mx.arange(kv_len)
        inside = k_idx >= (kv_len - kv_valid_len)
        bias = mx.where(
            inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
        )
        return bias[None, None, None, :]
    k_idx = mx.arange(kv_len)[None, :]
    inside = k_idx >= (kv_len - kv_valid_len[:, None])
    bias = mx.where(inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
    return bias[:, None, None, :]


def bidirectional_swa_mask(
    query_len: int,
    query_offset: Union[int, mx.array],
    kv_len: int,
    window: int,
    kv_valid_len: Optional[Union[int, mx.array]] = None,
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Bidirectional sliding-window mask.

    For each query at RoPE position ``q``, allow attention to keys whose
    RoPE positions ``r`` satisfy ``q - window < r < q + window``.

    With left-padding, key buffer index ``k`` maps to RoPE position
    ``r = k - (kv_len - kv_valid_len)`` (since RoPE was applied with
    offset ``-left_padding`` at write time). Real keys have ``r >= 0``;
    padded slots have ``r < 0`` and are excluded by the validity check.
    """
    if (
        isinstance(query_offset, int)
        and (kv_valid_len is None or isinstance(kv_valid_len, int))
        and kv_len <= window
        and query_offset + query_len <= kv_len + window
        and (kv_valid_len is None or kv_valid_len >= kv_len)
    ):
        return None

    if isinstance(query_offset, int):
        q_idx = mx.arange(query_offset, query_offset + query_len)[:, None]
        k_idx = mx.arange(kv_len)[None, :]
        if kv_valid_len is not None:
            left_pad = kv_len - int(kv_valid_len)
        else:
            left_pad = 0
        k_rel = k_idx - left_pad
        dist = q_idx - k_rel
        inside = (dist > -window) & (dist < window) & (k_rel >= 0)
        bias = mx.where(
            inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
        )
        return bias[None, None, :, :]

    q_idx = query_offset[:, None] + mx.arange(query_len)[None, :]
    k_idx = mx.arange(kv_len)[None, None, :]
    if kv_valid_len is not None:
        valid = (
            kv_valid_len
            if isinstance(kv_valid_len, mx.array)
            else mx.array(kv_valid_len)
        )
        left_pad = kv_len - valid[:, None, None]
    else:
        left_pad = 0
    k_rel = k_idx - left_pad
    dist = q_idx[:, :, None] - k_rel
    inside = (dist > -window) & (dist < window)
    if kv_valid_len is not None:
        inside = inside & (k_rel >= 0)
    # SDPA expects an additive mask broadcastable to (B, H, L, S).
    bias = mx.where(inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
    return bias[:, None, :, :]


def make_drafter_masks(
    shared_kv_states: dict,
    query_len: int,
    query_offset: Union[int, mx.array],
    sliding_window: int,
    dtype: mx.Dtype = mx.float32,
) -> dict:
    """Build masks per layer-type for the drafter forward."""
    masks = {}
    for layer_type, kv in shared_kv_states.items():
        kv_len = _kv_len(kv)
        kv_valid_len = query_offset
        if layer_type == "sliding_attention":
            masks[layer_type] = bidirectional_swa_mask(
                query_len, query_offset, kv_len, sliding_window, kv_valid_len, dtype
            )
        else:
            masks[layer_type] = bidirectional_full_mask(
                query_len, kv_len, kv_valid_len, dtype
            )
    return masks


def _kv_len(kv: Tuple[mx.array, mx.array]) -> int:
    K, _ = kv
    return int(K.shape[-2])
