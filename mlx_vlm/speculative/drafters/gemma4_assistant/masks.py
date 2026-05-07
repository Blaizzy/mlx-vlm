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
from mlx_lm.models.cache import dynamic_roll


def normalize_batched_shared_kv_states(
    shared_kv_states: dict,
    kv_valid_len: Union[int, mx.array],
    left_padding: Optional[Union[int, mx.array]] = None,
) -> dict:
    """Normalize batched shared K/V into the drafter's prefix-valid layout.

    The target cache may be left-padded (initial batched prefill) and can also
    carry per-row rollback slack in the tail after mixed speculative accepts.
    The Gemma drafter expects the simpler invariant used by the unbatched path:
    each row's real keys occupy ``[0, kv_valid_len)`` and any invalid slots are
    zeroed in the tail.
    """
    if left_padding is None or shared_kv_states is None:
        return shared_kv_states

    normalized = {}
    for layer_type, (keys, values) in shared_kv_states.items():
        normalized[layer_type] = (
            _normalize_shared_kv_tensor(keys, kv_valid_len, left_padding),
            _normalize_shared_kv_tensor(values, kv_valid_len, left_padding),
        )
    return normalized


def bidirectional_full_mask(
    query_len: int,
    kv_len: int,
    kv_valid_len: Optional[Union[int, mx.array]] = None,
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Full-attention mask.

    In the unbatched, no-padding case this is a no-op. In batched MTP, rows
    can have different prompt lengths while the K/V tensor is padded to a
    common length, so each row must mask keys beyond its own valid prefix.
    """
    del query_len
    if kv_valid_len is None or isinstance(kv_valid_len, int):
        if kv_valid_len is None or kv_valid_len >= kv_len:
            return None
        k_idx = mx.arange(kv_len)
        inside = k_idx < kv_valid_len
        bias = mx.where(
            inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
        )
        return bias[None, None, None, :]
    k_idx = mx.arange(kv_len)[None, :]
    inside = k_idx < kv_valid_len[:, None]
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

    For each query position ``q ∈ [query_offset, query_offset + query_len)``,
    allow attention to KV positions ``k ∈ (q - window, q + window)``.
    Returns ``None`` when no masking is needed (the entire KV fits in the
    bidirectional window of every query).
    """
    if (
        isinstance(query_offset, int)
        and (kv_valid_len is None or isinstance(kv_valid_len, int))
        and kv_len <= window
        and query_offset + query_len <= kv_len + window
    ):
        return None

    if isinstance(query_offset, int):
        q_idx = mx.arange(query_offset, query_offset + query_len)[:, None]
        k_idx = mx.arange(kv_len)[None, :]
        dist = q_idx - k_idx
        inside = (dist > -window) & (dist < window)
        if kv_valid_len is not None:
            inside = inside & (k_idx < int(kv_valid_len))
        bias = mx.where(
            inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
        )
        return bias[None, None, :, :]

    q_idx = query_offset[:, None] + mx.arange(query_len)[None, :]
    k_idx = mx.arange(kv_len)[None, None, :]
    dist = q_idx[:, :, None] - k_idx
    inside = (dist > -window) & (dist < window)
    if kv_valid_len is not None:
        valid = (
            kv_valid_len
            if isinstance(kv_valid_len, mx.array)
            else mx.array(kv_valid_len)
        )
        inside = inside & (k_idx < valid[:, None, None])
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


def _normalize_shared_kv_tensor(
    tensor: mx.array,
    kv_valid_len: Union[int, mx.array],
    left_padding: Union[int, mx.array],
) -> mx.array:
    if tensor.ndim != 4:
        return tensor

    batch = tensor.shape[0]
    seq_len = tensor.shape[-2]
    valid = _broadcast_batch_vector(kv_valid_len, batch, seq_len)
    left = _broadcast_batch_vector(left_padding, batch, seq_len)

    if batch == 1 and int(left[0].item()) == 0 and int(valid[0].item()) >= seq_len:
        return tensor

    rolled = dynamic_roll(tensor, -left[:, None], axis=2)
    keep = mx.arange(seq_len)[None, :] < valid[:, None]
    keep = keep.astype(tensor.dtype)[:, None, :, None]
    return rolled * keep


def _broadcast_batch_vector(
    value: Union[int, mx.array],
    batch: int,
    limit: int,
) -> mx.array:
    if isinstance(value, int):
        vector = mx.array([value], dtype=mx.int32)
    elif isinstance(value, mx.array):
        vector = value.astype(mx.int32)
    else:
        vector = mx.array(value, dtype=mx.int32)

    if vector.ndim == 0:
        vector = vector[None]
    elif vector.ndim > 1:
        vector = vector.reshape(-1)

    if vector.shape[0] == 1 and batch != 1:
        vector = mx.repeat(vector, batch, axis=0)
    if vector.shape[0] != batch:
        raise ValueError(
            f"Expected batch metadata of length {batch}, got {vector.shape[0]}"
        )

    return mx.clip(vector, 0, limit)
