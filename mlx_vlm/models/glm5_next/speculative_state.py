import mlx.core as mx

from ..cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    CacheList,
    KVCache,
    PoolingCache,
)


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_tree(item) for item in value]
    if type(value) is dict:
        return {key: _clone_cache_tree(item) for key, item in value.items()}
    return value


def _snapshot_single_cache(cache, incoming_tokens):
    if isinstance(cache, ArraysCache):
        # GLM's recurrent layer replaces its convolution and GDN arrays instead
        # of updating them in-place. Retaining those references avoids copying
        # roughly one full FP32 recurrent state per linear-attention layer.
        return (
            "arrays",
            list(cache.state),
            cache._left_padding,
            cache._left_padding_advance,
            cache._lengths,
            cache._lengths_advance,
        )
    if isinstance(cache, CacheList):
        return (
            "list",
            [_snapshot_single_cache(child, incoming_tokens) for child in cache.caches],
        )
    if isinstance(cache, (BatchKVCache, BatchQuantizedKVCache)):
        # Verification only appends to these buffers. Keep the original buffer
        # references and logical metadata; appended slots are outside ``_idx``
        # and are overwritten if rollback reuses them.
        return (
            "batch_append",
            cache.keys,
            cache.values,
            int(cache._idx),
            _clone_cache_tree(cache.offset),
            _clone_cache_tree(cache.left_padding),
            _clone_cache_tree(cache._right_padding),
        )
    if isinstance(cache, BatchPoolingCache):
        # Pool buffers are small (one compression window) but are updated
        # in-place. Preserve them while sharing the append-only pooled array.
        return (
            "batch_pooling",
            _clone_cache_tree(cache.buf_kv),
            _clone_cache_tree(cache.buf_gate),
            cache.pooled,
            list(cache.remainder),
            list(cache._pool_lengths),
            list(cache._lengths),
            list(cache._processed),
            list(cache.left_padding),
        )
    if isinstance(cache, PoolingCache):
        remainder = int(cache.remainder)
        total = remainder + int(incoming_tokens)
        overwrite = total % cache.ratio if total >= cache.ratio else total
        preserve = min(remainder, overwrite)
        buf_kv = (
            None
            if preserve == 0 or cache.buf_kv is None
            else _clone_cache_tree(cache.buf_kv[:, :preserve])
        )
        buf_gate = (
            None
            if preserve == 0 or cache.buf_gate is None
            else _clone_cache_tree(cache.buf_gate[:, :preserve])
        )
        return (
            "pooling",
            remainder,
            buf_kv,
            buf_gate,
            None if cache.pooled is None else cache.pooled.shape[1],
        )
    if isinstance(cache, KVCache):
        return ("append", cache.empty(), int(cache.offset))
    return (
        "full",
        _clone_cache_tree(cache.state),
        _clone_cache_tree(cache.meta_state),
    )


def _snapshot_cache(caches, incoming_tokens):
    return [_snapshot_single_cache(cache, incoming_tokens) for cache in caches]


def _restore_single_cache(cache, snapshot):
    kind = snapshot[0]
    if kind == "arrays":
        (
            _,
            state,
            left_padding,
            left_padding_advance,
            lengths,
            lengths_advance,
        ) = snapshot
        cache.state = list(state)
        cache._left_padding = left_padding
        cache._left_padding_advance = left_padding_advance
        cache._lengths = lengths
        cache._lengths_advance = lengths_advance
        return
    if kind == "list":
        for child, child_snapshot in zip(cache.caches, snapshot[1]):
            _restore_single_cache(child, child_snapshot)
        return
    if kind == "batch_append":
        (
            _,
            cache.keys,
            cache.values,
            cache._idx,
            cache.offset,
            cache.left_padding,
            cache._right_padding,
        ) = snapshot
        return
    if kind == "batch_pooling":
        (
            _,
            cache.buf_kv,
            cache.buf_gate,
            cache.pooled,
            remainder,
            pool_lengths,
            lengths,
            processed,
            left_padding,
        ) = snapshot
        cache.remainder = list(remainder)
        cache._pool_lengths = list(pool_lengths)
        cache._lengths = list(lengths)
        cache._processed = list(processed)
        cache.left_padding = list(left_padding)
        return
    if kind == "pooling":
        _, remainder, buf_kv, buf_gate, pooled_length = snapshot
        cache.remainder = remainder
        if buf_kv is not None:
            cache.buf_kv[:, : buf_kv.shape[1]] = buf_kv
            cache.buf_gate[:, : buf_gate.shape[1]] = buf_gate
        cache.pooled = (
            None if pooled_length is None else cache.pooled[:, :pooled_length]
        )
        return
    if kind == "append":
        _, was_empty, offset = snapshot
        if was_empty:
            cache.keys = cache.values = None
        cache.offset = offset
        return
    _, state, meta_state = snapshot
    cache.meta_state = _clone_cache_tree(meta_state)
    cache.state = _clone_cache_tree(state)


def _restore_cache(caches, snapshots):
    for cache, snapshot in zip(caches, snapshots):
        _restore_single_cache(cache, snapshot)
