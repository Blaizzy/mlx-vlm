"""Shared cache snapshots for speculative verification rollback."""

from typing import Any, Iterable, Optional

import mlx.core as mx

from ..models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    CacheList,
    KVCache,
    PoolingCache,
    RotatingKVCache,
)


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_cache_tree(item) for key, item in value.items()}
    return value


def _snapshot_single_cache(cache, incoming_tokens: int):
    if cache is None:
        return None
    if isinstance(cache, ArraysCache):
        # Recurrent layers replace these arrays rather than mutating them.
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
            "cache_list",
            [_snapshot_single_cache(child, incoming_tokens) for child in cache.caches],
        )
    if isinstance(cache, (BatchKVCache, BatchQuantizedKVCache)):
        # Appended slots remain outside _idx after restore and can be reused.
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
        # Batch pooling updates its remainder buffers in place.
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
        overwrite = total % cache.ratio if total >= cache.ratio else 0
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
    if isinstance(cache, RotatingKVCache):
        return (
            "rotating",
            _clone_cache_tree(cache.offset),
            int(cache._idx),
            getattr(cache, "start_position", None),
        )
    if isinstance(cache, KVCache):
        return ("append", cache.empty(), int(cache.offset))
    return (
        "full",
        _clone_cache_tree(getattr(cache, "state", None)),
        _clone_cache_tree(getattr(cache, "meta_state", None)),
    )


def snapshot_cache_state(caches: Iterable[Any], incoming_tokens: int = 0):
    """Capture the minimum state needed to undo a verifier block."""
    return [_snapshot_single_cache(cache, incoming_tokens) for cache in caches]


def _clear_cache_state(cache) -> None:
    if isinstance(cache, CacheList):
        for child in cache.caches:
            _clear_cache_state(child)
        return
    for name, value in (
        ("keys", None),
        ("values", None),
        ("offset", 0),
        ("_idx", 0),
        ("start_position", 0),
        ("buf_kv", None),
        ("buf_gate", None),
        ("remainder", 0),
        ("pooled", None),
    ):
        if hasattr(cache, name):
            setattr(cache, name, value)


def _restore_single_cache(cache, snapshot) -> None:
    if cache is None or snapshot is None:
        return
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
    if kind == "cache_list":
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
        cache.remainder = int(remainder)
        if buf_kv is not None:
            restore_length = int(buf_kv.shape[1])
            if cache.buf_kv is None or cache.buf_kv.shape[1] < cache.ratio:
                cache.buf_kv = mx.zeros(
                    (buf_kv.shape[0], cache.ratio, buf_kv.shape[2]),
                    dtype=buf_kv.dtype,
                )
            if cache.buf_gate is None or cache.buf_gate.shape[1] < cache.ratio:
                cache.buf_gate = mx.zeros(
                    (buf_gate.shape[0], cache.ratio, buf_gate.shape[2]),
                    dtype=buf_gate.dtype,
                )
            cache.buf_kv[:, :restore_length] = buf_kv
            cache.buf_gate[:, :restore_length] = buf_gate
        if pooled_length is None:
            cache.pooled = None
        elif cache.pooled is not None:
            cache.pooled = cache.pooled[:, :pooled_length]
        return
    if kind == "rotating":
        _, offset, idx, start_position = snapshot
        cache.offset = _clone_cache_tree(offset)
        cache._idx = int(idx)
        if start_position is not None and hasattr(cache, "start_position"):
            cache.start_position = int(start_position)
        return
    if kind == "append":
        _, was_empty, offset = snapshot
        if was_empty:
            cache.keys = cache.values = None
        cache.offset = offset
        return

    _, state, meta_state = snapshot
    if state is None:
        _clear_cache_state(cache)
        return
    if meta_state is not None and hasattr(type(cache), "meta_state"):
        cache.meta_state = _clone_cache_tree(meta_state)
    cache.state = _clone_cache_tree(state)


def restore_cache_state(caches: Iterable[Any], snapshots) -> None:
    """Restore caches from :func:`snapshot_cache_state`."""
    for cache, snapshot in zip(caches, snapshots):
        _restore_single_cache(cache, snapshot)


def iter_leaf_caches(caches: Iterable[Any]):
    """Yield non-container caches from a possibly nested cache sequence."""
    for cache in caches:
        if cache is None:
            continue
        if isinstance(cache, CacheList):
            yield from iter_leaf_caches(cache.caches)
        else:
            yield cache


class SpeculativeCacheTransaction:
    """A bounded transaction over caches that retain temporal state."""

    def __init__(self, entries):
        self._entries = entries
        self._active = True

    @property
    def active(self):
        return self._active

    def validate(self, lengths) -> None:
        if not self._active:
            raise RuntimeError("Speculative cache transaction is no longer active.")
        for cache, generation in self._entries:
            cache.validate_speculation(lengths, generation)

    def _commit_validated(self, lengths) -> None:
        for cache, generation in self._entries:
            cache.commit_speculation(lengths, generation)
        self._active = False

    def commit(self, lengths) -> None:
        self.validate(lengths)
        self._commit_validated(lengths)

    def abort(self) -> None:
        if not self._active:
            return
        for cache, generation in self._entries:
            cache.abort_speculation(generation)
        self._active = False


def start_speculative_cache(
    caches: Iterable[Any], length: int, cache_types: Optional[tuple] = None
):
    """Start a temporal transaction on every capable cache in ``caches``."""
    entries = []
    for cache in iter_leaf_caches(caches):
        if cache_types is not None and not isinstance(cache, cache_types):
            continue
        start = getattr(cache, "start_speculation", None)
        if callable(start):
            entries.append((cache, start(length)))
    return SpeculativeCacheTransaction(entries)


def rollback_speculative_cache(
    caches: Iterable[Any],
    transaction: SpeculativeCacheTransaction,
    accepted,
    block_size: int,
) -> int:
    """Commit accepted prefixes and rewind ordinary append-only caches."""
    if isinstance(accepted, int):
        accepted_values = [int(accepted)]
    elif isinstance(accepted, mx.array):
        accepted_values = [int(value) for value in accepted.reshape(-1).tolist()]
    else:
        accepted_values = [int(value) for value in accepted]

    max_accepted = max(accepted_values)
    retained = [value + 1 for value in accepted_values]
    trim = int(block_size) - (max_accepted + 1)
    is_batch = len(accepted_values) > 1
    right_padding = [max_accepted - value for value in accepted_values]
    has_ragged_tail = is_batch and any(right_padding)

    if isinstance(transaction, SpeculativeCacheTransaction):
        transaction.validate(retained)

    for cache in iter_leaf_caches(caches):
        if getattr(cache, "is_speculating", False):
            continue
        trim_cache = getattr(cache, "trim", None)
        if not callable(trim_cache):
            raise RuntimeError(
                f"{type(cache).__name__} cannot roll back a speculative block."
            )
        if trim > 0:
            trim_cache(trim)

        right_trimmed = False
        if has_ragged_tail:
            prepare = getattr(cache, "prepare", None)
            finalize = getattr(cache, "finalize", None)
            if (
                getattr(cache, "keys", None) is not None
                and callable(prepare)
                and callable(finalize)
            ):
                prepare(right_padding=right_padding)
                finalize()
                right_trimmed = True
        if (
            has_ragged_tail
            and not right_trimmed
            and hasattr(cache, "_idx")
            and getattr(cache, "keys", None) is not None
        ):
            raise RuntimeError(
                "Batched speculative rollback requires uniform acceptance or "
                "a cache with per-row tail trimming; got "
                f"{type(cache).__name__}."
            )

    if isinstance(transaction, SpeculativeCacheTransaction):
        transaction._commit_validated(retained)
    return max_accepted


__all__ = [
    "SpeculativeCacheTransaction",
    "iter_leaf_caches",
    "rollback_speculative_cache",
    "restore_cache_state",
    "snapshot_cache_state",
    "start_speculative_cache",
]
