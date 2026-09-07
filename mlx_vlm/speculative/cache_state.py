"""Shared cache transactions for speculative verification rollback."""

from typing import Any, Iterable, Optional

import mlx.core as mx

from ..models.cache import CacheList


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
    """A bounded transaction over temporal and append-only caches."""

    def __init__(self, entries, positions):
        self._entries = entries
        self._positions = positions
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
        for cache, initial in self._positions.values():
            advance = _cache_position(cache) - initial
            if advance > 0:
                cache.trim(advance)
        self._active = False

    def cache_advance(self, cache) -> Optional[int]:
        entry = self._positions.get(id(cache))
        if entry is None:
            return None
        return _cache_position(cache) - entry[1]


def _cache_position(cache) -> Optional[int]:
    offset = getattr(cache, "offset", None)
    if isinstance(offset, int):
        return offset
    physical_offset = getattr(cache, "_offset", None)
    if isinstance(physical_offset, int):
        return physical_offset
    physical_index = getattr(cache, "_idx", None)
    if isinstance(physical_index, int):
        return physical_index
    size = getattr(cache, "size", None)
    if callable(size):
        value = size()
        if isinstance(value, int):
            return value
    return None


def start_speculative_cache(
    caches: Iterable[Any], length: int, cache_types: Optional[tuple] = None
):
    """Track append positions and start each capable temporal cache."""
    entries = []
    positions = {}
    temporal = set()
    for cache in iter_leaf_caches(caches):
        position = _cache_position(cache)
        if position is not None and callable(getattr(cache, "trim", None)):
            positions.setdefault(id(cache), (cache, position))
        if cache_types is not None and not isinstance(cache, cache_types):
            continue
        start = getattr(cache, "start_speculation", None)
        if callable(start) and id(cache) not in temporal:
            entries.append((cache, start(length)))
            temporal.add(id(cache))
    return SpeculativeCacheTransaction(entries, positions)


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

    actions = []
    for cache in iter_leaf_caches(caches):
        if getattr(cache, "is_speculating", False):
            continue
        trim_cache = getattr(cache, "trim", None)
        if not callable(trim_cache):
            raise RuntimeError(
                f"{type(cache).__name__} cannot roll back a speculative block."
            )
        cache_trim = trim
        if isinstance(transaction, SpeculativeCacheTransaction):
            advance = transaction.cache_advance(cache)
            if advance is not None:
                if advance == 0:
                    continue
                cache_trim = max(0, advance - (max_accepted + 1))
                if has_ragged_tail and advance != int(block_size):
                    raise RuntimeError(
                        f"{type(cache).__name__} advanced by {advance} tokens "
                        f"during a {block_size}-token speculative block."
                    )

        right_trimmed = False
        if has_ragged_tail:
            prepare = getattr(cache, "prepare", None)
            finalize = getattr(cache, "finalize", None)
            if (
                getattr(cache, "keys", None) is not None
                and callable(prepare)
                and callable(finalize)
            ):
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
        actions.append((cache, trim_cache, cache_trim, right_trimmed))

    for cache, trim_cache, cache_trim, right_trimmed in actions:
        if cache_trim > 0:
            trim_cache(cache_trim)
        if right_trimmed:
            cache.prepare(right_padding=right_padding)
            cache.finalize()

    if isinstance(transaction, SpeculativeCacheTransaction):
        transaction._commit_validated(retained)
    return max_accepted


__all__ = [
    "SpeculativeCacheTransaction",
    "iter_leaf_caches",
    "rollback_speculative_cache",
    "start_speculative_cache",
]
