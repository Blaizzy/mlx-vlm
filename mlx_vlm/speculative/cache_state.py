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

    def __init__(self, entries, positions, caches, length):
        self.caches = tuple(caches)
        self.length = int(length)
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.abort()

    def commit(self, lengths) -> None:
        """Retain exactly these input positions, including append-only caches."""
        if isinstance(lengths, int):
            batch = next((c.batch_size for c, _ in self._entries), 1)
            lengths = [lengths] * batch
        elif isinstance(lengths, mx.array):
            lengths = lengths.reshape(-1).tolist()
        lengths = [int(value) for value in lengths]
        if not lengths or any(value < 0 or value > self.length for value in lengths):
            raise ValueError(f"Commit lengths must be between 0 and {self.length}.")
        self.validate(lengths)
        _trim_append_caches(self.caches, lengths, self.length, self)
        for cache, generation in self._entries:
            cache.commit_speculation(lengths, generation)
        self._active = False

    def abort(self) -> None:
        if not self._active:
            return
        for cache, generation in self._entries:
            cache.abort_speculation(generation)
        for cache, read_position, initial in self._positions.values():
            advance = read_position() - initial
            if advance > 0:
                cache.trim(advance)
        self._active = False

    def cache_advance(self, cache) -> Optional[int]:
        entry = self._positions.get(id(cache))
        if entry is None:
            return None
        return entry[1]() - entry[2]


def _cache_position_reader(cache):
    """Resolve each legacy cache's physical cursor once when binding a round."""
    for name in ("offset", "_offset", "_idx"):
        if isinstance(getattr(cache, name, None), int):
            return lambda name=name: getattr(cache, name)
    size = getattr(cache, "size", None)
    return size if callable(size) and isinstance(size(), int) else None


def start_speculative_cache(
    caches: Iterable[Any], length: int, cache_types: Optional[tuple] = None
):
    """Track append positions and start each capable temporal cache."""
    leaves = tuple({id(c): c for c in iter_leaf_caches(caches)}.values())
    entries = []
    positions = {}
    transaction = SpeculativeCacheTransaction(entries, positions, leaves, length)
    try:
        for cache in leaves:
            start = getattr(cache, "start_speculation", None)
            if callable(start) and (
                cache_types is None or isinstance(cache, cache_types)
            ):
                entries.append((cache, start(length)))
            else:
                read_position = _cache_position_reader(cache)
                if read_position is not None and callable(getattr(cache, "trim", None)):
                    positions[id(cache)] = (cache, read_position, read_position())
    except BaseException:
        transaction.abort()
        raise
    return transaction


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

    retained = [value + 1 for value in accepted_values]
    if isinstance(transaction, SpeculativeCacheTransaction):
        transaction.commit(retained)
    else:
        # Compatibility for model adapters that still return legacy state.
        _trim_append_caches(iter_leaf_caches(caches), retained, block_size)
    return max(accepted_values)


def abort_speculative_round(state):
    """Release and restore an unfinished transaction returned by a target."""
    if isinstance(state, SpeculativeCacheTransaction):
        state.abort()


def commit_speculative_round(model, caches, state, accepted, block_size):
    """Translate accepted draft counts once at the legacy target boundary."""
    if isinstance(accepted, int):
        values = [accepted]
    elif isinstance(accepted, mx.array):
        values = accepted.reshape(-1).tolist()
    else:
        values = list(accepted)
    if isinstance(state, SpeculativeCacheTransaction):
        state.commit([value + 1 for value in values])
    elif any(value < block_size - 1 for value in values):
        model.rollback_speculative_cache(caches, state, accepted, block_size)


def _trim_append_caches(caches, retained, block_size, transaction=None):
    max_retained = max(retained)
    trim = int(block_size) - max_retained
    right_padding = [max_retained - value for value in retained]
    has_ragged_tail = any(right_padding)

    actions = []
    for cache in caches:
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
                cache_trim = max(0, advance - max_retained)
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


__all__ = [
    "SpeculativeCacheTransaction",
    "abort_speculative_round",
    "commit_speculative_round",
    "iter_leaf_caches",
    "rollback_speculative_cache",
    "start_speculative_cache",
]
