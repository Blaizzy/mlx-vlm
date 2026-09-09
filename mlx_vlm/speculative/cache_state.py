"""Request-owned speculative state and bounded cache transactions.

Models perform ordinary forwards. The cache records temporal states, discards
unaccepted inputs, and aligns the MTP tokens with verified target features.
"""

from dataclasses import dataclass

import mlx.core as mx

from ..models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    CacheList,
    KVCache,
    PoolingCache,
    QuantizedKVCache,
)


def iter_leaf_caches(caches):
    for cache in caches:
        if isinstance(cache, CacheList):
            yield from iter_leaf_caches(cache.caches)
        elif cache is not None:
            yield cache


class CacheTransaction:
    """Retain a per-row input prefix, using each cache's own rollback logic.

    Append-only caches need only a cursor. Recurrent and pooling caches own
    their bounded histories. Unsupported cache types fail before any forward.
    """

    temporal_types = (ArraysCache, PoolingCache, BatchPoolingCache)
    append_types = (KVCache, QuantizedKVCache, BatchKVCache, BatchQuantizedKVCache)
    batch_types = (BatchKVCache, BatchQuantizedKVCache)

    @classmethod
    def check_types(cls, caches):
        for cache in iter_leaf_caches(caches):
            if type(cache) not in cls.temporal_types + cls.append_types:
                raise ValueError(
                    f"Speculative decoding does not support {type(cache).__name__}."
                )

    def __init__(self, caches, length):
        if length < 1:
            raise ValueError("A cache transaction requires a positive input length.")
        self.length = length
        self.active = True
        self.temporal = []
        self.append = []
        leaves = tuple({id(c): c for c in iter_leaf_caches(caches)}.values())
        self.check_types(leaves)
        try:
            for cache in leaves:
                if isinstance(cache, self.temporal_types):
                    self.temporal.append((cache, cache.start_speculation(length)))
                else:
                    cursor = "_idx" if isinstance(cache, self.batch_types) else "offset"
                    self.append.append((cache, cursor, getattr(cache, cursor)))
        except BaseException:
            self.abort()
            raise

    def validate(self, lengths):
        if not self.active:
            raise RuntimeError("The cache transaction has already finished.")
        if not lengths or any(n < 0 or n > self.length for n in lengths):
            raise ValueError(f"Retained lengths must be between 0 and {self.length}.")
        for cache, generation in self.temporal:
            cache.validate_speculation(lengths, generation)
        for cache, cursor, initial in self.append:
            advance = getattr(cache, cursor) - initial
            if advance not in (0, self.length):
                raise RuntimeError(
                    "Cache did not consume the complete verification block."
                )
            if len(set(lengths)) > 1 and not isinstance(cache, self.batch_types):
                raise ValueError("Ragged acceptance requires a batch cache.")

    def commit(self, lengths):
        lengths = list(lengths)
        self.validate(lengths)
        keep = max(lengths)
        padding = [keep - n for n in lengths]
        for cache, generation in self.temporal:
            cache.commit_speculation(lengths, generation)
        for cache, cursor, initial in self.append:
            if getattr(cache, cursor) == initial:
                continue  # Optional attention side cache was not used.
            cache.trim(self.length - keep)
            if any(padding):
                cache.prepare(right_padding=padding)
                cache.finalize()
        self.active = False

    def abort(self):
        if not self.active:
            return
        for cache, generation in self.temporal:
            cache.abort_speculation(generation)
        for cache, cursor, initial in self.append:
            cache.trim(getattr(cache, cursor) - initial)
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.abort()


class SpeculativePrefill:
    """Collect target features across chunks for the shifted MTP prefill."""

    def __init__(self, draft_kind, drafter, tokens=None):
        self.kwargs = {"return_hidden": True} if drafter is not None else {}
        self.tokens = tokens
        self.chunks = []

    def append(self, output):
        if self.kwargs:
            hidden = output.hidden_states[-1]
            mx.async_eval(hidden)
            self.chunks.append(hidden)

    def finish(self, output):
        if self.chunks:
            output.hidden_states = [
                mx.concatenate([*self.chunks, output.hidden_states[-1]], axis=1)
            ]
            self.chunks.clear()
        return output


@dataclass
class DraftState:
    token: mx.array
    hidden: mx.array


class SpeculativeCache:
    """One request's target cache, MTP cache, positions, and hidden states.

    Both caches end each round at the same logical position. Draft expansions
    are temporary: after verification, MTP extends only with accepted tokens
    and *target* hidden states. No model holds a seed or a rollback snapshot.
    """

    def __init__(self, target_cache, draft_cache, position, bonus):
        self.target = target_cache
        self.draft = draft_cache
        self.position = mx.array(position, dtype=mx.int32).reshape(-1)
        self.bonus = bonus.reshape(-1, 1)
        self.seed = None
        self._target_round = None
        self._draft_round = None
        self._verified_hidden = None

    def prefill(self, tokens, hidden, forward):
        if tokens.shape[:2] != hidden.shape[:2] or tokens.shape[1] == 0:
            raise ValueError(
                "MTP requires target hidden states for every prompt token."
            )
        if max((c.size() for c in self.target), default=0) > tokens.shape[1]:
            raise ValueError(
                "MTP requires a complete prompt prefill; target-only prefix caches cannot restore its draft state."
            )
        shifted = mx.concatenate([tokens[:, 1:], self.bonus], axis=1)
        logits, draft_hidden = forward(shifted, hidden, self.draft, self.position)
        self.position = self.position + tokens.shape[1]
        self.seed = DraftState(mx.argmax(logits, axis=-1), draft_hidden[:, -1:])

    def propose(self, count, forward):
        if self._target_round is not None:
            raise RuntimeError("The previous speculative round has not finished.")
        if count < 0 or self.seed is None:
            raise ValueError("Prefill the MTP cache before proposing tokens.")
        self._target_round = CacheTransaction(self.target, count + 1)
        try:
            if count == 0:
                return self.bonus[:, :0]
            token, hidden = self.seed.token, self.seed.hidden
            proposals = [token]
            if count > 1:
                self._draft_round = CacheTransaction(self.draft, count - 1)
            for step in range(count - 1):
                logits, hidden = forward(
                    token, hidden, self.draft, self.position + step
                )
                token = mx.argmax(logits, axis=-1)
                proposals.append(token)
            return mx.concatenate(proposals, axis=1).astype(self.bonus.dtype)
        except BaseException:
            self.abort()
            raise

    def verify_inputs(self, proposals):
        return mx.concatenate([self.bonus, proposals], axis=1)

    def record_verification(self, hidden):
        self._verified_hidden = hidden

    def commit(self, tokens, forward):
        """Commit exactly the outputs delivered to each row, even on close()."""
        lengths = [len(row) for row in tokens]
        if not any(lengths):
            self.abort()
            return
        try:
            self._target_round.validate(lengths)
            if self._draft_round is not None:
                self._draft_round.abort()
                self._draft_round = None
            width = max(lengths)
            padding = [width - n for n in lengths]
            inputs = mx.array(
                [row + [0] * pad for row, pad in zip(tokens, padding)],
                dtype=self.bonus.dtype,
            )
            # The same transaction handles ragged padding and replay failure.
            # Replay only runs the small MTP head, never the target model.
            with CacheTransaction(self.draft, width) as replay:
                logits, hidden = forward(
                    inputs,
                    self._verified_hidden[:, :width],
                    self.draft,
                    self.position,
                    lengths=lengths,
                )
                replay.validate(lengths)
                self._target_round.commit(lengths)
                replay.commit(lengths)
            indices = mx.maximum(mx.array(lengths), 1)[:, None, None] - 1
            self.seed = DraftState(
                mx.argmax(logits, axis=-1),
                mx.take_along_axis(hidden, indices, axis=1),
            )
            self.bonus = mx.where(
                mx.array(lengths)[:, None] > 0,
                mx.take_along_axis(inputs, indices.squeeze(-1), axis=1),
                self.bonus,
            )
            self.position = self.position + mx.array(lengths)
        finally:
            self.abort()

    def abort(self):
        if self._target_round is not None:
            self._target_round.abort()
        if self._draft_round is not None:
            self._draft_round.abort()
        self._target_round = self._draft_round = None
        self._verified_hidden = None
