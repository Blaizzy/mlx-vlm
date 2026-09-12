"""Request-owned speculative state and bounded cache transactions.

Models perform ordinary forwards. The cache records temporal states, discards
unaccepted inputs, and aligns the MTP tokens with verified target features.
"""

from dataclasses import dataclass
from functools import partial

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
from .stats import SpeculativeStats


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
        self.caches = caches
        leaves = tuple({id(c): c for c in iter_leaf_caches(caches)}.values())
        self.identities = {id(c) for c in leaves}
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
        if {id(c) for c in iter_leaf_caches(self.caches)} != self.identities:
            raise RuntimeError(
                "A forward replaced cache objects during an active transaction."
            )
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
    """Stream target features into the shifted MTP cache, retaining one seed."""

    def __init__(self, draft_kind, drafter, tokens=None):
        self.kwargs = {"return_hidden": True} if drafter is not None else {}
        self.tokens = tokens
        self.state = None
        self.consumed = 0
        self.checkpoint = None

    def start(
        self,
        model,
        target_cache,
        drafter,
        *,
        state=None,
        checkpoint=None,
        position_offset=None,
    ):
        """Stream target features into MTP, keeping only the next draft seed."""
        self.forward = partial(
            drafter, target_model=getattr(model, "language_model", model)
        )
        self.state = state or SpeculativeCache.create(
            target_cache, drafter, self.tokens.shape[0]
        )
        self.checkpoint = checkpoint
        if position_offset is not None:
            self.state.position_offset = position_offset.reshape(-1)

    def append(self, output):
        if self.kwargs:
            hidden = output.hidden_states[-1]
            if self.state is not None:
                end = self.consumed + hidden.shape[1]
                self.state.bonus = self.tokens[:, end : end + 1]
                self.state.prefill(
                    self.tokens[:, self.consumed : end], hidden, self.forward
                )
                mx.async_eval(
                    [entry.state for entry in self.state.draft],
                    self.state.seed.token,
                    self.state.seed.hidden,
                )
                self.consumed = end
                if self.checkpoint:
                    self.checkpoint(self.state)
                return
            raise RuntimeError("Initialize the speculative cache before prefill.")

    def finish(self, output, first_bonus=None):
        if self.state is not None:
            self.state.bonus = first_bonus.reshape(-1, 1)
            self.state.prefill(
                self.tokens[:, self.consumed :], output.hidden_states[-1], self.forward
            )
            return output
        if self.kwargs:
            raise RuntimeError("Initialize the speculative cache before prefill.")
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
        self.position_offset = mx.zeros_like(self.position)
        self.bonus = bonus.reshape(-1, 1)
        self.seed = None
        self.tokens = None
        self.stats = [SpeculativeStats() for _ in range(self.position.size)]
        self._target_round = None
        self._draft_round = None
        self._verified_hidden = None
        self._proposals = None

    @classmethod
    def create(cls, target_cache, drafter, batch):
        padding = next(
            (
                c.left_padding.tolist()
                for c in iter_leaf_caches(target_cache)
                if isinstance(c, CacheTransaction.batch_types)
            ),
            None,
        )
        if batch > 1 and padding is None:
            raise ValueError("Batched MTP requires batch prompt caches.")
        return cls(
            target_cache,
            drafter.make_cache(padding),
            [0] * batch if padding is None else [-p for p in padding],
            mx.zeros((batch, 1), dtype=mx.int32),
        )

    def checkpoint(self, row=0):
        """Return an atomic target/draft/seed checkpoint using native APC types."""
        from ..apc import snapshot_prompt_cache_row

        if self._target_round is not None or self.seed is None:
            raise RuntimeError("Only committed speculative state can be checkpointed.")
        target = snapshot_prompt_cache_row(self.target, row, clone=False)
        draft = snapshot_prompt_cache_row(self.draft, row, clone=False)
        if target is None or draft is None:
            raise ValueError("Cache cannot extract a prefix checkpoint row.")
        metadata = ArraysCache(5)
        metadata.cache = [
            self.position[row : row + 1, None],
            self.bonus[row : row + 1],
            self.seed.token[row : row + 1],
            self.seed.hidden[row : row + 1],
            self.position_offset[row : row + 1, None],
        ]
        return [CacheList(*target), CacheList(*draft), metadata]

    @classmethod
    def restore(cls, checkpoint):
        target, draft, metadata = checkpoint
        position, bonus, token, hidden, position_offset = metadata.cache
        state = cls(target.caches, draft.caches, position, bonus)
        state.position_offset = position_offset.reshape(-1)
        state.seed = DraftState(token, hidden)
        return state

    @classmethod
    def merge(cls, states):
        """Join independently prefilled rows without replaying either model."""
        from ..apc import make_warm_batch_exact_cache_multi

        checkpoints = [state.checkpoint() for state in states]
        positions = [int(state.position.item()) for state in states]
        target, _ = make_warm_batch_exact_cache_multi(
            [c[0].caches for c in checkpoints], positions
        )
        draft, _ = make_warm_batch_exact_cache_multi(
            [c[1].caches for c in checkpoints], positions
        )
        if target is None or draft is None:
            raise ValueError("Cache types cannot merge speculative request rows.")
        state = cls(target, draft, positions, mx.concatenate([s.bonus for s in states]))
        state.seed = DraftState(
            mx.concatenate([s.seed.token for s in states]),
            mx.concatenate([s.seed.hidden for s in states]),
        )
        state.position_offset = mx.concatenate([s.position_offset for s in states])
        state.stats = [stats for s in states for stats in s.stats]
        return state

    def positions(self, length):
        return (self.position + self.position_offset)[:, None] + mx.arange(length)[None]

    def prefill(self, tokens, hidden, forward):
        if tokens.shape[:2] != hidden.shape[:2] or tokens.shape[1] == 0:
            raise ValueError(
                "MTP requires target hidden states for every prompt token."
            )
        shifted = mx.concatenate([tokens[:, 1:], self.bonus], axis=1)
        logits, draft_hidden = forward(
            shifted, hidden, self.draft, self.position + self.position_offset
        )
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
                self._proposals = self.bonus[:, :0]
                return self._proposals
            token, hidden = self.seed.token, self.seed.hidden
            proposals = [token]
            if count > 1:
                self._draft_round = CacheTransaction(self.draft, count - 1)
            for step in range(count - 1):
                logits, hidden = forward(
                    token,
                    hidden,
                    self.draft,
                    self.position + self.position_offset + step,
                )
                token = mx.argmax(logits, axis=-1)
                proposals.append(token)
            self._proposals = mx.concatenate(proposals, axis=1).astype(self.bonus.dtype)
            return self._proposals
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
                    self.position + self.position_offset,
                    lengths=lengths,
                )
                replay.validate(lengths)
                self._target_round.commit(lengths)
                replay.commit(lengths)
            indices = mx.maximum(mx.array(lengths), 1)[:, None, None] - 1
            active = mx.array(lengths)[:, None] > 0
            self.seed = DraftState(
                mx.where(active, mx.argmax(logits, axis=-1), self.seed.token),
                mx.where(
                    active[..., None],
                    mx.take_along_axis(hidden, indices, axis=1),
                    self.seed.hidden,
                ),
            )
            self.bonus = mx.where(
                active,
                mx.take_along_axis(inputs, indices.squeeze(-1), axis=1),
                self.bonus,
            )
            self.position = self.position + mx.array(lengths)
            if self.tokens is not None:
                for context, emitted in zip(self.tokens, tokens):
                    context.extend(emitted)
            for stats, draft, output in zip(
                self.stats, self._proposals.tolist(), tokens
            ):
                stats.record(draft, output)
        finally:
            self.abort()

    def abort(self):
        if self._target_round is not None:
            self._target_round.abort()
        if self._draft_round is not None:
            self._draft_round.abort()
        self._target_round = self._draft_round = None
        self._verified_hidden = self._proposals = None
