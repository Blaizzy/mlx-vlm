"""Request-owned speculative state and bounded cache transactions.

Models perform ordinary forwards. The cache records temporal states, discards
unaccepted inputs, and aligns the MTP tokens with verified target features.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

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
        self.lengths = None
        self.tail_hidden = None

    def start(
        self,
        model,
        target_cache,
        drafter,
        *,
        state=None,
        lengths=None,
        position_offset=None,
    ):
        """Stream target features into MTP, keeping only the next draft seed."""
        self.forward = partial(
            drafter, target_model=getattr(model, "language_model", model)
        )
        self.state = state or (
            target_cache
            if isinstance(target_cache, SpeculativeCache)
            else SpeculativeCache.create(target_cache, drafter, self.tokens.shape[0])
        )
        self.lengths = lengths or [self.tokens.shape[1]] * self.tokens.shape[0]
        if position_offset is not None:
            self.state.position_offset = position_offset.reshape(-1)

    def append(self, output):
        if not self.kwargs:
            return
        if self.state is None:
            raise RuntimeError("Initialize the speculative cache before prefill.")
        hidden = output.hidden_states[-1]
        width = hidden.shape[1]
        end = self.consumed + width
        lengths = [min(width, max(0, n - 1 - self.consumed)) for n in self.lengths]
        eval_targets = []
        # A right-padded row can finish before the rest of the batch. Retain
        # its last target feature until the first output token has been sampled.
        if any(self.consumed < n <= end for n in self.lengths):
            indices = mx.array(self.lengths) - self.consumed - 1
            last = mx.take_along_axis(
                hidden, mx.clip(indices, 0, width - 1)[:, None, None], axis=1
            )
            if self.tail_hidden is None:
                self.tail_hidden = mx.zeros_like(last)
            self.tail_hidden = mx.where(
                ((indices >= 0) & (indices < width))[:, None, None],
                last,
                self.tail_hidden,
            )
            # Materialize the retained vector so it cannot keep old chunks'
            # full hidden-state graphs alive while longer rows finish.
            eval_targets.append(self.tail_hidden)
        if any(lengths):
            next_token = self.tokens[:, end : end + 1]
            self.state.bonus = (
                next_token if next_token.size else mx.zeros_like(self.state.bonus)
            )
            self.state.prefill(
                self.tokens[:, self.consumed : end],
                hidden,
                self.forward,
                lengths=lengths,
            )
            eval_targets.extend(
                [
                    [entry.state for entry in self.state.draft],
                    self.state.seed.token,
                    self.state.seed.hidden,
                ]
            )
        if eval_targets:
            mx.async_eval(eval_targets)
        self.consumed = end

    def finish(self, output, first_bonus=None):
        if not self.kwargs:
            return output
        self.append(output)
        self.state.bonus = first_bonus.reshape(-1, 1)
        self.state.prefill(self.state.bonus, self.tail_hidden, self.forward)
        self.tail_hidden = None
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

    # Expose the target's ordinary layer-cache sequence to model forwards.
    def __len__(self):
        return len(self.target)

    def __iter__(self):
        return iter(self.target)

    def __getitem__(self, index):
        return self.target[index]

    def __setitem__(self, index, value):
        self.target[index] = value

    prefix_replay_tokens = 1

    @property
    def prefix_cache_components(self):
        return [CacheList(*self.target), CacheList(*self.draft), ArraysCache(3)]

    @classmethod
    def for_model(cls, model, drafter, target_cache):
        """Cache factory consumed by the normal APC coordinator."""
        from ..apc import semantic_extra_hash

        def identity(weights):
            path = getattr(weights, "model_path", None)
            if path is None:
                return id(weights)
            path = Path(path).resolve()
            files = sorted([*path.glob("*.safetensors"), path / "config.json"])
            return [
                str(path),
                [
                    (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                    for p in files
                    if p.exists()
                ],
            ]

        state = cls.create(target_cache, drafter, 1)
        state.prefix_cache_identity = {
            "mtp_schema": 3,
            "target": identity(getattr(model, "language_model", model)),
            "draft": identity(drafter),
            "draft_dependencies": semantic_extra_hash(model=drafter),
        }
        return state

    def prefix_cache_key(self, tokens, row=0):
        """MTP has already consumed the pending target token at this boundary."""
        position = int(self.position[row].item())
        if position < 1 or position >= len(tokens):
            return None
        if self.bonus[row].item() != tokens[position]:
            raise ValueError("Prefix checkpoint must include its pending target token.")
        return list(tokens[: position + 1])

    def validate_prefix(self, tokens, position):
        if (
            int(self.position.item()) != position
            or self.bonus.item() != tokens[position]
        ):
            raise ValueError("Request cache is not aligned with its prefix key.")

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
        """Snapshot committed caches and alignment; suffix prefill rebuilds the seed."""
        from ..apc import snapshot_prompt_cache_row

        if self._target_round is not None:
            raise RuntimeError("Only committed speculative state can be checkpointed.")
        target = snapshot_prompt_cache_row(self.target, row, clone=False)
        draft = snapshot_prompt_cache_row(self.draft, row, clone=False)
        if target is None or draft is None:
            raise ValueError("Cache cannot extract a prefix checkpoint row.")
        metadata = ArraysCache(3)
        metadata.cache = [
            self.position[row : row + 1, None],
            self.bonus[row : row + 1],
            self.position_offset[row : row + 1, None],
        ]
        return [CacheList(*target), CacheList(*draft), metadata]

    @classmethod
    def restore(cls, checkpoint):
        """Restore prefix state without a decode seed, including older checkpoints."""
        target, draft, metadata = checkpoint
        values = metadata.cache
        if len(values) == 5:
            # Older entries also saved a prediction and hidden state; ignore them.
            values = [values[0], values[1], values[4]]
        position, bonus, position_offset = values
        state = cls(target.caches, draft.caches, position, bonus)
        state.position_offset = position_offset.reshape(-1)
        return state

    @classmethod
    def merge(cls, states, *, kv_quant_config=None):
        """Merge warm and empty rows before ordinary batched prefill."""
        from ..apc import make_warm_batch_exact_cache_multi

        positions = [int(state.position.item()) for state in states]
        target, _ = make_warm_batch_exact_cache_multi(
            [s.target for s in states], positions, kv_quant_config=kv_quant_config
        )
        draft, _ = make_warm_batch_exact_cache_multi(
            [s.draft for s in states], positions
        )
        if target is None or draft is None:
            raise ValueError("Cache types cannot merge request rows.")
        state = cls(target, draft, positions, mx.concatenate([s.bonus for s in states]))
        state.position_offset = mx.concatenate([s.position_offset for s in states])
        state.stats = [stats for s in states for stats in s.stats]
        return state

    def positions(self, length):
        return (self.position + self.position_offset)[:, None] + mx.arange(length)[None]

    def prefill(self, tokens, hidden, forward, *, lengths=None):
        if tokens.shape[:2] != hidden.shape[:2] or tokens.shape[1] == 0:
            raise ValueError(
                "MTP requires target hidden states for every prompt token."
            )
        width = tokens.shape[1]
        lengths = lengths or [width] * tokens.shape[0]
        shifted = mx.concatenate([tokens[:, 1:], self.bonus], axis=1)
        width = max(lengths)
        shifted, hidden = shifted[:, :width], hidden[:, :width]
        # Reuse native prefill padding support. Each chunk ends with canonical
        # caches, even when some rows finished and receive only padding.
        padding = [width - n for n in lengths]
        if any(padding):
            for entry in self.draft:
                entry.prepare(lengths=lengths, right_padding=padding)
        logits, draft_hidden = forward(
            shifted,
            hidden,
            self.draft,
            self.position + self.position_offset,
            lengths=lengths,
        )
        if any(padding):
            for entry in self.draft:
                entry.finalize()
        self.position = self.position + mx.array(lengths)
        last = mx.take_along_axis(
            draft_hidden, mx.maximum(mx.array(lengths), 1)[:, None, None] - 1, axis=1
        )
        token = mx.argmax(logits, axis=-1)
        active = mx.array(lengths)[:, None] > 0
        previous = self.seed or DraftState(mx.zeros_like(token), mx.zeros_like(last))
        self.seed = DraftState(
            mx.where(active, token, previous.token),
            mx.where(active[..., None], last, previous.hidden),
        )
        # The shift at the last retained position, not a padded batch column.
        self.bonus = mx.where(
            active,
            mx.take_along_axis(
                shifted, mx.maximum(mx.array(lengths), 1)[:, None] - 1, axis=1
            ),
            self.bonus,
        )

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
