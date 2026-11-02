"""Resumable speculative rounds for continuous request admission."""

import mlx.core as mx

from .cache_state import abort_speculative_round, commit_speculative_round
from .common import (
    _dflash_block_total,
    _record_speculative_round,
    _speculative_walk_batch,
    _SpeculativeSamplerRNG,
)
from .dflash import (
    _dflash_verify,
    _dflash_verify_greedy,
    _PositionedDraftSampler,
    _sample_dflash_target_walk,
    _supports_positioned_target_sampling,
)


def unpad_hidden(hidden, cache):
    """Return real target context only; padding has no position in a draft cache."""
    batch = hidden.shape[0]
    lengths = [hidden.shape[1]] * batch
    offsets = getattr(cache[0], "offset", None) if cache else None
    if isinstance(offsets, mx.array) and offsets.size == batch:
        lengths = [
            min(hidden.shape[1], max(0, n)) for n in offsets.reshape(-1).tolist()
        ]
    return [
        hidden[i : i + 1, hidden.shape[1] - length :]
        for i, length in enumerate(lengths)
    ]


class DFlashBatch:
    """Own each request's draft state and one shared, mutable target batch.

    A step completes its cache transaction before returning. Admission and
    cancellation can therefore change the batch at every step boundary.
    All device work stays on the caller's serving stream, including drafting,
    verification, and rollback, to preserve distributed collective ordering.
    """

    def __init__(
        self, model, draft, cache, hidden, bonus, block_size, sampler, dtype, greedy
    ):
        self.model = getattr(model, "language_model", model)
        self.draft = draft
        self.cache = cache
        self.hidden = unpad_hidden(hidden, cache)
        self.bonus = bonus.reshape(-1).tolist()
        self.block_size = _dflash_block_total(draft, block_size)
        self.sampler = sampler
        self.dtype = dtype
        self.greedy = greedy
        self.positioned_sampling = _supports_positioned_target_sampling(sampler)
        draft.reset(model)
        self.sampler_rng = _SpeculativeSamplerRNG(
            draft, enabled=not greedy and not self.positioned_sampling
        )
        self.draft_caches = [draft.make_cache() for _ in self.bonus]

    def extend(self, cache, hidden, bonus):
        incoming_hidden = unpad_hidden(hidden, cache)
        # Finish pending rollback copies before mutating batch layouts.
        mx.eval(
            [c.state for c in self.cache],
            [c.state for c in cache],
            self.hidden,
            incoming_hidden,
        )
        for current, incoming in zip(self.cache, cache):
            current.extend(incoming)
        self.hidden.extend(incoming_hidden)
        values = bonus.reshape(-1).tolist()
        self.bonus.extend(values)
        self.draft_caches.extend(self.draft.make_cache() for _ in values)

    def filter(self, keep):
        if len(keep) == len(self.bonus):
            return
        if keep:
            mx.eval([c.state for c in self.cache], self.hidden)
            indices = mx.array(keep, mx.int32)
            for cache in self.cache:
                cache.filter(indices)
        else:
            self.cache.clear()
        self.hidden = [self.hidden[i] for i in keep]
        self.bonus = [self.bonus[i] for i in keep]
        self.draft_caches = [self.draft_caches[i] for i in keep]

    def step(self, budgets, positions):
        if not budgets or min(budgets) <= 0:
            raise ValueError("Only live requests with remaining tokens can decode")
        block_size = min(self.block_size, max(budgets) + 1)
        sampler = self.sampler
        row_ids = [0] * len(budgets)
        if not self.greedy and self.positioned_sampling:
            sampler = _PositionedDraftSampler(
                sampler, row_ids=row_ids, positions=positions
            )
        draft_tokens = self.sampler_rng.draft_tokens(
            self.draft.draft_batch,
            self.bonus,
            self.hidden,
            self.draft_caches,
            block_size,
            sampler,
            self.dtype,
        )
        inputs = mx.concatenate(
            [mx.array(self.bonus, self.dtype)[:, None], draft_tokens], axis=1
        )
        transaction = None
        try:
            if self.greedy:
                captured, transaction, target_tokens = _dflash_verify_greedy(
                    self.model,
                    inputs,
                    self.cache,
                    list(self.draft.config.target_layer_ids),
                    self.sampler,
                )
            else:
                output, transaction = _dflash_verify(
                    self.model,
                    inputs,
                    self.cache,
                    list(self.draft.config.target_layer_ids),
                )
                captured = output.hidden_states
            hidden = mx.concatenate(captured, axis=-1)
            if self.greedy:
                mx.async_eval(target_tokens, hidden)
                accepted, tokens = _speculative_walk_batch(
                    draft_tokens, target_tokens, budgets
                )
            else:
                mx.async_eval(output.logits, hidden)
                accepted, tokens = _sample_dflash_target_walk(
                    output.logits,
                    draft_tokens,
                    self.sampler,
                    budgets,
                    row_ids=row_ids,
                    base_positions=positions,
                )
                self.sampler_rng.target_sampled(sync_draft=not self.positioned_sampling)
            # A clipped final round must never commit past the emitted prefix.
            accepted = [min(n, len(row) - 1) for n, row in zip(accepted, tokens)]
            commit_speculative_round(
                self.model, self.cache, transaction, accepted, block_size
            )
        except BaseException:
            abort_speculative_round(transaction)
            raise
        self.hidden = [hidden[i : i + 1, : len(row)] for i, row in enumerate(tokens)]
        self.bonus = [row[-1] for row in tokens]
        for count in accepted:
            _record_speculative_round(self.draft, count, block_size - 1)
        return tokens
