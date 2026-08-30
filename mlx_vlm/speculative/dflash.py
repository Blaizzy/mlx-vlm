from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .common import (
    _dflash_block_total,
    _record_speculative_round,
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_uniform_acceptance,
    _SpeculativeSamplerRNG,
    generation_stream,
)


def _dflash_next_block_size(
    draft_model: nn.Module,
    requested_block_total: int,
    remaining_budget: int,
    initial_block_size: Optional[int] = None,
) -> int:
    """Choose the next DFlash verify block size from recent acceptance.

    DFlash checkpoints advertise a trained block size, usually 16. Treat that
    as the ceiling and back off quickly when deeper positions are mostly
    rejected. When acceptance is strong at the current depth, grow back toward
    the configured ceiling.
    """
    block_total = min(requested_block_total, remaining_budget)
    if block_total <= 1:
        return block_total
    if getattr(draft_model, "prefer_requested_block_size", False):
        return block_total

    accept_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    recent = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-8:], draft_lens[-8:])
        if int(d) > 0
    ]
    if not recent:
        if initial_block_size is not None:
            return min(block_total, max(2, int(initial_block_size)))
        return block_total

    current = min(block_total, max(2, recent[-1][1] + 1))
    min_total = min(
        block_total,
        max(2, int(getattr(draft_model, "dflash_min_block_size", 4))),
    )
    drafted = sum(d for _, d in recent)
    accepted = sum(a for a, _ in recent)
    accept_rate = accepted / drafted
    mean_accept = accepted / len(recent)

    if accept_rate < 0.30 or mean_accept < 2.0:
        if current >= 8:
            return max(min_total, min(block_total, current // 2))
        return max(min_total, min(block_total, current - 2))

    if accept_rate < 0.50:
        return max(min_total, min(block_total, current - 2))

    full_hits = sum(1 for a, d in recent if a >= d)
    full_hit_rate = full_hits / len(recent)
    if accept_rate >= 0.85 and full_hit_rate >= 0.75:
        return min(block_total, current + 2)

    return min(block_total, current)


def _dflash_committed_hidden_segments(
    hidden_full: mx.array, new_tokens_list: List[List[int]]
) -> List[mx.array]:
    return [
        hidden_full[i : i + 1, : len(new_tokens), :]
        for i, new_tokens in enumerate(new_tokens_list)
    ]


def _target_requires_uniform_dflash_acceptance(model: nn.Module) -> bool:
    target = getattr(model, "language_model", model)
    return bool(getattr(target, "requires_uniform_dflash_acceptance", False))


def _dflash_uniform_acceptance(
    model: nn.Module,
    draft_tokens: mx.array,
    accepted_list: List[int],
    new_tokens_list: List[List[int]],
    budgets: List[int],
    target_tokens: Optional[mx.array] = None,
) -> Tuple[List[int], List[List[int]]]:
    if (
        len(accepted_list) <= 1
        or not _target_requires_uniform_dflash_acceptance(model)
        or len(set(accepted_list)) == 1
    ):
        return accepted_list, new_tokens_list

    if target_tokens is not None:
        return _speculative_walk_batch_uniform_acceptance(
            draft_tokens, target_tokens, accepted_list, budgets
        )

    accepted = min(accepted_list)
    draft_rows = draft_tokens.tolist()
    uniform_tokens = []
    for row, budget in enumerate(budgets):
        tokens = draft_rows[row][:accepted]
        if len(tokens) < budget:
            if accepted_list[row] == accepted:
                tokens.append(new_tokens_list[row][accepted])
            else:
                tokens.append(draft_rows[row][accepted])
        uniform_tokens.append(tokens[:budget])
    return [accepted] * len(accepted_list), uniform_tokens


def _supports_positioned_target_sampling(sampler: Callable) -> bool:
    return callable(getattr(sampler, "sample_target", None))


class _PositionedDraftSampler:
    def __init__(
        self,
        sampler: Callable,
        *,
        row_ids: List[int],
        positions: List[int],
    ):
        self.sampler = sampler
        self.row_ids = [int(row_id) for row_id in row_ids]
        self.positions = [int(position) for position in positions]

    def __call__(self, logits: mx.array) -> mx.array:
        if logits.ndim == 1:
            batch, length = 1, 1
        elif logits.ndim == 2:
            batch, length = logits.shape[0], 1
        else:
            batch, length = logits.shape[0], logits.shape[1]
        if batch != len(self.row_ids):
            raise ValueError(
                "Draft sampler row count does not match logits batch size."
            )

        rows = [row_id for row_id in self.row_ids for _ in range(length)]
        positions = [
            position + offset for position in self.positions for offset in range(length)
        ]
        sampled = self.sampler.sample_target(
            logits.reshape(batch * length, logits.shape[-1]),
            row_ids=rows,
            positions=positions,
        )
        self.positions = [position + length for position in self.positions]
        return sampled.reshape(logits.shape[:-1])

    def sample_proposal(self, logits: mx.array) -> mx.array:
        sample_proposal = getattr(self.sampler, "sample_proposal", None)
        if not callable(sample_proposal):
            return mx.argmax(logits, axis=-1)
        if logits.ndim == 1:
            batch, length = 1, 1
        elif logits.ndim == 2:
            batch, length = logits.shape[0], 1
        else:
            batch, length = logits.shape[0], logits.shape[1]
        if batch != len(self.row_ids):
            raise ValueError(
                "Draft sampler row count does not match logits batch size."
            )
        rows = [row_id for row_id in self.row_ids for _ in range(length)]
        positions = [
            position + offset for position in self.positions for offset in range(length)
        ]
        sampled = sample_proposal(
            logits.reshape(batch * length, logits.shape[-1]),
            row_ids=rows,
            positions=positions,
        )
        self.positions = [position + length for position in self.positions]
        return sampled.reshape(logits.shape[:-1])


def _sample_dflash_target_block(
    logits: mx.array,
    sampler: Callable[[mx.array], mx.array],
    *,
    row_ids: List[int],
    base_positions: List[int],
) -> mx.array:
    batch, length, vocab_size = logits.shape
    logprobs = _dflash_target_logprobs(logits)
    flat_logprobs = logprobs.reshape(batch * length, vocab_size)
    positions = [
        int(base_position) + position
        for base_position in base_positions
        for position in range(length)
    ]
    rows = [int(row_id) for row_id in row_ids for _ in range(length)]
    return sampler.sample_target(
        flat_logprobs,
        row_ids=rows,
        positions=positions,
    ).reshape(batch, length)


def _dflash_target_logprobs(logits: mx.array) -> mx.array:
    return mx.stack(
        [
            row - mx.logsumexp(row, axis=-1, keepdims=True)
            for row in (logits[:, position, :] for position in range(logits.shape[1]))
        ],
        axis=1,
    )


def _sample_dflash_target_walk(
    logits: mx.array,
    draft_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    budgets: List[int],
    *,
    row_ids: List[int],
    base_positions: List[int],
) -> Tuple[List[int], List[List[int]]]:
    if _supports_positioned_target_sampling(sampler):
        target_tokens = _sample_dflash_target_block(
            logits,
            sampler,
            row_ids=row_ids,
            base_positions=base_positions,
        )
        mx.async_eval(target_tokens)
        return _speculative_walk_batch(draft_tokens, target_tokens, budgets)

    batch, length, _ = logits.shape
    draft_count = int(draft_tokens.shape[1])
    draft_rows = draft_tokens.tolist()
    logprobs = _dflash_target_logprobs(logits)

    for position in range(length):
        target_tokens = sampler(logprobs[:, position, :])
        mx.eval(target_tokens)
        target_rows = [int(token) for token in target_tokens.reshape(-1).tolist()]
        if position < draft_count and all(
            target_rows[row] == draft_rows[row][position] for row in range(batch)
        ):
            continue

        new_tokens = []
        for row, budget in enumerate(budgets):
            tokens = draft_rows[row][:position]
            if len(tokens) < budget:
                tokens.append(target_rows[row])
            new_tokens.append(tokens[:budget])
        return [position] * batch, new_tokens

    return [draft_count] * batch, [
        draft_rows[row][: budgets[row]] for row in range(batch)
    ]


def _dflash_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    use_model_initial_block_size: bool = True,
    greedy_sampling: bool = True,
) -> Generator[Tuple[int, None], None, None]:
    """DFlash speculative-decoding **round loop**.

    draft → verify → walk → rollback. ``generate_step`` is responsible
    for prefill, sampling the first bonus token, and packaging the
    captured hidden states into ``hidden``.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "This target does not currently support DFlash speculative decoding."
        )

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    draft_cache = draft_model.reset(model)
    positioned_sampling = _supports_positioned_target_sampling(sampler)
    sampler_rng = _SpeculativeSamplerRNG(
        draft_model,
        enabled=not greedy_sampling and not positioned_sampling,
    )
    prepare_target_hidden = getattr(draft_model, "prepare_target_hidden", None)
    hidden_is_prepared = callable(prepare_target_hidden)
    if hidden_is_prepared:
        hidden = prepare_target_hidden(hidden)
        mx.async_eval(hidden)

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        bs = _dflash_next_block_size(
            draft_model,
            block_total,
            max_tokens - emitted + 1,
            (
                getattr(draft_model, "dflash_initial_block_size", None)
                if use_model_initial_block_size
                else None
            ),
        )
        if bs <= 1:
            break

        draft_kwargs = {"target_hidden_prepared": True} if hidden_is_prepared else {}
        draft_sampler = (
            _PositionedDraftSampler(
                sampler,
                row_ids=[0],
                positions=[emitted],
            )
            if not greedy_sampling and positioned_sampling
            else sampler
        )
        draft_tokens = sampler_rng.draft_tokens(
            draft_model.draft_block,
            b,
            hidden,
            draft_cache,
            bs,
            draft_sampler,
            token_dtype,
            **draft_kwargs,
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens],
                axis=1,
            )
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
                speculative_verify=True,
            )
            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            if greedy_sampling:
                target_tokens = sampler(verify_out.logits)
        if greedy_sampling:
            mx.async_eval(target_tokens, hidden)
        else:
            mx.async_eval(hidden)

        if greedy_sampling:
            accepted, new_tokens = _speculative_walk(
                draft_tokens, target_tokens, max_tokens - emitted
            )
        else:
            accepted_list, new_tokens_list = _sample_dflash_target_walk(
                verify_out.logits,
                draft_tokens,
                sampler,
                [max_tokens - emitted],
                row_ids=[0],
                base_positions=[emitted],
            )
            accepted = accepted_list[0]
            new_tokens = new_tokens_list[0]
            sampler_rng.target_sampled(sync_draft=not positioned_sampling)
        _record_speculative_round(draft_model, accepted, bs - 1)

        if accepted < bs - 1:
            hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted, bs
                )

        if hidden_is_prepared and emitted + len(new_tokens) < max_tokens:
            hidden = prepare_target_hidden(hidden)
            mx.async_eval(hidden)

        # Emit after scheduling the next context projection so its execution
        # can overlap server-side detokenization and response handling.
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        verify_out = None


def _dflash_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    greedy_sampling: bool = True,
    row_ids: Optional[List[int]] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DFlash speculative-decoding round loop (B > 1).

    Supports continuous batching: when a sequence finishes (EOS or
    max_tokens), it is filtered out of the target caches and the
    drafter cache is reinitialized for the new batch size.

    ``stop_check(seq_idx, token_id) -> bool`` is an optional callback
    that returns True to stop a sequence (e.g. EOS detection).

    Yields ``(tokens_list, None)`` where ``tokens_list[i]`` is the
    token for sequence ``i`` (or ``None`` if that sequence has nothing
    to emit this step).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement " "rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    row_ids = list(range(B)) if row_ids is None else list(row_ids)
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    draft_model.reset(model)
    positioned_sampling = _supports_positioned_target_sampling(sampler)
    sampler_rng = _SpeculativeSamplerRNG(
        draft_model,
        enabled=not greedy_sampling and not positioned_sampling,
    )
    draft_caches = [draft_model.make_cache() for _ in range(B)]

    # Per-sequence state tracked by ORIGINAL index so the caller sees
    # stable indices in the yielded token lists.
    b = first_bonus.tolist()  # active bonus tokens
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))  # maps active-slot → original-index
    hidden_by_orig = [hidden[i : i + 1] for i in range(B)]

    total_emitted = sum(emitted)

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = _dflash_next_block_size(draft_model, block_total, min(remaining))
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft rowwise: the DFlash drafter cache is scalar-offset and has
        # proven unsafe as a single batched cache on MLX/Metal. Target verify
        # remains batched below.
        def draft_active_rows():
            return mx.concatenate(
                [
                    draft_model.draft_block(
                        int(b_active[j]),
                        hidden_by_orig[active_idx[j]],
                        draft_caches[active_idx[j]],
                        bs,
                        (
                            _PositionedDraftSampler(
                                sampler,
                                row_ids=[row_ids[active_idx[j]]],
                                positions=[emitted[active_idx[j]]],
                            )
                            if not greedy_sampling and positioned_sampling
                            else sampler
                        ),
                        token_dtype,
                    )
                    for j in range(n_active)
                ],
                axis=0,
            )

        draft_tokens = sampler_rng.draft_tokens(
            draft_active_rows,
        )

        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
                speculative_verify=True,
            )
            hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            if greedy_sampling:
                target_tokens = sampler(verify_out.logits)
        if greedy_sampling:
            mx.async_eval(target_tokens, hidden_full)
        else:
            mx.async_eval(hidden_full)

        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        if greedy_sampling:
            accepted_list, new_tokens_list = _speculative_walk_batch(
                draft_tokens, target_tokens, budgets
            )
        else:
            accepted_list, new_tokens_list = _sample_dflash_target_walk(
                verify_out.logits,
                draft_tokens,
                sampler,
                budgets,
                row_ids=[row_ids[active_idx[j]] for j in range(n_active)],
                base_positions=[emitted[active_idx[j]] for j in range(n_active)],
            )
            sampler_rng.target_sampled(sync_draft=not positioned_sampling)

        accepted_list, new_tokens_list = _dflash_uniform_acceptance(
            model,
            draft_tokens,
            accepted_list,
            new_tokens_list,
            budgets,
            target_tokens if greedy_sampling else None,
        )

        min_accepted = min(accepted_list)
        accepted_arr = mx.array(accepted_list)

        hidden_segments = _dflash_committed_hidden_segments(
            hidden_full, new_tokens_list
        )
        for j in range(n_active):
            orig = active_idx[j]
            if hidden_segments[j].shape[1] > 0:
                hidden_by_orig[orig] = hidden_segments[j]

        for a in accepted_list:
            _record_speculative_round(draft_model, a, bs - 1)

        # Emit (map active slots back to original indices)
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if min_accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted_arr, bs
                )

        # --- Continuous batching: filter out finished sequences ---
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            if len(keep_slots) == 0:
                break
            # Filter target caches (BatchKVCache supports this)
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_mx)
            # Update active index mapping
            active_idx = [active_idx[j] for j in keep_slots]

        verify_out = None
        total_emitted = sum(emitted)
