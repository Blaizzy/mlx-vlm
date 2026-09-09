"""MTP proposal / target verification / prefix acceptance / cache commit."""

from functools import partial

import mlx.core as mx

from ..models.cache import BatchKVCache, BatchQuantizedKVCache
from .cache_state import SpeculativeCache, iter_leaf_caches
from .sampling import accept_greedy, accept_sampled
from .stats import record_round


def mtp_rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    prompt_tokens,
    first_bonus,
    max_tokens,
    sampler,
    draft_block_size=None,
    token_dtype=mx.int32,
    stop_check=None,
    eos_token_ids=None,
    greedy_sampling=False,
    row_ids=None,
):
    """Yield one token per live row. ``max_tokens`` includes ``first_bonus``.

    The drafter is deterministic, so matching a draw from the target is exact
    rejection sampling for a point-mass proposal. Sampling needs no draft RNG.
    Both batch-one and ragged batches use this same loop.
    """
    batch = first_bonus.size
    limits = [max_tokens] * batch if isinstance(max_tokens, int) else list(max_tokens)
    if len(limits) != batch:
        raise ValueError("max_tokens must provide one limit per row.")
    if max(limits, default=0) <= 1:
        return
    count = (
        draft_model.config.block_size if draft_block_size is None else draft_block_size
    ) - 1
    if count < 1:
        raise ValueError(
            "MTP block size must be at least 2 (one draft plus one target token)."
        )
    target = getattr(model, "language_model", model)
    row_ids = list(range(batch)) if row_ids is None else row_ids
    padding = None
    for entry in iter_leaf_caches(prompt_cache):
        if isinstance(entry, (BatchKVCache, BatchQuantizedKVCache)):
            padding = entry.left_padding.tolist()
            break
    if batch > 1 and padding is None:
        raise ValueError("Batched MTP requires batch prompt caches.")
    state = SpeculativeCache(
        prompt_cache,
        draft_model.make_cache(padding),
        [0] * batch if padding is None else [-p for p in padding],
        first_bonus.astype(token_dtype),
    )
    forward = partial(draft_model, target_model=target)
    state.prefill(prompt_tokens, hidden, forward)
    produced = [1] * batch
    stopped = [False] * batch
    eos = eos_token_ids or set()
    for row, token in enumerate(first_bonus.reshape(-1).tolist()):
        stopped[row] = token in eos or bool(stop_check and stop_check(row, token))

    try:
        while any(
            not done and n < limit for done, n, limit in zip(stopped, produced, limits)
        ):
            budgets = [
                0 if stopped[i] else max(0, limits[i] - n)
                for i, n in enumerate(produced)
            ]
            depth = min(count, max(budgets) - 1)
            proposals = state.propose(depth, forward)
            output = target(
                state.verify_inputs(proposals), cache=state.target, return_hidden=True
            )
            state.record_verification(output.hidden_states[-1])
            if greedy_sampling:
                rows = accept_greedy(proposals, output.logits, budgets)
            else:
                rows = accept_sampled(
                    proposals, output.logits, budgets, sampler, row_ids, produced
                )
            for row, values in enumerate(rows):
                for pos, token in enumerate(values):
                    if token in eos or (stop_check and stop_check(row, token)):
                        rows[row] = values[: pos + 1]
                        stopped[row] = True
                        break
            emitted = [[] for _ in rows]
            width = max(map(len, rows))
            committed = False
            try:
                for pos in range(width):
                    tokens = []
                    for row, values in enumerate(rows):
                        token = values[pos] if pos < len(values) else None
                        tokens.append(token)
                        if token is not None:
                            emitted[row].append(token)
                            produced[row] += 1
                    if pos + 1 == width:
                        state.commit(emitted, forward)
                        record_round(draft_model, proposals, emitted)
                        committed = True
                    yield tokens, {"round_pos": pos, "round_len": width}
            finally:
                if not committed:
                    state.commit(emitted, forward)
                    record_round(draft_model, proposals, emitted)
    finally:
        state.abort()
