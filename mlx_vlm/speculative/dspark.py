"""DSpark self-speculative round loops (Gemma 4 target).

DSpark drafts a block of ``K`` tokens from the anchor's own position outward, biased by a
low-rank Markov head, then the target verifies ``[anchor, d_1..d_K]`` in one forward and a
greedy walk accepts the longest matching prefix. Output is bit-identical to base greedy
decoding for any draft proposal — the proposal only changes how many tokens are confirmed
per target forward.

When ``MLX_VLM_DRAFT_CONFIDENCE_THRESHOLD`` > 0, the (advisory) confidence head truncates the
draft block where ``P(accept)`` falls below the threshold. This only shortens *speculation
depth* before verify — the emitted stream stays lossless — so it trades acceptance for fewer
wasted draft positions. Confidence gating is applied on the single-sequence path; batched
continuous decoding uses a fixed block (still lossless).
"""

import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .common import (
    _dflash_block_total,
    _record_speculative_round,
    _speculative_walk,
    _speculative_walk_batch,
    generation_stream,
)
from .dflash import _dflash_committed_hidden_segments


def _dspark_confidence_threshold() -> float:
    raw = os.environ.get("MLX_VLM_DRAFT_CONFIDENCE_THRESHOLD")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _confident_prefix_length(
    confidence_logits: mx.array, block_size: int, threshold: float
) -> int:
    """Keep the draft prefix while ``sigmoid(confidence) >= threshold``.

    ``threshold <= 0`` disables gating (always the full block). Truncating here is lossless:
    the target still verifies whatever is drafted, so the emitted tokens are unchanged.
    """
    if threshold <= 0.0:
        return block_size
    probs = 1.0 / (
        1.0 + np.exp(-np.array(confidence_logits).astype(np.float32).reshape(-1))
    )
    below = np.nonzero(probs < threshold)[0]
    return int(below[0]) if below.size else block_size


def _require_rollback(lm) -> None:
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "DSpark speculative decoding requires a Gemma 4 (or compatible) target."
        )


def _dspark_rounds(
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
) -> Generator[Tuple[int, None], None, None]:
    """Single-sequence DSpark round loop: draft → (confidence gate) → verify → walk → rollback.

    ``generate_step`` handles prefill, sampling the first bonus token, and packaging the
    captured target hidden states into ``hidden``.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    _require_rollback(lm)

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    threshold = _dspark_confidence_threshold()
    draft_cache = draft_model.reset(model)

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        K = min(block_total, max_tokens - emitted)
        if K < 1:
            break

        drafts = draft_model.draft_block(
            b, hidden, draft_cache, K, sampler, token_dtype
        )
        if threshold > 0.0 and draft_model._last_confidence is not None:
            keff = _confident_prefix_length(draft_model._last_confidence, K, threshold)
            if keff < K:
                drafts = drafts[:, :keff]
        n_draft = int(drafts.shape[1])
        mx.async_eval(drafts)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), drafts], axis=1
            )
            verify_out = lm(
                verify_input, cache=prompt_cache, capture_layer_ids=target_layer_ids
            )
            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden)

        accepted, new_tokens = _speculative_walk(
            drafts, target_tokens, max_tokens - emitted
        )
        _record_speculative_round(draft_model, accepted, n_draft)

        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        if accepted < n_draft:
            hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < n_draft:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache,
                    getattr(verify_out, "gdn_states", None),
                    accepted,
                    n_draft + 1,
                )

        if emitted % 256 == 0:
            mx.clear_cache()


def _dspark_rounds_batch(
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
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DSpark round loop (B > 1) with continuous batching.

    Lossless, fixed block size (confidence gating is single-sequence only). When a sequence
    finishes it is filtered out of the target caches and dropped from the active set.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    _require_rollback(lm)

    B = first_bonus.shape[0]
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    draft_model.reset(model)
    draft_caches = [draft_model.make_cache() for _ in range(B)]

    b = first_bonus.tolist()
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))
    hidden_by_orig = [hidden[i : i + 1] for i in range(B)]
    total_emitted = sum(emitted)

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]]) for j in range(len(active_idx))
        ]
        K = min(block_total, min(remaining))
        if K < 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft rowwise: the drafter cache is scalar-offset; verify stays batched below.
        draft_tokens = mx.concatenate(
            [
                draft_model.draft_block(
                    int(b_active[j]),
                    hidden_by_orig[active_idx[j]],
                    draft_caches[active_idx[j]],
                    K,
                    sampler,
                    token_dtype,
                )
                for j in range(n_active)
            ],
            axis=0,
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input, cache=prompt_cache, capture_layer_ids=target_layer_ids
            )
            hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, target_tokens, budgets
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
            _record_speculative_round(draft_model, a, K)

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

        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if min_accepted < K:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache,
                    getattr(verify_out, "gdn_states", None),
                    accepted_arr,
                    K + 1,
                )

        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            if len(keep_slots) == 0:
                break
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_mx)
            active_idx = [active_idx[j] for j in keep_slots]

        new_total = sum(emitted)
        if new_total // 256 > total_emitted // 256:
            mx.clear_cache()
        total_emitted = new_total
