import os
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .common import (
    _dflash_block_total,
    _record_speculative_round,
    _speculative_walk,
    _speculative_walk_batch,
    generation_stream,
)


def _env_flag(name: str, default: bool) -> bool:
    """Call-time env flag with a default. Deliberately NOT hoisted to import
    time so benches can toggle per-arm in-process (2026-08-11)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


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
    min_total = min(block_total, 4)
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


# --- Round-1 drafter-context assembly (2026-08-11) --------------------------
#
# RoPE-offset design (Fixes 1+2). The DFlash drafter is trained with ABSOLUTE
# positions: HF's DFlashTokenCandidateGenerator sets noise_position_ids =
# arange + last_ctx_pos + 1, and context keys carry their true sequence
# positions. The z-lab MLX reference (model_mlx.py:85-91) keeps that invariant
# when it drops context: it slices the round-1 context to the last
# sliding_window-1 positions and bumps its drafter cache offset by the number
# skipped, so kept keys still get absolute RoPE positions.
#
# We implement the same semantics, but the shift is carried as an additive
# ``pos_shift`` attribute on each drafter layer cache (read by the DFlash
# drafter attentions: rope offset = cache.offset + pos_shift) instead of
# mutating cache.offset directly. Rationale: RotatingKVCache's ``offset``
# doubles as storage bookkeeping — pre-seeding it while the buffer is not
# exactly full corrupts _update_in_place (growth size ``max_size - offset``
# can go negative, and the write index rewinds to ``offset``, past the buffer
# end, leaving attended zero-KV garbage). The additive shift keeps every cache
# code path byte-identical to the validated cold flow, and shifts compose: an
# APC warm hit contributes cached_tokens, a round-1 truncation adds the
# skipped count.


def _drafter_pos_shift(cache_list, extra: int) -> None:
    """Add ``extra`` to each drafter layer-cache's absolute-position shift."""
    for c in cache_list:
        c.pos_shift = int(getattr(c, "pos_shift", 0)) + int(extra)


def _round1_window_limit(draft_model: nn.Module) -> Optional[int]:
    """Max round-1 context positions = sliding_window - 1 (the drafter cache's
    max_size), or None when the drafter has no window (nothing to enforce) or
    MUSE_ROUND1_TRUNC=0 (call-time A/B kill switch restoring the old
    full-length round-1 context)."""
    if not _env_flag("MUSE_ROUND1_TRUNC", True):
        return None
    window = getattr(getattr(draft_model, "config", None), "sliding_window", None)
    if not window or int(window) <= 1:
        return None
    return int(window) - 1


def _apply_round1_window(
    hidden: mx.array, cache_list, keep: int
) -> mx.array:
    """Enforce the drafter's training-time sliding window on the round-1
    context: keep the LAST ``keep`` positions and bump ``pos_shift`` by the
    number skipped so the kept keys' RoPE positions stay absolute. Off-window
    context is off-training-distribution AND transiently stores a 5-layer x
    prompt_len KV spike in the rotating cache's first _update_concat.
    Continuation rounds carry accepted+1 positions (far below the window) and
    are naturally unaffected."""
    skipped = int(hidden.shape[1]) - int(keep)
    if skipped <= 0:
        return hidden
    _drafter_pos_shift(cache_list, skipped)
    return hidden[:, -int(keep):, :]


def assemble_warm_drafter_context(
    hidden_states: mx.array,
    right_pad: List[int],
    prefix_lens: List[int],
):
    """APC warm-hit drafter context (Fix 1, 2026-08-11).

    After a warm hit only the suffix (prompt_len - cached) was prefilled, and
    the forward captured aux hidden states for ALL suffix positions — but the
    old code handed the drafter a single position, versus ~2047 in the z-lab/HF
    reference flow (the top acceptance lever found by the reference audit).

    Default (MUSE_WARMCTX unset/on): returns ``(rows, context_offsets)`` where
    ``rows[i]`` is row i's FULL un-padded suffix hidden ``[1, S_i, D]`` (ragged
    per-row is fine — drafting is row-wise in the batched loop) and
    ``context_offsets[i] = prefix_lens[i]`` seeds the drafter cache's absolute
    position shift, so suffix context keys get RoPE positions
    cached..cached+S_i-1 exactly as the cold path gives 0..P-1.

    MUSE_WARMCTX=0 (call-time read, per-request A/B): the exact old behavior —
    one hidden position per row (each row's last real position) and no offsets,
    byte-identical to the pre-fix take_along_axis slice.
    """
    B = int(hidden_states.shape[0])
    seq = int(hidden_states.shape[1])
    if _env_flag("MUSE_WARMCTX", True):
        rows = [
            hidden_states[i : i + 1, : seq - int(right_pad[i]), :]
            for i in range(B)
        ]
        return rows, [int(p) for p in prefix_lens]
    li = mx.array([seq - 1 - int(rp) for rp in right_pad], dtype=mx.int32)
    idx = mx.broadcast_to(
        li[:, None, None], (B, 1, int(hidden_states.shape[-1]))
    )
    return mx.take_along_axis(hidden_states, idx, axis=1), None


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
        draft_tokens = draft_model.draft_block(
            b,
            hidden,
            draft_cache,
            bs,
            sampler,
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
            )
            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden)

        # Walk
        accepted, new_tokens = _speculative_walk(
            draft_tokens, target_tokens, max_tokens - emitted
        )
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
    context_offsets: Optional[List[int]] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DFlash speculative-decoding round loop (B > 1).

    Supports continuous batching: when a sequence finishes (EOS or
    max_tokens), it is filtered out of the target caches and the
    drafter cache is reinitialized for the new batch size.

    ``stop_check(seq_idx, token_id) -> bool`` is an optional callback
    that returns True to stop a sequence (e.g. EOS detection).

    ``hidden`` may be a single ``[B, S, D]`` array (cold path) or a per-row
    ragged list of ``[1, S_i, D]`` arrays (APC warm hits with per-row suffix
    lengths — drafting is row-wise, so raggedness is free here).

    ``context_offsets[i]``: absolute position of row i's first context
    position (cached_tokens on a warm hit); seeds that row's drafter-cache
    RoPE position shift. None/0 == historical behavior.

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
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    draft_model.reset(model)
    draft_caches = [draft_model.make_cache() for _ in range(B)]

    # Per-sequence state tracked by ORIGINAL index so the caller sees
    # stable indices in the yielded token lists.
    b = first_bonus.tolist()  # active bonus tokens
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))  # maps active-slot → original-index
    if isinstance(hidden, (list, tuple)):
        hidden_by_orig = [hidden[i] for i in range(B)]
    else:
        hidden_by_orig = [hidden[i : i + 1] for i in range(B)]
    if context_offsets:
        for i in range(B):
            off = int(context_offsets[i]) if i < len(context_offsets) else 0
            if off > 0:
                _drafter_pos_shift(draft_caches[i], off)
    # Round-1 window truncation (see _apply_round1_window). Runs AFTER the
    # warm-context offsets above so a warm suffix longer than the window
    # composes: pos_shift = cached_tokens + skipped.
    _keep = _round1_window_limit(draft_model)
    if _keep is not None:
        for i in range(B):
            hidden_by_orig[i] = _apply_round1_window(
                hidden_by_orig[i], draft_caches[i], _keep
            )

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
        draft_tokens = mx.concatenate(
            [
                draft_model.draft_block(
                    int(b_active[j]),
                    hidden_by_orig[active_idx[j]],
                    draft_caches[active_idx[j]],
                    bs,
                    sampler,
                    token_dtype,
                )
                for j in range(n_active)
            ],
            axis=0,
        )
        mx.async_eval(draft_tokens)

        # Verify
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
            )
            hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        # Walk (per-sequence)
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
