from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models import cache

generation_stream = mx.new_thread_local_stream(mx.default_device())


def _speculative_walk(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budget: int,
) -> Tuple[int, List[int]]:
    """Exact-greedy speculative-decoding walk.

    Accept drafted tokens up to the first mismatch with the target's
    greedy choice, then take the target's bonus at that position.
    Returns ``(accepted_count, new_tokens)`` with ``new_tokens``
    truncated to ``budget``.
    """
    n_draft = draft_tokens.shape[1]
    mismatch = draft_tokens[:, :n_draft] != target_tokens[:, :n_draft]
    mismatch = mismatch.reshape(-1)
    has_mismatch = bool(mx.any(mismatch).item())
    accepted = (
        int(mx.argmax(mismatch.astype(mx.int32)).item()) if has_mismatch else n_draft
    )
    accepted_prefix = draft_tokens[:, :accepted]
    bonus = target_tokens[:, accepted : accepted + 1]
    new_tokens = (
        mx.concatenate([accepted_prefix, bonus], axis=1)[:, :budget]
        .reshape(-1)
        .tolist()
    )
    return accepted, new_tokens


def _speculative_walk_batch(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Per-sequence speculative walk for B > 1.

    Returns ``(accepted_list, new_tokens_list)`` where each entry
    corresponds to one sequence in the batch.
    """
    B = draft_tokens.shape[0]
    n_draft = draft_tokens.shape[1]
    mismatch = draft_tokens[:, :n_draft] != target_tokens[:, :n_draft]
    has_mismatch = mx.any(mismatch, axis=1)
    first_mismatch = mx.argmax(mismatch.astype(mx.int32), axis=1)
    accepted_arr = mx.where(
        has_mismatch,
        first_mismatch,
        mx.full((B,), n_draft, dtype=mx.int32),
    )
    accepted_list = [int(x) for x in accepted_arr.tolist()]
    new_tokens_list: List[List[int]] = []
    for i in range(B):
        accepted = accepted_list[i]
        accepted_prefix = draft_tokens[i : i + 1, :accepted]
        bonus = target_tokens[i : i + 1, accepted : accepted + 1]
        new = (
            mx.concatenate([accepted_prefix, bonus], axis=1)[:, : budgets[i]]
            .reshape(-1)
            .tolist()
        )
        new_tokens_list.append(new)
    return accepted_list, new_tokens_list


def _speculative_walk_batch_uniform_acceptance(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    accepted_list: List[int],
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Clamp a batch to the earliest rejection with verifier-token fallback."""
    accepted = min(accepted_list)
    new_tokens_list: List[List[int]] = []
    for i, budget in enumerate(budgets):
        accepted_prefix = draft_tokens[i : i + 1, :accepted]
        bonus = target_tokens[i : i + 1, accepted : accepted + 1]
        new = (
            mx.concatenate([accepted_prefix, bonus], axis=1)[:, :budget]
            .reshape(-1)
            .tolist()
        )
        new_tokens_list.append(new)
    return [accepted] * len(accepted_list), new_tokens_list


@dataclass
class _MTPVerifyResult:
    hidden: mx.array
    shared_kv_states: dict
    target_tokens: Optional[mx.array] = None
    gdn_states: Optional[list] = None


def _mtp_shared_kv_from_prompt_cache(
    lm: nn.Module,
    prompt_cache: List[Any],
) -> dict:
    layers = getattr(getattr(lm, "model", None), "layers", [])
    if len(prompt_cache) != len(layers):
        return {}

    shared_kv_states = {}
    for layer, layer_cache in zip(layers, prompt_cache):
        if layer_cache is None or not hasattr(layer_cache, "state"):
            continue
        state = layer_cache.state
        if state is None or len(state) < 2:
            continue
        keys, values = state[:2]
        if keys is None or values is None:
            continue
        if (
            isinstance(layer_cache, cache.RotatingKVCache)
            and not isinstance(layer_cache, cache.BufferedRotatingKVCache)
            and hasattr(layer_cache, "_temporal_order")
        ):
            keys = layer_cache._temporal_order(keys)
            values = layer_cache._temporal_order(values)
        shared_kv_states[layer.layer_type] = (keys, values)
    return shared_kv_states


def _mtp_verify_without_logits(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
) -> Optional[_MTPVerifyResult]:
    verify_hidden = getattr(lm, "speculative_verify_hidden", None)
    if callable(verify_hidden):
        result = verify_hidden(verify_input, prompt_cache)
        if isinstance(result, tuple):
            if len(result) == 3:
                hidden, shared_kv_states, gdn_states = result
            elif len(result) == 2:
                hidden, shared_kv_states = result
                gdn_states = None
            else:
                raise ValueError(
                    "speculative_verify_hidden() must return "
                    "(hidden, shared_kv_states) or "
                    "(hidden, shared_kv_states, gdn_states)."
                )
        else:
            hidden = result
            shared_kv_states = {}
            gdn_states = None
        return _MTPVerifyResult(
            hidden=hidden,
            shared_kv_states=shared_kv_states or {},
            gdn_states=gdn_states,
        )

    layers = getattr(getattr(lm, "model", None), "layers", [])
    if len(prompt_cache) == len(layers):
        hidden = lm.model(
            verify_input,
            cache=prompt_cache,
            skip_final_norm=True,
        )
        shared_kv_states = _mtp_shared_kv_from_prompt_cache(lm, prompt_cache)
        if shared_kv_states:
            return _MTPVerifyResult(hidden=hidden, shared_kv_states=shared_kv_states)

    shared_kv_sink: dict = {}
    hidden = lm.model(
        verify_input,
        cache=prompt_cache,
        shared_kv_sink=shared_kv_sink,
        skip_final_norm=True,
    )
    if not shared_kv_sink:
        return None
    return _MTPVerifyResult(hidden=hidden, shared_kv_states=shared_kv_sink)


def _mtp_verify_with_model_method(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
    sampler: Callable[[mx.array], mx.array],
) -> Optional[_MTPVerifyResult]:
    verify_logits = getattr(lm, "speculative_verify_logits", None)
    if not callable(verify_logits):
        return None

    result = verify_logits(verify_input, prompt_cache, sampler)
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError(
            "speculative_verify_logits() must return "
            "(hidden, shared_kv_states, gdn_states, target_tokens)."
        )

    hidden, shared_kv_states, gdn_states, target_tokens = result
    return _MTPVerifyResult(
        hidden=hidden,
        shared_kv_states=shared_kv_states or {},
        target_tokens=target_tokens,
        gdn_states=gdn_states,
    )


def _mtp_verify_target(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
    sampler: Callable[[mx.array], mx.array],
) -> _MTPVerifyResult:
    result = _mtp_verify_with_model_method(lm, verify_input, prompt_cache, sampler)
    if result is not None:
        return result

    if hasattr(lm, "speculative_logits_from_hidden"):
        result = _mtp_verify_without_logits(lm, verify_input, prompt_cache)
        if result is not None:
            return result

    verify_out = lm(
        verify_input,
        cache=prompt_cache,
        return_hidden=True,
        return_shared_kv=True,
    )
    return _MTPVerifyResult(
        hidden=verify_out.hidden_states[-1],
        shared_kv_states=verify_out.shared_kv_states,
        target_tokens=sampler(verify_out.logits),
        gdn_states=verify_out.gdn_states,
    )


def _mtp_draft_hidden(lm: nn.Module, hidden: mx.array) -> mx.array:
    prepare = getattr(lm, "speculative_draft_hidden", None)
    return prepare(hidden) if callable(prepare) else hidden


def _speculative_walk_deferred_greedy(
    lm: nn.Module,
    target_hidden: mx.array,
    draft_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    budget: int,
) -> Tuple[int, List[int]]:
    """Greedy MTP walk that projects target logits only until rejection."""
    n_draft = draft_tokens.shape[1]
    draft_list = [int(x) for x in draft_tokens.reshape(-1).tolist()]
    accepted = 0
    new_tokens: List[int] = []

    for pos in range(n_draft + 1):
        with mx.stream(generation_stream):
            logits = lm.speculative_logits_from_hidden(
                target_hidden[:, pos : pos + 1, :]
            )
            target_token = sampler(logits)
        mx.eval(target_token)
        token = int(target_token.reshape(-1).item())

        if pos < n_draft and token == draft_list[pos]:
            accepted += 1
            if len(new_tokens) < budget:
                new_tokens.append(token)
            continue

        if len(new_tokens) < budget:
            new_tokens.append(token)
        break

    return accepted, new_tokens


def _speculative_walk_batch_deferred_greedy(
    lm: nn.Module,
    target_hidden: mx.array,
    draft_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Batched greedy walk that projects target logits only until all rows stop."""
    B = draft_tokens.shape[0]
    n_draft = draft_tokens.shape[1]
    draft_lists = [[int(token) for token in row] for row in draft_tokens.tolist()]
    budgets = [int(budget) for budget in budgets]
    accepted = [0] * B
    new_tokens: List[List[int]] = [[] for _ in range(B)]
    done = [False] * B

    for pos in range(n_draft + 1):
        if all(done):
            break
        with mx.stream(generation_stream):
            logits = lm.speculative_logits_from_hidden(
                target_hidden[:, pos : pos + 1, :]
            )
            target_tokens = sampler(logits)
        mx.eval(target_tokens)
        target_list = [int(token) for token in target_tokens.reshape(-1).tolist()]

        for row, token in enumerate(target_list):
            if done[row]:
                continue
            if pos < n_draft and token == draft_lists[row][pos]:
                accepted[row] += 1
                if len(new_tokens[row]) < budgets[row]:
                    new_tokens[row].append(token)
                continue

            if len(new_tokens[row]) < budgets[row]:
                new_tokens[row].append(token)
            done[row] = True

    return accepted, new_tokens


def _mtp_acceptance_walk(
    lm: nn.Module,
    verify: _MTPVerifyResult,
    draft_tokens: mx.array,
    sampler: Callable[[mx.array], mx.array],
    budget: int,
) -> Tuple[int, List[int]]:
    if verify.target_tokens is not None:
        mx.async_eval(verify.target_tokens, verify.hidden)
        return _speculative_walk(draft_tokens, verify.target_tokens, budget)

    mx.async_eval(verify.hidden)
    return _speculative_walk_deferred_greedy(
        lm,
        verify.hidden,
        draft_tokens,
        sampler,
        budget,
    )


def _slice_shared_kv_after_reject(shared_kv_states: dict, rejected: int) -> dict:
    if rejected <= 0:
        return shared_kv_states

    next_shared_kv = {}
    for k, kv in shared_kv_states.items():
        K, V = kv
        valid = K.shape[-2] - rejected
        if valid <= 0 or valid >= K.shape[-2]:
            next_shared_kv[k] = (
                (K, V) if valid >= K.shape[-2] else (K[..., :1, :], V[..., :1, :])
            )
        else:
            next_shared_kv[k] = (K[..., :valid, :], V[..., :valid, :])
    return next_shared_kv


def _record_speculative_round(
    draft_model: nn.Module, accepted: float, draft_count: int
) -> None:
    draft_model.accept_lens.append(accepted)
    if hasattr(draft_model, "draft_lens"):
        draft_model.draft_lens.append(int(draft_count))


def _dflash_block_total(draft_model: nn.Module, draft_block_size: Optional[int]) -> int:
    if draft_block_size is not None:
        return int(draft_block_size)

    configured = int(draft_model.config.block_size)
    runtime = getattr(draft_model.config, "runtime_block_size", None)
    if runtime is None:
        return configured
    return min(configured, max(1, int(runtime)))


def _dflash_next_block_size(
    draft_model: nn.Module,
    requested_block_total: int,
    remaining_budget: int,
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


def _format_speculative_stats(draft_model: nn.Module) -> Optional[str]:
    accepted_lens = getattr(draft_model, "accept_lens", None) or []
    if not accepted_lens:
        return None

    rounds = len(accepted_lens)
    mean_accept = sum(accepted_lens) / rounds
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    if len(draft_lens) == rounds and sum(draft_lens) > 0:
        accept_rate = 100 * sum(accepted_lens) / sum(draft_lens)
        mean_draft = sum(draft_lens) / rounds
        return (
            "Speculative decoding: "
            f"{mean_accept:.2f} accepted tokens/round "
            f"({accept_rate:.1f}% of drafted, "
            f"avg draft {mean_draft:.2f}) over {rounds} rounds"
        )

    return (
        "Speculative decoding: "
        f"{mean_accept:.2f} accepted tokens over {rounds} rounds"
    )


def _effective_mtp_block_size(
    requested_block_total: int,
    configured_block_total: int,
    accept_lens: List[int],
    remaining_budget: int,
) -> int:
    """Choose the MTP block size for the next round.

    Treat user-provided block sizes above the assistant's configured depth as a
    ceiling. Larger tails are useful only if the prefix reaches the configured
    depth often enough; otherwise each round pays extra autoregressive drafter
    forwards for tokens that cannot be accepted.
    """
    block_total = min(requested_block_total, remaining_budget)
    configured_block_total = min(configured_block_total, block_total)
    if block_total <= configured_block_total or configured_block_total <= 1:
        return block_total

    if len(accept_lens) < 8:
        return configured_block_total

    recent = accept_lens[-32:]
    configured_draft_count = configured_block_total - 1
    configured_prefix_hits = sum(
        1 for accepted in recent if accepted >= configured_draft_count
    )
    configured_prefix_hit_rate = configured_prefix_hits / len(recent)
    if configured_prefix_hit_rate < 0.65:
        return configured_block_total

    return block_total


def _mtp_next_block_size(
    draft_model: nn.Module,
    requested_block_total: int,
    configured_block_total: int,
    remaining_budget: int,
) -> int:
    if getattr(draft_model, "prefer_requested_block_size", False):
        return min(requested_block_total, remaining_budget)
    return _effective_mtp_block_size(
        requested_block_total,
        configured_block_total,
        draft_model.accept_lens,
        remaining_budget,
    )


def _buffer_mtp_target_cache(
    prompt_cache: List[Any],
    draft_model: nn.Module,
    draft_block_size: Optional[int],
) -> None:
    configured = int(getattr(draft_model.config, "block_size", draft_block_size or 1))
    requested = int(draft_block_size or configured)
    buffer_size = max(32, min(128, max(configured, requested) * 8))

    for idx, entry in enumerate(prompt_cache):
        if isinstance(entry, cache.BufferedRotatingKVCache):
            entry.buffer_size = max(entry.buffer_size, buffer_size)
        elif (
            isinstance(entry, cache.RotatingKVCache) and getattr(entry, "keep", 0) == 0
        ):
            prompt_cache[idx] = cache.BufferedRotatingKVCache.from_cache(
                entry, buffer_size=buffer_size
            )


def _mtp_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    prompt_tokens: Optional[mx.array] = None,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    greedy_sampling: bool = False,
) -> Generator[Tuple[int, None], None, None]:
    """Gemma 4 MTP (Single-Position Multi-Token) speculative-decoding round loop.

    Mirrors ``_dflash_rounds`` but with three differences:
    (1) the drafter consumes the target's last-layer hidden + last-layer
    shared K/V per layer-type rather than concatenated multi-layer hiddens;
    (2) ``draft_block`` is autoregressive (K small forwards) rather than a
    single masked forward; (3) ``rollback_speculative_cache`` ignores
    ``gdn_states`` (Gemma 4 has no SSM/GDN state).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "MTP speculative decoding currently only supports gemma4."
        )

    block_total = _dflash_block_total(draft_model, draft_block_size)
    configured_block_total = int(getattr(draft_model.config, "block_size", block_total))
    draft_model.reset(model)

    # Hidden from prefill is full prompt-length; reduce to a single slot.
    # The semantically-correct choice is the *last* prompt token's hidden:
    # the just-sampled bonus is the next-token prediction from that position,
    # so its embedding paired with that hidden is what the drafter expects.
    # (HF's literal ``[:, n_last_matches:n_last_matches+1]`` with ``n_matches=0``
    # on the first round picks position 0, which is BOS — they get away with
    # it because subsequent rounds slice into the per-call verify hidden, but
    # the round-1 acceptance is wasted. We don't replicate that quirk.)
    prefill_draft = getattr(draft_model, "prefill_from_target_hidden", None)
    if callable(prefill_draft) and prompt_tokens is not None:
        prefill_draft(
            prompt_tokens,
            hidden,
            first_bonus,
            sampler,
            token_dtype,
            **_mtp_draft_kwargs(draft_model, greedy_sampling),
        )

    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    hidden = _mtp_draft_hidden(lm, hidden)

    kv_offset = _mtp_cache_offset_max(prompt_cache)
    draft_model.set_shared_kv(
        shared_kv_states,
        kv_offset,
        position=_mtp_draft_position(kv_offset),
        kv_valid_len=kv_offset,
    )

    b = first_bonus
    emitted = 1  # caller already yielded the first bonus

    while emitted < max_tokens:
        bs = _mtp_next_block_size(
            draft_model,
            block_total,
            configured_block_total,
            max_tokens - emitted + 1,
        )
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b,
            hidden,
            None,
            bs,
            sampler,
            token_dtype,
            **_mtp_draft_kwargs(draft_model, greedy_sampling),
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
            )
            verify = _mtp_verify_target(
                lm,
                verify_input,
                prompt_cache,
                sampler,
            )
        accepted, new_tokens = _mtp_acceptance_walk(
            lm,
            verify,
            draft_tokens,
            sampler,
            max_tokens - emitted,
        )
        _record_speculative_round(draft_model, accepted, bs - 1)

        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        accept_verified = getattr(draft_model, "accept_verified_tokens", None)
        if callable(accept_verified):
            accept_verified(
                verify.hidden,
                draft_tokens,
                accepted,
                new_tokens,
                sampler,
                token_dtype,
                **_mtp_draft_kwargs(draft_model, greedy_sampling),
            )

        # Hidden for next round: pick the slot of the newly accepted bonus.
        hidden = _mtp_draft_hidden(lm, verify.hidden[:, accepted : accepted + 1, :])
        b = new_tokens[-1] if new_tokens else b

        rollback = getattr(lm, "rollback_speculative_cache", None)
        if accepted < bs - 1 and callable(rollback):
            with mx.stream(generation_stream):
                rollback(prompt_cache, verify.gdn_states, accepted, bs)

        next_shared_kv = _slice_shared_kv_after_reject(
            verify.shared_kv_states, bs - (accepted + 1)
        )
        kv_offset = _mtp_cache_offset_max(prompt_cache)
        draft_model.set_shared_kv(
            next_shared_kv,
            kv_offset,
            position=_mtp_draft_position(kv_offset),
            kv_valid_len=kv_offset,
        )

        if emitted % 256 == 0:
            mx.clear_cache()


def _batch_cache_left_padding(prompt_cache: List[Any]) -> Optional[mx.array]:
    for cache_entry in prompt_cache:
        left_padding = getattr(cache_entry, "left_padding", None)
        if left_padding is not None:
            return left_padding
    return None


def _mtp_cache_offset(prompt_cache: List[Any]) -> Any:
    for cache_entry in prompt_cache:
        offset = getattr(cache_entry, "offset", None)
        if offset is not None:
            return offset
    for cache_entry in prompt_cache:
        idx = getattr(cache_entry, "_idx", None)
        if idx is not None:
            return idx
    return 0


def _mtp_cache_offset_max(prompt_cache: List[Any]) -> int:
    offset = _mtp_cache_offset(prompt_cache)
    return int(offset.max().item()) if isinstance(offset, mx.array) else int(offset)


def _mtp_cache_positions(
    prompt_cache: List[Any], batch_size: int
) -> Tuple[int, List[int]]:
    offset = _mtp_cache_offset(prompt_cache)
    if isinstance(offset, mx.array):
        return int(offset.max().item()), [int(x) for x in offset.tolist()]
    max_offset = int(offset)
    return max_offset, [max_offset] * batch_size


def _mtp_draft_position(kv_valid_len: Any) -> Any:
    if isinstance(kv_valid_len, int):
        return max(kv_valid_len - 1, 0)
    if isinstance(kv_valid_len, mx.array):
        return mx.maximum(kv_valid_len.astype(mx.int32) - 1, 0)
    return mx.maximum(mx.array(kv_valid_len, dtype=mx.int32) - 1, 0)


def _mtp_draft_kwargs(draft_model: nn.Module, greedy_sampling: bool) -> Dict[str, bool]:
    if greedy_sampling and getattr(draft_model, "supports_greedy_draft_argmax", False):
        return {"greedy": True}
    return {}


def _mtp_draft_block_active(
    draft_model,
    bonus_tokens: List[int],
    hidden: mx.array,
    block_size: int,
    sampler: Callable[[mx.array], mx.array],
    token_dtype: mx.Dtype,
    positions: List[int],
    greedy_sampling: bool = False,
) -> mx.array:
    """Draft an active MTP batch, falling back to rowwise for mixed positions.

    ``positions`` stores each active row's valid target-KV length. The drafter's
    RoPE position is derived from that length and rows with mixed lengths are
    drafted independently to preserve singleton position semantics.
    """
    if hidden.shape[0] <= 1:
        return draft_model.draft_block(
            mx.array(bonus_tokens, dtype=token_dtype),
            hidden,
            None,
            block_size,
            sampler,
            token_dtype,
            **_mtp_draft_kwargs(draft_model, greedy_sampling),
        )

    positions_list = [int(position) for position in positions]
    shared_kv = getattr(draft_model, "_shared_kv", None)
    if len(set(positions_list)) == 1 or shared_kv is None:
        return draft_model.draft_block(
            mx.array(bonus_tokens, dtype=token_dtype),
            hidden,
            None,
            block_size,
            sampler,
            token_dtype,
            **_mtp_draft_kwargs(draft_model, greedy_sampling),
        )

    rowwise_tokens = []
    draft_round = getattr(draft_model, "_draft_round", None)
    for row_idx, (bonus_token, position) in enumerate(zip(bonus_tokens, positions)):
        if draft_round is not None:
            draft_model._draft_round = draft_round
        per_row_shared_kv = {
            layer_type: (keys[row_idx : row_idx + 1], values[row_idx : row_idx + 1])
            for layer_type, (keys, values) in shared_kv.items()
        }
        draft_model.set_shared_kv(
            per_row_shared_kv,
            kv_offset=position,
            position=_mtp_draft_position(position),
            kv_valid_len=position,
            left_padding=None,
        )
        rowwise_tokens.append(
            draft_model.draft_block(
                bonus_token,
                hidden[row_idx : row_idx + 1],
                None,
                block_size,
                sampler,
                token_dtype,
                **_mtp_draft_kwargs(draft_model, greedy_sampling),
            )
        )

    if draft_round is not None:
        draft_model._draft_round = draft_round + 1
    draft_model.set_shared_kv(
        shared_kv,
        kv_offset=max(positions_list),
        position=_mtp_draft_position(mx.array(positions_list)),
        kv_valid_len=mx.array(positions_list),
        left_padding=None,
    )
    return mx.concatenate(rowwise_tokens, axis=0)


def _mtp_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    eos_token_ids: Optional[set] = None,
    greedy_sampling: bool = False,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batched Gemma 4 MTP round loop (B > 1).

    Mirrors ``_dflash_rounds_batch``: per-row state tracked by original
    index, continuous-batching filter on row finish. Differences vs DFlash
    batched: drafter consumes ``shared_kv_states`` (per-layer-type K/V
    snapshot) instead of multi-layer hidden capture, ``draft_block`` is
    autoregressive, and the per-round ``shared_kv`` snapshot is normalized
    back to the unbatched prefix-valid layout before each drafter rebind.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    block_total = _dflash_block_total(draft_model, draft_block_size)
    configured_block_total = int(getattr(draft_model.config, "block_size", block_total))
    draft_model.reset(model)

    # First-round hidden: prefill output may have shape [B, L, H]; reduce
    # to a single slot per row (last prompt token's hidden — see comment in
    # ``_mtp_rounds`` for rationale).
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    hidden = _mtp_draft_hidden(lm, hidden)

    # Per-row state. ``positions`` stores each row's valid target-KV length.
    # All rows start at ``L_prefill`` and advance by ``accepted_i + 1`` per
    # round.
    L_prefill, positions = _mtp_cache_positions(prompt_cache, B)
    draft_model.set_shared_kv(
        shared_kv_states,
        kv_offset=L_prefill,
        position=_mtp_draft_position(mx.array(positions)),
        kv_valid_len=mx.array(positions),
        left_padding=_batch_cache_left_padding(prompt_cache),
    )

    b = first_bonus.tolist()
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = _mtp_next_block_size(
            draft_model,
            block_total,
            configured_block_total,
            min(remaining),
        )
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        positions_active = [positions[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft (autoregressive K-step). hidden / shared_kv state was set
        # via set_shared_kv above; the drafter pulls it from there.
        draft_tokens = _mtp_draft_block_active(
            draft_model,
            b_active,
            hidden,
            bs,
            sampler,
            token_dtype,
            positions_active,
            greedy_sampling=greedy_sampling,
        )
        mx.async_eval(draft_tokens)

        # Verify
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify = _mtp_verify_target(
                lm,
                verify_input,
                prompt_cache,
                sampler,
            )
            hidden_full = verify.hidden  # [B_active, bs, H]

        # Walk per-row
        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        if verify.target_tokens is not None:
            mx.async_eval(verify.target_tokens, hidden_full)
            accepted_list, new_tokens_list = _speculative_walk_batch(
                draft_tokens, verify.target_tokens, budgets
            )
            if (
                n_active > 1
                and getattr(draft_model, "requires_uniform_batch_acceptance", False)
                and len(set(accepted_list)) > 1
            ):
                accepted_list, new_tokens_list = (
                    _speculative_walk_batch_uniform_acceptance(
                        draft_tokens,
                        verify.target_tokens,
                        accepted_list,
                        budgets,
                    )
                )
        else:
            mx.async_eval(hidden_full)
            accepted_list, new_tokens_list = _speculative_walk_batch_deferred_greedy(
                lm,
                hidden_full,
                draft_tokens,
                sampler,
                budgets,
            )
        # Keep the adaptive block-size history on a per-round basis so
        # batched MTP reacts like the singleton loop instead of letting
        # batch size change the controller signal.
        _record_speculative_round(
            draft_model,
            sum(accepted_list) / len(accepted_list),
            bs - 1,
        )

        max_a = max(accepted_list)
        accepted_arr = mx.array(accepted_list)

        accept_verified = getattr(draft_model, "accept_verified_tokens_batch", None)
        if callable(accept_verified):
            accept_verified(
                hidden_full,
                draft_tokens,
                accepted_list,
                new_tokens_list,
                sampler,
                token_dtype,
                **_mtp_draft_kwargs(draft_model, greedy_sampling),
            )

        # Per-row hidden: each row picks its own accepted slot from
        # hidden_full. Build [B_active, 1, H] with row-i's hidden at
        # position accepted_list[i].
        if max_a < bs - 1 or any(a < max_a for a in accepted_list):
            row_idx = mx.arange(n_active)
            col_idx = mx.array(accepted_list)
            # gather: hidden_full[row_idx, col_idx, :] -> [B_active, H]
            hidden = hidden_full[row_idx, col_idx, :][:, None, :]
        else:
            hidden = hidden_full[:, -1:, :]
        hidden = _mtp_draft_hidden(lm, hidden)

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
                    if eos_token_ids is not None and tok in eos_token_ids:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens and per-row positions
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]
            positions[orig] = positions[orig] + accepted_list[j] + 1

        # Rollback target cache (uniform trim by ``bs - max_a - 1`` plus
        # per-row tail-zero on rows that accepted less).
        if max_a < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify.gdn_states, accepted_arr, bs
                )

        # Slice + tail-zero ``verify.shared_kv_states`` to match the
        # post-rollback target cache. ``set_shared_kv()`` will normalize the
        # resulting hybrid layout back into a prefix-valid drafter view.
        rejected_global = bs - (max_a + 1)
        next_shared_kv = {}
        for k, kv in verify.shared_kv_states.items():
            K, V = kv
            valid = K.shape[-2] - rejected_global
            if valid >= K.shape[-2]:
                K_next, V_next = K, V
            elif valid <= 0:
                K_next = K[..., :1, :]
                V_next = V[..., :1, :]
            else:
                K_next = K[..., :valid, :]
                V_next = V[..., :valid, :]
            # Per-row tail-zero on rows that accepted less than max_a.
            if any(a < max_a for a in accepted_list):
                # K_next/V_next shape: [B_active, H, valid, D]
                # For row i, zero positions [valid - max_a + accepted_i, valid).
                # (verify_start = valid - (max_a + 1), and tail begins at
                # verify_start + accepted_i + 1 = valid - max_a + accepted_i.)
                K_arr = mx.array(K_next)  # ensure materialized for slicing
                V_arr = mx.array(V_next)
                K_arr = mx.array(K_arr)
                V_arr = mx.array(V_arr)
                mask_rows = mx.arange(K_next.shape[-2])  # [valid]
                # Build per-row mask: True where position should be kept.
                # Shape [B_active, valid]. Row i keeps positions [0, valid - max_a + accepted_i).
                keep_lens = mx.array(
                    [valid - max_a + a for a in accepted_list], dtype=mx.int32
                )  # [B_active]
                keep_mask = mask_rows[None, :] < keep_lens[:, None]  # [B_active, valid]
                keep_f = keep_mask.astype(K_next.dtype)[:, None, :, None]  # broadcast
                K_next = K_next * keep_f
                V_next = V_next * keep_f
            next_shared_kv[k] = (K_next, V_next)

        # Continuous batching: filter finished sequences. Only safe when
        # the caches expose a .filter() method (e.g. BatchKVCache); the
        # plain KVCache / RotatingKVCache do not, so we keep all rows
        # in the batch and just stop emitting for finished rows. End the
        # round-loop when every row has finished.
        cache_filterable = all(hasattr(c, "filter") for c in prompt_cache)
        if all(finished[active_idx[j]] for j in range(n_active)):
            break
        if cache_filterable:
            keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
            if len(keep_slots) < n_active:
                keep_mx = mx.array(keep_slots, dtype=mx.int32)
                for c in prompt_cache:
                    c.filter(keep_mx)
                filter_drafter = getattr(draft_model, "filter_batch", None)
                if callable(filter_drafter):
                    filter_drafter(keep_mx)
                hidden = hidden[keep_mx]
                for k in next_shared_kv:
                    K_next, V_next = next_shared_kv[k]
                    next_shared_kv[k] = (K_next[keep_mx], V_next[keep_mx])
                active_idx = [active_idx[j] for j in keep_slots]

        # Re-bind drafter with new shared_kv and per-row positions.
        positions_active = [positions[active_idx[j]] for j in range(len(active_idx))]
        new_kv_offset = _mtp_cache_offset_max(prompt_cache)
        draft_model.set_shared_kv(
            next_shared_kv,
            kv_offset=new_kv_offset,
            position=_mtp_draft_position(mx.array(positions_active)),
            kv_valid_len=mx.array(positions_active),
            left_padding=_batch_cache_left_padding(prompt_cache),
        )

        if sum(emitted) % 256 == 0:
            mx.clear_cache()


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
            "Speculative decoding with a DFlash drafter currently only "
            "supports mlx_vlm.models.qwen3_5."
        )

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = _dflash_block_total(draft_model, draft_block_size)
    draft_cache = draft_model.reset(model)

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        bs = _dflash_next_block_size(
            draft_model,
            block_total,
            max_tokens - emitted + 1,
        )
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b, hidden, draft_cache, bs, sampler, token_dtype
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

        # Emit
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        if accepted < bs - 1:
            hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted, bs
                )

        if emitted % 256 == 0:
            mx.clear_cache()


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

        new_total = sum(emitted)
        if new_total // 256 > total_emitted // 256:
            mx.clear_cache()
        total_emitted = new_total


def format_speculative_stats(draft_model: nn.Module) -> Optional[str]:
    return _format_speculative_stats(draft_model)


def get_speculative_rounds_batch(draft_kind: str):
    if draft_kind == "mtp":
        return _mtp_rounds_batch
    if draft_kind == "dflash":
        return _dflash_rounds_batch
    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def speculative_prefill_kwargs(draft_kind: str, drafter) -> dict:
    if draft_kind == "mtp":
        return {"return_hidden": True, "return_shared_kv": True}
    if draft_kind == "dflash":
        return {"capture_layer_ids": list(drafter.config.target_layer_ids)}
    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def speculative_hidden_state(draft_kind: str, outputs):
    if draft_kind == "mtp":
        return outputs.hidden_states[-1]
    if draft_kind == "dflash":
        return mx.concatenate(outputs.hidden_states, axis=-1)
    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def make_speculative_prompt_cache(
    lm,
    *,
    draft_kind: str,
    batch_size: int,
    left_padding,
    make_cache: Callable,
):
    if draft_kind == "mtp" and batch_size == 1:
        return cache.make_prompt_cache(lm)
    return make_cache(lm, left_padding)


def run_speculative_server_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    draft_kind: str,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    greedy_sampling: bool = False,
    shared_kv_states: Optional[dict] = None,
    eos_token_ids: Optional[set] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    batch_size = int(first_bonus.shape[0]) if first_bonus.ndim > 0 else 1

    if draft_kind == "mtp":
        if batch_size == 1:
            yield from (
                ([tok], state)
                for tok, state in _mtp_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    shared_kv_states,
                    first_bonus=int(first_bonus.reshape(-1).item()),
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=token_dtype,
                    greedy_sampling=greedy_sampling,
                )
            )
            return

        yield from _mtp_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            shared_kv_states,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            eos_token_ids=eos_token_ids,
            greedy_sampling=greedy_sampling,
        )
        return

    if draft_kind == "dflash":
        yield from _dflash_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
        )
        return

    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def run_speculative_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    input_ids: mx.array,
    first_token: mx.array,
    logprobs: mx.array,
    last_outputs: Any,
    *,
    draft_kind: str,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    sampler_is_greedy: bool = False,
) -> Generator[Tuple[Any, mx.array], None, None]:
    B = input_ids.shape[0]

    if draft_kind == "mtp":
        shared_kv_states = last_outputs.shared_kv_states
        hidden = last_outputs.hidden_states[-1]
        if B == 1:
            _buffer_mtp_target_cache(prompt_cache, draft_model, draft_block_size)
            mx.eval(first_token)
            bonus = first_token.item()
            yield bonus, logprobs
            yield from _mtp_rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                shared_kv_states,
                prompt_tokens=input_ids,
                first_bonus=bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
                greedy_sampling=sampler_is_greedy,
            )
        else:
            mx.eval(first_token)
            first_bonus = (
                first_token if first_token.ndim == 1 else first_token.reshape(-1)
            )
            yield first_bonus.tolist(), logprobs
            eos = getattr(model.config, "eos_token_id", None)
            if isinstance(eos, int):
                eos_set = {eos}
            elif eos is None:
                eos_set = None
            else:
                eos_set = set(int(x) for x in eos)
            yield from _mtp_rounds_batch(
                model,
                draft_model,
                prompt_cache,
                hidden,
                shared_kv_states,
                first_bonus=first_bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
                eos_token_ids=eos_set,
                greedy_sampling=sampler_is_greedy,
            )
        return

    if draft_kind != "dflash":
        raise ValueError(
            f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']"
        )

    hidden = mx.concatenate(last_outputs.hidden_states, axis=-1)
    if B == 1:
        mx.eval(first_token)
        bonus = first_token.item()
        yield bonus, logprobs
        yield from _dflash_rounds(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=input_ids.dtype,
        )
    else:
        mx.eval(first_token)
        first_bonus = first_token.squeeze(-1)
        yield first_bonus.tolist(), logprobs
        yield from _dflash_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=input_ids.dtype,
        )
