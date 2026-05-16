from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .common import (
    _batch_cache_left_padding,
    _record_speculative_round,
    generation_stream,
)


def _eagle3_walk(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budget: int,
) -> Tuple[int, List[int]]:
    n_draft = int(draft_tokens.shape[1])
    draft_row = draft_tokens.reshape(-1).tolist()[:n_draft]
    target_row = target_tokens.reshape(-1).tolist()

    accepted = n_draft
    for i, (draft_tok, target_tok) in enumerate(zip(draft_row, target_row)):
        if draft_tok != target_tok:
            accepted = i
            break

    new_tokens = draft_row[:accepted] + target_row[accepted : accepted + 1]
    return accepted, new_tokens[:budget]


def _eagle3_walk_batch(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    B = int(draft_tokens.shape[0])
    n_draft = int(draft_tokens.shape[1])
    draft_rows = draft_tokens.tolist()
    target_rows = target_tokens.tolist()
    accepted_list = []
    new_tokens_list: List[List[int]] = []
    for i in range(B):
        accepted = n_draft
        for j, (draft_tok, target_tok) in enumerate(
            zip(draft_rows[i][:n_draft], target_rows[i])
        ):
            if draft_tok != target_tok:
                accepted = j
                break
        accepted_list.append(accepted)
        new_tokens = draft_rows[i][:accepted] + target_rows[i][accepted : accepted + 1]
        new_tokens_list.append(new_tokens[: budgets[i]])
    return accepted_list, new_tokens_list


def _eagle3_accept_counts(draft_tokens: mx.array, target_tokens: mx.array) -> mx.array:
    n_draft = int(draft_tokens.shape[1])
    mismatches = draft_tokens != target_tokens[:, :n_draft]
    first_mismatch = mx.argmax(mismatches.astype(mx.int32), axis=1)
    has_mismatch = mx.any(mismatches, axis=1)
    return mx.where(
        has_mismatch,
        first_mismatch,
        mx.full((int(draft_tokens.shape[0]),), n_draft, dtype=mx.int32),
    ).astype(mx.int32)


def _eagle3_walk_batch_uniform_acceptance(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    accepted_list: List[int],
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
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


def _eagle3_draft_kwargs(draft_model: nn.Module, greedy_sampling: bool) -> dict:
    if greedy_sampling and getattr(draft_model, "supports_greedy_draft_argmax", False):
        return {"greedy": True}
    return {}


def _eagle3_block_settings(
    draft_model: nn.Module,
    draft_block_size: Optional[int],
) -> Tuple[int, int, bool]:
    configured = int(draft_model.config.block_size)
    if draft_block_size is not None:
        return int(draft_block_size), configured, False

    auto_max = getattr(draft_model.config, "adaptive_max_block_size", None)
    if auto_max is None:
        return configured, configured, False
    auto_max = max(configured, int(auto_max))
    return auto_max, configured, auto_max > configured


def _eagle3_block_tiers(configured_block_total: int, max_block_total: int) -> List[int]:
    tiers = [configured_block_total]
    for tier in (8, 12, 16):
        if configured_block_total < tier < max_block_total:
            tiers.append(tier)
    if max_block_total not in tiers:
        tiers.append(max_block_total)
    return sorted(set(max(2, int(tier)) for tier in tiers))


def _eagle3_next_block_size(
    draft_model: nn.Module,
    max_block_total: int,
    configured_block_total: int,
    remaining_budget: int,
    *,
    adaptive: bool,
) -> int:
    if not adaptive:
        return min(max_block_total, remaining_budget)

    tiers = _eagle3_block_tiers(configured_block_total, max_block_total)
    current = getattr(draft_model, "_adaptive_block_size", None)
    if current is None:
        current = tiers[0]

    accepted = getattr(draft_model, "accept_lens", None) or []
    drafted = getattr(draft_model, "draft_lens", None) or []
    if len(accepted) >= 6 and len(drafted) >= 6:
        recent_accept = [int(a) for a in accepted[-6:]]
        recent_draft = [max(1, int(d)) for d in drafted[-6:]]
        mean_output = sum(a + 1 for a in recent_accept) / len(recent_accept)
        full_rate = sum(1 for a, d in zip(recent_accept, recent_draft) if a >= d) / len(
            recent_accept
        )

        tier_idx = tiers.index(min(tiers, key=lambda tier: abs(tier - current)))
        if len(accepted) == 6 and current == configured_block_total and len(tiers) > 1:
            tier_idx = len(tiers) - 1
        elif mean_output < 2.0 or (mean_output < 3.0 and full_rate == 0):
            tier_idx = max(0, tier_idx - 1)
        elif full_rate >= 0.33 or mean_output >= current * 0.75:
            tier_idx = min(len(tiers) - 1, tier_idx + 1)
        current = tiers[tier_idx]

    current = min(current, max_block_total, remaining_budget)
    draft_model._adaptive_block_size = current
    return current


def _eagle3_verify_target(
    lm: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
    sampler: Callable[[mx.array], mx.array],
    target_layer_ids: List[int],
):
    if verify_input.shape[1] > 1 and "gemma4" in type(lm).__module__:
        first_out = lm(
            verify_input[:, :1],
            cache=prompt_cache,
            capture_layer_ids=target_layer_ids,
        )
        hidden_chunks = [mx.concatenate(first_out.hidden_states, axis=-1)]
        target_chunks = [sampler(first_out.logits)]
        gdn_states = first_out.gdn_states

        if verify_input.shape[1] > 1:
            tail_out = lm(
                verify_input[:, 1:],
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
            )
            hidden_chunks.append(mx.concatenate(tail_out.hidden_states, axis=-1))
            target_chunks.append(sampler(tail_out.logits))
            gdn_states = tail_out.gdn_states
        return (
            mx.concatenate(hidden_chunks, axis=1),
            mx.concatenate(target_chunks, axis=1),
            gdn_states,
        )

    verify_out = lm(
        verify_input,
        cache=prompt_cache,
        capture_layer_ids=target_layer_ids,
    )
    hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
    target_tokens = sampler(verify_out.logits)
    return hidden, target_tokens, verify_out.gdn_states


def _eagle3_eos_token_ids(lm: nn.Module) -> List[int]:
    config = getattr(lm, "config", None)
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        return []
    if isinstance(eos, (list, tuple, set)):
        return [int(tok) for tok in eos if tok is not None]
    return [int(eos)]


def _eagle3_hot_token_ids(
    lm: nn.Module,
    draft_model: nn.Module,
    eos_token_ids: Optional[List[int]] = None,
) -> Optional[mx.array]:
    d2t = getattr(draft_model, "d2t", None)
    if d2t is None:
        return None

    eos_ids = eos_token_ids if eos_token_ids is not None else _eagle3_eos_token_ids(lm)
    cache_key = tuple(eos_ids)
    cached_key = getattr(draft_model, "_hot_token_ids_cache_key", None)
    cached_ids = getattr(draft_model, "_hot_token_ids_cache", None)
    if cached_ids is not None and cached_key == cache_key:
        return cached_ids

    hot_ids = mx.arange(d2t.shape[0], dtype=mx.int32) + d2t.astype(mx.int32)
    if eos_ids:
        hot_ids = mx.concatenate([hot_ids, mx.array(eos_ids, dtype=mx.int32)], axis=0)
    hot_ids = hot_ids.astype(mx.int32)
    draft_model._hot_token_ids_cache_key = cache_key
    draft_model._hot_token_ids_cache = hot_ids
    return hot_ids


def _eagle3_hot_logits_from_hidden(
    lm: nn.Module,
    draft_model: nn.Module,
    hidden: mx.array,
    hot_ids: mx.array,
) -> mx.array:
    embed = lm.model.embed_tokens
    cache_key = (id(embed), int(hot_ids.shape[0]))
    cached_key = getattr(draft_model, "_hot_lm_head_cache_key", None)
    cached_head = getattr(draft_model, "_hot_lm_head_cache", None)

    if cached_head is None or cached_key != cache_key:
        if hasattr(embed, "scales"):
            cached_head = (
                embed.weight[hot_ids],
                embed.scales[hot_ids],
                embed.biases[hot_ids] if hasattr(embed, "biases") else None,
                embed.group_size,
                embed.bits,
                getattr(embed, "mode", "affine"),
                True,
            )
        else:
            cached_head = (embed.weight[hot_ids], None, None, None, None, None, False)
        draft_model._hot_lm_head_cache_key = cache_key
        draft_model._hot_lm_head_cache = cached_head

    weight, scales, biases, group_size, bits, mode, quantized = cached_head

    if quantized:
        logits = mx.quantized_matmul(
            hidden,
            weight,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
    else:
        logits = hidden @ weight.T

    softcap = getattr(lm, "final_logit_softcapping", None)
    if softcap is not None:
        logits = mx.tanh(logits / softcap) * softcap
    return logits


def _eagle3_verify_target_hot(
    lm: nn.Module,
    draft_model: nn.Module,
    verify_input: mx.array,
    prompt_cache: List[Any],
    sampler: Callable[[mx.array], mx.array],
    target_layer_ids: List[int],
    eos_token_ids: Optional[List[int]] = None,
) -> Optional[Tuple[mx.array, mx.array, Any]]:
    if not hasattr(lm, "model") or not hasattr(lm, "logits_from_hidden"):
        return None

    hot_ids = _eagle3_hot_token_ids(lm, draft_model, eos_token_ids)
    if hot_ids is None:
        return None

    hidden_sink: List[mx.array] = []
    final_hidden = lm.model(
        verify_input,
        cache=prompt_cache,
        capture_layer_ids=target_layer_ids,
        hidden_sink=hidden_sink,
    )
    if not hidden_sink:
        raise RuntimeError("EAGLE-3 verification did not produce hidden states.")

    hidden = mx.concatenate(hidden_sink, axis=-1)
    hot_logits = _eagle3_hot_logits_from_hidden(lm, draft_model, final_hidden, hot_ids)
    hot_idx = mx.argmax(hot_logits, axis=-1).astype(mx.int32)
    target_tokens = hot_ids[hot_idx]

    draft_tokens = verify_input[:, 1:]
    full_pos = mx.minimum(
        _eagle3_accept_counts(draft_tokens, target_tokens),
        int(target_tokens.shape[1]) - 1,
    )
    full_hidden = mx.take_along_axis(final_hidden, full_pos[:, None, None], axis=1)
    full_token = sampler(lm.logits_from_hidden(full_hidden)).astype(target_tokens.dtype)
    target_tokens = mx.put_along_axis(
        target_tokens,
        full_pos[:, None],
        full_token,
        axis=1,
    )
    return hidden, target_tokens, None


def _eagle3_capture_layer_ids(draft_model: nn.Module) -> List[int]:
    return list(
        getattr(
            draft_model.config,
            "capture_layer_ids",
            draft_model.config.target_layer_ids,
        )
    )


def _eagle3_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    prompt_tokens: Optional[mx.array] = None,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    greedy_sampling: bool = False,
) -> Generator[Tuple[int, None], None, None]:
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache."
        )

    target_layer_ids = _eagle3_capture_layer_ids(draft_model)
    block_total, configured_block_total, adaptive_block_size = _eagle3_block_settings(
        draft_model, draft_block_size
    )
    if draft_block_size is None:
        block_total = min(block_total, 2)
        configured_block_total = min(configured_block_total, block_total)
        adaptive_block_size = False
    draft_cache = draft_model.reset(model)

    prefill_draft = getattr(draft_model, "prefill_from_target_hidden", None)
    if callable(prefill_draft) and prompt_tokens is not None:
        prefill_draft(
            prompt_tokens,
            hidden,
            first_bonus,
            sampler,
            token_dtype,
            **_eagle3_draft_kwargs(draft_model, greedy_sampling),
        )

    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    b = first_bonus
    emitted = 1

    while emitted < max_tokens:
        bs = _eagle3_next_block_size(
            draft_model,
            block_total,
            configured_block_total,
            max_tokens - emitted + 1,
            adaptive=adaptive_block_size,
        )
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b,
            hidden,
            draft_cache,
            bs,
            sampler,
            token_dtype,
            **_eagle3_draft_kwargs(draft_model, greedy_sampling),
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens],
                axis=1,
            )
            hot_verify = _eagle3_verify_target_hot(
                lm,
                draft_model,
                verify_input,
                prompt_cache,
                sampler,
                target_layer_ids,
                _eagle3_eos_token_ids(model),
            )
            if hot_verify is None:
                verify_hidden, target_tokens, gdn_states = _eagle3_verify_target(
                    lm,
                    verify_input,
                    prompt_cache,
                    sampler,
                    target_layer_ids,
                )
            else:
                verify_hidden, target_tokens, gdn_states = hot_verify
        mx.async_eval(target_tokens, verify_hidden)

        accepted, new_tokens = _eagle3_walk(
            draft_tokens,
            target_tokens,
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
                verify_hidden,
                draft_tokens,
                accepted,
                new_tokens,
                sampler,
                token_dtype,
                **_eagle3_draft_kwargs(draft_model, greedy_sampling),
            )

        hidden = verify_hidden[:, accepted : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(prompt_cache, gdn_states, accepted, bs)

        if emitted % 256 == 0:
            mx.clear_cache()


def _eagle3_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    prompt_tokens: Optional[mx.array] = None,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    eos_token_ids: Optional[set] = None,
    greedy_sampling: bool = False,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    cache_left_padding = _batch_cache_left_padding(prompt_cache)
    if cache_left_padding is None:
        left_padding = [0] * int(B)
    else:
        padding_values = (
            cache_left_padding.tolist()
            if hasattr(cache_left_padding, "tolist")
            else cache_left_padding
        )
        left_padding = [int(pad) for pad in padding_values]
    target_layer_ids = _eagle3_capture_layer_ids(draft_model)
    block_total, configured_block_total, adaptive_block_size = _eagle3_block_settings(
        draft_model, draft_block_size
    )
    if draft_block_size is None:
        block_total = min(block_total, 2)
        configured_block_total = min(configured_block_total, block_total)
        adaptive_block_size = False
    draft_cache = draft_model.reset(model, left_padding=left_padding)

    prefill_draft = getattr(draft_model, "prefill_from_target_hidden", None)
    if callable(prefill_draft) and prompt_tokens is not None:
        prefill_draft(
            prompt_tokens,
            hidden,
            first_bonus,
            sampler,
            token_dtype,
            **_eagle3_draft_kwargs(draft_model, greedy_sampling),
        )

    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    b = first_bonus.tolist()
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))
    cache_slots = list(range(B))

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = _eagle3_next_block_size(
            draft_model,
            block_total,
            configured_block_total,
            min(remaining),
            adaptive=adaptive_block_size,
        )
        if bs <= 1:
            break

        n_active = len(active_idx)
        draft_batch_size = getattr(draft_model, "batch_size", lambda: None)()
        if draft_batch_size is not None and draft_batch_size != n_active:
            filter_drafter = getattr(draft_model, "filter_batch", None)
            if callable(filter_drafter):
                filter_drafter(mx.array(cache_slots, dtype=mx.int32))
                cache_slots = list(range(n_active))

        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        draft_tokens = draft_model.draft_block(
            b_arr,
            hidden,
            draft_cache,
            bs,
            sampler,
            token_dtype,
            **_eagle3_draft_kwargs(draft_model, greedy_sampling),
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            hot_verify = _eagle3_verify_target_hot(
                lm,
                draft_model,
                verify_input,
                prompt_cache,
                sampler,
                target_layer_ids,
                _eagle3_eos_token_ids(model),
            )
            if hot_verify is None:
                verify_hidden, target_tokens, gdn_states = _eagle3_verify_target(
                    lm,
                    verify_input,
                    prompt_cache,
                    sampler,
                    target_layer_ids,
                )
            else:
                verify_hidden, target_tokens, gdn_states = hot_verify
        mx.async_eval(target_tokens, verify_hidden)

        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _eagle3_walk_batch(
            draft_tokens, target_tokens, budgets
        )
        if (
            n_active > 1
            and getattr(draft_model, "requires_uniform_batch_acceptance", False)
            and len(set(accepted_list)) > 1
        ):
            accepted_list, new_tokens_list = _eagle3_walk_batch_uniform_acceptance(
                draft_tokens,
                target_tokens,
                accepted_list,
                budgets,
            )
        _record_speculative_round(
            draft_model,
            sum(accepted_list) / len(accepted_list),
            bs - 1,
        )

        accept_verified = getattr(draft_model, "accept_verified_tokens_batch", None)
        if callable(accept_verified):
            accept_verified(
                verify_hidden,
                draft_tokens,
                accepted_list,
                new_tokens_list,
                sampler,
                token_dtype,
                **_eagle3_draft_kwargs(draft_model, greedy_sampling),
            )

        row_idx = mx.arange(n_active)
        col_idx = mx.array(accepted_list)
        hidden = verify_hidden[row_idx, col_idx, :][:, None, :]

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

        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if any(accepted < bs - 1 for accepted in accepted_list):
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache,
                    gdn_states,
                    mx.array(accepted_list),
                    bs,
                )

        if all(finished[active_idx[j]] for j in range(n_active)):
            break
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            if all(hasattr(c, "filter") for c in prompt_cache):
                for c in prompt_cache:
                    c.filter(keep_mx)
            filter_drafter = getattr(draft_model, "filter_batch", None)
            if callable(filter_drafter):
                filter_drafter(keep_mx)
            hidden = hidden[keep_mx]
            active_idx = [active_idx[j] for j in keep_slots]
            cache_slots = [cache_slots[j] for j in keep_slots]

        if sum(emitted) % 256 == 0:
            mx.clear_cache()
