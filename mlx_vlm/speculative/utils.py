from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models import cache
from .common import (
    _dflash_block_total,
    _format_speculative_stats,
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_uniform_acceptance,
)
from .dflash import (
    _dflash_committed_hidden_segments,
    _dflash_next_block_size,
    _dflash_rounds,
    _dflash_rounds_batch,
)
from .eagle3 import _eagle3_capture_layer_ids, _eagle3_rounds, _eagle3_rounds_batch
from .mtp import (
    _buffer_mtp_target_cache,
    _effective_mtp_block_size,
    _mtp_draft_block_active,
    _mtp_draft_hidden,
    _mtp_next_block_size,
    _mtp_rounds,
    _mtp_rounds_batch,
    _mtp_shared_kv_from_prompt_cache,
    _mtp_verify_target,
    _MTPVerifyResult,
    _speculative_walk_batch_deferred_greedy,
    _speculative_walk_deferred_greedy,
)

__all__ = [
    "_MTPVerifyResult",
    "_dflash_block_total",
    "_dflash_committed_hidden_segments",
    "_dflash_next_block_size",
    "_dflash_rounds",
    "_dflash_rounds_batch",
    "_effective_mtp_block_size",
    "_format_speculative_stats",
    "_mtp_draft_block_active",
    "_mtp_draft_hidden",
    "_mtp_next_block_size",
    "_mtp_rounds",
    "_mtp_rounds_batch",
    "_mtp_shared_kv_from_prompt_cache",
    "_mtp_verify_target",
    "_speculative_walk",
    "_speculative_walk_batch",
    "_speculative_walk_batch_deferred_greedy",
    "_speculative_walk_batch_uniform_acceptance",
    "_speculative_walk_deferred_greedy",
    "format_speculative_stats",
    "get_speculative_rounds_batch",
    "make_speculative_prompt_cache",
    "run_speculative_rounds",
    "run_speculative_server_rounds",
    "speculative_hidden_state",
    "speculative_prefill_kwargs",
]


def format_speculative_stats(draft_model: nn.Module) -> Optional[str]:
    return _format_speculative_stats(draft_model)


def get_speculative_rounds_batch(draft_kind: str):
    if draft_kind == "eagle3":
        return _eagle3_rounds_batch
    if draft_kind == "mtp":
        return _mtp_rounds_batch
    if draft_kind == "dflash":
        return _dflash_rounds_batch
    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'mtp']"
    )


def speculative_prefill_kwargs(draft_kind: str, drafter) -> dict:
    if draft_kind == "mtp":
        return {"return_hidden": True, "return_shared_kv": True}
    if draft_kind == "eagle3":
        return {"capture_layer_ids": _eagle3_capture_layer_ids(drafter)}
    if draft_kind == "dflash":
        return {"capture_layer_ids": list(drafter.config.target_layer_ids)}
    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'mtp']"
    )


def speculative_hidden_state(draft_kind: str, outputs):
    if draft_kind == "mtp":
        return outputs.hidden_states[-1]
    if draft_kind in ("dflash", "eagle3"):
        return mx.concatenate(outputs.hidden_states, axis=-1)
    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'mtp']"
    )


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
    prompt_tokens: Optional[mx.array] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    batch_size = int(first_bonus.shape[0]) if first_bonus.ndim > 0 else 1

    if draft_kind == "eagle3":
        if batch_size == 1:
            yield from (
                ([tok], state)
                for tok, state in _eagle3_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    prompt_tokens=prompt_tokens,
                    first_bonus=int(first_bonus.reshape(-1).item()),
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=token_dtype,
                    greedy_sampling=greedy_sampling,
                )
            )
            return

        yield from _eagle3_rounds_batch(
            model,
            draft_model,
            prompt_cache,
            hidden,
            prompt_tokens=prompt_tokens,
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

    raise ValueError(
        f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'mtp']"
    )


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

    if draft_kind == "eagle3":
        hidden = mx.concatenate(last_outputs.hidden_states, axis=-1)
        if B == 1:
            mx.eval(first_token)
            bonus = first_token.item()
            yield bonus, logprobs
            yield from _eagle3_rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
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
            first_bonus = first_token.squeeze(-1)
            yield first_bonus.tolist(), logprobs
            eos = getattr(model.config, "eos_token_id", None)
            if isinstance(eos, int):
                eos_set = {eos}
            elif eos is None:
                eos_set = None
            else:
                eos_set = set(int(x) for x in eos)
            yield from _eagle3_rounds_batch(
                model,
                draft_model,
                prompt_cache,
                hidden,
                prompt_tokens=input_ids,
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
            f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'eagle3', 'mtp']"
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
