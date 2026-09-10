"""Generation entry points for native MTP adapters."""

from .cache_state import SpeculativePrefill
from .drafters import validate_drafter_compatibility
from .mtp import mtp_rounds
from .stats import (
    format_speculative_stats,
    speculative_stats_since,
    speculative_stats_snapshot,
)

__all__ = [
    "SpeculativePrefill",
    "format_speculative_stats",
    "make_speculative_prompt_cache",
    "run_speculative_rounds",
    "run_speculative_server_rounds",
    "speculative_hidden_state",
    "speculative_prefill_kwargs",
    "speculative_stats_since",
    "speculative_stats_snapshot",
]


def speculative_prefill_kwargs(draft_kind, drafter):
    if draft_kind != "mtp":
        raise ValueError("Only native MTP speculative decoding is supported.")
    return {"return_hidden": True}


def speculative_hidden_state(draft_kind, outputs):
    return outputs.hidden_states[-1]


def make_speculative_prompt_cache(
    lm, *, draft_kind, batch_size, left_padding, make_cache
):
    return make_cache(lm, left_padding)


def run_speculative_server_rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    draft_kind,
    **kwargs,
):
    validate_drafter_compatibility(model, draft_model, draft_kind)
    yield from mtp_rounds(model, draft_model, prompt_cache, hidden, **kwargs)


def run_speculative_rounds(
    model,
    draft_model,
    prompt_cache,
    input_ids,
    first_token,
    logprobs,
    last_outputs,
    *,
    draft_kind,
    max_tokens,
    sampler,
    draft_block_size=None,
    sampler_is_greedy=False,
    logits_processors=None,
    token_context=None,
    state=None,
):
    if max_tokens <= 0:
        return
    validate_drafter_compatibility(model, draft_model, draft_kind)
    batch = input_ids.shape[0]
    first_token = first_token.reshape(-1)
    values = first_token.tolist()
    yield values[0] if batch == 1 else values, logprobs
    target = getattr(model, "language_model", model)
    eos = getattr(target.config, "eos_token_id", None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    rounds = mtp_rounds(
        model,
        draft_model,
        prompt_cache,
        last_outputs.hidden_states[-1],
        prompt_tokens=input_ids,
        first_bonus=first_token,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=draft_block_size,
        token_dtype=input_ids.dtype,
        eos_token_ids=eos,
        greedy_sampling=sampler_is_greedy,
        logits_processors=[logits_processors or []] * batch,
        token_context=token_context,
        state=state,
    )
    try:
        for tokens, _ in rounds:
            yield tokens[0] if batch == 1 else tokens, None
    finally:
        rounds.close()
