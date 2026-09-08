from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

generation_stream = mx.new_thread_local_stream(mx.default_device())


def _copy_rng_state() -> List[mx.array]:
    return [mx.array(state) for state in mx.random.state]


def _restore_rng_state(state: List[mx.array]) -> None:
    for i, value in enumerate(state):
        mx.random.state[i][:] = value


def _append_arrays(value: Any, arrays: List[mx.array]) -> None:
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _append_arrays(item, arrays)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _append_arrays(item, arrays)


def _draft_sampler_state_arrays(draft_model: nn.Module) -> List[mx.array]:
    state_fn = getattr(draft_model, "draft_eval_state", None)
    if callable(state_fn):
        arrays: List[mx.array] = []
        _append_arrays(state_fn(), arrays)
        return arrays

    attrs = getattr(draft_model, "sampler_state_attrs", ("_seed_token",))
    if isinstance(attrs, str):
        attrs = (attrs,)

    arrays = []
    for attr in attrs:
        _append_arrays(getattr(draft_model, attr, None), arrays)
    return arrays


class _SpeculativeSamplerRNG:
    """Keep target and drafter sampler RNG streams independent."""

    def __init__(self, draft_model: nn.Module, *, enabled: bool):
        self.draft_model = draft_model
        self.enabled = bool(enabled)
        self._target_rng_state = _copy_rng_state() if self.enabled else None
        self._draft_rng_state = _copy_rng_state() if self.enabled else None

    def draft_call(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ):
        if not self.enabled:
            result = fn(*args, **kwargs)
            arrays = []
            _append_arrays(result, arrays)
            arrays.extend(_draft_sampler_state_arrays(self.draft_model))
            if arrays:
                mx.async_eval(*arrays)
            return result

        self._target_rng_state = _copy_rng_state()
        _restore_rng_state(self._draft_rng_state)
        result = fn(*args, **kwargs)

        arrays = _draft_sampler_state_arrays(self.draft_model)
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)

        self._draft_rng_state = _copy_rng_state()
        _restore_rng_state(self._target_rng_state)
        return result

    def draft_tokens(self, fn: Callable, *args, **kwargs):
        if not self.enabled:
            result = fn(*args, **kwargs)
            arrays: List[mx.array] = []
            _append_arrays(result, arrays)
            if arrays:
                mx.async_eval(*arrays)
            return result

        self._target_rng_state = _copy_rng_state()
        _restore_rng_state(self._draft_rng_state)
        result = fn(*args, **kwargs)

        arrays = []
        _append_arrays(result, arrays)
        arrays.extend(_draft_sampler_state_arrays(self.draft_model))
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)

        self._draft_rng_state = _copy_rng_state()
        _restore_rng_state(self._target_rng_state)
        return result

    def target_sampled(self, *, sync_draft: bool = False) -> None:
        if self.enabled:
            self._target_rng_state = _copy_rng_state()
            if sync_draft:
                self._draft_rng_state = _copy_rng_state()

    def sync_draft_to_target(self) -> None:
        if self.enabled:
            self._draft_rng_state = _copy_rng_state()

    def target_eval(self, *values: Any) -> None:
        if not self.enabled:
            arrays: List[mx.array] = []
            for value in values:
                _append_arrays(value, arrays)
            if arrays:
                mx.async_eval(*arrays)
            return

        arrays = []
        for value in values:
            _append_arrays(value, arrays)
        arrays.extend(mx.random.state)
        if arrays:
            mx.async_eval(*arrays)
        self._target_rng_state = _copy_rng_state()


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


def _speculative_walk_batch(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Per-sequence speculative walk for B > 1.

    Returns ``(accepted_list, new_tokens_list)`` where each entry
    corresponds to one sequence in the batch.
    """
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


def _requires_uniform_batch_acceptance(
    draft_model: nn.Module, target_model: Optional[nn.Module] = None
) -> bool:
    """Whether ragged per-row acceptance must be clamped to a uniform count.

    The flag is honored on either the drafter or the target model. A target's
    dedicated drafter (e.g. qwen3_5 -> qwen3_5_mtp) may not advertise it, yet
    the target's rectangular batch KV cache (single ``_idx``) cannot represent
    ragged accepts without leaving phantom zeroed keys attended (issue #1962),
    so the target may require it independently of the drafter.
    """
    if getattr(draft_model, "requires_uniform_batch_acceptance", False):
        return True
    return bool(getattr(target_model, "requires_uniform_batch_acceptance", False))


def _record_speculative_round(
    draft_model: nn.Module, accepted: float, draft_count: int
) -> None:
    draft_model.accept_lens.append(accepted)
    if hasattr(draft_model, "draft_lens"):
        draft_model.draft_lens.append(int(draft_count))
    # Monotonic lifetime counters for per-request attribution. Unlike the
    # ``accept_lens`` history, these survive ``reset()`` between requests,
    # so callers can snapshot-and-diff across a request's lifetime.
    draft_model.speculative_total_rounds = (
        getattr(draft_model, "speculative_total_rounds", 0) + 1
    )
    draft_model.speculative_total_accepted = getattr(
        draft_model, "speculative_total_accepted", 0.0
    ) + float(accepted)
    draft_model.speculative_total_drafted = getattr(
        draft_model, "speculative_total_drafted", 0
    ) + int(draft_count)


def speculative_stats_snapshot(draft_model: nn.Module) -> Tuple[int, float, int]:
    """Capture the drafter's lifetime round counters for later diffing."""
    return (
        getattr(draft_model, "speculative_total_rounds", 0),
        getattr(draft_model, "speculative_total_accepted", 0.0),
        getattr(draft_model, "speculative_total_drafted", 0),
    )


def speculative_stats_since(
    draft_model: nn.Module, snapshot: Tuple[int, float, int]
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (rounds, accepted, drafted) recorded since ``snapshot``.

    Rounds are batch-wide, so the attribution is exact for batch size 1 and
    shared across concurrent requests otherwise.
    """
    rounds0, accepted0, drafted0 = snapshot
    rounds = getattr(draft_model, "speculative_total_rounds", 0) - rounds0
    if rounds <= 0:
        return None, None, None
    accepted = getattr(draft_model, "speculative_total_accepted", 0.0) - accepted0
    drafted = getattr(draft_model, "speculative_total_drafted", 0) - drafted0
    return rounds, int(round(accepted)), int(drafted)


def _dflash_block_total(draft_model: nn.Module, draft_block_size: Optional[int]) -> int:
    if draft_block_size is not None:
        return int(draft_block_size)

    configured = int(draft_model.config.block_size)
    runtime = getattr(draft_model.config, "runtime_block_size", None)
    if runtime is None:
        return configured
    return min(configured, max(1, int(runtime)))


def _batch_cache_left_padding(prompt_cache: List[Any]) -> Optional[mx.array]:
    for cache_entry in prompt_cache:
        left_padding = getattr(cache_entry, "left_padding", None)
        if left_padding is not None:
            return left_padding
    return None


def _format_speculative_stats(draft_model: nn.Module) -> Optional[str]:
    accepted_lens = getattr(draft_model, "accept_lens", None) or []
    if not accepted_lens:
        return None

    rounds = len(accepted_lens)
    accepted_drafts = sum(accepted_lens)
    mean_accept = accepted_drafts / rounds
    mean_accepted_tokens = (accepted_drafts + rounds) / rounds
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    if len(draft_lens) == rounds and sum(draft_lens) > 0:
        accept_rate = 100 * accepted_drafts / sum(draft_lens)
        mean_draft = sum(draft_lens) / rounds
        return (
            "Speculative decoding: "
            f"{mean_accepted_tokens:.2f} accepted tokens/round "
            f"({mean_accept:.2f} accepted drafts/round, "
            f"{accept_rate:.1f}% of drafted, "
            f"avg draft {mean_draft:.2f}) over {rounds} rounds"
        )

    return (
        "Speculative decoding: "
        f"{mean_accepted_tokens:.2f} accepted tokens over {rounds} rounds"
    )
