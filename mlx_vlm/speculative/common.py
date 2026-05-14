from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

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
