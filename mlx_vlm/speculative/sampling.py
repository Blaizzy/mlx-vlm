"""Model-independent acceptance for deterministic draft proposals."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class AcceptedTokens:
    tokens: list
    logprobs: list | None = None


@mx.compile
def _prefix_decisions(proposals, target_tokens):
    matches = proposals == target_tokens[:, :-1]
    accepted = mx.sum(mx.cumprod(matches.astype(mx.int32), axis=1), axis=1)
    return mx.concatenate([accepted[:, None], target_tokens], axis=1)


def _accepted(proposals, target_tokens, budgets, logprobs=None):
    decisions = _prefix_decisions(proposals, target_tokens).tolist()
    rows = [
        row[1 : 1 + min(row[0] + 1, budget)] for row, budget in zip(decisions, budgets)
    ]
    return AcceptedTokens(
        rows,
        (
            [[logprobs[i, j] for j in range(len(row))] for i, row in enumerate(rows)]
            if logprobs is not None
            else None
        ),
    )


def accept_greedy(proposals, logits, budgets, *, compute_logprobs=False):
    """One device-to-host decision transfer for the entire verification block."""
    logprobs = (
        logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if compute_logprobs
        else None
    )
    return _accepted(proposals, mx.argmax(logits, axis=-1), budgets, logprobs)


def token_logprobs(logprobs, token, top_k=0):
    """Extract the target's selected-token and optional top-K probabilities."""
    if logprobs is None:
        return None, None
    selected = logprobs[token]
    if top_k:
        indices = mx.argsort(logprobs)[-min(top_k, logprobs.size) :][::-1]
        values = logprobs[indices]
        mx.eval(selected, indices, values)
        return selected.item(), list(zip(indices.tolist(), values.tolist()))
    return selected.item(), None


def accept_sampled(
    proposals,
    logits,
    budgets,
    sampler,
    row_ids,
    positions,
    *,
    processors=None,
    contexts=None,
    greedy=False,
    compute_logprobs=False,
):
    """Accept a target draw iff it matches the deterministic draft proposal.

    Positioned RNG allows independent draws for the entire block in one call.
    History-dependent masks instead advance along the accepted prefix only.
    Returned logprobs use the processed target logits, before temperature/top-P,
    matching ordinary AR reporting.
    """
    if not greedy and not any(processors or []) and hasattr(sampler, "sample_target"):
        batch, length, vocab = logits.shape
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = sampler.sample_target(
            logprobs.reshape(-1, vocab),
            row_ids=[row for row in row_ids for _ in range(length)],
            positions=[p + j for p in positions for j in range(length)],
        ).reshape(batch, length)
        return _accepted(
            proposals, sampled, budgets, logprobs if compute_logprobs else None
        )

    drafts = proposals.tolist()
    rows, distributions = [], []
    for row, budget in enumerate(budgets):
        tokens, row_logprobs = [], []
        row_processors = processors[row] if processors else []
        context = list(contexts[row]) if row_processors else []
        for pos in range(min(len(drafts[row]) + 1, budget)):
            scores = logits[row : row + 1, pos]
            for processor in row_processors or []:
                scores = processor(mx.array(context, dtype=mx.int32), scores)
            logprobs = (
                scores - mx.logsumexp(scores, axis=-1, keepdims=True)
                if compute_logprobs or not greedy
                else None
            )
            if greedy:
                token = mx.argmax(scores, axis=-1)
            elif hasattr(sampler, "sample_target"):
                token = sampler.sample_target(
                    logprobs, row_ids=[row_ids[row]], positions=[positions[row] + pos]
                )
            else:
                token = sampler(logprobs)
            token = int(token.item())
            tokens.append(token)
            if compute_logprobs:
                row_logprobs.append(logprobs[0])
            if row_processors:
                context.append(token)
            if pos == len(drafts[row]) or token != drafts[row][pos]:
                break
        rows.append(tokens)
        distributions.append(row_logprobs)
    return AcceptedTokens(rows, distributions if compute_logprobs else None)
