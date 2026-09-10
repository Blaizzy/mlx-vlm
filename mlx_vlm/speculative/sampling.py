"""Model-independent acceptance for deterministic draft proposals."""

import mlx.core as mx


@mx.compile
def _greedy_decisions(proposals, target_tokens):
    matches = proposals == target_tokens[:, :-1]
    accepted = mx.sum(mx.cumprod(matches.astype(mx.int32), axis=1), axis=1)
    return mx.concatenate([accepted[:, None], target_tokens], axis=1)


def accept_greedy(proposals, logits, budgets):
    """One device-to-host decision transfer for the entire verification block."""
    decisions = _greedy_decisions(proposals, mx.argmax(logits, axis=-1)).tolist()
    return [
        row[1 : 1 + min(row[0] + 1, budget)] for row, budget in zip(decisions, budgets)
    ]


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
):
    """Sample the target only through the first rejection in each row.

    A deterministic draft has q(d)=1. Accepting iff a target draw equals d,
    otherwise emitting that draw, implements both acceptance and the residual
    distribution without a second sampler or a model-specific kernel.
    """
    drafts = proposals.tolist()
    rows = []
    for row, budget in enumerate(budgets):
        tokens = []
        row_processors = processors[row] if processors else []
        context = list(contexts[row]) if row_processors else []
        for pos in range(min(len(drafts[row]) + 1, budget)):
            scores = logits[row : row + 1, pos]
            for processor in row_processors or []:
                scores = processor(mx.array(context, dtype=mx.int32), scores)
            if greedy:
                token = mx.argmax(scores, axis=-1)
            else:
                logprobs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
                if hasattr(sampler, "sample_target"):
                    token = sampler.sample_target(
                        logprobs,
                        row_ids=[row_ids[row]],
                        positions=[positions[row] + pos],
                    )
                else:
                    token = sampler(logprobs)
            token = int(token.item())
            tokens.append(token)
            if row_processors:
                context.append(token)
            if pos == len(drafts[row]) or token != drafts[row][pos]:
                break
        rows.append(tokens)
    return rows
