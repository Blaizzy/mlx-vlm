"""Sampling-time reasoning limits shared by AR and speculative decoding."""

import mlx.core as mx


def _last_sequence(tokens, sequence):
    for start in range(len(tokens) - len(sequence), -1, -1):
        if tokens[start : start + len(sequence)] == sequence:
            return start
    return -1


class ThinkingBudgetLogitsProcessor:
    """Force the reasoning terminator from the committed sampling prefix.

    This policy has no mutable token cursor: a speculative verifier may examine
    a prefix and discard it without changing any request state. The request
    cache already owns the committed history used for the next invocation.
    """

    def __init__(self, budget, start_ids, end_ids, *, prompt_length, preopened=False):
        if budget < 0:
            raise ValueError("thinking_budget must be non-negative")
        if not start_ids or not end_ids:
            raise ValueError(
                "Thinking delimiters must encode to nonempty token sequences"
            )
        self.budget = budget
        self.start_ids = list(start_ids)
        self.end_ids = list(end_ids)
        self.prompt_length = prompt_length
        self.preopened = preopened

    def __call__(self, tokens, logits):
        context = tokens.tolist()
        start = _last_sequence(context, self.start_ids)
        end = _last_sequence(context, self.end_ids)
        if start >= 0:
            if end >= start:
                return logits
            first = start + len(self.start_ids)
        elif self.preopened and end < self.prompt_length:
            first = self.prompt_length
        else:
            return logits
        if len(context) - first < self.budget:
            return logits
        # A natural partial delimiter can already be present at the boundary.
        progress = 0
        for size in range(1, min(len(self.end_ids), len(context) - first + 1)):
            if context[-size:] == self.end_ids[:size]:
                progress = size
        token = self.end_ids[progress]
        return mx.where(
            mx.arange(logits.shape[-1]) == token, mx.zeros_like(logits), -mx.inf
        )
