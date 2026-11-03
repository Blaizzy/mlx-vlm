# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black; the position logits and `spans` in NumPy,
# without torch; `query_plan` and `read_logits` (torch batching) removed.

"""Parameter-free readouts over a row's state positions, with BOS0 for nothing."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from .json_spans import (  # torch-free, so code without torch reads spans too
    json_text,
    string_paths,
)

SPAN_CAPS = {"text": 512, "number": 32, "date": 32}  # a read's longest span, in tokens


@dataclass(frozen=True)
class StatePositions:
    """Token positions mapped to the caller's rendered state; None means not a text position."""

    text: str
    offsets: tuple[tuple[int, int] | None, ...]
    paths: tuple[tuple[int, int, str], ...] = ()

    @property
    def valid(self) -> list[bool]:
        return [i == 0 or span is not None for i, span in enumerate(self.offsets)]

    @property
    def pointer_mask(self) -> list[bool]:
        """`valid` up to its last valid position: the keys a read's pointer scores."""
        valid = self.valid
        return valid[: max(i for i, v in enumerate(valid) if v) + 1]

    def read(self, first: int, last: int, max_tokens: int = SPAN_CAPS["text"]) -> dict:
        if first == last == 0:
            return {"text": None, "span": None}
        if not 0 < first <= last < len(self.offsets) or last - first >= max_tokens:
            raise ValueError("invalid pointer endpoints")
        a, b = self.offsets[first], self.offsets[last]
        if a is None or b is None:
            raise ValueError("pointer leaves the state text")
        span = [a[0], b[1]]
        # A token can carry what is around a value: the separating space, or a JSON string's quote (` €",`). Trim
        # boundary space and clip to the one string the span starts or ends in; never alter a character inside.
        while span[0] < span[1] and self.text[span[0]].isspace():
            span[0] += 1
        while span[1] > span[0] and self.text[span[1] - 1].isspace():
            span[1] -= 1
        held = [
            (start, end, p)
            for start, end, p in self.paths
            if start < span[1] and span[0] < end
        ]
        if len(held) == 1:
            start, end, path = held[0]
            span = [max(span[0], start), min(span[1], end)]
            # inside one JSON string: the caller's characters, not their escapes
            return {
                "text": json_text(self.text[slice(*span)]),
                "span": span,
                "path": path,
            }
        return {"text": self.text[slice(*span)], "span": span}


def state_positions(
    tokenizer,
    state,
    prefix: str,
    full_text: str,
    style: str = "json_only",
    *,
    encoded=None,
) -> StatePositions:
    """Align one unchanged rendered prompt to text tokens; template, picture and question tokens stay invalid."""
    from .prompt import plain_offsets, question_header, state_block, state_text

    if not full_text.startswith(prefix):
        raise ValueError("the tokenized prompt must start with the state prefix")
    if state is None:
        body, escaped, boundaries = "", "", [0]
        start = len(prefix)
    else:
        body = state_block(state, style)
        escaped, boundaries = plain_offsets(tokenizer, body)
        tail = escaped + "\n" + question_header(tokenizer, state)
        if not prefix.endswith(tail):
            raise ValueError("state rendering differs from the prompt prefix")
        start = len(prefix) - len(tail)
    encoded = (
        tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded is None
        else encoded
    )
    text = (
        state_text(state, style) if state is not None else ""
    )  # the blank line after it is the template's
    stop = start + bisect_right(boundaries, len(text)) - 1
    offsets = []
    for a, b in encoded["offset_mapping"]:
        span = None
        lo, hi = max(a, start), min(b, stop)
        if lo < hi and not full_text[a:lo].strip() and not full_text[hi:b].strip():
            left, right = boundaries[lo - start], boundaries[hi - start]
            if left < right <= len(text) and text[left:right].strip():
                span = (left, right)
        offsets.append(span)
    if not offsets or offsets[0] is not None:
        raise ValueError("position 0 must be a template token for the nothing readout")
    paths = (
        string_paths(text)
        if state is not None
        and not isinstance(state, str)
        and style.endswith("_only")
        and style.startswith("json")
        else []
    )
    return StatePositions(text, tuple(offsets), tuple(paths))


def position_logits(
    query: np.ndarray, state: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """[slots, D] against [T, D]; valid [T], boolean, includes BOS0."""
    dtype = np.float64 if query.dtype == state.dtype == np.float64 else np.float32
    logits = (query.astype(dtype) @ state.astype(dtype).T) / dtype(
        math.sqrt(query.shape[-1])
    )
    return np.where(valid[None], logits, -np.inf).astype(dtype)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    top = x.max(-1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(-1, keepdims=True))


def spans(
    logits: np.ndarray, max_tokens: int = SPAN_CAPS["text"]
) -> tuple[np.ndarray, np.ndarray]:
    """Each read's span from its start and end scores [R, 2, T]: found when P(found) >= 0.5, then the likeliest span
    within `max_tokens`; (0, 0) for nothing."""
    logp = _log_softmax(logits)
    start, end = logp[:, 0], logp[:, 1]
    found = -np.expm1(start[:, 0])
    if logits.shape[2] == 1:
        return np.zeros((logits.shape[0], 2), dtype=np.int64), found
    cap = min(max_tokens, logits.shape[2] - 1)
    padded = np.pad(end[:, 1:], ((0, 0), (0, cap - 1)), constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, cap, axis=-1)
    best = windows.argmax(-1)  # the first end on ties, as max_pool1d
    scores = start[:, 1:] + np.take_along_axis(windows, best[..., None], -1)[..., 0]
    index = scores.argmax(-1)
    at = np.stack((index + 1, index + best[np.arange(len(index)), index] + 1), -1)
    return np.where((found < 0.5)[:, None], 0, at), found
