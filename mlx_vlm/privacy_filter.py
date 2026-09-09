"""High-level MLX inference and span decoding for OpenAI Privacy Filter."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from .utils import get_model_path, load_model

VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


@dataclass(frozen=True)
class PrivacySpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class PrivacyFilterResult:
    text: str
    spans: tuple[PrivacySpan, ...]
    redacted_text: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spans": [asdict(span) for span in self.spans],
            "redacted_text": self.redacted_text,
        }


@dataclass(frozen=True)
class _LabelInfo:
    labels: tuple[str, ...]
    background: int
    span_names: tuple[str, ...]
    span_index: Mapping[str, int]
    token_to_span: Mapping[int, int]
    boundaries: Mapping[int, Optional[str]]
    states_by_span: Mapping[str, Mapping[str, int]]


def _build_label_info(labels: Sequence[str]) -> _LabelInfo:
    background = None
    span_names = ["O"]
    span_index = {"O": 0}
    token_to_span = {}
    boundaries = {}
    states_by_span: dict[str, dict[str, int]] = {}

    for index, label in enumerate(labels):
        if label == "O":
            background = index
            token_to_span[index] = 0
            boundaries[index] = None
            continue
        if "-" not in label:
            raise ValueError(f"Invalid token label {label!r}; expected BIOES labels")
        boundary, span_name = label.split("-", 1)
        if boundary not in {"B", "I", "E", "S"} or not span_name:
            raise ValueError(f"Invalid token label {label!r}; expected BIOES labels")
        if span_name not in span_index:
            span_index[span_name] = len(span_names)
            span_names.append(span_name)
        token_to_span[index] = span_index[span_name]
        boundaries[index] = boundary
        states_by_span.setdefault(span_name, {})[boundary] = index

    if background is None:
        raise ValueError("Privacy Filter labels must include the background label 'O'")
    for span_name, states in states_by_span.items():
        missing = {"B", "I", "E", "S"} - set(states)
        if missing:
            raise ValueError(
                f"Privacy Filter labels for {span_name!r} are missing {sorted(missing)}"
            )
    return _LabelInfo(
        labels=tuple(labels),
        background=background,
        span_names=tuple(span_names),
        span_index=span_index,
        token_to_span=token_to_span,
        boundaries=boundaries,
        states_by_span=states_by_span,
    )


class ViterbiDecoder:
    """Linear-time constrained BIOES decoder used by Privacy Filter."""

    def __init__(
        self,
        labels: Sequence[str],
        transition_biases: Optional[Mapping[str, float]] = None,
    ):
        self.label_info = _build_label_info(labels)
        supplied = dict(transition_biases or {})
        unknown = set(supplied) - set(VITERBI_BIAS_KEYS)
        if unknown:
            raise ValueError(f"Unknown Viterbi transition biases: {sorted(unknown)}")
        self.biases = {key: float(supplied.get(key, 0.0)) for key in VITERBI_BIAS_KEYS}

        states = self.label_info.states_by_span
        self._b = np.asarray([value["B"] for value in states.values()], dtype=np.int32)
        self._i = np.asarray([value["I"] for value in states.values()], dtype=np.int32)
        self._e = np.asarray([value["E"] for value in states.values()], dtype=np.int32)
        self._s = np.asarray([value["S"] for value in states.values()], dtype=np.int32)
        self._start_states = np.concatenate(
            (np.asarray([self.label_info.background], dtype=np.int32), self._b, self._s)
        )
        self._end_states = np.concatenate(
            (np.asarray([self.label_info.background], dtype=np.int32), self._e, self._s)
        )
        self._closed_states = np.concatenate((self._e, self._s))

    def decode(self, emissions: np.ndarray) -> list[int]:
        emissions = np.asarray(emissions, dtype=np.float32)
        if emissions.ndim != 2:
            raise ValueError("emissions must have shape [tokens, labels]")
        length, num_labels = emissions.shape
        if num_labels != len(self.label_info.labels):
            raise ValueError(
                f"Expected {len(self.label_info.labels)} labels, got {num_labels}"
            )
        if length == 0:
            return []

        negative_infinity = np.float32(-1e9)
        scores = np.full((num_labels,), negative_infinity, dtype=np.float32)
        scores[self._start_states] = emissions[0, self._start_states]
        backpointers = np.full((length - 1, num_labels), -1, dtype=np.int16)
        background = self.label_info.background
        biases = self.biases

        for step in range(1, length):
            next_scores = np.full_like(scores, negative_infinity)
            pointers = backpointers[step - 1]

            closed_values = scores[self._closed_states]
            closed_offset = int(np.argmax(closed_values))
            best_closed = float(closed_values[closed_offset])
            best_closed_state = int(self._closed_states[closed_offset])

            background_options = (
                float(scores[background]) + biases["transition_bias_background_stay"],
                best_closed + biases["transition_bias_end_to_background"],
            )
            if background_options[0] >= background_options[1]:
                background_score, background_prev = background_options[0], background
            else:
                background_score, background_prev = (
                    background_options[1],
                    best_closed_state,
                )
            next_scores[background] = background_score + emissions[step, background]
            pointers[background] = background_prev

            start_options = (
                float(scores[background])
                + biases["transition_bias_background_to_start"],
                best_closed + biases["transition_bias_end_to_start"],
            )
            if start_options[0] >= start_options[1]:
                start_score, start_prev = start_options[0], background
            else:
                start_score, start_prev = start_options[1], best_closed_state
            next_scores[self._b] = start_score + emissions[step, self._b]
            next_scores[self._s] = start_score + emissions[step, self._s]
            pointers[self._b] = start_prev
            pointers[self._s] = start_prev

            open_scores = np.stack((scores[self._b], scores[self._i]), axis=1)
            choose_inside = open_scores[:, 1] > open_scores[:, 0]
            best_open = np.where(choose_inside, open_scores[:, 1], open_scores[:, 0])
            best_open_state = np.where(choose_inside, self._i, self._b)
            next_scores[self._i] = (
                best_open
                + biases["transition_bias_inside_to_continue"]
                + emissions[step, self._i]
            )
            next_scores[self._e] = (
                best_open
                + biases["transition_bias_inside_to_end"]
                + emissions[step, self._e]
            )
            pointers[self._i] = best_open_state
            pointers[self._e] = best_open_state
            scores = next_scores

        end_scores = scores[self._end_states]
        last_label = int(self._end_states[int(np.argmax(end_scores))])
        path = np.empty((length,), dtype=np.int32)
        path[-1] = last_label
        for step in range(length - 2, -1, -1):
            last_label = int(backpointers[step, last_label])
            path[step] = last_label
        return path.tolist()


def _load_transition_biases(model_path: Path, operating_point: str) -> dict[str, float]:
    calibration_path = model_path / "viterbi_calibration.json"
    if not calibration_path.is_file():
        return {key: 0.0 for key in VITERBI_BIAS_KEYS}
    with calibration_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        raw_biases = payload["operating_points"][operating_point]["biases"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown Privacy Filter operating point {operating_point!r}"
        ) from exc
    missing = set(VITERBI_BIAS_KEYS) - set(raw_biases)
    if missing:
        raise ValueError(f"Viterbi calibration is missing biases: {sorted(missing)}")
    return {key: float(raw_biases[key]) for key in VITERBI_BIAS_KEYS}


def _placeholder(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


def _token_spans(path: Sequence[int], labels: _LabelInfo):
    spans = []
    current_span = None
    current_start = None
    for token_index, token_label in enumerate(path):
        span_index = labels.token_to_span[token_label]
        boundary = labels.boundaries[token_label]
        if token_label == labels.background:
            if current_span is not None:
                spans.append((current_span, current_start, token_index))
            current_span = current_start = None
        elif boundary == "S":
            if current_span is not None:
                spans.append((current_span, current_start, token_index))
            spans.append((span_index, token_index, token_index + 1))
            current_span = current_start = None
        elif boundary == "B":
            if current_span is not None:
                spans.append((current_span, current_start, token_index))
            current_span, current_start = span_index, token_index
        elif boundary == "I":
            if current_span != span_index:
                if current_span is not None:
                    spans.append((current_span, current_start, token_index))
                current_span, current_start = span_index, token_index
        elif boundary == "E":
            if current_span == span_index:
                spans.append((span_index, current_start, token_index + 1))
            else:
                if current_span is not None:
                    spans.append((current_span, current_start, token_index))
                spans.append((span_index, token_index, token_index + 1))
            current_span = current_start = None
    if current_span is not None:
        spans.append((current_span, current_start, len(path)))
    return spans


def _redact(text: str, spans: Sequence[PrivacySpan], replacement: Optional[str]) -> str:
    output = text
    for span in reversed(spans):
        value = replacement if replacement is not None else span.placeholder
        output = output[: span.start] + value + output[span.end :]
    return output


class PrivacyFilter:
    """Tokenize text, run the MLX model, and return coherent privacy spans."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        transition_biases: Optional[Mapping[str, float]] = None,
        context_size: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        id2label = model.config.id2label
        labels = tuple(id2label[index] for index in range(model.config.num_labels))
        self.decoder = ViterbiDecoder(labels, transition_biases)
        self.label_info = self.decoder.label_info
        self.context_size = int(
            context_size
            or getattr(model.config, "default_n_ctx", None)
            or getattr(model.config, "max_position_embeddings", 4096)
        )
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        self.model.eval()

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        if "offset_mapping" not in encoded:
            raise ValueError("Privacy Filter requires a fast tokenizer with offsets")
        return list(encoded["input_ids"]), [
            tuple(item) for item in encoded["offset_mapping"]
        ]

    def __call__(
        self,
        text: str,
        *,
        decode: str = "viterbi",
        replacement: Optional[str] = None,
        trim_whitespace: bool = True,
    ) -> PrivacyFilterResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if decode not in {"viterbi", "argmax"}:
            raise ValueError("decode must be 'viterbi' or 'argmax'")
        input_ids, offsets = self._tokenize(text)
        if not input_ids:
            return PrivacyFilterResult(text=text, spans=(), redacted_text=text)

        score_chunks = []
        for start in range(0, len(input_ids), self.context_size):
            token_chunk = input_ids[start : start + self.context_size]
            ids = mx.array([token_chunk], dtype=mx.int32)
            attention_mask = mx.ones(ids.shape, dtype=mx.bool_)
            logits = self.model(ids, attention_mask=attention_mask).logits
            log_probs = logits.astype(mx.float32)
            log_probs = log_probs - mx.logsumexp(log_probs, axis=-1, keepdims=True)
            log_probs = log_probs[0]
            mx.eval(log_probs)
            score_chunks.append(np.asarray(log_probs))
        scores = np.concatenate(score_chunks, axis=0)

        if decode == "viterbi":
            path = self.decoder.decode(scores)
        else:
            path = np.argmax(scores, axis=-1).tolist()

        detected = []
        for span_index, token_start, token_end in _token_spans(path, self.label_info):
            char_start = int(offsets[token_start][0])
            char_end = int(offsets[token_end - 1][1])
            if trim_whitespace:
                while char_start < char_end and text[char_start].isspace():
                    char_start += 1
                while char_end > char_start and text[char_end - 1].isspace():
                    char_end -= 1
            if char_end <= char_start:
                continue
            label = self.label_info.span_names[span_index]
            detected.append(
                PrivacySpan(
                    label=label,
                    start=char_start,
                    end=char_end,
                    text=text[char_start:char_end],
                    placeholder=_placeholder(label),
                )
            )

        # Byte-level tokenizers can give several tokens the same character
        # range. Keep one deterministic, globally non-overlapping span set.
        detected.sort(
            key=lambda span: (span.start, -(span.end - span.start), span.label)
        )
        non_overlapping = []
        cursor = 0
        for span in detected:
            if span.start < cursor:
                continue
            non_overlapping.append(span)
            cursor = span.end
        spans = tuple(non_overlapping)
        return PrivacyFilterResult(
            text=text,
            spans=spans,
            redacted_text=_redact(text, spans, replacement),
        )


def load_privacy_filter(
    path_or_hf_repo: str = "openai/privacy-filter",
    *,
    revision: Optional[str] = None,
    force_download: bool = False,
    lazy: bool = False,
    strict: bool = True,
    operating_point: str = "default",
    transition_biases: Optional[Mapping[str, float]] = None,
    context_size: Optional[int] = None,
) -> PrivacyFilter:
    """Load a local or Hugging Face Privacy Filter checkpoint."""

    model_path = get_model_path(
        path_or_hf_repo,
        revision=revision,
        force_download=force_download,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "viterbi_calibration.json",
        ],
    )
    model = load_model(model_path, lazy=lazy, strict=strict)
    if model.model_type != "openai_privacy_filter":
        raise ValueError(
            f"Expected an OpenAI Privacy Filter checkpoint, got {model.model_type!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    biases = (
        dict(transition_biases)
        if transition_biases is not None
        else _load_transition_biases(model_path, operating_point)
    )
    return PrivacyFilter(
        model,
        tokenizer,
        transition_biases=biases,
        context_size=context_size,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Detect and redact PII with MLX")
    parser.add_argument("text", help="Text to inspect")
    parser.add_argument("--model", default="openai/privacy-filter")
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--replacement", default=None)
    args = parser.parse_args(argv)

    detector = load_privacy_filter(args.model)
    result = detector(
        args.text,
        decode="argmax" if args.argmax else "viterbi",
        replacement=args.replacement,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
