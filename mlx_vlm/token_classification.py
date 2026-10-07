"""Shared token-classification pipeline: windowed inference, span decoding, redaction.

``load_token_classifier`` runs ``*ForTokenClassification`` encoder checkpoints
(BERT family) through ``TokenClassifier``. ``mlx_vlm.privacy_filter`` builds on
the same pipeline and adds OpenAI Privacy Filter's constrained Viterbi decoder.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from .encoder_loader import load_encoder_model
from .utils import get_model_path, load_config

TOKEN_CLASSIFIER_MODEL_REMAPPING = {"xlm-roberta": "xlm_roberta"}
BOUNDARIES = {"B", "I", "E", "S"}


@dataclass(frozen=True)
class TokenSpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class TokenClassificationResult:
    text: str
    spans: tuple[TokenSpan, ...]
    redacted_text: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spans": [asdict(span) for span in self.spans],
            "redacted_text": self.redacted_text,
        }


@dataclass(frozen=True)
class LabelInfo:
    labels: tuple[str, ...]
    background: int
    span_names: tuple[str, ...]
    span_index: Mapping[str, int]
    token_to_span: Mapping[int, int]
    boundaries: Mapping[int, Optional[str]]
    states_by_span: Mapping[str, Mapping[str, int]]


def build_label_info(labels: Sequence[str]) -> LabelInfo:
    """Index a BIO or BIOES tag set (``O``, ``B-x``, ``I-x``, ``E-x``, ``S-x``)."""
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
        boundary, _, span_name = label.partition("-")
        if boundary not in BOUNDARIES or not span_name:
            raise ValueError(
                f"Invalid token label {label!r}; expected BIO or BIOES labels"
            )
        if span_name not in span_index:
            span_index[span_name] = len(span_names)
            span_names.append(span_name)
        token_to_span[index] = span_index[span_name]
        boundaries[index] = boundary
        states_by_span.setdefault(span_name, {})[boundary] = index

    if background is None:
        raise ValueError("Token labels must include the background label 'O'")
    return LabelInfo(
        labels=tuple(labels),
        background=background,
        span_names=tuple(span_names),
        span_index=span_index,
        token_to_span=token_to_span,
        boundaries=boundaries,
        states_by_span=states_by_span,
    )


def token_spans(path: Sequence[int], labels: LabelInfo):
    """Return ``(span_index, token_start, token_end)`` runs for a label path."""
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


def group_words(
    path: Sequence[int], word_ids: Sequence[Optional[int]], labels: LabelInfo
) -> list[int]:
    """Give every sub-token of a word one tag so the word forms a single span.

    The word takes the tag of its first non-``O`` sub-token, so a word is never
    dropped because only a later piece was tagged. The remaining sub-tokens are
    relabeled as the span's inside state.
    """
    grouped = list(path)
    start = 0
    while start < len(path):
        end = start + 1
        if word_ids[start] is not None:
            while end < len(path) and word_ids[end] == word_ids[start]:
                end += 1
        tags = [tag for tag in path[start:end] if tag != labels.background]
        if tags:
            span_name = labels.span_names[labels.token_to_span[tags[0]]]
            inside = labels.states_by_span[span_name].get("I", tags[0])
            grouped[start:end] = [tags[0]] + [inside] * (end - start - 1)
        start = end
    return grouped


def placeholder(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


def decode_spans(
    text: str,
    path: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    labels: LabelInfo,
    *,
    word_ids: Optional[Sequence[Optional[int]]] = None,
    trim_whitespace: bool = True,
) -> tuple[TokenSpan, ...]:
    """Turn a per-token label path into non-overlapping character spans.

    With ``word_ids`` (WordPiece-style tokenizers), sub-tokens are grouped into
    words first, and same-entity spans that touch with no gap (``617`` ``-``
    ``555``) are merged, because the pre-tokenizer splits punctuation into
    separate words that checkpoints may tag ``B-`` individually.
    """
    if word_ids is not None:
        path = group_words(path, word_ids, labels)

    detected = []
    for span_index, token_start, token_end in token_spans(path, labels):
        char_start = int(offsets[token_start][0])
        char_end = int(offsets[token_end - 1][1])
        if trim_whitespace:
            while char_start < char_end and text[char_start].isspace():
                char_start += 1
            while char_end > char_start and text[char_end - 1].isspace():
                char_end -= 1
        if char_end <= char_start:
            continue
        detected.append((labels.span_names[span_index], char_start, char_end))

    # Byte-level tokenizers can give several tokens the same character
    # range. Keep one deterministic, globally non-overlapping span set.
    detected.sort(key=lambda span: (span[1], -(span[2] - span[1]), span[0]))
    spans = []
    cursor = 0
    for label, start, end in detected:
        if start < cursor:
            continue
        if word_ids is not None and spans and spans[-1][0] == label:
            if spans[-1][2] == start:
                spans[-1][2] = end
                cursor = end
                continue
        spans.append([label, start, end])
        cursor = end
    return tuple(
        TokenSpan(
            label=label,
            start=start,
            end=end,
            text=text[start:end],
            placeholder=placeholder(label),
        )
        for label, start, end in spans
    )


def redact(
    text: str,
    spans: Sequence[TokenSpan],
    replacement: Optional[str] = None,
    keep_labels: Iterable[str] = (),
) -> str:
    keep = set(keep_labels)
    output = text
    for span in reversed(spans):
        if span.label in keep:
            continue
        value = replacement if replacement is not None else span.placeholder
        output = output[: span.start] + value + output[span.end :]
    return output


class TokenClassifier:
    """Tokenize text, run a token-classification model, and return coherent spans.

    ``special_tokens`` are the ids wrapped around every window; ``None`` uses the
    tokenizer's ``[CLS]``/``[SEP]`` when it defines both. ``group_words`` groups
    sub-tokens by ``word_ids`` before decoding spans.
    """

    decode_modes: tuple[str, ...] = ("argmax",)
    default_decode = "argmax"

    def __init__(
        self,
        model,
        tokenizer,
        *,
        context_size: Optional[int] = None,
        special_tokens: Optional[tuple[Sequence[int], Sequence[int]]] = None,
        group_words: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        id2label = {int(k): v for k, v in (model.config.id2label or {}).items()}
        if len(id2label) != model.config.num_labels:
            raise ValueError(
                f"id2label has {len(id2label)} entries but the classifier has "
                f"{model.config.num_labels} outputs"
            )
        self.labels = tuple(id2label[index] for index in range(len(id2label)))
        self.label_info = build_label_info(self.labels)
        if special_tokens is None:
            cls_id = getattr(tokenizer, "cls_token_id", None)
            sep_id = getattr(tokenizer, "sep_token_id", None)
            special_tokens = (
                ([cls_id], [sep_id]) if None not in (cls_id, sep_id) else ((), ())
            )
        self.prefix_ids, self.suffix_ids = map(list, special_tokens)
        self.group_words = group_words
        self.context_size = int(
            context_size
            or getattr(model.config, "default_n_ctx", None)
            or getattr(model.config, "max_position_embeddings", 4096)
        )
        self.window = self.context_size - len(self.prefix_ids) - len(self.suffix_ids)
        if self.window <= 0:
            raise ValueError("context_size must leave room for at least one token")
        self.model.eval()

    def _tokenize(self, text: str):
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        if "offset_mapping" not in encoded:
            raise ValueError(
                "Token classification requires a fast tokenizer with offsets"
            )
        offsets = [tuple(item) for item in encoded["offset_mapping"]]
        word_ids = encoded.word_ids() if self.group_words else None
        return list(encoded["input_ids"]), offsets, word_ids

    def scores(self, input_ids: Sequence[int]) -> np.ndarray:
        """Per-token log-probabilities ``[tokens, labels]``, windowed by context."""
        head, tail = len(self.prefix_ids), len(self.suffix_ids)
        chunks = []
        for start in range(0, len(input_ids), self.window):
            window = self.prefix_ids + list(input_ids[start : start + self.window])
            ids = mx.array([window + self.suffix_ids], dtype=mx.int32)
            attention_mask = mx.ones(ids.shape, dtype=mx.bool_)
            logits = self.model(ids, attention_mask=attention_mask).logits
            log_probs = logits.astype(mx.float32)
            log_probs = log_probs - mx.logsumexp(log_probs, axis=-1, keepdims=True)
            log_probs = log_probs[0, head : log_probs.shape[1] - tail]
            mx.eval(log_probs)
            chunks.append(np.asarray(log_probs))
        return np.concatenate(chunks, axis=0)

    def decode_path(self, scores: np.ndarray, decode: str) -> list[int]:
        return np.argmax(scores, axis=-1).tolist()

    def __call__(
        self,
        text: str,
        *,
        decode: Optional[str] = None,
        replacement: Optional[str] = None,
        keep_labels: Iterable[str] = (),
        trim_whitespace: bool = True,
    ) -> TokenClassificationResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        decode = decode or self.default_decode
        if decode not in self.decode_modes:
            raise ValueError(
                "decode must be " + " or ".join(repr(m) for m in self.decode_modes)
            )
        input_ids, offsets, word_ids = self._tokenize(text)
        if not input_ids:
            return TokenClassificationResult(text=text, spans=(), redacted_text=text)
        path = self.decode_path(self.scores(input_ids), decode)
        spans = decode_spans(
            text,
            path,
            offsets,
            self.label_info,
            word_ids=word_ids,
            trim_whitespace=trim_whitespace,
        )
        return TokenClassificationResult(
            text=text,
            spans=spans,
            redacted_text=redact(text, spans, replacement, keep_labels),
        )


def is_token_classifier_config(config: dict) -> bool:
    return any(
        str(architecture).endswith("ForTokenClassification")
        for architecture in config.get("architectures") or []
    )


def load_token_classification_model(
    model_path: Path,
    lazy: bool = False,
    config: Optional[dict] = None,
    **kwargs,
):
    config = dict(config) if config is not None else load_config(model_path, **kwargs)
    if not is_token_classifier_config(config):
        raise ValueError("The model is not a token-classification checkpoint.")
    id2label = config.get("id2label")
    if not isinstance(id2label, dict) or not id2label:
        raise ValueError("Token-classification checkpoints must define id2label.")
    return load_encoder_model(
        model_path,
        model_remapping=TOKEN_CLASSIFIER_MODEL_REMAPPING,
        model_class_name="TokenClassificationModel",
        config=config,
        config_overrides={"num_labels": len(id2label)},
        lazy=lazy,
        **kwargs,
    )


def load_token_classifier(
    path_or_hf_repo: Union[str, Path],
    *,
    revision: Optional[str] = None,
    force_download: bool = False,
    lazy: bool = False,
    strict: bool = True,
    context_size: Optional[int] = None,
) -> TokenClassifier:
    """Load a local or Hugging Face ``*ForTokenClassification`` encoder checkpoint."""
    model_path = get_model_path(
        path_or_hf_repo, revision=revision, force_download=force_download
    )
    model = load_token_classification_model(model_path, lazy=lazy, strict=strict)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return TokenClassifier(model, tokenizer, context_size=context_size)


def run_cli(
    argv: Optional[Sequence[str]],
    load: Callable[[str], TokenClassifier],
    *,
    description: str,
    default_model: Optional[str] = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("text", help="Text to inspect")
    parser.add_argument(
        "--model", default=default_model, required=default_model is None
    )
    parser.add_argument(
        "--argmax", action="store_true", help="Independent per-token decoding"
    )
    parser.add_argument("--replacement", default=None)
    parser.add_argument(
        "--keep",
        default="",
        help="Comma-separated entity labels to report but leave unredacted",
    )
    args = parser.parse_args(argv)

    classifier = load(args.model)
    result = classifier(
        args.text,
        decode="argmax" if args.argmax else None,
        replacement=args.replacement,
        keep_labels=[label for label in args.keep.split(",") if label],
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(
        argv,
        load_token_classifier,
        description="Tag and redact entities with an MLX token classifier",
    )


if __name__ == "__main__":
    main()
