"""Native MLX inference for Fastino GLiNER2.5 boundary extractors."""

import bisect
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import mlx.core as mx

from .tokenizer_utils import load_tokenizer
from .utils import get_model_path, load_model

SPECIAL_TOKENS = [
    "[SEP_STRUCT]",
    "[SEP_TEXT]",
    "[P]",
    "[C]",
    "[E]",
    "[R]",
    "[L]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[DESCRIPTION]",
]


class _WhitespaceSplitter:
    pattern = re.compile(
        r"(?:https?://[^\s]+|www\.[^\s]+)"
        r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
        r"|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S",
        re.IGNORECASE,
    )

    def __call__(self, text):
        return [
            (match.group().lower(), match.start(), match.end())
            for match in self.pattern.finditer(text)
        ]


class _CharSplitter:
    pattern = re.compile(r"[A-Za-z0-9@._\-+]+|\S")

    def __call__(self, text):
        return [
            (match.group().lower(), match.start(), match.end())
            for match in self.pattern.finditer(text)
        ]


@dataclass
class _PreparedInput:
    input_ids: mx.array
    attention_mask: mx.array
    word_positions: List[int]
    marker_positions: List[int]
    words: List[str]
    offsets: List[tuple]
    text: str


def _schema_tokens(parent, labels, marker, prompt=None, descriptions=None):
    prompt_text = f"{parent}: {prompt}" if prompt else parent
    descriptions = descriptions or {}
    for label in labels:
        if label in descriptions:
            prompt_text += f" [DESCRIPTION] {label}: {descriptions[label]}"
    output = ["(", "[P]", prompt_text, "("]
    for label in labels:
        output.extend((marker, label))
    output.extend((")", ")"))
    return output


def _resolve_flat_overlaps(spans):
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span[2], span[1], -span[0]))
    ends = [span[2] for span in ordered]
    predecessors = [
        bisect.bisect_right(ends, span[1], 0, index) - 1
        for index, span in enumerate(ordered)
    ]
    best = [(0.0, ())]
    for index, span in enumerate(ordered):
        previous_score, previous = best[predecessors[index] + 1]
        selected = (previous_score + span[0], previous + (index,))
        skipped = best[index]
        best.append(selected if selected[0] > skipped[0] else skipped)
    return sorted(
        (ordered[index] for index in best[-1][1]),
        key=lambda span: (-span[0], span[1], span[2]),
    )


class GLiNER2:
    """High-level entity extraction and classification facade."""

    def __init__(self, model, tokenizer, *, word_splitter="whitespace"):
        self.model = model
        self.tokenizer = tokenizer
        if word_splitter == "whitespace":
            self.word_splitter = _WhitespaceSplitter()
        elif word_splitter == "char":
            self.word_splitter = _CharSplitter()
        else:
            raise ValueError("word_splitter must be 'whitespace' or 'char'")
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        if added:
            raise ValueError("checkpoint tokenizer is missing GLiNER2.5 special tokens")

    def _prepare(self, text, schema, max_len=None):
        if not text.rstrip().endswith((".", "!", "?")):
            text = text + "."
        max_len = max_len or self.model.config.max_len
        combined = list(schema) + ["[SEP_TEXT]"]
        marker_slots = {1, *range(4, len(schema) - 2, 2)}
        subwords = []
        marker_positions = []
        for index, token in enumerate(combined):
            position = len(subwords)
            pieces = self.tokenizer.tokenize(token)
            if index < len(schema) and index in marker_slots:
                marker_positions.append(position)
            subwords.extend(pieces)
        if len(subwords) >= max_len:
            raise ValueError("schema alone exceeds max_len")

        words = []
        offsets = []
        word_positions = []
        for word, start, end in self.word_splitter(text):
            pieces = self.tokenizer.tokenize(word)
            if not pieces:
                pieces = [self.tokenizer.unk_token]
            if len(subwords) + len(pieces) > max_len:
                break
            word_positions.append(len(subwords))
            words.append(word)
            offsets.append((start, end))
            subwords.extend(pieces)
        if not words:
            raise ValueError("no text tokens fit within max_len")
        input_ids = mx.array(
            [self.tokenizer.convert_tokens_to_ids(subwords)], dtype=mx.int32
        )
        return _PreparedInput(
            input_ids=input_ids,
            attention_mask=mx.ones(input_ids.shape, dtype=mx.bool_),
            word_positions=word_positions,
            marker_positions=marker_positions,
            words=words,
            offsets=offsets,
            text=text,
        )

    def extract_entities(
        self,
        text: str,
        entity_types: Union[Sequence[str], Dict[str, str]],
        threshold: float = 0.5,
        *,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: Optional[int] = None,
    ):
        if isinstance(entity_types, dict):
            labels = list(entity_types)
            descriptions = {
                key: value
                for key, value in entity_types.items()
                if isinstance(value, str)
            }
        else:
            labels = list(entity_types)
            descriptions = {}
        if not labels:
            return {"entities": {}}
        schema = _schema_tokens("entities", labels, "[E]", descriptions=descriptions)
        prepared = self._prepare(text, schema, max_len)
        encoded = self.model.encode(prepared.input_ids, prepared.attention_mask)
        text_states = encoded[:, prepared.word_positions]
        query_states = encoded[:, prepared.marker_positions[1:]]
        text_mask = mx.ones(text_states.shape[:2], dtype=mx.bool_)
        query_mask = mx.ones(query_states.shape[:2], dtype=mx.bool_)
        pooled, logits, null_logits = self.model.extract(
            text_states, text_mask, query_states, query_mask
        )
        probabilities = mx.sigmoid(logits.astype(mx.float32))
        abstain = mx.sigmoid(null_logits.astype(mx.float32)) > 0.5
        mx.eval(probabilities, pooled.indices, pooled.mask, abstain)
        output = {}
        for query_index, label in enumerate(labels):
            candidates = []
            if not bool(abstain[0, query_index].item()):
                for candidate_index in range(pooled.indices.shape[1]):
                    score = float(probabilities[0, candidate_index, query_index].item())
                    if (
                        not bool(pooled.mask[0, candidate_index].item())
                        or score < threshold
                    ):
                        continue
                    start = int(pooled.indices[0, candidate_index, 0].item())
                    end = int(pooled.indices[0, candidate_index, 1].item())
                    if start >= end or end > len(prepared.offsets):
                        continue
                    candidates.append((score, start, end))
            spans = _resolve_flat_overlaps(candidates)
            formatted = []
            for score, start, end in spans:
                char_start = prepared.offsets[start][0]
                char_end = prepared.offsets[end - 1][1]
                value = prepared.text[char_start:char_end]
                if include_confidence or include_spans:
                    item = {"text": value}
                    if include_confidence:
                        item["confidence"] = score
                    if include_spans:
                        item.update(start=char_start, end=char_end)
                    formatted.append(item)
                else:
                    formatted.append(value)
            output[label] = formatted
        return {"entities": output}

    def classify_text(
        self,
        text: str,
        tasks: Dict[str, Union[Sequence[str], Dict]],
        threshold: float = 0.5,
        *,
        max_len: Optional[int] = None,
    ):
        results = {}
        for task, spec in tasks.items():
            if isinstance(spec, dict):
                labels = list(spec["labels"])
                prompt = spec.get("prompt")
                descriptions = spec.get("label_descriptions")
                multi_label = spec.get("multi_label", False)
                task_threshold = spec.get("threshold", threshold)
            else:
                labels = list(spec)
                prompt = None
                descriptions = None
                multi_label = False
                task_threshold = threshold
            schema = _schema_tokens(
                task, labels, "[L]", prompt=prompt, descriptions=descriptions
            )
            prepared = self._prepare(text, schema, max_len)
            encoded = self.model.encode(prepared.input_ids, prepared.attention_mask)
            choices = encoded[:, prepared.marker_positions[1:]]
            probabilities = mx.sigmoid(self.model.classify(choices).astype(mx.float32))[
                0
            ]
            mx.eval(probabilities)
            scores = [
                float(probabilities[index].item()) for index in range(len(labels))
            ]
            if multi_label:
                results[task] = [
                    label
                    for label, score in zip(labels, scores)
                    if score >= task_threshold
                ]
            else:
                results[task] = labels[max(range(len(labels)), key=scores.__getitem__)]
        return results


def load_gliner(
    path_or_hf_repo: str,
    *,
    revision: Optional[str] = None,
    lazy: bool = False,
    word_splitter: str = "whitespace",
):
    """Download and load a GLiNER2.5 checkpoint and its tokenizer."""
    model_path = get_model_path(path_or_hf_repo, revision=revision)
    model = load_model(model_path, lazy=lazy)
    tokenizer = load_tokenizer(model_path)
    return GLiNER2(model, tokenizer, word_splitter=word_splitter)


__all__ = ["GLiNER2", "load_gliner"]
