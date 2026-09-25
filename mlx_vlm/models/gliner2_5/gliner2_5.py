import bisect
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .boundary import BoundaryHead
from .config import ModelConfig
from .deberta import DebertaModel

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
    word_positions: list[int]
    marker_positions: list[int]
    offsets: list[tuple]
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


class BoundaryExtractor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        settings = config.boundary_head
        if settings.get("candidate_pool", "shared") != "shared":
            raise ValueError("GLiNER2.5 MLX currently supports shared candidate pools")
        if settings.get("candidate_attention_layers", 0) != 0:
            raise ValueError("candidate attention layers are not supported yet")
        if settings.get("query_attention_layers", 0) != 0:
            raise ValueError("query attention layers are not supported yet")
        self.encoder = DebertaModel(config.encoder_config)
        hidden_size = config.encoder_config.hidden_size
        self.classifier = [
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_size * 2, 1),
        ]
        self.boundary_head = BoundaryHead(hidden_size, settings)

    def encode(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)

    def classify(self, choice_states):
        logits = choice_states
        for layer in self.classifier:
            logits = layer(logits)
        return logits.squeeze(-1)

    def extract(self, text_states, text_mask, query_states, query_mask):
        return self.boundary_head(text_states, text_mask, query_states, query_mask)

    def __call__(self, input_ids, attention_mask=None):
        return self.encode(input_ids, attention_mask)

    def sanitize(self, weights):
        unsupported = (
            "boundary_head.boundary_proposer.",
            "boundary_head.pair_scorer.",
            "boundary_head.candidate_encoder.",
            "record_decoder.",
            "relation_scorer.",
        )
        remapped = {}
        for key, value in weights.items():
            if key.startswith(unsupported):
                continue
            key = key.replace("encoder.encoder.layer.", "encoder.encoder.layers.")
            key = key.replace(".attention.self.", ".attention.self_attn.")
            key = key.replace(".LayerNorm.", ".layer_norm.")
            remapped[key] = value
        return remapped


class Model(BoundaryExtractor):
    def _prepare(
        self, processor, text, schema, max_len=None, word_splitter="whitespace"
    ):
        if word_splitter == "whitespace":
            splitter = _WhitespaceSplitter()
        elif word_splitter == "char":
            splitter = _CharSplitter()
        else:
            raise ValueError("word_splitter must be 'whitespace' or 'char'")
        added = processor.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        if added:
            raise ValueError("checkpoint tokenizer is missing GLiNER2.5 special tokens")
        if not text.rstrip().endswith((".", "!", "?")):
            text = text + "."
        max_len = max_len or self.config.max_len
        combined = list(schema) + ["[SEP_TEXT]"]
        marker_slots = {1, *range(4, len(schema) - 2, 2)}
        subwords = []
        marker_positions = []
        for index, token in enumerate(combined):
            position = len(subwords)
            pieces = processor.tokenize(token)
            if index < len(schema) and index in marker_slots:
                marker_positions.append(position)
            subwords.extend(pieces)
        if len(subwords) >= max_len:
            raise ValueError("schema alone exceeds max_len")

        offsets = []
        word_positions = []
        for word, start, end in splitter(text):
            pieces = processor.tokenize(word)
            if not pieces:
                pieces = [processor.unk_token]
            if len(subwords) + len(pieces) > max_len:
                break
            word_positions.append(len(subwords))
            offsets.append((start, end))
            subwords.extend(pieces)
        if not offsets:
            raise ValueError("no text tokens fit within max_len")
        input_ids = mx.array(
            [processor.convert_tokens_to_ids(subwords)], dtype=mx.int32
        )
        return _PreparedInput(
            input_ids=input_ids,
            word_positions=word_positions,
            marker_positions=marker_positions,
            offsets=offsets,
            text=text,
        )

    def extract_entities(
        self,
        processor,
        text: str,
        entity_types: Union[Sequence[str], Dict[str, str]],
        threshold: float = 0.5,
        *,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: Optional[int] = None,
        word_splitter: str = "whitespace",
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
        prepared = self._prepare(processor, text, schema, max_len, word_splitter)
        encoded = self.encode(prepared.input_ids)
        text_states = encoded[:, prepared.word_positions]
        query_states = encoded[:, prepared.marker_positions[1:]]
        text_mask = mx.ones(text_states.shape[:2], dtype=mx.bool_)
        query_mask = mx.ones(query_states.shape[:2], dtype=mx.bool_)
        pooled, logits, null_logits = self.extract(
            text_states, text_mask, query_states, query_mask
        )
        probabilities = mx.sigmoid(logits.astype(mx.float32))
        abstain = mx.sigmoid(null_logits.astype(mx.float32)) > 0.5
        scores = probabilities[0].tolist()
        keeps = pooled.mask[0].tolist()
        spans_index = pooled.indices[0].tolist()
        abstains = abstain[0].tolist()
        word_count = len(prepared.offsets)

        output = {}
        for query_index, label in enumerate(labels):
            candidates = []
            if not abstains[query_index]:
                for candidate_index, keep in enumerate(keeps):
                    if not keep:
                        continue
                    score = scores[candidate_index][query_index]
                    if score < threshold:
                        continue
                    start, end = spans_index[candidate_index]
                    if start >= end or end > word_count:
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
        processor,
        text: str,
        tasks: Dict[str, Union[Sequence[str], Dict]],
        threshold: float = 0.5,
        *,
        max_len: Optional[int] = None,
        word_splitter: str = "whitespace",
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
            prepared = self._prepare(processor, text, schema, max_len, word_splitter)
            encoded = self.encode(prepared.input_ids)
            choices = encoded[:, prepared.marker_positions[1:]]
            probabilities = mx.sigmoid(self.classify(choices).astype(mx.float32))[0]
            scores = probabilities.tolist()[: len(labels)]
            if multi_label:
                results[task] = [
                    label
                    for label, score in zip(labels, scores)
                    if score >= task_threshold
                ]
            else:
                results[task] = labels[max(range(len(labels)), key=scores.__getitem__)]
        return results


__all__ = ["Model", "ModelConfig"]
