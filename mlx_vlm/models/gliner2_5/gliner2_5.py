import bisect
import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

from .boundary import BoundaryHead
from .config import ModelConfig
from .deberta import DebertaModel

SPECIAL_TOKENS = [
    "[SEP_TEXT]",
    "[SEP_STRUCT]",
    "[P]",
    "[E]",
    "[L]",
    "[DESCRIPTION]",
]

WORD_PATTERNS = {
    "whitespace": re.compile(
        r"(?:https?://[^\s]+|www\.[^\s]+)"
        r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
        r"|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S",
        re.IGNORECASE,
    ),
    "char": re.compile(r"[A-Za-z0-9@._\-+]+|\S"),
}


def _split_words(text, mode):
    try:
        pattern = WORD_PATTERNS[mode]
    except KeyError:
        raise ValueError("word_splitter must be 'whitespace' or 'char'") from None
    return [
        (match.group().lower(), match.start(), match.end())
        for match in pattern.finditer(text)
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


class Extractor(nn.Module):
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
        if config.architecture == "boundary":
            self.boundary_head = BoundaryHead(hidden_size, settings)
        elif config.architecture != "span":
            raise ValueError(
                f"Unsupported GLiNER architecture: {config.architecture!r}"
            )

    def encode(self, input_ids, attention_mask=None):
        return self.encoder(input_ids, attention_mask)

    def classify(self, choice_states):
        logits = choice_states
        for layer in self.classifier:
            logits = layer(logits)
        return logits.squeeze(-1)

    def extract(self, text_states, text_mask, query_states, query_mask):
        if self.config.architecture != "boundary":
            raise ValueError("Span checkpoints currently support classification only")
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
        if self.config.architecture == "span":
            unsupported += ("span_rep.", "count_embed.", "count_pred.")
        remapped = {}
        for key, value in weights.items():
            if key.startswith(unsupported):
                continue
            if self.config.architecture == "span" and key.startswith("classifier.2."):
                key = key.replace("classifier.2.", "classifier.3.", 1)
            key = key.replace("encoder.encoder.layer.", "encoder.encoder.layers.")
            key = key.replace(".attention.self.", ".attention.self_attn.")
            key = key.replace(".LayerNorm.", ".layer_norm.")
            remapped[key] = value
        return remapped


class Model(Extractor):
    decision_types = ("choice", "multi_label")

    def _prepare(
        self, processor, text, schema, max_len=None, word_splitter="whitespace"
    ):
        words = _split_words(text, word_splitter)
        added = processor.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        if added:
            raise ValueError("checkpoint tokenizer is missing GLiNER2.5 special tokens")
        if not text.rstrip().endswith((".", "!", "?")):
            text = text + "."
        max_len = max_len or self.config.max_len or processor.model_max_length
        schemas = schema if schema and isinstance(schema[0], list) else [schema]
        combined, marker_slots = [], set()
        for item in schemas:
            offset = len(combined)
            marker_slots.update(
                [offset + 1, *range(offset + 4, offset + len(item) - 2, 2)]
            )
            combined.extend([*item, "[SEP_STRUCT]"])
        combined[-1:] = ["[SEP_TEXT]"]
        subwords = []
        marker_positions = []
        for index, token in enumerate(combined):
            position = len(subwords)
            pieces = processor.tokenize(token)
            if index in marker_slots:
                marker_positions.append(position)
            subwords.extend(pieces)
        if len(subwords) >= max_len:
            raise ValueError("schema alone exceeds max_len")

        offsets = []
        word_positions = []
        for word, start, end in words:
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
        return_scores: bool = False,
        max_len: Optional[int] = None,
        word_splitter: str = "whitespace",
    ):
        results, specs, schemas = {}, [], []
        for task, spec in tasks.items():
            if isinstance(spec, dict):
                labels = list(spec["labels"])
                prompt = spec.get("prompt")
                descriptions = spec.get("label_descriptions")
                if descriptions is None and isinstance(spec["labels"], dict):
                    descriptions = spec["labels"]
                multi_label = spec.get("multi_label", False)
                task_threshold = spec.get(
                    "cls_threshold", spec.get("threshold", threshold)
                )
            else:
                labels = list(spec)
                prompt = None
                descriptions = None
                multi_label = False
                task_threshold = threshold
            schema = _schema_tokens(
                task, labels, "[L]", prompt=prompt, descriptions=descriptions
            )
            schemas.append(schema)
            specs.append((task, labels, multi_label, task_threshold))
        if not specs:
            return results
        prepared = self._prepare(processor, text, schemas, max_len, word_splitter)
        encoded = self.encode(prepared.input_ids)
        offset = 0
        for task, labels, multi_label, task_threshold in specs:
            positions = prepared.marker_positions[offset + 1 : offset + 1 + len(labels)]
            offset += 1 + len(labels)
            choices = encoded[:, positions]
            logits = self.classify(choices).astype(mx.float32)[0]
            probabilities = mx.sigmoid(logits) if multi_label else mx.softmax(logits)
            scores = probabilities.tolist()[: len(labels)]
            if return_scores:
                results[task] = dict(zip(labels, scores))
            elif multi_label:
                results[task] = [
                    label
                    for label, score in zip(labels, scores)
                    if score >= task_threshold
                ]
                if self.config.architecture == "span" and not results[task]:
                    results[task] = [
                        labels[max(range(len(labels)), key=scores.__getitem__)]
                    ]
            else:
                results[task] = labels[max(range(len(labels)), key=scores.__getitem__)]
        return results

    def predict(self, processor, state, questions, **kwargs):
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        tasks = {}
        for name, spec in questions.items():
            criteria = spec["criteria"]
            tasks[name] = {
                "labels": list(criteria),
                "prompt": spec.get("instructions"),
                "multi_label": spec["type"] == "multi_label",
                "label_descriptions": (
                    {k: v for k, v in criteria.items() if v is not None}
                    if isinstance(criteria, dict)
                    else None
                ),
            }
        scores = self.classify_text(
            processor, state, tasks, return_scores=True, **kwargs
        )
        answers = {}
        for name, values in scores.items():
            spec = questions[name]
            if spec["type"] == "multi_label":
                threshold = spec.get("threshold", 0.5)
                value = [label for label, score in values.items() if score >= threshold]
            else:
                value = max(values, key=values.get)
            if (
                self.config.architecture == "span"
                and spec["type"] == "multi_label"
                and not value
            ):
                value = [max(values, key=values.get)]
            score_key = "scores" if spec["type"] == "multi_label" else "probabilities"
            answers[name] = {"type": spec["type"], "value": value, score_key: values}
        return {"model": self.config.model_type, "answers": answers}


__all__ = ["Model", "ModelConfig"]
