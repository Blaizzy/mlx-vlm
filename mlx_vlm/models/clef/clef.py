import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..qwen3_5 import Model as Qwen3_5Model

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
MEDIA_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
BACKBONE_PREFIXES = ("model.", "lm_head", "mtp.", "language_model.", "vision_tower.")


def _render(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _options(kind, criteria):
    if kind == "noul":
        defaults = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        defaults.update(criteria or {})
        return [(key, defaults[key]) for key in ("true", "false")]
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        return sorted((str(key), value) for key, value in criteria.items())
    return [(str(index), value) for index, value in enumerate(criteria)]


def _normalize(x, eps=1e-12):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


def _span_means(spans, length):
    weights = np.zeros((len(spans), length), dtype=np.float32)
    for row, (start, end) in enumerate(spans):
        weights[row, start:end] = 1.0 / (end - start)
    return mx.array(weights)


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiHeadAttention(width, heads, bias=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = [
            nn.Linear(width, feedforward),
            nn.Linear(feedforward, width),
        ]

    def __call__(self, queries, memory):
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        hidden = nn.gelu(self.feedforward[0](self.feedforward_norm(queries)))
        return queries + self.feedforward[1](hidden)


class FieldDecoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(width, heads, bias=True)
        self.multihead_attn = nn.MultiHeadAttention(width, heads, bias=True)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x, memory):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward)
            for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [
            FieldDecoderLayer(width, heads, feedforward) for _ in range(layers)
        ]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = [nn.Linear(width * 4, width), nn.Linear(width, 1)]
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def __call__(self, hidden, question_spans, option_spans, lexical, types, counts):
        length = hidden.shape[0]
        memory = self.memory_projection(hidden)[None]
        global_vector = hidden[-1]
        questions = (_span_means(question_spans, length) @ hidden).astype(hidden.dtype)
        contexts = (_span_means(option_spans, length) @ hidden).astype(hidden.dtype)
        owner = mx.array([i for i, count in enumerate(counts) for _ in range(count)])

        routed = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(questions)[owner]
        )[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]

        base = self.question_projection(questions)
        scores = mx.sum(routed * base[owner], axis=-1) / math.sqrt(routed.shape[-1])
        bounds = np.cumsum([0, *counts]).tolist()
        summaries = mx.stack(
            [
                mx.sum(
                    mx.softmax(scores[s:e], precise=True)[:, None] * routed[s:e], axis=0
                )
                for s, e in zip(bounds[:-1], bounds[1:])
            ]
        )
        fields = (
            base
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector)
            + self.type_embedding(types)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[owner]

        anchor = _normalize(questions + global_vector)[owner]
        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        prior = prior_scale * mx.sum(_normalize(lexical) * anchor, axis=-1)
        options = self.option_norm(routed)
        cosine = mx.sum(fields * options, axis=-1) / mx.maximum(
            mx.linalg.norm(fields, axis=-1) * mx.linalg.norm(options, axis=-1), 1e-8
        )
        features = mx.concatenate(
            [fields, options, fields * options, mx.abs(fields - options)], axis=-1
        )
        residual = self.residual_scorer[1](
            nn.gelu(self.residual_scorer[0](features))
        ).squeeze(-1)
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        joint = joint_scale * cosine + residual
        return prior + mx.sigmoid(self.residual_gate) * joint


def _sanitize_head(weights):
    result = {}
    for key, value in weights.items():
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            prefix, suffix = key.rsplit(".in_proj_", 1)
            for name, part in zip(
                ("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)
            ):
                result[f"{prefix}.{name}.{suffix}"] = part
            continue
        for layer in ("feedforward", "residual_scorer"):
            key = key.replace(f"{layer}.3.", f"{layer}.1.")
        result[key] = value
    return result


class Model(Qwen3_5Model):
    decision_types = ("choice", "score", "bool", "noul")

    def __init__(self, config):
        super().__init__(config)
        self.head = JointSchemaHead(**config.head_config)

    def sanitize(self, weights):
        backbone = {k: v for k, v in weights.items() if k.startswith(BACKBONE_PREFIXES)}
        head = {
            k.removeprefix("head."): v
            for k, v in weights.items()
            if not k.startswith(BACKBONE_PREFIXES)
        }
        weights = super().sanitize(backbone)
        weights.update({f"head.{k}": v for k, v in _sanitize_head(head).items()})
        return weights

    @property
    def quant_predicate(self):
        base = super().quant_predicate

        def predicate(path, module):
            if path.startswith("head."):
                return False
            return True if base is None else base(path, module)

        return predicate

    def _output_rows(self, ids):
        layer = getattr(self.language_model, "lm_head", None)
        if layer is None:
            layer = self.language_model.model.embed_tokens
        if "scales" not in layer:
            return layer.weight[ids]
        biases = layer.get("biases")
        return mx.dequantize(
            layer.weight[ids],
            layer.scales[ids],
            None if biases is None else biases[ids],
            group_size=layer.group_size,
            bits=layer.bits,
            mode=layer.mode,
        )

    def encode(
        self,
        processor,
        state,
        questions,
        images=None,
        videos=None,
        media_kwargs=None,
        max_length=16384,
        max_state_tokens=None,
    ):
        tokenizer = getattr(processor, "tokenizer", processor)

        def tokens(text):
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        schema = tokens("\n\nSCHEMA FIELDS:\n")
        rows = []
        for index, (name, question) in enumerate(questions.items()):
            kind = "noul" if question["type"] == "bool" else question["type"]
            schema += tokens(
                f"\nFIELD {index + 1}\nID: {name}\nTYPE: {kind}\nINSTRUCTION: "
            )
            start = len(schema)
            instructions = question.get("instructions")
            schema += tokens(_render(instructions or str(name)))
            question_span = (start, len(schema))
            schema += tokens("\nALLOWED OPTIONS:\n")
            spans, ids = [], []
            for number, (option, description) in enumerate(
                _options(kind, question.get("criteria")), 1
            ):
                schema += tokens(f"OPTION {number}: ")
                start = len(schema)
                semantics = {"option_id": option}
                if description is not None:
                    semantics["description"] = description
                schema += tokens(_render(semantics))
                spans.append((start, len(schema)))
                ids.append(option)
                schema += tokens("\n")
            schema += tokens("END FIELD\n")
            rows.append((name, kind, question_span, spans, ids))

        prefix = tokens(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        )
        suffix = tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "JOINT SCHEMA DECISIONS:"
        )
        media = {}
        images, videos = list(images or []), list(videos or [])
        if images or videos:
            encoded = processor(
                text=[
                    IMAGE_PLACEHOLDER * len(images)
                    + VIDEO_PLACEHOLDER * len(videos)
                    + "\n"
                ],
                images=images or None,
                videos=videos or None,
                **(media_kwargs or {}),
            )
            prefix += np.asarray(encoded["input_ids"])[0].tolist()
            media = {
                key: mx.array(np.asarray(encoded[key]))
                for key in MEDIA_KEYS
                if encoded.get(key) is not None
            }
        state_ids = tokens(_render(state))[:max_state_tokens]
        fixed = len(prefix) + len(schema) + len(suffix)
        if fixed > max_length:
            raise ValueError(
                f"schema requires {fixed} tokens before state; maximum is {max_length}"
            )
        state_ids = state_ids[: max_length - fixed]
        offset = len(prefix) + len(state_ids)
        rows = [
            (
                name,
                kind,
                (q[0] + offset, q[1] + offset),
                [(s + offset, e + offset) for s, e in spans],
                ids,
            )
            for name, kind, q, spans, ids in rows
        ]
        return prefix + state_ids + schema + suffix, rows, media

    def decide(self, input_ids, rows, media=None):
        input_ids = mx.array([input_ids])
        features = self.get_input_embeddings(input_ids, **(media or {}))
        hidden = self.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
        )
        hidden = self.head.hidden_norm(hidden[0])
        option_spans = [span for row in rows for span in row[3]]
        flat = np.asarray(input_ids[0])
        lexical_ids = np.concatenate([flat[s:e] for s, e in option_spans])
        lexical_spans = np.cumsum([0, *(e - s for s, e in option_spans)])
        lexical = _span_means(
            list(zip(lexical_spans[:-1], lexical_spans[1:])), len(lexical_ids)
        ) @ self._output_rows(mx.array(lexical_ids))
        logits = self.head(
            hidden,
            [row[2] for row in rows],
            option_spans,
            lexical.astype(hidden.dtype),
            mx.array([QUESTION_TYPES[row[1]] for row in rows]),
            [len(row[3]) for row in rows],
        )
        bounds = np.cumsum([0, *(len(row[3]) for row in rows)]).tolist()
        return [logits[s:e] for s, e in zip(bounds[:-1], bounds[1:])]

    def predict(self, processor, state, questions, **kwargs):
        input_ids, rows, media = self.encode(processor, state, questions, **kwargs)
        logits = self.decide(input_ids, rows, media)
        mx.eval(logits)
        answers = {}
        for (name, kind, _, _, ids), values in zip(rows, logits):
            p = dict(zip(ids, mx.softmax(values.astype(mx.float32)).tolist()))
            question = questions[name]
            if kind == "noul":
                probability = round(p["true"], 4)
                answers[name] = {
                    "type": "bool",
                    "value": probability >= 0.5,
                    "probability": probability,
                }
                continue
            labels = (
                list(question["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            best = max(labels, key=p.__getitem__)
            answer = {
                "type": kind,
                "value": (
                    best
                    if kind == "choice"
                    else round(sum(i * p[label] for i, label in enumerate(labels)), 4)
                ),
                "probabilities": {label: round(p[label], 4) for label in labels},
                "metadata": {"confidence": round(p[best], 4)},
            }
            if kind == "score":
                answer["metadata"]["legend"] = dict(zip(labels, question["criteria"]))
            answers[name] = answer
        return {
            "model": str(getattr(self.config, "model_path", None) or "clef"),
            "answers": answers,
            "usage": {"input_tokens": len(input_ids), "output_tokens": 0},
        }
