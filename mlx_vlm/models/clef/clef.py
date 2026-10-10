import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...decision import context_limit
from ..base import InputEmbeddingsFeatures
from ..qwen3_5 import Model as Qwen3_5Model
from .processing_clef import encode_record


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

    def __call__(
        self,
        hidden,
        question_spans,
        option_spans,
        lexical,
        types,
        counts,
        checkpoint=None,
    ):
        length = hidden.shape[0]
        memory = self.memory_projection(hidden)[None]
        if checkpoint is not None:
            checkpoint(memory)
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
            if checkpoint is not None:
                checkpoint(routed)
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
            if checkpoint is not None:
                checkpoint(fields)
        fields = self.field_norm(fields[0])[owner]

        anchor = questions + global_vector
        anchor = anchor / mx.maximum(
            mx.linalg.norm(anchor, axis=-1, keepdims=True), 1e-12
        )
        lexical = lexical / mx.maximum(
            mx.linalg.norm(lexical, axis=-1, keepdims=True), 1e-12
        )
        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        prior = prior_scale * mx.sum(lexical * anchor[owner], axis=-1)
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


class Model(Qwen3_5Model):
    decision_types = ("choice", "score", "bool", "noul")
    decision_media = ("images", "videos")

    def __init__(self, config):
        super().__init__(config)
        self.head = JointSchemaHead(**config.head_config)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        """Scatter each modality separately, while sharing multimodal positions."""
        embeds = self.language_model.model.embed_tokens(input_ids)
        for pixels, grid, token in (
            (pixel_values, kwargs.get("image_grid_thw"), self.config.image_token_index),
            (
                kwargs.get("pixel_values_videos"),
                kwargs.get("video_grid_thw"),
                self.config.video_token_index,
            ),
        ):
            if pixels is not None:
                dtype = self.vision_tower.patch_embed.proj.weight.dtype
                features, _ = self.vision_tower(pixels.astype(dtype), grid)
                embeds, _ = self.merge_input_ids_with_image_features(
                    features, embeds, input_ids, token, token
                )
        positions, deltas = self.language_model.get_rope_index(
            input_ids,
            kwargs.get("image_grid_thw"),
            kwargs.get("video_grid_thw"),
            kwargs.get("mask"),
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=embeds, position_ids=positions, rope_deltas=deltas
        )

    def _output_embeddings(self, ids):
        layer = (
            self.language_model.model.embed_tokens
            if self.config.text_config.tie_word_embeddings
            else self.language_model.lm_head
        )
        if hasattr(layer, "scales"):
            # Gather first: never dequantize the full vocabulary matrix.
            biases = getattr(layer, "biases", None)
            return mx.dequantize(
                layer.weight[ids],
                layer.scales[ids],
                biases[ids] if biases is not None else None,
                group_size=layer.group_size,
                bits=layer.bits,
                mode=layer.mode,
            )
        return layer.weight[ids]

    def __call__(self, input_ids, question_spans, option_spans, qtype, **media):
        features = self.get_input_embeddings(input_ids, **media)
        hidden = self.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
        )
        return self._score_hidden(
            hidden, input_ids, question_spans, option_spans, qtype
        )

    def _score_hidden(
        self, hidden, input_ids, question_spans, option_spans, qtype, checkpoint=None
    ):
        hidden = self.head.hidden_norm(hidden[0])
        flat = np.asarray(input_ids[0])
        lexical_ids = np.concatenate([flat[s:e] for s, e in option_spans])
        lexical_spans = np.cumsum([0, *(e - s for s, e in option_spans)])
        embeddings = self._output_embeddings(mx.array(lexical_ids))
        lexical = (
            _span_means(
                list(zip(lexical_spans[:-1], lexical_spans[1:])), len(lexical_ids)
            )
            @ embeddings
        )
        return self.head(
            hidden,
            [span for span, _ in question_spans],
            option_spans,
            lexical.astype(hidden.dtype),
            qtype,
            [count for _, count in question_spans],
            checkpoint=checkpoint,
        )

    def score_record(self, hidden, input_ids, record, checkpoint=None):
        counts = [len(q.option_spans) for q in record.questions]
        logits = self._score_hidden(
            hidden,
            input_ids,
            [(q.question_span, count) for q, count in zip(record.questions, counts)],
            [span for q in record.questions for span in q.option_spans],
            mx.array([q.question_type for q in record.questions]),
            checkpoint=checkpoint,
        )
        return mx.split(logits, np.cumsum(counts[:-1]).tolist())

    def decide(self, record):
        ids = mx.array(record.input_ids)[None]
        media = {
            k: mx.array(v)
            for k, v in (record.media or {}).items()
            if k
            in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            )
        }
        features = self.get_input_embeddings(ids, **media)
        hidden = self.language_model.model(
            ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
        )
        return self.score_record(hidden, ids, record)

    def make_decision_engine(self, processor, **settings):
        from .inference import DecisionEngine

        return DecisionEngine(self, processor, **settings)

    def sanitize(self, weights):
        backbone, head = {}, {}
        for key, value in weights.items():
            if key.startswith(
                ("model.", "lm_head", "language_model.", "vision_tower.")
            ):
                backbone[key] = value
                continue
            key = "head." + key.removeprefix("head.")
            key = key.replace(".feedforward.layers.", ".feedforward.").replace(
                "head.residual_scorer.layers.", "head.residual_scorer."
            )
            if key.endswith(("in_proj_weight", "in_proj_bias")):
                prefix, suffix = key.rsplit(".in_proj_", 1)
                for name, part in zip(
                    ("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)
                ):
                    head[f"{prefix}.{name}.{suffix}"] = part
                continue
            for layer in ("feedforward", "residual_scorer"):
                key = key.replace(f"{layer}.3.", f"{layer}.1.")
            head[key] = value
        return {**super().sanitize(backbone), **head}

    @property
    def quant_predicate(self):
        base = super().quant_predicate
        lexical_layer = (
            "language_model.model.embed_tokens"
            if self.config.text_config.tie_word_embeddings
            else "language_model.lm_head"
        )

        def predicate(path, module):
            if path.startswith("head."):
                return False
            if (
                self.config.decision_quantization == "preserve-output"
                and path == lexical_layer
            ):
                return False
            return base(path, module) if base else True

        return predicate

    def predict(self, processor, state, questions, **kwargs):
        questions = {
            name: {**spec, "type": "noul" if spec["type"] == "bool" else spec["type"]}
            for name, spec in questions.items()
        }
        return Clef(self, processor).predict(state, questions, **kwargs)


def format_result(record, questions, logits):
    probabilities = [
        mx.softmax(value.astype(mx.float32), precise=True) for value in logits
    ]
    mx.eval(probabilities)
    raw = {
        question.question_id: dict(zip(question.option_ids, p.tolist()))
        for question, p in zip(record.questions, probabilities)
    }
    if not all(math.isfinite(p) for values in raw.values() for p in values.values()):
        raise RuntimeError("Decision model produced non-finite probabilities")
    answers = {}
    for name, question in questions.items():
        p = raw[name]
        kind = question["type"]
        if kind in ("bool", "noul"):
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
        answers[name] = {
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
            answers[name]["metadata"]["legend"] = dict(
                zip(labels, question["criteria"])
            )
    return {
        "response": {
            "model": "clef",
            "answers": answers,
            "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0},
        },
        "probabilities": raw,
    }


class Clef:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor

    def predict(
        self, state, questions, max_length=None, max_state_tokens=None, **media
    ):
        record = encode_record(
            getattr(self.processor, "tokenizer", self.processor),
            {"state": state, "questions": questions, **media},
            max_length=context_limit(self.model.config, max_length),
            max_state_tokens=max_state_tokens,
            processor=self.processor,
        )
        return format_result(record, questions, self.model.decide(record))["response"]
