import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_vl.qwen3_vl import masked_scatter

QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


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

    def __init__(self, config):
        super().__init__(config)
        self.head = JointSchemaHead(**config.head_config)

    def __call__(self, input_ids, question_spans, option_spans, qtype, **media):
        embeds = self.language_model.model.embed_tokens(input_ids)
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        for pixels, grid, token in (
            ("pixel_values", "image_grid_thw", self.config.image_token_index),
            ("pixel_values_videos", "video_grid_thw", self.config.video_token_index),
        ):
            if media.get(pixels) is not None:
                states, _ = self.vision_tower(media[pixels].astype(dtype), media[grid])
                embeds = masked_scatter(
                    embeds,
                    mx.broadcast_to((input_ids == token)[..., None], embeds.shape),
                    states,
                )
        position_ids, _ = self.language_model.get_rope_index(
            input_ids, media.get("image_grid_thw"), media.get("video_grid_thw")
        )
        hidden = self.language_model.model(
            input_ids, inputs_embeds=embeds, position_ids=position_ids
        )
        hidden = self.head.hidden_norm(hidden[0])
        flat = np.asarray(input_ids[0])
        lexical_ids = np.concatenate([flat[s:e] for s, e in option_spans])
        lexical_spans = np.cumsum([0, *(e - s for s, e in option_spans)])
        lm_head, ids = self.language_model.lm_head, mx.array(lexical_ids)
        embeddings = lm_head.weight[ids]
        if "scales" in lm_head:
            biases = lm_head.get("biases")
            embeddings = mx.dequantize(
                embeddings,
                lm_head.scales[ids],
                None if biases is None else biases[ids],
                group_size=lm_head.group_size,
                bits=lm_head.bits,
                mode=lm_head.mode,
            )
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
        )

    def sanitize(self, weights):
        backbone, head = {}, {}
        for key, value in weights.items():
            if key.startswith(
                ("model.", "lm_head", "language_model.", "vision_tower.")
            ):
                backbone[key] = value
                continue
            key = "head." + key.removeprefix("head.")
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

        def predicate(path, module):
            if path.startswith("head."):
                return False
            return True if base is None else base(path, module)

        return predicate

    def predict(self, processor, state, questions, **kwargs):
        questions = {
            name: {**spec, "type": "noul" if spec["type"] == "bool" else spec["type"]}
            for name, spec in questions.items()
        }
        return Clef(self, processor).predict(state, questions, **kwargs)


def _render(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _options(question):
    kind = question["type"]
    criteria = question.get("criteria")
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
    if kind == "score":
        return [(str(index), value) for index, value in enumerate(criteria)]
    raise ValueError(f"Unsupported question type: {kind!r}")


class Clef:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def _tokens(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _sequence(
        self,
        state,
        questions,
        images=None,
        videos=None,
        media_kwargs=None,
        max_length=16384,
        max_state_tokens=None,
    ):
        tokens = self._tokens
        schema = tokens("\n\nSCHEMA FIELDS:\n")
        rows = []
        for index, (name, question) in enumerate(questions.items()):
            schema += tokens(
                f"\nFIELD {index + 1}\nID: {name}\nTYPE: {question['type']}\n"
                "INSTRUCTION: "
            )
            start = len(schema)
            schema += tokens(_render(question.get("instructions") or str(name)))
            question_span = (start, len(schema))
            schema += tokens("\nALLOWED OPTIONS:\n")
            spans, labels = [], []
            for number, (option, description) in enumerate(_options(question), 1):
                schema += tokens(f"OPTION {number}: ")
                start = len(schema)
                semantics = {"option_id": option}
                if description is not None:
                    semantics["description"] = description
                schema += tokens(_render(semantics))
                spans.append((start, len(schema)))
                labels.append(option)
                schema += tokens("\n")
            schema += tokens("END FIELD\n")
            rows.append((name, question, question_span, spans, labels))

        prefix = tokens(
            "<|im_start|>system\nRead the complete state and schema. Decide every "
            "field jointly. Each answer must be exactly one of that field's allowed "
            "options.<|im_end|>\n<|im_start|>user\nSTATE:\n"
        )
        suffix = tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "JOINT SCHEMA DECISIONS:"
        )
        media = {}
        images, videos = list(images or []), list(videos or [])
        media_kwargs = dict(media_kwargs or {})
        if videos and "video_metadata" not in media_kwargs:
            video_processor = self.processor.video_processor
            fps = media_kwargs.pop("fps", None)
            num_frames = media_kwargs.pop("num_frames", None)
            metadata = []
            for index, frames in enumerate(videos):
                total = len(frames)
                count = num_frames
                if count is None:
                    count = int(total / 24 * (fps or video_processor.fps))
                    count = min(
                        max(count, video_processor.min_frames),
                        video_processor.max_frames,
                        total,
                    )
                indices = np.linspace(0, total - 1, count).round().astype(int)
                videos[index] = [frames[i] for i in indices]
                metadata.append({"frames_indices": indices.tolist(), "fps": 24})
            media_kwargs["video_metadata"] = metadata
        if images or videos:
            encoded = self.processor(
                text=[
                    "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
                    # transformers 5 keeps the outer vision tokens around timestamped frames
                    + "<|vision_start|><|vision_start|><|video_pad|><|vision_end|><|vision_end|>"
                    * len(videos)
                    + "\n"
                ],
                images=images or None,
                videos=videos or None,
                **media_kwargs,
            )
            prefix += np.asarray(encoded["input_ids"])[0].tolist()
            media = {
                key: mx.array(np.asarray(encoded[key]))
                for key in (
                    "pixel_values",
                    "image_grid_thw",
                    "pixel_values_videos",
                    "video_grid_thw",
                )
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
                question,
                (span[0] + offset, span[1] + offset),
                [(s + offset, e + offset) for s, e in spans],
                labels,
            )
            for name, question, span, spans, labels in rows
        ]
        return prefix + state_ids + schema + suffix, rows, media

    def predict(self, state, questions, **kwargs):
        ids, rows, media = self._sequence(state, questions, **kwargs)
        logits = self.model(
            mx.array([ids]),
            [(row[2], len(row[3])) for row in rows],
            [span for row in rows for span in row[3]],
            mx.array([QUESTION_TYPES[row[1]["type"]] for row in rows]),
            **media,
        )
        mx.eval(logits)
        answers = {}
        start = 0
        for name, question, _, spans, labels in rows:
            values = logits[start : start + len(spans)].astype(mx.float32)
            start += len(spans)
            p = dict(zip(labels, mx.softmax(values).tolist()))
            kind = question["type"]
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
            "model": "clef",
            "answers": answers,
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
        }
