import json
import math

import mlx.core as mx
import mlx.nn as nn

from ..modernbert import ModelConfig as EncoderConfig
from ..modernbert.modernbert import Model as ModernBert

QUESTION_TYPES = {"choice": 0, "score": 1, "noul": 2}


class DecisionLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.MultiHeadAttention(width, width // 64, bias=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, 4 * width)
        self.linear2 = nn.Linear(4 * width, width)
        self.activation = nn.ReLU()

    def __call__(self, hidden, mask):
        normalized = self.norm1(hidden)
        hidden = hidden + self.attention(normalized, normalized, normalized, mask)
        return hidden + self.linear2(self.activation(self.linear1(self.norm2(hidden))))


class DecisionHead(nn.Module):
    def __init__(self, width, count):
        super().__init__()
        self.layers = [DecisionLayer(width) for _ in range(count)]

    def __call__(self, hidden, mask):
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden


class Model(nn.Module):
    decision_types = ("choice", "score", "bool", "noul")

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ModernBert(EncoderConfig.from_dict(config.encoder_config))
        width = config.encoder_config["hidden_size"]
        self.type_emb = nn.Embedding(3, width)
        self.head = DecisionHead(width, config.head_layers)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(width + 4, 256), nn.GELU(), nn.Linear(256, 2)
        )
        self.temperature = mx.ones((3,), dtype=mx.float32)

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        hidden = self.encoder(input_ids, attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        mask = mx.where(attention_mask[:, None, None, :] == 1, 0.0, -1e9).astype(
            hidden.dtype
        )
        hidden = self.head(hidden, mask)
        batch = mx.arange(hidden.shape[0])[:, None]
        markers = hidden[batch, marker_pos]
        logits = self.scorer(markers).squeeze(-1).astype(mx.float32)
        logits = mx.where(marker_mask, logits, -1e4)
        probabilities = mx.softmax(logits, axis=-1)
        count = mx.maximum(mx.sum(marker_mask, axis=-1), 2).astype(mx.float32)
        entropy = -mx.sum(
            probabilities * mx.log(mx.maximum(probabilities, 1e-9)), axis=-1
        ) / mx.log(count)
        top = mx.sort(probabilities, axis=-1)[:, -2:]
        features = mx.stack(
            (top[:, 1], top[:, 1] - top[:, 0], entropy, count / 255), axis=-1
        )
        pooled = hidden[:, 0].astype(mx.float32)
        act = self.act_head(mx.concatenate((pooled, features), axis=-1))
        return logits, act

    def sanitize(self, weights):
        result = {}
        for key, value in weights.items():
            if ".self_attn.in_proj_" in key:
                prefix, suffix = key.split(".self_attn.in_proj_")
                for name, split in zip(
                    ("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)
                ):
                    result[f"{prefix}.attention.{name}.{suffix}"] = split
            elif ".self_attn.out_proj." in key:
                result[key.replace(".self_attn.out_proj.", ".attention.out_proj.")] = (
                    value
                )
            elif key.startswith(("scorer.", "act_head.")) and not key.startswith(
                ("scorer.layers.", "act_head.layers.")
            ):
                prefix, index, suffix = key.split(".", 2)
                result[f"{prefix}.layers.{index}.{suffix}"] = value
            else:
                result[key] = value
        return result

    def predict(self, processor, state, questions, **kwargs):
        questions = {
            name: {**spec, "type": "noul" if spec["type"] == "bool" else spec["type"]}
            for name, spec in questions.items()
        }
        return Laya(self, processor).predict(state, questions, **kwargs)


def _render_criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _options(question):
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice needs at least two options")
        return list(criteria), [
            (
                key
                if value is None or value == ""
                else f"{key}: {_render_criterion(value)}"
            )
            for key, value in criteria.items()
        ]
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score needs at least two levels")
        return list(range(len(criteria))), [
            f"level {i}: {_render_criterion(value)}" for i, value in enumerate(criteria)
        ]
    if kind == "noul":
        criteria = criteria or {}
        return [False, True], [
            label
            + ": "
            + (
                default
                if criteria.get(label) is None or criteria.get(label) == ""
                else _render_criterion(criteria[label])
            )
            for label, default in (
                ("false", "no, the statement does not hold"),
                ("true", "yes, the statement holds"),
            )
        ]
    raise ValueError(f"Unsupported question type: {kind!r}")


class Laya:
    def __init__(self, model, tokenizer):
        self.config = model.config.decision_config
        self.tokenizer = tokenizer
        self.model = model

    def _sequence(self, state, question, option_text):
        tok = self.tokenizer
        mask = tok.mask_token
        instruction = question["instructions"]
        if not isinstance(instruction, str):
            instruction = json.dumps(instruction)
        head = tok(
            f'{question["type"]} question: {instruction.replace(mask, " ")}',
            add_special_tokens=False,
        )["input_ids"]
        option_ids = [
            [tok.mask_token_id]
            + tok(" " + text.replace(mask, " "), add_special_tokens=False)["input_ids"][
                :48
            ]
            for text in option_text
        ]
        budget = self.config["head_max_len"] - sum(map(len, option_ids))
        if budget < 16:
            per = max(4, (self.config["head_max_len"] - 16) // len(option_ids))
            option_ids = [ids[:per] for ids in option_ids]
            budget = self.config["head_max_len"] - sum(map(len, option_ids))
        ids = [tok.cls_token_id] + head[: max(8, budget)] + [tok.sep_token_id]
        markers = []
        for option in option_ids:
            markers.append(len(ids))
            ids.extend(option)
        ids.append(tok.sep_token_id)
        room = max(0, self.config["max_len"] - len(ids) - 1)
        text = (
            state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        )
        ids.extend(
            tok(text.replace(mask, " "), add_special_tokens=False)["input_ids"][:room]
        )
        ids.append(tok.sep_token_id)
        markers = [
            position for position in markers if position < self.config["max_len"]
        ]
        if len(markers) != len(option_text):
            raise ValueError("Options do not fit the Laya question budget")
        return ids[: self.config["max_len"]], markers

    def predict(self, state, questions):
        if not questions:
            raise ValueError("At least one question is required")
        rows = []
        for key, question in questions.items():
            labels, option_text = _options(question)
            ids, markers = self._sequence(state, question, option_text)
            rows.append((key, question, labels, ids, markers))
        length = max(len(row[3]) for row in rows)
        options = max(len(row[4]) for row in rows)
        ids, attention, positions, valid = [], [], [], []
        for _, _, _, tokens, markers in rows:
            padding = length - len(tokens)
            missing = options - len(markers)
            ids.append(tokens + [self.tokenizer.pad_token_id] * padding)
            attention.append([1] * len(tokens) + [0] * padding)
            positions.append(markers + [0] * missing)
            valid.append([True] * len(markers) + [False] * missing)
        logits, act = self.model(
            mx.array(ids, dtype=mx.int32),
            mx.array(attention, dtype=mx.int32),
            mx.array(positions, dtype=mx.int32),
            mx.array(valid, dtype=mx.bool_),
            mx.array([QUESTION_TYPES[row[1]["type"]] for row in rows]),
        )
        act = mx.softmax(act.astype(mx.float32), -1)
        mx.eval(logits, act)
        answers = {}
        for i, (key, question, labels, _, markers) in enumerate(rows):
            kind = question["type"]
            count = len(markers)
            bucket = (
                "2"
                if count <= 2
                else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
            )
            temperature = self.config.get("temperature_by_options", {}).get(
                f"{kind}:{bucket}", self.config["temperature"][QUESTION_TYPES[kind]]
            )
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                temperature = 1.0
            temperature = (
                min(5.0, max(0.5, temperature)) if math.isfinite(temperature) else 1.0
            )
            z = logits[i, :count] / temperature
            probabilities = mx.exp(z - z.max())
            probabilities /= probabilities.sum()
            confidence = round(
                (
                    1
                    + mx.sum(probabilities * mx.log(mx.clip(probabilities, 1e-12, 1)))
                    / math.log(count)
                ).item(),
                4,
            )
            metadata = {"rl_agent": {"act_probability": act[i, 0].item()}}
            if kind == "noul":
                probability = round(probabilities[1].item(), 4)
                answers[key] = {
                    "type": "bool",
                    "value": probability >= 0.5,
                    "probability": probability,
                    "metadata": metadata,
                }
            else:
                metadata["confidence"] = confidence
                answers[key] = {
                    "type": kind,
                    "value": (
                        labels[probabilities.argmax().item()]
                        if kind == "choice"
                        else round(mx.sum(mx.arange(count) * probabilities).item(), 4)
                    ),
                    "probabilities": {
                        label if kind == "choice" else str(label): round(p.item(), 4)
                        for label, p in zip(labels, probabilities)
                    },
                    "metadata": metadata,
                }
                if kind == "score":
                    metadata["legend"] = {
                        str(n): text for n, text in enumerate(question["criteria"])
                    }
        return {
            "model": self.config.get("model_name", "laya"),
            "answers": answers,
            "usage": {
                "input_tokens": sum(len(row[3]) for row in rows),
                "output_tokens": 0,
            },
        }
