import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from ..modernbert import ModelConfig
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
        self.activation = nn.GELU(approx="precise")

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


class DecisionModel(nn.Module):
    def __init__(self, encoder_config, head_layers=2):
        super().__init__()
        self.encoder = ModernBert(ModelConfig.from_dict(encoder_config))
        width = encoder_config["hidden_size"]
        self.type_emb = nn.Embedding(3, width)
        self.head = DecisionHead(width, head_layers)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(approx="precise"),
            nn.Linear(width, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(width + 4, 256), nn.GELU(approx="precise"), nn.Linear(256, 2)
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


def _weights(path):
    weights = mx.load(str(path))
    result = {}
    for key, value in weights.items():
        if ".self_attn.in_proj_" in key:
            prefix, suffix = key.split(".self_attn.in_proj_")
            for name, split in zip(
                ("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)
            ):
                result[f"{prefix}.attention.{name}.{suffix}"] = split
        elif ".self_attn.out_proj." in key:
            result[key.replace(".self_attn.out_proj.", ".attention.out_proj.")] = value
        elif key.startswith(("scorer.", "act_head.")):
            prefix, index, suffix = key.split(".", 2)
            result[f"{prefix}.layers.{index}.{suffix}"] = value
        else:
            result[key] = value
    return result


def _options(question):
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice needs at least two options")
        return list(criteria), [
            key if not value else f"{key}: {value}" for key, value in criteria.items()
        ]
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score needs at least two levels")
        return list(range(len(criteria))), [
            f"level {i}: {value}" for i, value in enumerate(criteria)
        ]
    if kind == "noul":
        criteria = criteria or {}
        return [False, True], [
            "false: " + (criteria.get("false") or "no, the statement does not hold"),
            "true: " + (criteria.get("true") or "yes, the statement holds"),
        ]
    raise ValueError(f"Unsupported question type: {kind!r}")


class Laya:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / "rl_agent_config.json").read_text())
        encoder_config = json.loads((self.path / "encoder" / "config.json").read_text())
        rope = encoder_config.get("rope_parameters", {})
        if isinstance(rope, dict):
            encoder_config["global_rope_theta"] = rope.get("full_attention", {}).get(
                "rope_theta", 160000
            )
            encoder_config["local_rope_theta"] = rope.get("sliding_attention", {}).get(
                "rope_theta", 10000
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path / "tokenizer")
        self.model = DecisionModel(encoder_config, self.config["head_layers"])
        self.model.load_weights(
            list(_weights(self.path / "model.safetensors").items()), strict=True
        )
        self.model.eval()
        mx.eval(self.model.parameters())

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
        ids = np.full((len(rows), length), self.tokenizer.pad_token_id, dtype=np.int32)
        attention = np.zeros_like(ids)
        positions = np.zeros((len(rows), options), dtype=np.int32)
        valid = np.zeros((len(rows), options), dtype=bool)
        for i, (_, _, _, tokens, markers) in enumerate(rows):
            ids[i, : len(tokens)] = tokens
            attention[i, : len(tokens)] = 1
            positions[i, : len(markers)] = markers
            valid[i, : len(markers)] = True
        types = mx.array([QUESTION_TYPES[row[1]["type"]] for row in rows])
        logits, act = self.model(
            mx.array(ids),
            mx.array(attention),
            mx.array(positions),
            mx.array(valid),
            types,
        )
        mx.eval(logits, act)
        logits, act = np.asarray(logits), np.asarray(
            mx.softmax(act.astype(mx.float32), -1)
        )
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
            z = logits[i, :count] / temperature
            probabilities = np.exp(z - z.max())
            probabilities /= probabilities.sum()
            confidence = round(
                float(
                    1
                    + np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)))
                    / math.log(count)
                ),
                4,
            )
            extra = {"rl_agent": {"act_probability": float(act[i, 0])}}
            if kind == "choice":
                answers[key] = {
                    "type": kind,
                    "choice": labels[int(probabilities.argmax())],
                    "probabilities": dict(
                        zip(labels, [round(float(p), 4) for p in probabilities])
                    ),
                    "confidence": confidence,
                    **extra,
                }
            elif kind == "score":
                answers[key] = {
                    "type": kind,
                    "score": round(float(np.dot(np.arange(count), probabilities)), 4),
                    "legend": {
                        str(n): text for n, text in enumerate(question["criteria"])
                    },
                    "probabilities": {
                        str(n): round(float(p), 4) for n, p in enumerate(probabilities)
                    },
                    "confidence": confidence,
                    **extra,
                }
            else:
                answers[key] = {
                    "type": kind,
                    "noul": round(float(probabilities[1]), 4),
                    **extra,
                }
        return {
            "model": self.config.get("model_name", "laya"),
            "answers": answers,
            "usage": {"input_tokens": int(attention.sum()), "output_tokens": 0},
        }


def load(path_or_repo, *, subfolder=None, revision=None):
    from ...utils import get_model_path

    prefix = f"{subfolder}/" if subfolder else ""
    path = get_model_path(
        path_or_repo,
        revision=revision,
        allow_patterns=[
            f"{prefix}model.safetensors",
            f"{prefix}rl_agent_config.json",
            f"{prefix}encoder/*",
            f"{prefix}tokenizer/*",
        ],
    )
    return Laya(path / subfolder if subfolder else path)
