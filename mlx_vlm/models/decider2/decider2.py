import json
import math
import re
import string
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from ..qwen3_5.config import TextConfig
from ..qwen3_5.language import LanguageModel


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _annotate(value):
    if isinstance(value, list):
        if len(value) >= 8:
            return [
                (
                    {"_index": i, **_annotate(item)}
                    if isinstance(item, dict)
                    else {"_index": i, "value": _annotate(item)}
                )
                for i, item in enumerate(value)
            ]
        return [_annotate(item) for item in value]
    if isinstance(value, dict):
        return {key: _annotate(item) for key, item in value.items()}
    return value


def _render_question(spec):
    kind = spec.get("type", "choice")
    instruction = _text(spec.get("instructions", spec.get("question", "")))
    if not instruction:
        raise ValueError("Question needs instructions")
    criteria = spec.get("criteria", spec.get("options"))
    if kind == "choice":
        if isinstance(criteria, (list, tuple)):
            criteria = dict.fromkeys(map(str, criteria))
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError("choice needs 2 to 255 options")
        names = list(criteria)
        options = [
            name if criteria[name] in (None, "") else f"{name}: {_text(criteria[name])}"
            for name in names
        ]
        legend = None
    elif kind == "score":
        if isinstance(criteria, dict):
            criteria = [criteria[key] for key in sorted(criteria, key=float)]
        if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
            raise ValueError("score needs 2 to 10 levels")
        names = list(range(len(criteria)))
        options = [f"{i}: {_text(value)}" for i, value in enumerate(criteria)]
        legend = [_text(value) for value in criteria]
    elif kind in ("noul", "bool"):
        criteria = criteria or {}
        false = criteria.get("false", criteria.get(False))
        true = criteria.get("true", criteria.get(True))
        names = [False, True]
        options = [
            "no" if false in (None, "") else f"no: {_text(false)}",
            "yes" if true in (None, "") else f"yes: {_text(true)}",
        ]
        legend = None
        kind = "noul"
    else:
        raise ValueError(f"Unsupported question type: {kind!r}")
    return {
        "type": kind,
        "instruction": instruction,
        "names": names,
        "options": options,
        "legend": legend,
        "isolated": spec.get("isolated", True),
    }


def _mapped_weights(weights):
    result = {}
    for key, value in weights.items():
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
        elif key == "lm_head.weight":
            continue
        if "conv1d.weight" in key and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)
        if key.endswith(
            (
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                "model.norm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
            )
        ):
            value = value + 1
        result[key] = value
    return result


class Decider2:
    def __init__(self, path):
        self.path = Path(path)
        config = json.loads((self.path / "config.json").read_text())
        self.settings = json.loads((self.path / "decider_config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = LanguageModel(TextConfig.from_dict(config))
        weights = _mapped_weights(mx.load(str(self.path / "model.safetensors")))
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        names = list(string.ascii_uppercase) + [
            a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
        ]
        labels = [
            (name, tokens[0])
            for name in names
            if len(tokens := self.tokenizer.encode(name, add_special_tokens=False)) == 1
        ][:255]
        if len(labels) != 255 or len({token for _, token in labels}) != 255:
            raise ValueError(
                "The decider tokenizer does not provide 255 distinct label tokens"
            )
        self.labels, token_ids = zip(*labels)
        self.label_token_ids = token_ids
        self.label_ids = mx.array(token_ids)
        self.label_weights = mx.take(
            self.model.model.embed_tokens.weight, self.label_ids, axis=0
        )

    def _prompt(self, context, question, options, max_state_tokens):
        tok = self.tokenizer
        ids = tok.encode("Context:\n" + context, add_special_tokens=False)[
            :max_state_tokens
        ]
        head = f"\n\nQuestion: {question}\nOptions:"
        tail = "\nAnswer: ("
        if len(options) <= 10:
            rendered = (
                head
                + "".join(
                    f"\n({self.labels[i]}) {value}" for i, value in enumerate(options)
                )
                + tail
            )
            ids += tok.encode(rendered, add_special_tokens=False)
        else:
            ids += tok.encode(head, add_special_tokens=False)
            open_ids = tok.encode("\n(", add_special_tokens=False)
            for i, value in enumerate(options):
                ids += (
                    open_ids
                    + [self.label_token_ids[i]]
                    + tok.encode(f") {value}", add_special_tokens=False)
                )
            ids += tok.encode(tail, add_special_tokens=False)
        return ids

    def _packed_prompt(self, context, rows, max_state_tokens):
        tok = self.tokenizer
        ids = tok.encode("Context:\n" + context, add_special_tokens=False)[
            :max_state_tokens
        ]
        slots = []
        multi = len(rows) > 1
        for index, (question, options) in enumerate(rows, 1):
            number = f" {index}" if multi else ""
            head = f"\n\nQuestion{number}: {question}\nOptions:"
            tail = f"\nAnswer{number}: ("
            if len(options) <= 10:
                piece = (
                    head
                    + "".join(
                        f"\n({self.labels[i]}) {value}"
                        for i, value in enumerate(options)
                    )
                    + tail
                )
                ids.extend(tok.encode(piece, add_special_tokens=False))
            else:
                ids.extend(tok.encode(head, add_special_tokens=False))
                open_ids = tok.encode("\n(", add_special_tokens=False)
                for i, value in enumerate(options):
                    ids += (
                        open_ids
                        + [self.label_token_ids[i]]
                        + tok.encode(f") {value}", add_special_tokens=False)
                    )
                ids.extend(tok.encode(tail, add_special_tokens=False))
            slots.append(len(ids) - 1)
        return ids, slots

    def _score_packed(self, context, rows, max_state_tokens):
        tokens, positions = self._packed_prompt(context, rows, max_state_tokens)
        length = ((len(tokens) + 63) // 64) * 64
        ids = np.full((1, length), self.tokenizer.pad_token_id, dtype=np.int32)
        ids[0, : len(tokens)] = tokens
        rope = mx.broadcast_to(mx.arange(length)[None, None, :], (3, 1, length))
        hidden = self.model.model(mx.array(ids), position_ids=rope)
        logits = hidden[0, mx.array(positions)] @ self.label_weights.T
        mx.eval(logits)
        results = []
        for i, (_, options) in enumerate(rows):
            values = (
                np.asarray(logits[i, : len(options)].astype(mx.float32))
                / self.settings["temperature"]
            )
            probabilities = np.exp(values - values.max())
            results.append((probabilities / probabilities.sum()).tolist())
        return results, len(tokens)

    def _score_rows(self, context, rows, max_state_tokens):
        prompts = [
            self._prompt(context, question, options, max_state_tokens)
            for question, options in rows
        ]
        temperature = float(self.settings["temperature"])
        results = []
        start = 0
        while start < len(rows):
            end = start + 1
            while (
                end < len(rows)
                and (end - start + 1) * max(len(p) for p in prompts[start : end + 1])
                <= 8192
            ):
                end += 1
            group = prompts[start:end]
            length = ((max(map(len, group)) + 63) // 64) * 64
            ids = np.full(
                (len(group), length), self.tokenizer.pad_token_id, dtype=np.int32
            )
            for i, prompt in enumerate(group):
                ids[i, : len(prompt)] = prompt
            positions = mx.broadcast_to(
                mx.arange(length)[None, None, :], (3, len(group), length)
            )
            hidden = self.model.model(mx.array(ids), position_ids=positions)
            slots = hidden[
                mx.arange(len(group)), mx.array([len(prompt) - 1 for prompt in group])
            ]
            logits = slots @ self.label_weights.T
            mx.eval(logits)
            for i, (_, options) in enumerate(rows[start:end]):
                values = (
                    np.asarray(logits[i, : len(options)].astype(mx.float32))
                    / temperature
                )
                probabilities = np.exp(values - values.max())
                results.append((probabilities / probabilities.sum()).tolist())
            start = end
        common = 0
        while common < min(map(len, prompts)) and all(
            prompt[common] == prompts[0][common] for prompt in prompts
        ):
            common += 1
        return results, common + sum(len(prompt) - common for prompt in prompts)

    def predict(
        self,
        state,
        questions,
        *,
        independent=True,
        isolated=None,
        max_state_tokens=32768,
    ):
        """Return calibrated typed decisions using the published decider-2b checkpoint."""
        if not questions:
            raise ValueError("At least one question is required")
        if isolated is None:
            isolated = bool(self.settings.get("isolated_levels", False))
        isolated = isolated and independent
        context = (
            state
            if isinstance(state, str)
            else json.dumps(_annotate(state), ensure_ascii=False)
        )
        rendered = {key: _render_question(spec) for key, spec in questions.items()}
        rows, lookup = [], []
        for key, question in rendered.items():
            if isolated and question["type"] == "score" and question["isolated"]:
                start = len(rows)
                for level in question["legend"]:
                    level = re.sub(r"^\s*-?\d+\s*:\s*", "", level)
                    rows.append(
                        (
                            f'{question["instruction"]}\nProposed answer: {level}\nDoes the proposed answer fit?',
                            ["no", "yes"],
                        )
                    )
                lookup.append((key, start, len(question["legend"]), True))
            else:
                lookup.append((key, len(rows), 1, False))
                rows.append((question["instruction"], question["options"]))
        if independent:
            probabilities, tokens = self._score_rows(context, rows, max_state_tokens)
        else:
            probabilities, tokens = self._score_packed(context, rows, max_state_tokens)
        answers = {}
        for key, start, count, is_isolated in lookup:
            question = rendered[key]
            if is_isolated:
                fit = [probabilities[start + i][1] for i in range(count)]
                mass = sum(fit)
                p = [value / (mass or 1e-9) for value in fit]
            else:
                p = probabilities[start]
            kind = question["type"]
            if kind == "noul":
                answer = {"type": kind, "noul": round(p[1], 4)}
            else:
                best = int(np.argmax(p))
                entropy = -sum(value * math.log(value) for value in p if value > 0)
                answer = {
                    "type": kind,
                    "confidence": round(p[best], 4),
                    "certainty": round(max(0.0, 1 - entropy / math.log(len(p))), 4),
                }
                if kind == "choice":
                    answer.update(
                        choice=question["names"][best],
                        probabilities={
                            name: round(value, 4)
                            for name, value in zip(question["names"], p)
                        },
                    )
                else:
                    answer.update(
                        score=round(sum(i * value for i, value in enumerate(p)), 2),
                        legend={
                            str(i): value for i, value in enumerate(question["legend"])
                        },
                        probabilities={
                            str(i): round(value, 4) for i, value in enumerate(p)
                        },
                    )
            if is_isolated:
                answer.update(
                    level_fit={str(i): round(value, 4) for i, value in enumerate(fit)},
                    fit_mass=round(mass, 4),
                )
            answers[key] = answer
        return {
            "model": str(self.path),
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


def load(path_or_repo, *, revision=None):
    """Load decider-2b from a local directory or the Hub."""
    path = Path(path_or_repo)
    if not path.exists():
        path = Path(
            snapshot_download(
                path_or_repo,
                revision=revision,
                allow_patterns=[
                    "model.safetensors",
                    "config.json",
                    "decider_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
            )
        )
    return Decider2(path)
