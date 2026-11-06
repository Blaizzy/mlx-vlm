import json
import string

import mlx.core as mx
from PIL import Image

from ..qwen3_5 import Model as Qwen3_5Model


class Model(Qwen3_5Model):
    decision_types = ("choice", "score", "bool", "noul")

    def __call__(self, input_ids, **media):
        features = self.get_input_embeddings(input_ids, **media)
        hidden = self.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
        )
        return self.language_model.lm_head(hidden[:, -1])

    def sanitize(self, weights):
        return super().sanitize(merge_lora(weights, self.config.decision_config))

    @property
    def quant_predicate(self):
        base = super().quant_predicate
        return lambda path, module: path != "language_model.lm_head" and (
            base is None or base(path, module)
        )

    def predict(self, processor, state, questions, **kwargs):
        return Jev(self, processor).predict(state, questions, **kwargs)


def merge_lora(weights, settings):
    backbone, lora = {}, {}
    for key, value in weights.items():
        if key.startswith("base_model.model."):
            target, _, side = key.removeprefix("base_model.model.").rpartition(".lora_")
            lora.setdefault(f"{target}.weight", {})[side[0]] = value
        else:
            backbone[key] = value
    scale = settings["lora_alpha"] / settings["lora_r"]
    for key, pair in lora.items():
        weight = backbone[key]
        backbone[key] = (
            weight.astype(mx.float32) + scale * (pair["B"] @ pair["A"])
        ).astype(weight.dtype)
    return backbone


def _render_criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _state(state, image_token):
    if state is None:
        return "", []
    if not isinstance(state, list):
        return _render_criterion(state), []
    text, images = "", []
    for part in state:
        if isinstance(part, Image.Image):
            image = part
        elif isinstance(part, dict) and "image" in part:
            image = part["image"]
        elif isinstance(part, dict) and part.get("type") == "image_url":
            image = part["image_url"]["url"]
        elif isinstance(part, dict) and part.get("type") == "text":
            text += part["text"]
            continue
        else:
            text += _render_criterion(part)
            continue
        if image_token is None:
            raise ValueError("This model does not accept images")
        text += image_token
        images.append(image)
    return text, images


def _options(question):
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria:
            raise ValueError("bool questions take no criteria")
        return ["false", "true"], ["false", "true"]
    if kind == "score":
        if len(criteria) != 6:
            raise ValueError("score questions need six levels, 0 to 5")
        return [str(i) for i in range(6)], [str(i) for i in range(6)]
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if len(criteria) > 256:
            raise ValueError("choice questions take at most 256 options")
        return list(criteria), [
            key if value in (None, "") else f"{key}: {_render_criterion(value)}"
            for key, value in criteria.items()
        ]
    raise ValueError(f"Unsupported question type: {kind!r}")


class Jev:
    prefix = ""
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.settings = model.config.decision_config
        self.labels, self.label_ids = self._labels()
        self.tokens = 0

    def _labels(self):
        letters = string.ascii_uppercase
        labels = []
        for label in [*letters, *(a + b for a in letters for b in letters)]:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            context = self.tokenizer.encode(f"x\n{label}) y", add_special_tokens=False)
            if len(ids) == 1 and ids[0] in context:
                labels.append((label, ids[0]))
            if len(labels) == 256:
                break
        return tuple(zip(*labels))

    def _scores(self, kind, ids, media, count):
        start, end = self.settings["ranges"][kind]
        tokens = (
            self.label_ids[:count]
            if kind == "choice"
            else self.settings["verbalizer_ids"][start : start + count]
        )
        bias = [
            self.settings["bias"][start + i] if start + i < end else 0.0
            for i in range(count)
        ]
        logits = self.model(ids, **media)[0, mx.array(tokens)]
        return logits.astype(mx.float32) + mx.array(bias)

    def _pass(self, kind, state, images, question, options):
        lines = (
            [f"{self.labels[i]}) {o}" for i, o in enumerate(options)]
            if kind == "choice"
            else options
        )
        text = (
            f"{self.prefix}[kind] {kind}\n[state] {state}\n[question] {question}\n"
            "[options]\n" + "\n".join(lines) + "\n[decision]:"
        )
        media = {}
        if images:
            inputs = self.processor(
                text=[text], images=images, add_special_tokens=False
            )
            ids = mx.array(inputs["input_ids"])
            media = {
                key: mx.array(inputs[key])
                for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
                if inputs.get(key) is not None
            }
        else:
            ids = mx.array(
                [self.tokenizer(text, add_special_tokens=False)["input_ids"]]
            )
        self.tokens += ids.shape[1]
        scores = self._scores(kind, ids, media, len(options))
        temperature = self.settings["temperature_by_type"][kind]
        return mx.softmax(scores / temperature).tolist()

    def _choice(self, state, images, question, options):
        return self._pass("choice", state, images, question, options)

    def predict(self, state, questions):
        questions = {
            name: {**spec, "type": "noul" if spec["type"] == "bool" else spec["type"]}
            for name, spec in questions.items()
        }
        state, images = _state(state, self.image_token)
        answers = {}
        for name, question in questions.items():
            kind = question["type"]
            labels, options = _options(question)
            instruction = _render_criterion(question.get("instructions") or name)
            if kind == "choice":
                p = self._choice(state, images, instruction, options)
            else:
                p = self._pass(kind, state, images, instruction, options)
            if kind == "noul":
                probability = round(p[1], 4)
                answers[name] = {
                    "type": "bool",
                    "value": probability >= 0.5,
                    "probability": probability,
                }
                continue
            best = max(range(len(p)), key=p.__getitem__)
            answers[name] = {
                "type": kind,
                "value": (
                    labels[best]
                    if kind == "choice"
                    else round(sum(i * value for i, value in enumerate(p)), 4)
                ),
                "probabilities": {
                    label: round(value, 4) for label, value in zip(labels, p)
                },
                "metadata": {"confidence": round(p[best], 4)},
            }
            if kind == "score":
                answers[name]["metadata"]["legend"] = dict(
                    zip(labels, question["criteria"])
                )
        return {
            "model": self.model.config.model_type,
            "answers": answers,
            "usage": {"input_tokens": self.tokens, "output_tokens": 0},
        }
