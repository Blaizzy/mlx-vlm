import json
import math
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..qwen3_5 import Model as Qwen3_5Model

KINDS = {"bool": "noul", "noul": "noul", "choice": "choice", "score": "score"}
SCORE_LEVELS = 6
MAX_STATE_TOKENS = 32768


class SystemOneLinear(nn.Module):
    """A base projection with JEV's System 1 LoRA, applied only when enabled."""

    def __init__(self, linear, rank, scale):
        super().__init__()
        output_dims, input_dims = linear.weight.shape
        self.linear = linear
        self.lora_a = mx.zeros((input_dims, rank))
        self.lora_b = mx.zeros((rank, output_dims))
        self.scale = scale
        self.enabled = False

    def __call__(self, x):
        y = self.linear(x)
        if not self.enabled:
            return y
        update = (x @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)
        return y + (self.scale * update).astype(y.dtype)


class DecisionHead(nn.Module):
    """Linear float32 read-out of the final-norm hidden state into decision slots."""

    def __init__(self, hidden_size, rows):
        super().__init__()
        self.weight = mx.zeros((rows, hidden_size), dtype=mx.float32)
        self.bias = mx.zeros((rows,), dtype=mx.float32)

    def __call__(self, hidden):
        return hidden.astype(mx.float32) @ self.weight.T + self.bias


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _rows(kind, count, slots):
    """Decision-head rows for a question: trained slots first, then wide labels."""
    start, end = slots[kind]
    if kind != "choice":
        return list(range(start, end))
    native = min(count, end - start)
    return list(range(start, start + native)) + list(range(end, end + count - native))


class Model(Qwen3_5Model):
    decision_types = ("choice", "score", "bool", "noul")

    def __init__(self, config):
        super().__init__(config)
        # system_one() flips shared module state, so decisions on one model must
        # not overlap (the server runs each request in its own thread).
        self._decision_lock = threading.Lock()
        settings = config.decision_config
        adapter = settings.get("adapter") or {}
        self._targets = tuple(adapter.get("target_modules", ()))
        if adapter.get("rank"):
            for layer in self.language_model.model.layers:
                for block in ("self_attn", "linear_attn", "mlp"):
                    parent = getattr(layer, block, None)
                    for name in self._targets if parent is not None else ():
                        if isinstance(parent.get(name), nn.Linear):
                            setattr(
                                parent,
                                name,
                                SystemOneLinear(
                                    parent[name], adapter["rank"], adapter["scale"]
                                ),
                            )
        slots = settings.get("slots", {})
        if slots:
            rows = slots["choice"][0] + len(settings["choice_labels"])
            self.decision_head = DecisionHead(config.text_config.hidden_size, rows)

    @contextmanager
    def system_one(self):
        """Switch the System 1 adapter on; the base model is System 2.

        The switch is shared by every caller of this model: do not run
        ``generate()`` on the same object while a decision is in progress.
        """
        adapters = [
            module
            for _, module in self.language_model.named_modules()
            if isinstance(module, SystemOneLinear)
        ]
        for module in adapters:
            module.enabled = True
        try:
            yield
        finally:
            for module in adapters:
                module.enabled = False

    def sanitize(self, weights):
        weights = super().sanitize(weights)
        if not self._targets:
            return weights
        pattern = re.compile(
            r"^(language_model\.model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\."
            rf"(?:{'|'.join(map(re.escape, self._targets))}))\."
            r"(weight|bias|scales|biases)$"
        )
        return {
            pattern.sub(r"\1.linear.\2", key): value for key, value in weights.items()
        }

    def quantization_path_aliases(self, path):
        if not path.endswith(".linear"):
            return ()
        path = path[: -len(".linear")]
        prefix = "language_model."
        return (path, path[len(prefix) :]) if path.startswith(prefix) else (path,)

    @property
    def cast_predicate(self):
        base = super().cast_predicate

        def predicate(path):
            return not path.startswith("decision_head.") and base(path)

        return predicate

    def predict(self, processor, state, questions, **kwargs):
        from ...generate.common import wired_limit

        if getattr(self, "decision_head", None) is None:
            raise ValueError("This checkpoint has no JEV decision head")
        with self._decision_lock, wired_limit(self), self.system_one():
            try:
                return Jev(self, processor).predict(state, questions, **kwargs)
            finally:
                mx.clear_cache()


class Jev:
    """JEV's bare decision protocol: one forward pass, no generated tokens."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.settings = model.config.decision_config
        self.labels = self.settings["choice_labels"]
        self.slots = {
            kind: tuple(span) for kind, span in self.settings["slots"].items()
        }
        self.temperatures = self.settings["temperature_by_type"]
        self.native_choices = self.slots["choice"][1] - self.slots["choice"][0]

    def _state(self, state):
        """Split a state into text pieces and images, as JEV's /v1/decide does."""
        if isinstance(state, str):
            return [state], []
        if isinstance(state, Mapping):
            return [_text(dict(state))], []
        if not isinstance(state, (list, tuple)):
            raise ValueError("state must be a string, an object, or a list of parts")
        pieces, images = [], []
        for part in state:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, Mapping) and "image" in part:
                images.append(part["image"])
                pieces.append(None)
            else:
                pieces.append(_text(part))
        return pieces, images

    def _question(self, spec):
        kind = KINDS.get(spec.get("type"))
        if kind is None:
            raise ValueError(f"JEV does not support {spec.get('type')!r} questions")
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("JEV questions need instructions")
        criteria = spec.get("criteria")
        legend = None
        if kind == "noul":
            if criteria is not None and not isinstance(criteria, Mapping):
                raise ValueError(
                    "bool criteria must map false and true to descriptions"
                )
            if criteria and any(value not in (None, "") for value in criteria.values()):
                raise ValueError(
                    "JEV bool questions take no criteria descriptions; phrase the "
                    "instructions so that true is the outcome of interest"
                )
            names, lines = [False, True], ["false", "true"]
        elif kind == "score":
            if not isinstance(criteria, (list, tuple)) or len(criteria) != SCORE_LEVELS:
                raise ValueError(
                    "JEV scores on a fixed 0-5 scale: give six level descriptions"
                )
            names = list(range(SCORE_LEVELS))
            lines = [str(level) for level in names]
            legend = {str(level): _text(value) for level, value in enumerate(criteria)}
        else:
            if isinstance(criteria, (list, tuple)):
                if len(set(criteria)) != len(criteria):
                    raise ValueError("JEV choice options must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, Mapping) or not (
                2 <= len(criteria) <= len(self.labels)
            ):
                raise ValueError(f"JEV choice takes 2 to {len(self.labels)} options")
            names = list(criteria)
            lines = [
                f"{label}) {name}"
                + ("" if criteria[name] in (None, "") else f": {_text(criteria[name])}")
                for label, name in zip(self.labels, names)
            ]
            if any("\n" in line for line in lines):
                raise ValueError("JEV choice options must fit on one line each")
        tail = f"\n[question] {instructions}\n[options]\n" + "\n".join(lines)
        return {
            "kind": kind,
            "names": names,
            "head": f"[kind] {kind}\n[state] ",
            "tail": tail + "\n[decision]:",
            "rows": _rows(kind, len(names), self.slots),
            "legend": legend,
        }

    def _encode_images(self, images):
        """Preprocess the state's images once; every question reuses them."""
        from ...utils import load_image

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("Images in the state need the model's processor")
        features = image_processor(
            images=[load_image(image).convert("RGB") for image in images]
        )
        grid = features["image_grid_thw"]
        merge = image_processor.merge_size**2
        return {
            "pixel_values": mx.array(features["pixel_values"]),
            "image_grid_thw": mx.array(grid),
            "counts": [int(np.prod(thw)) // merge for thw in grid],
            "features": None,
        }

    def _image_prompt(self, pieces, images, question):
        config = self.model.config
        token = self.tokenizer.convert_ids_to_tokens
        image = token(config.image_token_id)
        counts = iter(images["counts"])
        text = "".join(
            (
                piece
                if piece is not None
                else token(config.vision_start_token_id)
                + image * next(counts)
                + token(config.vision_end_token_id)
            )
            for piece in pieces
        )
        return self.tokenizer.encode(
            question["head"] + text + question["tail"], add_special_tokens=False
        )

    def _hidden_with_images(self, ids, images):
        model = self.model
        input_ids = mx.array([ids])
        if images["features"] is None:
            pixels = images["pixel_values"].astype(
                model.vision_tower.patch_embed.proj.weight.dtype
            )
            images["features"], _ = model.vision_tower(pixels, images["image_grid_thw"])
        embedded = model.get_input_embeddings(
            input_ids,
            images["pixel_values"],
            image_grid_thw=images["image_grid_thw"],
            cached_image_features=images["features"],
        )
        hidden = model.language_model.model(
            input_ids,
            inputs_embeds=embedded.inputs_embeds,
            position_ids=embedded.position_ids,
        )
        return hidden[:, -1]

    def _hidden_text(self, ids):
        length = len(ids)
        positions = mx.broadcast_to(mx.arange(length)[None, None, :], (3, 1, length))
        hidden = self.model.language_model.model(
            mx.array([ids]), position_ids=positions
        )
        return hidden[:, -1]

    def _check_state_length(self, pieces, images, max_state_tokens):
        if max_state_tokens is None:
            return
        text = "".join(piece for piece in pieces if piece is not None)
        length = len(self.tokenizer.encode(text, add_special_tokens=False))
        if images:
            length += sum(images["counts"])
        if length > max_state_tokens:
            raise ValueError(
                f"The state has {length} tokens; JEV accepts at most "
                f"{max_state_tokens} (raise max_state_tokens to allow more)"
            )

    def probabilities(self, state, questions, max_state_tokens=MAX_STATE_TOKENS):
        """Calibrated probabilities per question, in option order, plus token count.

        Each question is its own forward pass, so an answer does not depend on the
        other questions in the request.
        """
        pieces, images = self._state(state)
        rendered = [self._question(spec) for spec in questions.values()]
        if images:
            images = self._encode_images(images)
        self._check_state_length(pieces, images, max_state_tokens)
        if images:
            prompts = [self._image_prompt(pieces, images, q) for q in rendered]
            hidden = mx.concatenate(
                [self._hidden_with_images(p, images) for p in prompts]
            )
        else:
            text = "".join(pieces)
            prompts = [
                self.tokenizer.encode(
                    q["head"] + text + q["tail"], add_special_tokens=False
                )
                for q in rendered
            ]
            hidden = mx.concatenate([self._hidden_text(p) for p in prompts])
        logits = self.model.decision_head(hidden)
        results = []
        for i, question in enumerate(rendered):
            z = (
                logits[i, mx.array(question["rows"])]
                / self.temperatures[question["kind"]]
            )
            results.append(mx.softmax(z))
        mx.eval(results)
        return rendered, [p.tolist() for p in results], sum(map(len, prompts))

    def predict(self, state, questions, *, max_state_tokens=MAX_STATE_TOKENS):
        """Return calibrated typed decisions using JEV's System 1 decision head."""
        rendered, probabilities, tokens = self.probabilities(
            state, questions, max_state_tokens
        )
        answers = {}
        for name, question, p in zip(questions, rendered, probabilities):
            kind = question["kind"]
            if kind == "noul":
                probability = round(p[1], 4)
                answers[name] = {
                    "type": "bool",
                    "value": probability >= 0.5,
                    "probability": probability,
                }
                continue
            best = int(np.argmax(p))
            entropy = -sum(value * math.log(value) for value in p if value > 0)
            metadata = {
                "confidence": round(p[best], 4),
                "certainty": round(max(0.0, 1 - entropy / math.log(len(p))), 4),
            }
            if kind == "choice":
                metadata["adaptation"] = (
                    "native" if len(p) <= self.native_choices else "wide-labels"
                )
            else:
                metadata["legend"] = question["legend"]
            answers[name] = {
                "type": kind,
                "value": (
                    question["names"][best]
                    if kind == "choice"
                    else round(sum(i * value for i, value in enumerate(p)), 4)
                ),
                "probabilities": {
                    option if kind == "choice" else str(option): round(value, 4)
                    for option, value in zip(question["names"], p)
                },
                "metadata": metadata,
            }
        return {
            "model": str(getattr(self.model, "model_path", "jev")),
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
