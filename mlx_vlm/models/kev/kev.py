"""Kev decision models (jaredpalmer/kev-27b).

The prompt, the pointer readout and the answer statistics follow the release's
code (kev/api.py and kev/model.py of https://github.com/jaredpalmer/kev,
Apache-2.0; see LICENSE in this directory).
"""

import math
import re
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..cache import ArraysCache
from ..qwen3_5_text import Model as Qwen3_5TextModel

# The state, question, option, option-end and decide delimiters (kev/model.py SPECIAL).
DELIMITERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
# The release's serving limits: a state of up to 65,536 tokens (its delimiter
# included), and a state plus one question of up to 8,192 tokens more.
MAX_STATE_TOKENS = 65536
MAX_ROW_TOKENS = MAX_STATE_TOKENS + 8192
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def render(value, indent=0):
    """The text the model reads for a JSON value; field names are kept as labels."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "\n".join(
            f"{pad}- {render(item, indent + 1).lstrip()}" for item in value
        )
    if not isinstance(value, Mapping):
        return str(value)
    return "\n".join(
        (
            f"{pad}{key}:\n{render(item, indent + 1)}"
            if isinstance(item, (Mapping, list, tuple))
            else f"{pad}{key}: {render(item)}"
        )
        for key, item in value.items()
    )


def _option(name, description):
    if description is None or description == "":
        return name
    return f"{name}: {render(description)}"


def _options(kind, criteria):
    """Probability keys and option texts in the release's order."""
    if kind == "bool":
        criteria = criteria or {}
        return ["false", "true"], [
            _option("no", criteria.get("false")),
            _option("yes", criteria.get("true")),
        ]
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        return list(criteria), [_option(k, v) for k, v in criteria.items()]
    if kind == "score":
        return [str(i) for i in range(len(criteria))], [render(c) for c in criteria]
    raise ValueError(f"Unsupported question type: {kind!r}")


def encode(tokenizer, state, questions):
    """The release's token layout: the state's ids, and per question a tuple of
    its name, kind, question, probability keys, branch ids and the read-out
    rows within the branch (<decide>, then each option's closing delimiter)."""
    delimiters = tokenizer.convert_tokens_to_ids(list(DELIMITERS))
    if any(i is None or i == tokenizer.unk_token_id for i in delimiters):
        raise ValueError("The tokenizer lacks Kev's delimiter tokens")
    start, question_id, option_id, option_end, decide = delimiters

    def tokens(value):
        # Caller text never yields a delimiter: <|name|> is read as <¦name¦>.
        text = _SPECIAL.sub(r"<¦\1¦>", render(value))
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    rows = []
    for name, question in questions.items():
        kind = "bool" if question["type"] == "noul" else question["type"]
        labels, options = _options(kind, question.get("criteria"))
        branch = [question_id] + tokens(question.get("instructions"))
        reads = []
        for option in options:
            branch += [option_id] + tokens(option) + [option_end]
            reads.append(len(branch) - 1)
        branch.append(decide)
        rows.append((name, kind, question, labels, branch, [len(branch) - 1] + reads))
    return [start] + tokens(state), rows


def _normalize(p):
    total = sum(p)
    return [1 / len(p)] * len(p) if total == 0 else [x / total for x in p]


def choice_confidence(p):
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    k = len(p)
    return 1.0 if k == 1 else (max(_normalize(p)) - 1 / k) / (1 - 1 / k)


def score_confidence(p):
    """1 - E|level - mode| / D, D being that spread for a uniform distribution."""
    n = len(p)
    if n == 1:
        return 1.0
    p = _normalize(p)
    mode = max(range(n), key=p.__getitem__)
    spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - sum(q * abs(i - mode) for i, q in enumerate(p)) / spread)


def load_release_head(path):
    """The pointer head and calibration temperature of the release's head.pt."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Reading Kev's head.pt needs PyTorch (pip install torch); "
            "converted checkpoints do not."
        ) from error
    meta = torch.load(path, map_location="cpu", weights_only=True)
    if meta.get("weights") != "full":
        raise ValueError(
            f"{path}: only full-weight Kev checkpoints load directly; merge a "
            "LoRA release first (the Kev repository's merge_lora_checkpoint.py)"
        )
    if meta.get("option_isolation") or meta.get("special_embeddings"):
        raise ValueError(f"{path}: this Kev head layout is not supported")
    weights = {
        f"head.{name}": mx.array(tensor.detach().float().numpy())
        for name, tensor in meta["head"].items()
    }
    weights["head.temperature"] = mx.array(
        [float(meta.get("temperature", 1.0))], dtype=mx.float32
    )
    return weights


class PointerHead(nn.Module):
    def __init__(self, hidden_size, dims):
        super().__init__()
        self.q = nn.Linear(hidden_size, dims)
        self.k = nn.Linear(hidden_size, dims)
        self.temperature = mx.ones((1,))

    def __call__(self, decide, options):
        scale = 1 / math.sqrt(self.q.weight.shape[0])
        return (self.k(options) @ self.q(decide)) * scale / self.temperature


class Model(Qwen3_5TextModel):
    decision_types = ("choice", "score", "bool", "noul")

    def __init__(self, config):
        super().__init__(config)
        # Kev reads final hidden states; the release has no LM head.
        self.language_model.pop("lm_head", None)
        self.head = PointerHead(config.hidden_size, config.pointer_dim)

    def __call__(self, input_ids, cache=None):
        """Final-normed hidden states."""
        if cache is None:
            cache = self.make_cache()
        return self.language_model.model(input_ids, cache=cache)

    def sanitize(self, weights):
        if not any(key.startswith("head.") for key in weights):
            path = self.config.model_path and Path(self.config.model_path) / "head.pt"
            if not path or not path.is_file():
                raise ValueError("Kev checkpoint without a pointer head (head.pt)")
            weights = {**weights, **load_release_head(path)}
        backbone, head = {}, {}
        for key, value in weights.items():
            if key.startswith("head."):
                head[key] = value
            elif "lm_head" in key:
                continue
            elif key.startswith(("embed_tokens.", "layers.", "norm.")):
                # transformers' Qwen3_5TextModel export
                backbone["model." + key] = value
            else:
                backbone[key] = value
        return {**super().sanitize(backbone), **head}

    @property
    def quant_predicate(self):
        base = super().quant_predicate

        def predicate(path, module):
            if path.startswith("head."):
                return False
            return True if base is None else base(path, module)

        return predicate

    @property
    def cast_predicate(self):
        base = super().cast_predicate
        return lambda path: not path.startswith("head.") and base(path)

    def predict(
        self,
        processor,
        state,
        questions,
        max_state_tokens=MAX_STATE_TOKENS,
        max_length=None,
        prefill_step_size=2048,
    ):
        max_length = max_length or MAX_ROW_TOKENS
        state_ids, rows = encode(
            getattr(processor, "tokenizer", processor), state, questions
        )
        if len(state_ids) > max_state_tokens:
            raise ValueError(
                f"state is {len(state_ids)} tokens, over the "
                f"{max_state_tokens}-token limit"
            )
        for name, _, _, _, branch, _ in rows:
            if len(state_ids) + len(branch) > max_length:
                raise ValueError(
                    f"question {name!r} needs {len(state_ids) + len(branch)} tokens "
                    f"with its state; maximum is {max_length}"
                )

        probabilities = self.probabilities(state_ids, rows, prefill_step_size)
        answers = {
            name: _answer(kind, question, labels, p)
            for (name, kind, question, labels, _, _), p in zip(rows, probabilities)
        }
        input_tokens = len(state_ids) + sum(len(row[4]) for row in rows)
        return {
            "model": "kev",
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }

    def probabilities(self, state_ids, rows, prefill_step_size=2048):
        """Unrounded option probabilities of encoded questions (see `encode`)."""
        # The state runs once; each question continues it as its own causal row,
        # so questions never see one another. After each question the attention
        # caches are trimmed back to the state and the DeltaNet states restored.
        cache = self.make_cache()
        for i in range(0, len(state_ids), prefill_step_size):
            self(mx.array([state_ids[i : i + prefill_step_size]]), cache=cache)
            mx.eval([entry.state for entry in cache])
        state = [list(e.cache) if isinstance(e, ArraysCache) else None for e in cache]
        out = []
        for *_, branch, reads in rows:
            hidden = self(mx.array([branch]), cache=cache)[0]
            picked = hidden[mx.array(reads)].astype(mx.float32)
            out.append(mx.softmax(self.head(picked[0], picked[1:])).tolist())
            for entry, arrays in zip(cache, state):
                if arrays is None:
                    entry.trim(len(branch))
                else:
                    entry.cache = list(arrays)
        return out


def _answer(kind, question, labels, p):
    if kind == "bool":
        probability = round(p[1], 4)
        return {"type": "bool", "value": probability >= 0.5, "probability": probability}
    probabilities = {label: round(q, 4) for label, q in zip(labels, p)}
    if kind == "choice":
        return {
            "type": "choice",
            "value": labels[max(range(len(p)), key=p.__getitem__)],
            "probabilities": probabilities,
            "metadata": {"confidence": round(choice_confidence(p), 4)},
        }
    return {
        "type": "score",
        "value": round(sum(i * q for i, q in enumerate(p)), 4),
        "probabilities": probabilities,
        "metadata": {
            "confidence": round(score_confidence(p), 4),
            "legend": dict(zip(labels, question["criteria"])),
        },
    }
