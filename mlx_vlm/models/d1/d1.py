import mlx.core as mx
import numpy as np
from PIL import Image

from ..cache import ArraysCache, KVCache
from ..lfm2_vl import Model as Lfm2VlModel
from .prompt import (
    as_question,
    cap_pixels,
    image_markup,
    prefix_text,
    readout,
    suffix_text,
)

# Longer trunks are read in chunks of this many tokens: one pass over a long trunk
# keeps every conv layer's full input alive.
PREFILL_STEP = 2048


def _tile(cache, count):
    """Fresh caches holding `count` copies of the trunk's state."""
    tiled = [KVCache() if isinstance(c, KVCache) else ArraysCache(1) for c in cache]
    for copy, source in zip(tiled, cache):
        copy.state = [mx.broadcast_to(x, (count, *x.shape[1:])) for x in source.state]
    return tiled


def _chunks(lengths, trunk, budget):
    """Branch batches, longest first, each in request order: at most `budget` padded
    tokens, trunk included, and no branch padded past twice its length."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out = []
    while order:
        longest, size = lengths[order[0]], 1
        while (
            size < len(order)
            and (size + 1) * (trunk + longest) <= budget
            and 2 * lengths[order[size]] >= longest
        ):
            size += 1
        out.append(sorted(order[:size]))
        order = order[size:]
    return out


def _answer(question, p):
    kind = question["type"]
    if kind == "noul":
        probability = round(p[0], 4)
        return {"type": "bool", "value": probability >= 0.5, "probability": probability}
    best = max(range(len(p)), key=p.__getitem__)
    names = list(question["criteria"]) if kind == "choice" else range(len(p))
    answer = {
        "type": kind,
        "value": (
            names[best]
            if kind == "choice"
            else round(sum(i * v for i, v in enumerate(p)), 4)
        ),
        "probabilities": {str(n): round(v, 4) for n, v in zip(names, p)},
        "metadata": {"confidence": round(p[best], 4)},
    }
    if kind == "score":
        answer["metadata"]["legend"] = {
            str(i): text for i, text in enumerate(question["criteria"])
        }
    return answer


class Model(Lfm2VlModel):
    decision_types = ("choice", "score", "bool", "noul")
    decision_media = ("images",)

    def predict(self, processor, state, questions, images=None, token_budget=65536):
        """Typed decisions read off the next-token logits at the answer slot."""
        questions = {name: as_question(q) for name, q in questions.items()}
        probabilities, tokens = self.probabilities(
            processor, state, list(questions.values()), images, token_budget
        )
        return {
            "model": str(getattr(self, "model_path", "d1")),
            "answers": {
                name: _answer(q, p)
                for (name, q), p in zip(questions.items(), probabilities)
            },
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    def decision_inputs(self, processor, state, questions, images=None):
        """Trunk ids, branch ids and vision inputs. One question is a single pass
        (no branches); several share the state as the trunk, one branch each."""
        from ...utils import load_image

        tokenizer = getattr(processor, "tokenizer", processor)
        if images is not None and not isinstance(images, (list, tuple)):
            images = [images]
        images = [
            cap_pixels(im if isinstance(im, Image.Image) else load_image(im))
            for im in images or ()
        ]
        if images and not hasattr(processor, "image_processor"):
            raise ValueError("d1 image decisions need the model's processor")
        bos = getattr(tokenizer, "bos_token", None)
        markup = image_markup(processor, len(images)) if images else ""
        prefix = prefix_text(state, bos if isinstance(bos, str) else "", markup)
        branches = [suffix_text(tokenizer, q) for q in questions]
        if len(branches) == 1:
            prefix, branches = prefix + branches[0], []
        branches = [tokenizer.encode(b, add_special_tokens=False) for b in branches]
        if not images:
            return tokenizer.encode(prefix, add_special_tokens=False), branches, {}
        inputs = processor(text=[prefix], images=[images], add_special_tokens=False)
        mask = np.asarray(inputs["pixel_attention_mask"])
        patches = int(mask.sum(1).max())
        vision = {
            "pixel_values": mx.array(np.asarray(inputs["pixel_values"])[:, :patches]),
            "pixel_attention_mask": mx.array(mask[:, :patches]),
            "spatial_shapes": mx.array(np.asarray(inputs["spatial_shapes"])),
        }
        return np.asarray(inputs["input_ids"])[0].tolist(), branches, vision

    def _logz(self, hidden):
        lm = self.language_model
        if lm.args.tie_word_embeddings:
            logits = lm.model.embed_tokens.as_linear(hidden)
        else:
            logits = lm.lm_head(hidden)
        logits = logits.astype(mx.float32)
        return np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))

    def answer_logprobs(self, trunk, branches, vision=None, token_budget=65536):
        """Log-probabilities at the answer slot: of `trunk` alone when there are no
        branches, else of every branch continuing the trunk, which is read once. A
        trunk longer than 2048 tokens, or than the budget, is read in chunks of
        that many, which bounds the memory a long state takes."""
        if token_budget is None or token_budget < 1:
            raise ValueError(f"token_budget must be at least 1, got {token_budget!r}")
        ids = mx.array([trunk])
        embeds = self.get_input_embeddings(ids, **(vision or {})).inputs_embeds
        model = self.language_model.model
        step = min(PREFILL_STEP, token_budget)
        if not branches and len(trunk) <= step:
            return list(self._logz(model(ids, input_embeddings=embeds)[:, -1]))
        cache = self.language_model.make_cache()
        start = 0
        while len(trunk) - start > step:
            end = start + step
            model(ids[:, start:end], cache=cache, input_embeddings=embeds[:, start:end])
            mx.eval([c.state for c in cache])
            start = end
        hidden = model(ids[:, start:], cache=cache, input_embeddings=embeds[:, start:])
        if not branches:
            return list(self._logz(hidden[:, -1]))
        mx.eval([c.state for c in cache])
        out = [None] * len(branches)
        for chunk in _chunks(list(map(len, branches)), len(trunk), token_budget):
            rows = [branches[j] for j in chunk]
            width = max(map(len, rows))
            padded = mx.array([row + [0] * (width - len(row)) for row in rows])
            hidden = model(padded, cache=_tile(cache, len(rows)))
            last = mx.array([len(row) - 1 for row in rows])
            for j, z in zip(chunk, self._logz(hidden[mx.arange(len(rows)), last])):
                out[j] = z
        return out

    def probabilities(
        self, processor, state, questions, images=None, token_budget=65536
    ):
        """Each question's option probabilities (`yes`, `no` for a noul) and the
        number of input tokens read."""
        tokenizer = getattr(processor, "tokenizer", processor)
        trunk, branches, vision = self.decision_inputs(
            processor, state, questions, images
        )
        logz = self.answer_logprobs(trunk, branches, vision, token_budget)
        probabilities = [readout(tokenizer, q, z) for q, z in zip(questions, logz)]
        return probabilities, len(trunk) + sum(map(len, branches))
