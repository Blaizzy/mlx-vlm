import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from . import processing  # noqa: F401
from .audio import Audio, sanitize_audio
from .config import ModelConfig
from .encoder import DecisionHead, Trunk
from .prompt import QTYPES, as_question, criterion, encode, options, temperature_key
from .vision import Vision

YES_NO = {"false": "no", "true": "yes"}  # how image and audio questions were trained


def _length(row):
    return (0 if row[0] is None else row[0].shape[0]) + len(row[1])


def _image(image):
    """A PIL image as given (no EXIF turn), as in the reference; else loaded."""
    from ...utils import load_image

    return image if isinstance(image, Image.Image) else load_image(image)


def _answer(question, probs):
    kind, _, criteria = question
    if kind == "noul":
        p = round(probs[1], 4)
        return {"type": "bool", "value": p >= 0.5, "probability": p}
    best = max(range(len(probs)), key=probs.__getitem__)
    labels = list(criteria) if kind == "choice" else list(range(len(criteria)))
    metadata = {"confidence": round(probs[best], 4)}
    answer = {
        "type": kind,
        "value": (
            labels[best]
            if kind == "choice"
            else round(sum(i * p for i, p in enumerate(probs)), 4)
        ),
        "probabilities": {str(k): round(p, 4) for k, p in zip(labels, probs)},
        "metadata": metadata,
    }
    if kind == "score":
        metadata["legend"] = {str(i): criterion(c) for i, c in enumerate(criteria)}
    return answer


class Model(nn.Module):
    decision_types = ("choice", "score", "bool", "noul")

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        d = config.text_config.hidden_size
        self.encoder = Trunk(config.text_config)
        self.head = DecisionHead(d, config.head_layers)
        self.vision = Vision(config.vision_config, config.projector_hidden_size, d)
        self.audio = Audio(config.audio_config, d)

    def __call__(
        self, inputs_embeds, pad, prefix, lengths, marker_pos, marker_mask, qtype
    ):
        """Option logits for right-padded rows of ``prefix`` media embeddings then text."""
        h = self.encoder(inputs_embeds, pad, prefix)
        T = max(lengths)
        text_pad = mx.arange(T)[None] < mx.array(lengths)[:, None]
        index = mx.minimum(
            mx.array(prefix)[:, None] + mx.arange(T)[None], h.shape[1] - 1
        )
        text = mx.take_along_axis(h, index[..., None], axis=1)
        text = mx.where(text_pad[..., None], text, 0)
        return self.head(text, text_pad, marker_pos, marker_mask, qtype)

    def sanitize(self, weights):
        result = {}
        for key, value in sanitize_audio(weights).items():
            if key.endswith(("num_batches_tracked", "position_ids")):
                continue
            if key.startswith("audio."):
                result[key] = value
                continue
            key = key.replace("vision.tower.vision_model.", "vision.tower.")
            if key.endswith(".conv.conv.weight") and value.shape[-1] > value.shape[1]:
                value = value.transpose(0, 2, 1)
            if key.startswith("head.head.") and ".self_attn.in_proj_" in key:
                prefix, suffix = key.split(".self_attn.in_proj_")
                for name, split in zip(
                    ("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)
                ):
                    result[f"{prefix}.attention.{name}.{suffix}"] = split
                continue
            if key.startswith("head.head."):
                key = key.replace(".self_attn.out_proj.", ".attention.out_proj.")
            if key.startswith("head.scorer.") and not key.startswith(
                "head.scorer.layers."
            ):
                key = key.replace("head.scorer.", "head.scorer.layers.", 1)
            result[key] = value
        return result

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.startswith(("head.type_emb", "head.scorer"))

        return predicate

    def predict(
        self, processor, state, questions, images=None, audio=None, token_budget=65536
    ):
        tokenizer = getattr(processor, "tokenizer", processor)
        parsed = {name: as_question(spec) for name, spec in questions.items()}
        [(probs, read)] = self.probabilities(
            tokenizer, [(state, list(parsed.values()), images, audio)], token_budget
        )
        return {
            "model": str(getattr(self, "model_path", "d1-omni")),
            "answers": {n: _answer(q, p) for (n, q), p in zip(parsed.items(), probs)},
            "usage": {"input_tokens": read, "output_tokens": 0},
        }

    def probabilities(self, tokenizer, requests, token_budget=65536):
        """``(state, questions[, images[, audio]])`` requests -> per request, each
        question's distribution over its options in option order (``[false, true]``
        for a noul) and the positions the trunk read. ``audio`` is one clip: 16 kHz
        mono samples, or a path, URL or binary file object of an audio file."""
        config, rows, read = self.config, [], []
        for state, questions, *media in requests:
            images = (media[0] if media else None) or None
            audio = media[1] if len(media) > 1 else None
            questions = [
                q if isinstance(q, tuple) else as_question(q) for q in questions
            ]
            if images is not None and audio is not None:
                raise ValueError("a request carries images or audio, not both")
            if images is not None:
                images = images if isinstance(images, (list, tuple)) else [images]
                prefix = self.vision([_image(image) for image in images])
                max_len, noul, calibrate = config.image_text_length, YES_NO, False
            elif audio is not None:
                prefix = self.audio(audio)[0]
                max_len, noul, calibrate = config.audio_text_length, YES_NO, False
                state = {} if state is None else state  # as audio was trained
            else:
                prefix, max_len, noul, calibrate = None, config.max_length, None, True
            p = 0 if prefix is None else prefix.shape[0]
            max_len = min(max_len, config.max_length - p)
            if max_len < 64:
                raise ValueError(
                    f"the media take {p} of the {config.max_length} positions; send fewer images"
                )
            state = "" if state is None else state
            spoken = audio is not None  # options as the audio questions were trained
            seqs = [
                encode(tokenizer, state, q, max_len, noul, spoken) for q in questions
            ]
            rows.append(
                [(prefix, ids, m, q, calibrate) for q, (ids, m) in zip(questions, seqs)]
            )
            read.append(sum(p + len(ids) for ids, _ in seqs))
        results = self._run([row for request in rows for row in request], token_budget)
        out = []
        for request, tokens in zip(rows, read):
            out.append((results[: len(request)], tokens))
            results = results[len(request) :]
        return out

    def _run(self, rows, token_budget=65536):
        """Rows (prefix, ids, markers, question, calibrate) -> probabilities, batched
        longest first under a token budget."""
        results = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: _length(rows[i]), reverse=True)
        while order:
            longest, size = _length(rows[order[0]]), 1
            while size < len(order) and (size + 1) * longest <= token_budget:
                size += 1
            batch, order = order[:size], order[size:]
            for i, probs in zip(batch, self._forward([rows[i] for i in batch])):
                results[i] = probs
        return results

    def _forward(self, rows):
        dtype = self.encoder.embedding_norm.weight.dtype
        seqs, prefix, lengths = [], [], []
        for media, ids, *_ in rows:
            text = self.encoder.embed_tokens(mx.array(ids)).astype(dtype)
            seqs.append(
                text if media is None else mx.concatenate([media.astype(dtype), text])
            )
            prefix.append(0 if media is None else media.shape[0])
            lengths.append(len(ids))
        L = max(s.shape[0] for s in seqs)
        h = mx.stack([mx.pad(s, [(0, L - s.shape[0]), (0, 0)]) for s in seqs])
        pad = mx.arange(L)[None] < mx.array([s.shape[0] for s in seqs])[:, None]
        k = max(len(row[2]) for row in rows)
        marker_pos = mx.array([row[2] + [0] * (k - len(row[2])) for row in rows])
        marker_mask = (
            mx.arange(k)[None] < mx.array([len(row[2]) for row in rows])[:, None]
        )
        qtype = mx.array([QTYPES[row[3][0]] for row in rows])
        logits = self(h, pad, prefix, lengths, marker_pos, marker_mask, qtype)
        temperatures = self.config.temperatures
        out = []
        for (_, _, _, question, calibrate), z in zip(rows, logits):
            z = z[: options(question)]
            if calibrate:
                z = z / temperatures.get(
                    temperature_key(question), temperatures.get(question[0], 1.0)
                )
            out.append(mx.softmax(z, axis=-1).tolist())
        return out
