"""d1's System One API on MLX: named questions over a state, answered in one
forward pass per group of questions, read at their answer slots.

    model, processor = mlx_vlm.load(path)
    model.system_one(state, {name: question}, media=None)

`pack_set`, `state_length` and `group_size` are adapted from the system_one code
shipped with the LiquidAI d1 checkpoints (LFM Open License v1.0; see LICENSE in
this folder); the forward pass and readout are written for MLX.
"""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..cache import create_causal_mask
from ..lfm3_vl.lfm3_vl import Model as Lfm3VlModel
from .api import SCORE_READOUT, SystemOneApi
from .calibration import Calibration
from .media import Reader, media_list
from .pointer import SPAN_CAPS, position_logits, spans, state_positions
from .prompt import (
    DEFAULT_STATE_STYLE,
    DEFAULT_SYSTEM,
    Box,
    Boxes,
    Choice,
    Read,
    prefix_text,
    probabilities,
    question_header,
    question_slots,
    readout_table,
    suffix_layout,
    typed,
)

LEAD = "<think></think>"  # LFM3's template closes the reasoning block before a non-thinking answer
REFUSAL = (
    "LiquidAI d1 checkpoints are decision models and do not generate text: ask "
    "them with model.system_one(state, questions) or "
    "`python -m mlx_vlm.decide --model PATH --state ... --questions ...`."
)


def group_size(n):
    """A group size (`set_size`): a whole number from 1 (each question alone), or None for the engine's own."""
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"set_size must be a whole number, not {n!r}")
    if n < 1:
        raise ValueError(f"set_size must be at least 1, not {n}")
    return n


def state_length(tokenizer, ids, suffix):
    """A prompt's tokens before its question: the suffix opens on a token boundary, so its own tokens are the
    prompt's last ones."""
    return len(ids) - len(tokenizer.encode(suffix, add_special_tokens=False))


def pack_set(tokenizer, prefix, suffixes, header, slot_offsets, first_ids):
    """`set`'s row for one state: `prefix` and the first question tokenized as one string, each further question its
    own group, `header` then its suffix. Returns the ids, the state's length, each token's question (1..N, -1 on the
    state) and each question's answer positions (`slot_offsets` count back from its suffix's last token).
    """
    if not suffixes:
        raise ValueError("a question set cannot be empty")
    if len(slot_offsets) != len(suffixes):
        raise ValueError("one slot layout is required per question")

    def positions(k, length, end):
        own = slot_offsets[k]
        if (
            not own
            or own != sorted(set(own))
            or own[-1] != 0
            or any(not -length < o <= 0 for o in own)
        ):
            raise ValueError(
                "slots must be increasing positions inside their question, ending at its last token"
            )
        return [end + o for o in own]

    ids = list(first_ids)
    state = state_length(tokenizer, ids, suffixes[0])
    question = [-1] * state + [1] * (len(ids) - state)
    ends = [positions(0, len(ids) - state, len(ids) - 1)]
    for k, suffix in enumerate(suffixes[1:], 2):
        own = tokenizer.encode(header + suffix, add_special_tokens=False)
        ids, question = ids + own, question + [k] * len(own)
        ends.append(positions(k - 1, len(own), len(ids) - 1))
    return ids, state, question, ends


def set_masks(length, state_len, question, window):
    """Each layer type's mask over one packed row: full attention reads the state
    and its own question both ways, sliding attention stays causal in its window,
    the convolutions causal over the row."""
    pos = mx.arange(length)
    group = mx.array(question)
    full = (
        (pos[:, None] < state_len)
        | (pos[None] < state_len)
        | (group[:, None] == group[None])
    )
    sliding = (
        "causal" if length <= window else create_causal_mask(length, window_size=window)
    )
    return {"full_attention": full, "sliding_attention": sliding, "conv": None}


class SystemOne(SystemOneApi):
    """The reference engine's `run` over an MLX d1 model: one unpadded row per
    group of `set_size` questions, each state media item encoded once."""

    def __init__(self, model, processor, calibration=None):
        config = model.config
        self.model, self.processor, self.calibration = model, processor, calibration
        self.tokenizer = tokenizer = processor.tokenizer
        # the prompt code reads the checkpoint's choices off the tokenizer
        tokenizer.d1_codes = config.d1_codes
        tokenizer.d1_numbers = config.d1_numbers
        tokenizer.d1_box_edges = config.d1_box_edges
        tokenizer.d1_prompt = config.d1_prompt
        pixels = os.environ.get("D1_VISION_MAX_PIXELS")
        self.media = Reader(
            processor, int(pixels) if pixels else config.d1_vision_max_pixels
        )
        self.set_size = config.d1_set_size
        bos = getattr(tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.state_style, self.system, self.lead = (
            DEFAULT_STATE_STYLE,
            DEFAULT_SYSTEM,
            LEAD,
        )

    def run(self, requests, set_size=None):
        size = group_size(set_size) or self.set_size
        return [
            self._request(state, qs, media_list(media), size)
            for state, qs, media in requests
        ]

    def _request(self, state, qs, media, size):
        if (
            any(isinstance(q, (Box, Boxes)) for q in qs)
            and (n := sum(m.type == "image" for m in media)) != 1
        ):
            raise ValueError(
                f"a box or boxes question reads exactly one picture in `media`; this request has {n}"
            )
        tokenizer = self.tokenizer
        markup, parts = self.media.encode(media) if media else ("", [])
        prefix = prefix_text(
            tokenizer, state, self.bos, self.state_style, self.system, markup
        )
        layouts = [suffix_layout(tokenizer, q, self.lead) for q in qs]
        placed = [self.media.place(text, q) for (text, _), q in zip(layouts, qs)]
        # the towers never see the text: one encoding serves every row
        state_media = [self.model.d1_features(p) for p in parts]
        voices = [[self.model.d1_features(p) for p in own] for _, own in placed]
        mx.eval(state_media, voices)
        header = question_header(tokenizer, state)
        probs, read = [None] * len(qs), 0
        for k in range(0, len(qs), size):
            group = qs[k : k + size]
            suffixes = [text for text, _ in placed[k : k + size]]
            text = prefix + suffixes[0]
            reads = any(isinstance(q, Read) for q in group)
            encoded = (
                tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                if reads
                else None
            )
            ids = (
                encoded["input_ids"]
                if reads
                else tokenizer.encode(text, add_special_tokens=False)
            )
            ids, state_len, question, ends = pack_set(
                tokenizer,
                prefix,
                suffixes,
                header,
                [offsets for _, offsets in layouts[k : k + size]],
                ids,
            )
            read += len(ids)
            row_media = state_media + [f for own in voices[k : k + size] for f in own]
            hidden = self.model.d1_hidden(ids, state_len, question, row_media)
            positions = (
                state_positions(
                    tokenizer, state, prefix, text, self.state_style, encoded=encoded
                )
                if reads
                else None
            )
            for j, p in enumerate(self._read(group, ends, hidden, positions)):
                probs[k + j] = p
        return probs, read

    def _read(self, group, ends, hidden, positions):
        """Each question's probabilities at its slots: option codes through the
        tied head (only the option rows), reads through the pointer."""
        codes, out = [], [None] * len(group)
        for j, (q, slots) in enumerate(zip(group, ends)):
            if isinstance(q, Read):
                out[j] = self._pointer(q, hidden, slots, positions)
            else:
                for r, at in zip(question_slots(q, self.tokenizer), slots):
                    codes.append((j, r, at, not typed(q)))
        tables = {}  # slots sharing a readout table (a box's edges) read in one matmul
        for i, (_, r, _, _) in enumerate(codes):
            table = readout_table(self.tokenizer, r)
            tables.setdefault(id(table), (table, []))[1].append(i)
        embed = self.model.language_model.model.embed_tokens
        scores = []
        for (tokens, forms, _), rows in tables.values():
            at = mx.array([codes[i][2] for i in rows])
            logits = hidden[at] @ embed(mx.array(tokens)).T
            scores.append((rows, forms, logits.astype(mx.float32)))
        mx.eval([logits for _, _, logits in scores])
        slot_probs = [None] * len(codes)
        for rows, forms, logits in scores:
            best = np.array(logits)[:, forms].max(-1)
            for i, values in zip(rows, best.tolist()):
                _, r, _, tempered = codes[i]
                slot_probs[i] = probabilities(
                    r, values, self.calibration if tempered else None
                )
        for j, q in enumerate(group):
            if isinstance(q, Read):
                continue
            own = [p for c, p in zip(codes, slot_probs) if c[0] == j]
            if isinstance(q, Choice) and q.multiple:
                out[j] = [p[0] for p in own]
            elif isinstance(q, (Box, Boxes)):
                out[j] = own
            else:
                out[j] = own[0]
        return out

    def _pointer(self, q, hidden, slots, positions):
        valid = np.asarray(positions.pointer_mask)
        both = np.array(
            mx.concatenate([hidden[mx.array(slots)], hidden[: len(valid)]]).astype(
                mx.float32
            )
        )
        logits = position_logits(both[:2], both[2:], valid)
        cap = SPAN_CAPS[q.type]
        at, found = spans(logits[None], max_tokens=cap)
        return {
            **positions.read(*at[0].tolist(), max_tokens=cap),
            "found": float(found[0]),
        }


class D1:
    """What a d1 checkpoint adds to its LFM3 backbone: `system_one` over set
    attention, and no text generation."""

    def __call__(self, *args, **kwargs):
        raise ValueError(REFUSAL)

    # generation prefills through it; system_one embeds in d1_hidden
    get_input_embeddings = __call__

    def attach_processor(self, processor, calibration=None):
        """The checkpoint's processor and calibration (`calibration.json` next to
        its weights, else temperature 1 and no tables) for `system_one`."""
        path = getattr(self, "model_path", None)
        if calibration is None and path is not None:
            file = Path(path) / "calibration.json"
            calibration = Calibration.load(file) if file.is_file() else None
        object.__setattr__(self, "_engine", SystemOne(self, processor, calibration))

    @property
    def engine(self):
        """The System One engine; it also takes `abstain_below` on `system_one`."""
        if getattr(self, "_engine", None) is None:
            from ...utils import load_processor

            self.attach_processor(load_processor(self.model_path, False))
        return self._engine

    def system_one(
        self,
        state,
        questions,
        media=None,
        score_readout=SCORE_READOUT,
        set_size=None,
    ):
        """Named questions over a state (text, JSON, or None with media alone) and
        its media (pictures, audio clips, videos): `{"answers": {name: answer},
        "usage": {"input_tokens": n, "output_tokens": 0}}`."""
        return self.engine.system_one(
            state, questions, media, score_readout, set_size=set_size
        )

    def system_one_batch(self, requests, score_readout=SCORE_READOUT, set_size=None):
        """`(state, questions)` or `(state, questions, media)` requests, one response each."""
        return self.engine.system_one_batch(requests, score_readout, set_size=set_size)

    def d1_features(self, part):
        """One media item's encoder features and the placeholder token they fill."""
        config = self.config
        if part["type"] == "image":
            return config.image_token_id, self.encode_images(
                part["pixel_values"], image_grid_thw=part["image_grid_thw"]
            )
        if part["type"] == "video":
            return config.video_token_id, self.encode_images(
                part["pixel_values_videos"], image_grid_thw=part["video_grid_thw"]
            )
        lengths = mx.array(part["feature_attention_mask"]).sum(-1)
        audio = self.audio_tower(part["input_features"], lengths)
        return config.audio_token_id, self.audio_projector(audio)

    def d1_hidden(self, ids, state_len, question, media=()):
        """The final hidden states [T, D] of one packed row under set attention,
        `media` (token, features) pairs scattered at their placeholders in order."""
        backbone = self.language_model.model
        tokens = np.asarray(ids)
        h = backbone.embed_tokens(mx.array(tokens))
        for token in dict.fromkeys(t for t, _ in media):
            at = np.flatnonzero(tokens == token)
            features = mx.concatenate([f for t, f in media if t == token])
            if len(at) != features.shape[0]:
                raise ValueError(
                    f"placeholders {token} ({len(at)}) and features ({features.shape[0]}) do not match"
                )
            h[mx.array(at)] = features.astype(h.dtype)
        # shortcut: one dense T x T mask a row; chunk the queries for very long states
        masks = set_masks(
            len(ids), state_len, question, self.config.text_config.sliding_window
        )
        h = h[None]
        for layer in backbone.layers:
            h = layer(h, masks[layer.layer_type], None)
        return backbone.embedding_norm(h)[0]


class Model(D1, Lfm3VlModel):
    pass
