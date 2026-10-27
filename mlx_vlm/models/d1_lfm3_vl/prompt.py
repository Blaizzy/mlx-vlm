# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black; the torch `readouts` removed.

"""The prompt and the readout. A System One decision is a single forward pass that stops at the answer
slot: everything the model sees is built here, and the answer is a softmax over its options' tokens.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from .media import SPOKEN, Media, as_media

# The system turn: none by default.
SYSTEMS: dict[str, str | None] = {
    "none": None,
    "d1": "You are d1, a decision model trained by Liquid AI.",
}
DEFAULT_SYSTEM = "none"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# --------------------------------------------------------------------------- #
# question types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Choice:
    instructions: str | Media  # a spoken question's clip (`media`)
    criteria: Mapping[str, str | None]
    type: str = "choice"
    multiple: bool = False  # each option its own yes or no


@dataclass(frozen=True)
class Noul:
    instructions: str | Media
    criteria: Mapping[str, Any] | None = None
    type: str = "noul"


@dataclass(frozen=True)
class Score:
    instructions: str | Media
    criteria: Sequence[str]
    type: str = "score"


@dataclass(frozen=True)
class Read:
    """A span read from the state: text, or a number or a date written there; `unit`, a number's."""

    instructions: str | Media
    type: str = "text"
    unit: str | None = None


@dataclass(frozen=True)
class Estimate:
    """A number estimated on a grid over the declared range: `step` apart, at most 128 values (a finer step
    is read on 128)."""

    instructions: str | Media
    low: float
    high: float
    step: float | None = None
    unit: str | None = None
    type: str = "number"

    @cached_property
    def grid(self):
        from .grid import Grid

        return Grid(
            self.low,
            self.high,
            (
                128
                if self.step is None
                else min(128, round((self.high - self.low) / self.step) + 1)
            ),
        )


@dataclass(frozen=True)
class Box:
    """Where something is in the picture: its four edges, or nothing."""

    instructions: str | Media
    type: str = "box"


@dataclass(frozen=True)
class Boxes:
    """Every instance of something in the picture, in reading order, none if absent: `max` blocks read, else
    `box.INSTANCES`. A readout choice: the question's text never says it."""

    instructions: str | Media
    type: str = "boxes"
    max: int | None = None

    @property
    def blocks(self) -> int:
        from .box import INSTANCES

        return INSTANCES if self.max is None else self.max


@dataclass(frozen=True)
class Position:
    name: str


@dataclass(frozen=True)
class Number:
    """A slot over a grid's values as the model writes them: value i times `scale`, an integer, is one token of the
    tokenizer's own (`number_ids`). A checkpoint that records `d1_numbers` reads its grids so.
    """

    instructions: str
    grid: Any
    scale: int = 1
    type: str = "grid"


Question = Choice | Noul | Score | Read | Estimate | Box | Boxes


TYPED = (
    "text",
    "number",
    "date",
    "box",
    "boxes",
)  # the typed answers' kinds; a choice is one when `multiple`


def typed(q: Question) -> bool:
    """A typed answer, read at its own slots or state positions and answered in its own shape (api.py):
    a read, an estimate, a box, a multiple choice."""
    return q.type in TYPED or isinstance(q, Choice) and q.multiple


def cardinality(q: Question) -> int:
    if isinstance(q, (Estimate, Number)):
        return q.grid.size
    return 2 if isinstance(q, Noul) else len(q.criteria)


# --------------------------------------------------------------------------- #
# verbalizer
# --------------------------------------------------------------------------- #


PROMPTS = (1, 2)


def version(tokenizer) -> int:
    """The prompt version the checkpoint behind `tokenizer` trained with (`PROMPTS`)."""
    return getattr(tokenizer, "d1_prompt", 1)


def most_blocks(tokenizer) -> int:
    """The most blocks a `boxes` question's `max` asks of this checkpoint: the 16 prompt 1 trained, else
    `box.MAX_INSTANCES`."""
    from .box import INSTANCES, MAX_INSTANCES

    return INSTANCES if version(tokenizer) == 1 else MAX_INSTANCES


LETTERS = tuple(chr(ord("A") + i) for i in range(26))


def option_codes(labels: Sequence[str], scheme: int = 1) -> list[str]:
    """Native letters when the labels already are letters, else A..Z, else the `scheme`'s codes: 00.. (1), A..Z then
    AA, AB.. (2); one rule for every cardinality."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return list(LETTERS[: len(labs)])
    if scheme == 2:
        return (list(LETTERS) + [a + b for a in LETTERS for b in LETTERS])[: len(labs)]
    return [f"{i:02d}" for i in range(len(labs))]


_FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    # Where a tokenizer splits digits, "00".."99" and "#i" are two tokens, and two capital letters are
    # often one. Last in the pool, so a tokenizer with digit pairs never reaches it.
    + [
        chr(a) + chr(b)
        for a in range(ord("A"), ord("Z") + 1)
        for b in range(ord("A"), ord("Z") + 1)
    ]
)


@lru_cache(maxsize=4096)
def _aliases_cached(
    tokenizer_key: int, codes: tuple[str, ...]
) -> tuple[tuple[str, int], ...]:
    tokenizer = _TOKENIZERS[tokenizer_key]
    used: set[int] = set()
    out: list[tuple[str, int]] = []

    def take(raw: str) -> bool:
        enc = tokenizer.encode(raw, add_special_tokens=False)
        if len(enc) != 1 or enc[0] in used:
            return False
        out.append((raw, enc[0]))
        used.add(enc[0])
        return True

    for code in codes:
        if take(code):
            continue
        if not any(take(raw) for raw in _FALLBACK_POOL):
            raise RuntimeError(f"no single-token alias left for {len(codes)} options")
    return tuple(out)


_TOKENIZERS: dict[int, object] = {}


def aliases(tokenizer, labels: Sequence[str]) -> list[tuple[str, int]]:
    """Every label's distinct single-token code: [(code, token_id)]. Memoised on the codes, which are positional unless
    the labels are letters, so every ordering of one option list shares an entry."""
    _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
    codes = tuple(option_codes(labels, getattr(tokenizer, "d1_codes", 1)))
    return list(_aliases_cached(id(tokenizer), codes))


@lru_cache(maxsize=8192)
def _ids_cached(tokenizer_key: int, texts: tuple[str, ...]) -> tuple[int, ...]:
    tokenizer = _TOKENIZERS[tokenizer_key]
    out, seen = [], set()
    for t in texts:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in seen:
            out.append(enc[0])
            seen.add(enc[0])
    return tuple(out)


def _ids(tokenizer, texts: Sequence[str]) -> list[int]:
    _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
    return list(_ids_cached(id(tokenizer), tuple(texts)))


QUESTION_TYPES = ("noul", "choice", "score", "text", "number", "date", "box", "boxes")
CHOICE = '"criteria" names the options: {"billing": "Charges and refunds", "shipping": "Delivery"}'
SCORE = '"criteria" lists 2 to 10 levels, lowest first: ["Can wait", "Today", "Now"]'


def check_question(q: Any, tokenizer=None) -> None:
    """A caller's question in the API's JSON, refused with what to change; a class passes. `tokenizer` carries the
    checkpoint's prompt version, which bounds a `boxes` question's `max` (`most_blocks`; prompt 1 without it).
    """
    if isinstance(q, (Choice, Noul, Score, Read, Estimate, Box, Boxes)):
        return
    if not isinstance(q, Mapping):
        raise ValueError(
            f'a question is a dict such as {{"type": "noul", "instructions": "Is it urgent?"}}, not '
            f"{type(q).__name__}"
        )
    kind, said, criteria = q.get("type"), q.get("instructions"), q.get("criteria")
    if kind not in QUESTION_TYPES:
        raise ValueError(f'"type" is one of {", ".join(QUESTION_TYPES)}, not {kind!r}')
    if not (isinstance(said, str) and said.strip() or isinstance(said, Mapping)):
        raise ValueError(
            '"instructions" is the question as text (or a spoken one, {"type": "audio", "path": ...})'
        )
    if kind == "choice" and not (
        isinstance(criteria, Mapping)
        and criteria
        and all(
            isinstance(k, str) and (v is None or isinstance(v, str))
            for k, v in criteria.items()
        )
    ):
        raise ValueError(
            f"a choice's {CHOICE}"
            + ("; a list of levels is a score" if isinstance(criteria, list) else "")
        )
    if kind == "score" and not (
        isinstance(criteria, (list, tuple))
        and 2 <= len(criteria) <= 10
        and all(isinstance(c, str) for c in criteria)
    ):
        raise ValueError(f"a score's {SCORE}")
    if kind == "noul" and criteria is not None and not isinstance(criteria, Mapping):
        raise ValueError(
            'a noul\'s "criteria", when given, is {"true": "what yes means", "false": "what no means"}'
        )
    if q.get("multiple") and kind != "choice":
        raise ValueError(
            '"multiple" belongs to a choice: {"type": "choice", "multiple": true, "criteria": {...}}'
        )
    if "max" in q and not (
        kind == "boxes"
        and isinstance(q["max"], int)
        and not isinstance(q["max"], bool)
        and 1 <= q["max"] <= most_blocks(tokenizer)
    ):
        raise ValueError(
            f'"max" belongs to boxes, a whole number from 1 to {most_blocks(tokenizer)}: '
            '{"type": "boxes", "instructions": "Every cup.", "max": 4}'
        )
    if "step" in q and "range" not in q:
        raise ValueError(
            '"step" needs a "range": {"type": "number", "range": [0, 10], "step": 1}'
        )
    if "range" in q:
        bounds = q["range"]
        if kind != "number" or not (
            isinstance(bounds, (list, tuple))
            and len(bounds) == 2
            and all(
                isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds
            )
        ):
            raise ValueError(
                'an estimate is {"type": "number", "range": [low, high]}, with an optional "step"'
            )


def as_question(q: Mapping | Question) -> Question:
    """A question in the API's JSON (`type`, `instructions`, `criteria`; `multiple`, `range`, `step`, `unit` for the
    typed answers; `instructions` a media dict for a spoken question), as one of the classes above; a class passes
    through."""
    if not isinstance(q, Mapping):
        return q
    if isinstance(q.get("instructions"), Mapping):
        q = {**q, "instructions": as_media(q["instructions"])}
    kind = q.get("type", "choice")
    if kind == "number" and "range" in q:
        low, high = q["range"]
        step = q.get("step")
        if (
            not low < high
            or step is not None
            and not (
                step > 0
                and math.isclose((high - low) / step, round((high - low) / step))
            )
        ):
            raise ValueError(
                f"an estimate's range runs low to high and its step, above 0, divides it: not "
                f"{[low, high]} in steps of {step}"
            )
        return Estimate(q["instructions"], low, high, step, q.get("unit"))
    if kind in ("text", "number", "date"):
        return Read(q["instructions"], kind, q.get("unit"))
    if kind == "box":
        return Box(q["instructions"])
    if kind == "boxes":
        return Boxes(q["instructions"], max=q.get("max"))
    if kind == "noul":
        return Noul(q["instructions"], q.get("criteria"))
    if kind == "score":
        return Score(q["instructions"], list(q["criteria"]))
    return Choice(
        q["instructions"], q["criteria"], multiple=bool(q.get("multiple", False))
    )


def question_slots(q: Question, tokenizer=None) -> tuple[Question, ...]:
    """Each answer position's readout, in question order; a grid's through the tokenizer's numbers where the
    checkpoint reads them so (`grid_slot`)."""
    if isinstance(q, Read):
        return Position("start"), Position("end")
    if isinstance(q, Estimate):
        return (grid_slot(tokenizer, q.instructions, q.grid),)
    if isinstance(q, Box):
        from .box import EDGES, edge_grid

        grid, scale = edge_grid(tokenizer)
        return (
            Noul(q.instructions),
            *(grid_slot(tokenizer, q.instructions, grid, scale) for _ in EDGES),
        )
    if isinstance(q, Boxes):
        return question_slots(Box(q.instructions), tokenizer) * q.blocks
    if isinstance(q, Choice) and q.multiple:
        if not q.criteria:
            raise ValueError("choice with no options")
        return tuple(Noul(q.instructions) for _ in q.criteria)
    return (q,)


YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")


def readout_ids(tokenizer, q: Question) -> list[list[int]]:
    """Token ids to score, one group per option, max-pooled: the answer is a softmax over these and
    nothing else."""
    if isinstance(q, Estimate):
        return readout_ids(tokenizer, question_slots(q, tokenizer)[0])
    if isinstance(q, Number):
        return [list(group) for group in number_ids(tokenizer, q.grid, q.scale)]
    if isinstance(q, (Read, Position)):
        raise ValueError("a pointer reads state positions, not vocabulary codes")
    if isinstance(q, Choice) and q.multiple:
        raise ValueError("a multiple choice reads one yes/no distribution per option")
    if isinstance(q, Noul):
        yes, no = _ids(tokenizer, YES_FORMS), _ids(tokenizer, NO_FORMS)
        if not yes or not no:
            raise RuntimeError("tokenizer has no single-token yes/no")
        return [yes, no]
    if isinstance(q, Score):
        groups = [_ids(tokenizer, [str(i)]) for i in range(len(q.criteria))]
        if any(not g for g in groups):
            raise RuntimeError(
                f"score with {len(q.criteria)} levels needs single-token digits; the primitive is defined "
                "for 2 to 10"
            )
        return groups
    groups = []
    for code, tid in aliases(tokenizer, list(q.criteria.keys())):
        extra = _ids(tokenizer, [f" {code}"])
        groups.append([tid] + [i for i in extra if i != tid])
    if not groups:
        raise RuntimeError("choice with no options")
    return groups


def readout(tokenizer, q: Question, logz, calibration=None) -> list[float]:
    """Option probabilities from the log-probabilities at the answer slot, `logz` (a vocabulary tensor, or a dict
    holding at least the option tokens). Each option scores its best form."""
    if not hasattr(logz, "tolist"):
        return probabilities(
            q,
            [max(float(logz[i]) for i in g) for g in readout_ids(tokenizer, q)],
            calibration,
        )
    tokens, forms, _ = readout_table(tokenizer, q)
    values = np.asarray(logz[list(tokens)].tolist())  # one device transfer
    return probabilities(q, values[forms].max(-1).tolist(), calibration)


def probabilities(q: Question, scores: list[float], calibration=None) -> list[float]:
    """Each option's best score, through the calibration's temperature, as a softmax."""
    if calibration is not None:
        scores = calibration.apply(q, scores)
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def readout_table(
    tokenizer, q: Question
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """`readout_ids` as one gather: every token once, each group's forms as positions among them (a short group
    repeats its first form, which leaves its best as it is), and the tokens as an array. Cached: a box edge scores
    1,000 numbers."""
    if isinstance(q, Number):
        _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
        return _number_table(
            id(tokenizer), q.grid.low, q.grid.high, q.grid.size, q.scale
        )
    return _table(tuple(map(tuple, readout_ids(tokenizer, q))))


@lru_cache(maxsize=4096)
def _table(
    groups: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    tokens = tuple(dict.fromkeys(i for g in groups for i in g))
    at = {t: k for k, t in enumerate(tokens)}
    width = max(map(len, groups))
    return (
        tokens,
        np.array([[at[i] for i in g] + [at[g[0]]] * (width - len(g)) for g in groups]),
        np.array(tokens),
    )


@lru_cache(maxsize=256)
def _number_table(tokenizer_key: int, low: float, high: float, size: int, scale: int):
    from .grid import Grid

    return _table(
        tuple(
            map(
                tuple,
                readout_ids(
                    _TOKENIZERS[tokenizer_key], Number("", Grid(low, high, size), scale)
                ),
            )
        )
    )


# --------------------------------------------------------------------------- #
# state and question rendering
# --------------------------------------------------------------------------- #


# How a state is rendered: `json_only`, the default, writes every state as the object it is; `json` keeps
# three shortcuts (`Message:`, `Passage:` / `Asked:`, a lone question's text); `sections` writes nested
# states as labelled blocks.
DEFAULT_STATE_STYLE = "json_only"


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool)) and "\n" not in str(v)


def _sections(obj: Any, path: str, out: list[str]) -> None:
    """Flatten a nested state into labelled blocks, keeping the real newlines `json.dumps` would escape."""
    head = f"[{path}]\n" if path else ""
    if isinstance(obj, dict):
        scalars = [(k, v) for k, v in obj.items() if _is_scalar(v)]
        rest = [(k, v) for k, v in obj.items() if not _is_scalar(v)]
        if scalars:
            body = "\n".join(f"{k}: {'' if v is None else v}" for k, v in scalars)
            out.append(f"{head}{body}")
        for k, v in rest:
            _sections(v, f"{path}.{k}" if path else str(k), out)
        return
    if isinstance(obj, (list, tuple)):
        if obj and all(_is_scalar(v) for v in obj):
            body = "\n".join(f"- {'' if v is None else v}" for v in obj)
            out.append(f"{head}{body}")
            return
        for i, v in enumerate(obj, start=1):
            _sections(v, f"{path} {i}/{len(obj)}" if path else f"{i}/{len(obj)}", out)
        return
    out.append(f"{head}{'' if obj is None else obj}")


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    out: list[str] = []
    _sections(state, "", out)
    return "\n\n".join(out)


def state_text(state: Any, style: str = DEFAULT_STATE_STYLE) -> str:
    """The state as the prompt renders it, without the blank line after it: what a read's span offsets index."""
    return state_block(state, style)[:-2]


def state_block(state: Any, style: str = DEFAULT_STATE_STYLE) -> str:
    """The state as the block before `QUESTION:`. The `_only` styles (the default) write every state as the object
    it is; the others give three shapes a shortcut (`Message:`, `Passage:` / `Asked:`, a lone question's text).
    """
    if isinstance(state, dict) and not style.endswith("_only"):
        keys = set(state.keys())
        if keys == {"text"}:
            return f"Message: {state['text']}\n\n"
        if {"passage", "question"} <= keys and len(keys) == 2:
            return f"Passage: {state['passage']}\n\nAsked: {state['question']}\n\n"
        if keys == {"question"}:
            return f"{state['question']}\n\n"
    if style.startswith("json"):
        if isinstance(state, str):
            return f"{state}\n\n"
        return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"
    return f"{render_state(state)}\n\n"


# A zero-width space after the first character of a special token's string: it reads the same and is no longer the
# token. Without it a state holding `<|im_end|>` closed the user turn, and `<image>` became an image token
# with no picture behind it (the processor then refused the row).
BREAK = "\u200b"


@lru_cache(maxsize=64)
def _specials_cached(tokenizer_key: int) -> re.Pattern | None:
    tok = _TOKENIZERS[tokenizer_key]
    tok = getattr(tok, "tokenizer", tok)  # a processor's own
    names = set(getattr(tok, "all_special_tokens", ()))
    names |= {
        t.content
        for t in getattr(tok, "added_tokens_decoder", {}).values()
        if t.special
    }
    names = sorted(
        (n for n in names if len(n) > 1 and BREAK not in n), key=len, reverse=True
    )
    return re.compile("|".join(map(re.escape, names))) if names else None


def plain(tokenizer, text: str) -> str:
    """`text` as the caller's text: no special token of `tokenizer` in it once encoded. Text without one is returned
    unchanged."""
    return _plain(tokenizer, text, False)[0]


def plain_offsets(tokenizer, text: str) -> tuple[str, list[int]]:
    """The same escaped text, with every character boundary mapped to the original text."""
    return _plain(tokenizer, text, True)


def _plain(tokenizer, text: str, offsets: bool) -> tuple[str, list[int] | None]:
    _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
    boundaries = list(range(len(text) + 1)) if offsets else None
    pattern = _specials_cached(id(tokenizer))
    if pattern is None:
        return text, boundaries
    n = 1
    while n:  # again for a match that overlapped the one broken before it
        if offsets:
            insertions = [m.start() + 1 for m in pattern.finditer(text)]
            for at in reversed(insertions):
                boundaries.insert(at, boundaries[at])
        text, n = pattern.subn(lambda m: m[0][0] + BREAK + m[0][1:], text)
    return text, boundaries


_PLACEHOLDER = re.compile(r"^opt\d+$")


def _option_line(code: str, label: str, desc: str | None, style: str) -> str:
    text = desc or label.replace("_", " ")
    if style == "name_desc" and not _PLACEHOLDER.match(label) and label != text:
        return f"{code} {label}: {text}"
    return f"{code} {text}"


def said(q: Question) -> str:
    """A question's instructions as its text writes them: their place (`media.SPOKEN`) when they are spoken."""
    return SPOKEN if isinstance(q.instructions, Media) else q.instructions


def edge_legend(tokenizer) -> str:
    """How a box's edges are written: number tokens on a checkpoint that reads numbers, else codes."""
    from .box import edge_grid

    grid, scale = edge_grid(tokenizer)
    if scale > 1:
        return (
            "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, "
            "left and right from its left edge."
        )
    codes = [
        code for code, _ in aliases(tokenizer, list(grid_question("", grid).criteria))
    ]
    return (
        f"Each edge is a code on a {grid.size}-value scale: {codes[0]} is 0 (top or left), {codes[-1]} is 1 "
        "(bottom or right). Values are equally spaced."
    )


def question_block(tokenizer, q: Question, option_style: str = "desc") -> str:
    if isinstance(q, Box):
        one = (
            "one number"
            if isinstance(question_slots(q, tokenizer)[-1], Number)
            else "one code"
        )
        return (
            f"{said(q)}\n\nPresence is yes or no. {edge_legend(tokenizer)} Reply with yes or no at present, "
            f"then {one} at top, bottom, left and right."
        )
    if isinstance(q, Boxes):
        from .box import INSTANCES

        most = f"{INSTANCES} at most, " if version(tokenizer) == 1 else ""
        return (
            f"{said(q)}\n\nEvery one of them, {most}in reading order (left edge, then top). "
            f"Present k is yes when there are at least k. {edge_legend(tokenizer)} Reply with yes or no at each "
            "present, then its top, bottom, left and right."
        )
    if isinstance(q, Estimate) and isinstance(question_slots(q, tokenizer)[0], Number):
        unit = f" {q.unit}" if q.unit is not None else ""
        return (
            f"{said(q)}\n\nEstimate from {q.grid.low:g} to {q.grid.high:g}{unit}, in steps of "
            f"{q.grid.step:g}. Reply with the number only."
        )
    if isinstance(q, Estimate):
        bins = grid_question("", q.grid)
        codes = (
            [str(i) for i in range(q.grid.size)]
            if isinstance(bins, Score)
            else [code for code, _ in aliases(tokenizer, list(bins.criteria))]
        )
        unit = f" {q.unit}" if q.unit is not None else ""
        return (
            f"{said(q)}\n\nEstimate from {q.grid.low:g} to {q.grid.high:g}{unit} on "
            f"{q.grid.size} equally spaced values, step {q.grid.step:g}: "
            f"{codes[0]} is {q.grid.low:g}, {codes[-1]} is {q.grid.high:g}. Reply with the code only."
        )
    if isinstance(q, Read):
        unit = f" (unit: {q.unit})" if q.type == "number" and q.unit else ""
        return f"{said(q)}\n\nRead the {q.type}{unit} from the state, or nothing if it is absent."
    if isinstance(q, Choice):
        labels = list(q.criteria.keys())
        if q.multiple:
            lines = "\n".join(
                f"{json.dumps(name, ensure_ascii=False)}: {q.criteria[name] or name}"
                for name in labels
            )
            return f"{said(q)}\n\nOptions:\n{lines}\n\nReply with yes or no for each option."
        lines = "\n".join(
            _option_line(code, lab, q.criteria[lab], option_style)
            for (code, _), lab in zip(aliases(tokenizer, labels), labels)
        )
        return f"{said(q)}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if isinstance(q, Noul):
        extra = ""
        if q.criteria:
            extra = f"\nYes: {q.criteria.get('true')}\nNo: {q.criteria.get('false')}"
        return f"{said(q)}{extra}\n\nReply with yes or no only."
    if isinstance(q, Score):
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(q.criteria))
        return f"{said(q)}\n\n{legend}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."
    raise TypeError(f"unknown question type {type(q)}")


def prefix_text(
    tokenizer,
    state: Any,
    bos: str = "",
    style: str = DEFAULT_STATE_STYLE,
    system: str = DEFAULT_SYSTEM,
    media: str = "",
) -> str:
    """Everything before the question, shared by all questions on one state: the media's markup (`media`, as
    `media.Reader` writes it) at the head of the user turn, then the state. With no state (`None`) the question
    follows the media directly. The state is the caller's text (`plain`)."""
    text = SYSTEMS[system]
    turn = "" if text is None else f"{IM_START}system\n{text}{IM_END}\n"
    if state is None:
        body = ""
    else:
        body = f"{plain(tokenizer, state_block(state, style))}\n{question_header(tokenizer, state)}"
    return f"{bos}{turn}{IM_START}user\n{media}{body}"


def question_header(tokenizer, state: Any) -> str:
    """What `prefix_text` ends with to open a question. `set` attention repeats it before each further question
    packed after the state (`attention.pack_set`)."""
    return "" if state is None else "QUESTION:\n"


# What sits between the assistant header and the answer slot. A family module registers its measured one
# (`LEADS`); any other family's is read off its chat template (`template_lead`).
DEFAULT_LEAD = ""
LEADS: dict[str, str] = {}


def default_lead(model_type: str | None, tokenizer=None) -> str:
    """What a checkpoint's own template writes before a non-thinking answer: `LEADS` for the families measured,
    else read off the tokenizer's chat template when one is given."""
    if model_type in LEADS or tokenizer is None:
        return LEADS.get(model_type, DEFAULT_LEAD)
    return template_lead(tokenizer)


def template_lead(tokenizer) -> str:
    """What a ChatML template writes after the assistant header for a non-thinking reply. The prompt is ChatML
    (`IM_START`, `IM_END`), so a model with another template is refused rather than mis-prompted.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    head = f"{IM_START}assistant\n"
    if head not in text:
        raise ValueError("not a ChatML chat template: d1 renders ChatML prompts only")
    return text[text.rindex(head) + len(head) :]


def suffix_text(
    tokenizer, q: Question, lead: str = DEFAULT_LEAD, option_style: str = "desc"
) -> str:
    """The question and the assistant header, up to the answer slot. The question is the caller's text (`plain`)."""
    return suffix_layout(tokenizer, q, lead, option_style)[0]


def suffix_layout(
    tokenizer, q: Question, lead: str = DEFAULT_LEAD, option_style: str = "desc"
) -> tuple[str, list[int]]:
    """The suffix and each slot's token offset from its last token; old questions have one slot at zero."""
    body = question_block(tokenizer, q, option_style)
    if isinstance(q, Read):
        return marked_suffix(tokenizer, body, ["start", "end"], lead)
    if isinstance(q, Box):
        from .box import EDGES

        return marked_suffix(tokenizer, body, ["present", *EDGES], lead)
    if isinstance(q, Boxes):
        from .box import EDGES, INSTANCES

        # the last block ends on the newline a next block would follow, where the colon and it are one token, so every
        # slot is the token the whole read has there: prompt 1 when `max` reads fewer than its 16, prompt 2 always
        return marked_suffix(
            tokenizer,
            body,
            [
                f"{name} {k}"
                for k in range(1, q.blocks + 1)
                for name in ("present", *EDGES)
            ],
            lead,
            "\n" if q.blocks < INSTANCES or version(tokenizer) > 1 else "",
        )
    if isinstance(q, Choice) and q.multiple:
        names = [json.dumps(name, ensure_ascii=False) for name in q.criteria]
        return marked_suffix(tokenizer, body, names, lead)
    return f"{plain(tokenizer, body)}{IM_END}\n{IM_START}assistant\n{lead}", [0]


def marked_suffix(
    tokenizer,
    body: str,
    names: Sequence[str],
    lead: str = DEFAULT_LEAD,
    after: str = "",
) -> tuple[str, list[int]]:
    """A question's named answer markers and their token offsets from its last token; `after`, whitespace the last
    marker's colon token takes where the tokenizer writes the two as one token."""
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n or "\n" in n for n in names)
    ):
        raise ValueError("slot names must be distinct nonempty single lines")
    text = f"{plain(tokenizer, body)}{IM_END}\n{IM_START}assistant\n{lead}"
    ends = []
    for i, name in enumerate(names):
        text += ("\n" if i else "") + plain(tokenizer, name) + ":"
        ends.append(len(text))
    if after and tokenizer.decode(
        tokenizer.encode(text + after, add_special_tokens=False)[-1:]
    ).rstrip().endswith(":"):
        text += after
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    slots = []
    for chars in ends:
        matching = [
            (i, stop)
            for i, (start, stop) in enumerate(encoded["offset_mapping"])
            if start <= chars - 1 < stop
        ]
        if len(matching) != 1:
            raise ValueError("a slot marker must have an exact colon token position")
        end, stop = matching[0]
        # The colon may share its token with following whitespace, never with the next marker's text.
        if text[chars:stop].strip() or not tokenizer.decode(
            [ids[end]]
        ).rstrip().endswith(":"):
            raise ValueError(
                "a slot marker token must end with its colon and optional whitespace"
            )
        slots.append(end - len(ids) + 1)
    return text, slots


def number_ids(tokenizer, grid, scale: int = 1) -> tuple[tuple[int, ...], ...] | None:
    """Each grid value times `scale` as the tokenizer writes it, plain and after a space, one token a form: None when
    a value is no integer or takes more than one token (a tokenizer of digits)."""
    _TOKENIZERS.setdefault(id(tokenizer), tokenizer)
    return _number_ids_cached(id(tokenizer), grid.low, grid.high, grid.size, scale)


@lru_cache(maxsize=256)
def _number_ids_cached(
    tokenizer_key: int, low: float, high: float, size: int, scale: int
):
    from .grid import Grid

    groups = []
    for value in Grid(low, high, size).values:
        n = round(value * scale)
        if not math.isclose(n, value * scale, abs_tol=1e-6):
            return None
        group = _ids_cached(tokenizer_key, (str(n), f" {n}"))
        if (
            not group
            or len(_TOKENIZERS[tokenizer_key].encode(str(n), add_special_tokens=False))
            != 1
        ):
            return None
        groups.append(group)
    return tuple(groups)


def grid_slot(
    tokenizer, instructions: str, grid, scale: int = 1
) -> Number | Choice | Score:
    """A grid's slot: the tokenizer's own numbers where the checkpoint reads them so (`d1_numbers`) and every value is
    one (`number_ids`), else the grid's codes (`grid_question`)."""
    if (
        getattr(tokenizer, "d1_numbers", False)
        and number_ids(tokenizer, grid, scale) is not None
    ):
        return Number(instructions, grid, scale)
    return grid_question(instructions, grid)


def grid_question(instructions: str, grid) -> Choice | Score:
    """The grid's ordered bins on the existing digit or option-code readout."""
    labels = [f"{v:g}" for v in grid.values]
    return (
        Score(instructions, labels)
        if grid.size <= 10
        else Choice(instructions, {str(i): v for i, v in enumerate(labels)})
    )


def render(
    tokenizer,
    state: Any,
    q: Question,
    bos: str = "",
    lead: str = DEFAULT_LEAD,
    style: str = DEFAULT_STATE_STYLE,
    system: str = DEFAULT_SYSTEM,
    option_style: str = "desc",
    media: str = "",
) -> str:
    return prefix_text(tokenizer, state, bos, style, system, media) + suffix_text(
        tokenizer, q, lead, option_style
    )
