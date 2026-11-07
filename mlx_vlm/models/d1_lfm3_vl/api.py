# Adapted from the system_one code shipped with the LiquidAI d1 checkpoints.
# Licensed under the LFM Open License v1.0; see LICENSE in this folder.
# Changed: reformatted with black.

"""The System One API: named questions over a state in, typed answers out.

    model.system_one(state, {"refund": {"type": "noul", "instructions": "Is this a refund request?"}})
    # {"answers": {"refund": {"type": "noul", "noul": p}}, "usage": {"input_tokens": n, "output_tokens": 0}}

A state is a string, any JSON value, or None with media alone; `media` holds its pictures, audio clips and videos
(`media`). A question is a dict in the Decision Index's schema (`type`, `instructions`, `criteria`; spoken
instructions a media dict) or one of `prompt`'s classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .media import Media, media_list
from .prompt import (
    DEFAULT_STATE_STYLE,
    Box,
    Boxes,
    Choice,
    Estimate,
    Noul,
    Question,
    Read,
    as_question,
    check_question,
    state_text,
    typed,
)

# How a score question's distribution becomes its answer. `median`, the default: the first level whose cumulative
# probability reaches one half, optimal for absolute error (ordinal MAE). `mean`: the expected level, which a caller
# rounds (the Decision Index's rule), optimal only for squared error. `argmax`: the likeliest level. `median` and
# `argmax` answer a level, which rounding leaves as it is.
SCORE_READOUTS = ("mean", "median", "argmax")
SCORE_READOUT = "median"
# An explicit "I don't know" option: a choice option whose text (its description, else its name) says the answerer
# does not know. Not an option saying the evidence is insufficient or that a request cannot be served: those are
# answers. `abstain_below` (off by default) picks it when the best other option's probability, renormalised over the
# other options, is below the threshold.
IDK_TEXTS = frozenset(
    {
        "i don't know",
        "i do not know",
        "i don't know the answer",
        "i do not know the answer",
    }
)


def abstain_option(criteria: Mapping[str, str | None] | None) -> str | None:
    """The name of a choice's explicit "I don't know" option (the English `IDK_TEXTS`), or None."""
    if not isinstance(criteria, Mapping):
        return None
    for name, text in criteria.items():
        if (
            str(text or name).strip().lower().replace("\u2019", "'").rstrip(".!")
            in IDK_TEXTS
        ):
            return name
    return None


def choice_index(
    probs: Sequence[float],
    abstain: int | None = None,
    abstain_below: float | None = None,
) -> int:
    """A choice's answer: the likeliest option; with `abstain` (the "I don't know" option's index) and
    `abstain_below`, that option when the best other option holds less than `abstain_below` of the other options'
    mass."""
    best = max(range(len(probs)), key=probs.__getitem__)
    if abstain is None or abstain_below is None:
        return best
    others = [p for i, p in enumerate(probs) if i != abstain]
    total = sum(others)
    return abstain if total <= 0 or max(others) < abstain_below * total else best


def score_level(probs: Sequence[float], readout: str = SCORE_READOUT) -> int:
    """A score question's answer as a level index under `readout` (`mean`: rounded, clamped to the levels)."""
    total = sum(probs) or 1.0
    if readout == "mean":
        return max(
            0,
            min(
                len(probs) - 1,
                int(round(sum(i * p / total for i, p in enumerate(probs)))),
            ),
        )
    if readout == "median":
        mass = 0.0
        for i, p in enumerate(probs):
            mass += p / total
            if mass >= 0.5:
                return i
        return len(probs) - 1
    if readout == "argmax":
        return max(range(len(probs)), key=probs.__getitem__)
    raise ValueError(f"score_readout {readout!r}: one of {SCORE_READOUTS}")


def answer(
    q: Question,
    probs: Sequence[float] | Sequence[Sequence[float]] | Mapping,
    score_readout: str = SCORE_READOUT,
    abstain_below: float | None = None,
    context: str = "",
    calibration=None,
    edges: str = "median",
) -> dict:
    """A noul's P(yes); a choice's pick (`choice_index`) and its probabilities; a score's level under
    `score_readout` (`mean`: the expected level, unrounded); a typed answer's probabilities through `calibration`'s
    tables (a read is found, a box drawn, an option chosen, at 0.5 after them; an estimate's quantiles read at the
    levels its table gives). `edges`: how a box's edges are read (`box.EDGE_READS`, the checkpoint's `d1_box_edges`).
    """
    through = (
        calibration.through if calibration is not None else lambda name, value: value
    )
    if isinstance(q, Box):
        from .box import answer as box_answer

        return box_answer(probs, through, edges)
    if isinstance(q, Boxes):
        from .box import answer_many

        return answer_many(probs, through, edges)
    if isinstance(q, Estimate):
        value = q.grid.read(
            probs,
            (
                None
                if calibration is None
                else lambda share: calibration.level("estimate", share)
            ),
        )
        return {
            "type": "number",
            "number": value["median"],
            "interval": [value["p10"], value["p90"]],
            "unit": q.unit,
        }
    if isinstance(q, Read):
        from .parsing import parse_date, parse_number, written_unit

        found = through(f"found:{q.type}", probs["found"])
        present = found >= 0.5
        text = probs["text"] if present else None
        span = probs["span"] if present else None
        value, unit = text, q.unit
        if text is not None and q.type == "number":
            unit = unit or written_unit(text)
            value = parse_number(text, context, unit=q.unit)
        elif text is not None and q.type == "date":
            value, _ = parse_date(text, context)
        result = {"type": q.type, q.type: value, "span": span, "found": found}
        if q.type != "text":
            result["text"] = text
        if present and "path" in probs:
            result["path"] = probs["path"]
        if q.type == "number":
            result["unit"] = unit
        return result
    if isinstance(q, Noul):
        return {"type": "noul", "noul": probs[0]}
    if isinstance(q, Choice) and q.multiple:
        probs = [through("multiple", p) for p in probs]
        return {
            "type": "choice",
            "choices": [name for name, p in zip(q.criteria, probs) if p >= 0.5],
            "probabilities": dict(zip(q.criteria, probs)),
        }
    best = max(range(len(probs)), key=probs.__getitem__)
    if isinstance(q, Choice):
        names = list(q.criteria)
        idk = abstain_option(q.criteria) if abstain_below is not None else None
        pick = choice_index(
            probs, None if idk is None else names.index(idk), abstain_below
        )
        return {
            "type": "choice",
            "choice": names[pick],
            "confidence": probs[pick],
            "probabilities": dict(zip(names, probs)),
        }
    level = (
        sum(i * p for i, p in enumerate(probs))
        if score_readout == "mean"
        else float(score_level(probs, score_readout))
    )
    return {
        "type": "score",
        "score": level,
        "confidence": probs[best],
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "legend": {str(i): text for i, text in enumerate(q.criteria)},
    }


def request_media(request: tuple) -> list[Media]:
    """A `(state, questions)` or `(state, questions, media)` request's media."""
    return media_list(request[2] if len(request) > 2 else None)


class SystemOneApi:
    """`system_one` and `system_one_batch`, and the plain probabilities under them, over a model's
    `run(requests)`: each `(state, questions, media)` request's probabilities (in its questions' order)
    and the tokens it read."""

    serves_typed = (
        True  # an engine without the slots and state positions of slots.py says False
    )

    def ask(self, q: Mapping | Question) -> Question:
        """A caller's question, refused with what to change, or when this engine cannot serve its type."""
        check_question(q, getattr(self, "tokenizer", None))
        q = as_question(q)
        if typed(q) and not self.serves_typed:
            raise ValueError(
                f"{type(self).__name__} serves noul, choice and score questions only"
            )
        return q

    def named(self, questions: Mapping[str, Any]) -> dict[str, Question]:
        """A request's named questions (`ask`), an error naming the question it is about."""
        if not isinstance(questions, Mapping):
            raise ValueError(
                f'questions are a dict of name to question, {{"refund": {{"type": "noul", "instructions": '
                f'"Is it a refund request?"}}}}, not {type(questions).__name__}'
            )
        out = {}
        for name, q in questions.items():
            try:
                out[name] = self.ask(q)
            except ValueError as e:
                raise ValueError(f"question {name!r}: {e}") from None
        return out

    def answer(
        self,
        state: Any,
        q: Question,
        probs,
        score_readout: str = SCORE_READOUT,
        abstain_below: float | None = None,
    ) -> dict:
        """One question's answer as `system_one` gives it: a read's value parsed in its state, a typed answer's
        probabilities through the served calibration's tables."""
        context = (
            state_text(state, getattr(self, "state_style", DEFAULT_STATE_STYLE))
            if isinstance(q, Read)
            else ""
        )
        return answer(
            q,
            probs,
            score_readout,
            abstain_below,
            context,
            getattr(self, "calibration", None),
            getattr(getattr(self, "tokenizer", None), "d1_box_edges", "median"),
        )

    def run(
        self,
        requests: Sequence[tuple[Any, list[Question], Sequence[Media]]],
        set_size: int | None = None,
    ) -> list[tuple[list[list[float]], int]]:
        raise NotImplementedError

    def probabilities(
        self, state: Any, questions: Sequence, media: Sequence | None = None
    ) -> list[list[float]]:
        """Each question's option probabilities; a multiple choice returns independent P(yes) per option."""
        return self.run([(state, [self.ask(q) for q in questions], media_list(media))])[
            0
        ][0]

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        media: Sequence | None = None,
        score_readout: str = SCORE_READOUT,
        abstain_below: float | None = None,
        set_size: int | None = None,
    ) -> dict:
        """Named questions over one state, and its media if any (pictures, audio clips, videos), in one forward
        pass."""
        return self.system_one_batch(
            [(state, questions, media)], score_readout, abstain_below, set_size
        )[0]

    def system_one_batch(
        self,
        requests: Sequence[tuple],
        score_readout: str = SCORE_READOUT,
        abstain_below: float | None = None,
        set_size: int | None = None,
    ) -> list[dict]:
        """Many `(state, questions)` or `(state, questions, media)` requests in one call, batched as the engine
        reads them (`run`); a request with several questions reads its state once a pass.
        `score_readout` (`SCORE_READOUTS`) sets how a score question answers; `abstain_below` (off by default)
        when a choice with an "I don't know" option picks it (`choice_index`); `set_size` how many questions are
        read together after the state, for this call (the model's trained size by default).
        """
        if score_readout not in SCORE_READOUTS:
            raise ValueError(
                f"score_readout {score_readout!r}: one of {SCORE_READOUTS}"
            )
        if abstain_below is not None and not 0 < abstain_below <= 1:
            raise ValueError(f"abstain_below {abstain_below!r}: a share in (0, 1]")
        named = [(r[0], self.named(r[1]), request_media(r)) for r in requests]
        done = self.run(
            [(state, list(qs.values()), media) for state, qs, media in named],
            set_size=set_size,
        )
        return [
            {
                "answers": {
                    n: self.answer(state, q, p, score_readout, abstain_below)
                    for (n, q), p in zip(qs.items(), probs)
                },
                "usage": {"input_tokens": read, "output_tokens": 0},
            }
            for (state, qs, _), (probs, read) in zip(named, done)
        ]
