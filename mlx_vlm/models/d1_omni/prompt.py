"""d1-omni prompts, ported from the reference ``prompt.py``.

    <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> ... <decide>

The head scores the hidden state at every ``<mask>``. ``<|...|>`` in caller text
is rewritten so a state can never forge a delimiter.
"""

import json
import re

QTYPES = {"choice": 0, "score": 1, "noul": 2}
DELIM = {
    "state": "<|reserved_7|>",
    "q": "<|reserved_8|>",
    "opt": "<|reserved_9|>",
    "opt_end": "<|reserved_10|>",
    "decide": "<|reserved_11|>",
}
MARKER = "<|mask|>"
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def as_question(spec):
    """A normalized question dict -> (type, instructions, criteria), checked as the reference does."""
    kind = "noul" if spec.get("type") == "bool" else spec.get("type")
    if kind not in QTYPES:
        raise ValueError(f"question type must be one of {sorted(QTYPES)}, got {kind!r}")
    if "instructions" not in spec:
        raise ValueError("a question needs `instructions`")
    criteria = spec.get("criteria")
    if kind == "choice" and isinstance(criteria, (list, tuple)):
        criteria = dict.fromkeys(criteria)
    if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
        raise ValueError(
            "a choice needs criteria {name: description} with at least two options"
        )
    if kind == "score" and (
        not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10
    ):
        raise ValueError(
            "a score needs criteria: a list of 2 to 10 level descriptions, lowest first"
        )
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        raise ValueError(
            'noul criteria are optional: {"true": "...", "false": "..."} (or "yes", "no")'
        )
    return kind, str(spec["instructions"]), criteria


def options(question):
    kind, _, criteria = question
    return 2 if kind == "noul" else len(criteria)


def escape(text):
    return _SPECIAL.sub(r"<¦\1¦>", text)


def serialize(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question, noul_default=None, audio=False):
    """Option texts in the model's order; a noul is read as [false, true]. After
    an audio prefix they are written as the audio questions were trained:
    ``option_000: text`` and ``false: no``, ``true: yes``."""
    kind, _, criteria = question
    if kind == "choice" and audio:
        return [
            f"option_{i:03d}: {criterion(k if v is None or v == '' else v)}"
            for i, (k, v) in enumerate(criteria.items())
        ]
    if kind == "choice":
        return [
            k if v is None or v == "" else f"{k}: {criterion(v)}"
            for k, v in criteria.items()
        ]
    if kind == "score":
        return [f"level {i}: {criterion(c)}" for i, c in enumerate(criteria)]
    if audio:
        return ["false: no", "true: yes"]
    crit = criteria or noul_default or {}
    false, true = crit.get("false", crit.get("no")), crit.get("true", crit.get("yes"))
    return [
        "false: "
        + (
            criterion(false)
            if false not in (None, "")
            else "no, the statement does not hold"
        ),
        "true: "
        + (criterion(true) if true not in (None, "") else "yes, the statement holds"),
    ]


def encode(
    tok, state, question, max_len, noul_default=None, audio=False, per_option=24
):
    """Token ids of one question over one state, and each option marker's position.

    The option block gets max(96, min(24k + 32, max_len / 2)) tokens, shared out
    evenly; the state is truncated on the right to the room that is left.
    """
    ids_of = tok.convert_tokens_to_ids

    def enc(text):
        return tok(escape(text), add_special_tokens=False)["input_ids"]

    opts = render_options(question, noul_default, audio)
    budget = max(96, min(len(opts) * per_option + 32, max_len // 2))
    per = max(2, (budget - 3 * len(opts)) // len(opts))
    ids = ([ids_of(DELIM["q"])] + enc(question[1]))[: max(16, budget)]
    markers = []
    for text in opts:
        markers.append(len(ids) + 1)
        ids += [ids_of(DELIM["opt"]), ids_of(MARKER)]
        ids += enc(" " + text)[:per] + [ids_of(DELIM["opt_end"])]
    ids.append(ids_of(DELIM["decide"]))
    room = max(0, max_len - len(ids) - 2)
    state_ids = [ids_of(DELIM["state"])] + enc(serialize(state))[:room]
    ids = ([tok.bos_token_id] + state_ids + ids)[:max_len]
    markers = [m + 1 + len(state_ids) for m in markers]
    if markers[-1] >= max_len:
        raise ValueError("the options do not fit in the context")
    return ids, markers


def temperature_key(question):
    k = options(question)
    bucket = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{question[0]}:{bucket}"
