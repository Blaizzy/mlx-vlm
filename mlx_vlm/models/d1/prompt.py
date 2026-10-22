"""The d1 System One prompt and readout, ported from the published D1Model defaults
(no system turn, `json_only` states, `desc` options, no lead, no calibration)."""

import json
import math
from collections.abc import Mapping

from PIL import Image

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_MAX_PIXELS = 1024 * 1024
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")

_FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [
        chr(a) + chr(b)
        for a in range(ord("A"), ord("Z") + 1)
        for b in range(ord("A"), ord("Z") + 1)
    ]
)


def as_question(question):
    """A normalized decision question as `{type, instructions, criteria}`, with
    `bool` read as `noul` and list choice criteria as labels without descriptions.
    Malformed questions raise ValueError; a score has 2 to 10 levels, as the
    reference documents (its readout is defined for those) but does not check."""
    kind = question.get("type", "choice")
    kind = "noul" if kind == "bool" else kind
    if "instructions" not in question:
        raise ValueError("a question needs `instructions`")
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, (list, tuple)):
        criteria = dict.fromkeys(criteria)
    if kind == "choice" and (not isinstance(criteria, Mapping) or len(criteria) < 2):
        raise ValueError(
            "a choice needs criteria {name: description} with at least two options"
        )
    if kind == "score":
        if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
            raise ValueError(
                "a score needs criteria: a list of 2 to 10 level descriptions, lowest first"
            )
        criteria = list(criteria)
    if kind == "noul" and criteria is not None and not isinstance(criteria, Mapping):
        raise ValueError('noul criteria are optional: {"true": "...", "false": "..."}')
    return {
        "type": kind,
        "instructions": question["instructions"],
        "criteria": criteria,
    }


def option_codes(labels):
    labels = [str(x).strip() for x in labels]
    if labels and all(len(k) == 1 and k.isalpha() for k in labels):
        return labels
    if len(labels) <= 26:
        return [chr(ord("A") + i) for i in range(len(labels))]
    return [f"{i:02d}" for i in range(len(labels))]


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def aliases(tokenizer, labels):
    """A distinct single-token code for every label: [(code, token_id)]."""
    used, out = set(), []

    def take(raw):
        ids = _encode(tokenizer, raw)
        if len(ids) != 1 or ids[0] in used:
            return False
        out.append((raw, ids[0]))
        used.add(ids[0])
        return True

    for code in option_codes(labels):
        if not take(code) and not any(take(raw) for raw in _FALLBACK_POOL):
            raise ValueError(f"no single-token alias left for {len(labels)} options")
    return out


def _ids(tokenizer, texts):
    out = []
    for text in texts:
        ids = _encode(tokenizer, text)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def readout_ids(tokenizer, q):
    """Token ids scored for each option, one group per option."""
    if q["type"] == "noul":
        yes, no = _ids(tokenizer, YES_FORMS), _ids(tokenizer, NO_FORMS)
        if not yes or not no:
            raise RuntimeError("tokenizer has no single-token yes/no")
        return [yes, no]
    if q["type"] == "score":
        groups = [_ids(tokenizer, [str(i)]) for i in range(len(q["criteria"]))]
        if any(not g for g in groups):
            raise RuntimeError(
                f"score with {len(q['criteria'])} levels needs single-token digits"
            )
        return groups
    groups = []
    for code, tid in aliases(tokenizer, list(q["criteria"])):
        groups.append([tid] + [i for i in _ids(tokenizer, [f" {code}"]) if i != tid])
    if not groups:
        raise RuntimeError("choice with no options")
    return groups


def readout(tokenizer, q, logz):
    """Option probabilities: each option's best token log-probability, softmaxed
    over the options (P(yes), P(no) for a noul)."""
    scores = [max(float(logz[i]) for i in g) for g in readout_ids(tokenizer, q)]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return [e / sum(exps) for e in exps]


def state_block(state):
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def question_block(tokenizer, q):
    if q["type"] == "choice":
        labels = list(q["criteria"])
        codes = aliases(tokenizer, labels)
        lines = "\n".join(
            f"{codes[i][0]} {q['criteria'][label] or label.replace('_', ' ')}"
            for i, label in enumerate(labels)
        )
        return (
            f"{q['instructions']}\n\nOptions:\n{lines}\n\n"
            "Reply with the option code only."
        )
    if q["type"] == "noul":
        extra = ""
        if q["criteria"]:
            extra = (
                f"\nYes: {q['criteria'].get('true')}\nNo: {q['criteria'].get('false')}"
            )
        return f"{q['instructions']}{extra}\n\nReply with yes or no only."
    if q["type"] == "score":
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(q["criteria"]))
        return (
            f"{q['instructions']}\n\n{legend}\n\n"
            f"Reply with a single digit 0-{len(q['criteria']) - 1} only."
        )
    raise ValueError(f"Unsupported question type: {q['type']!r}")


def prefix_text(state, bos="", images=""):
    """Everything before the question, shared by all questions on one state."""
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{bos}{IM_START}user\n{images}{body}"


def suffix_text(tokenizer, q):
    """The question and the assistant header, up to the answer slot."""
    return f"{question_block(tokenizer, q)}{IM_END}\n{IM_START}assistant\n"


def image_markup(processor, count):
    """What the chat template writes for `count` images at the head of a user turn."""
    owner = processor
    if not getattr(processor, "chat_template", None):
        owner = getattr(processor, "tokenizer", processor)
    content = [{"type": "image"}] * count + [{"type": "text", "text": "\x00"}]
    text = owner.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=False,
        tokenize=False,
    )
    head = f"{IM_START}user\n"
    return text[text.index(head) + len(head) : text.index("\x00")]


def cap_pixels(image, max_pixels=VISION_MAX_PIXELS):
    """A picture downscaled to at most `max_pixels`, bicubic."""
    image = image.convert("RGB")
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)
