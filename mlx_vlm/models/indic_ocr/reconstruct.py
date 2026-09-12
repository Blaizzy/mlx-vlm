"""Assemble transcribed blocks into the page's markdown.

Ported from upstream ``idp_reconstruct.py`` (bodhan-ai/indic-ocr).
stdlib-only -- no model, no PIL.
"""

import re

from .processing_indic_ocr import DROP_TYPES

_HYPHEN_BREAK = re.compile("(\\w)[-\\u2010-\\u2014\\u2212]\\n[ \\t]*(\\w)")
_MATH_DELIMITERS = ("$", "\\[", "\\(")
_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", re.S)
_NON_LATIN_RUN = re.compile(
    "[\\u0600-\\u06ff\\u0900-\\u0d7f\\u1c50-\\u1c7f]"
    "(?:[\\u0600-\\u06ff\\u0900-\\u0d7f\\u1c50-\\u1c7f \\u200c\\u200d]*"
    "[\\u0600-\\u06ff\\u0900-\\u0d7f\\u1c50-\\u1c7f])?"
)
_TEXT_CMD = re.compile(r"\\text\{[^{}]*\}")


def dehyphenate(text: str) -> str:
    """Rejoin words split by a hyphen at a line break (to a fixpoint)."""
    previous = None
    while previous != text:
        previous = text
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text


def _repair_expression(tex: str, display: bool) -> str:
    """Make one transcribed expression valid LaTeX.

    Wraps Indic-script runs in ``\\text{}`` (math mode has no glyphs for
    them) and turns bare newlines in display math into ``\\\\``.
    """
    protected: list = []

    def stash(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    tex = _TEXT_CMD.sub(stash, tex)
    tex = _NON_LATIN_RUN.sub(lambda m: f"\\text{{{m.group(0)}}}", tex)
    tex = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], tex)

    if display:
        tex = re.sub(r"\s*\n\s*", r" \\\\ ", tex.strip())
    return tex


def repair_math(text: str) -> str:
    """Repair every math span in a markdown string."""

    def fix(match: re.Match) -> str:
        display = match.group(1) is not None
        inner = _repair_expression(match.group(1) or match.group(2), display)
        return f"$${inner}$$" if display else f"${inner}$"

    return _MATH_SPAN.sub(fix, text)


def reconstruct(blocks: list, repair: bool = True) -> str:
    """Reading-ordered markdown. Blocks with no text contribute nothing but
    are not removed -- they still appear in the JSON with text ""."""
    kept = sorted(
        (b for b in blocks if b.type not in DROP_TYPES), key=lambda b: b.order
    )

    parts = []
    for block in kept:
        text = (block.text or "").strip()
        if not text:
            continue
        if (
            block.type == "Equation"
            and "$" not in text
            and not text.startswith(_MATH_DELIMITERS)
        ):
            text = f"$${text}$$"
        parts.append(text)

    markdown = dehyphenate("\n\n".join(parts))
    return repair_math(markdown) if repair else markdown


__all__ = ["dehyphenate", "repair_math", "reconstruct"]
