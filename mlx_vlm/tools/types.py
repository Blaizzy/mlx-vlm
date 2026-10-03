"""Neutral internal representation for the tool-calling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ParseResult:
    """The outcome of extracting tool calls from a model's text output.

    ``calls`` are OpenAI-shaped tool-call dicts ready for the wire;
    ``remaining_text`` is the model output with the tool-call spans removed.
    """

    calls: List[Dict[str, Any]]
    remaining_text: str
