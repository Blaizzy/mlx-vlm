"""Parse a gpt-oss harmony tool call, ``<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{json}<|call|>``."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

tool_call_start: str = "<|channel|>commentary to="
tool_call_end: str = "<|call|>"

_name_re = re.compile(r"(?:functions\.)?([^\s<|]+)")


def parse_tool_call(text: str, tools: Optional[Any] = None) -> dict:
    """Decode one harmony ``commentary`` tool call (markers already stripped) into ``{name, arguments}``."""
    header, sep, payload = text.partition("<|message|>")
    match = _name_re.match(header.strip())
    if not match:
        raise ValueError(f"No function recipient in harmony tool call: {text!r}")

    payload = payload.strip() if sep else ""
    try:
        arguments: Any = json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        arguments = payload
    return {"name": match.group(1), "arguments": arguments}
