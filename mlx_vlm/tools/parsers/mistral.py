# Vendored from mlx-lm 0.31.3 (mlx_lm/tool_parsers/mistral.py).
# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

# Mistral emits two official tool-call shapes after the [TOOL_CALLS] marker:
#   - v11/tekken tokenizer: ``name[ARGS]{json}`` (Mistral-Small-3.x, Devstral)
#   - classic v3 tokenizer: a JSON array ``[{"name": ..., "arguments": ...}]``
#     (Mistral-7B-v0.3, Mixtral, Ministral-8B)
_tool_call_regex = re.compile(r"\s*(\w+)\[ARGS\]\s*(\{.*\})", re.DOTALL)

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    match = _tool_call_regex.search(text)
    if match is not None:
        return dict(name=match.group(1), arguments=json.loads(match.group(2)))

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse tool call from: {text}") from exc

    calls = parsed if isinstance(parsed, list) else [parsed]
    if calls and all(isinstance(c, dict) and "name" in c for c in calls):
        return [dict(name=c["name"], arguments=c.get("arguments", {})) for c in calls]
    raise ValueError(f"Could not parse tool call from: {text}")
