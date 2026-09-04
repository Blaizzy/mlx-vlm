"""Parser for Poolside Laguna tool calls."""

import ast
import json
from typing import Any

import regex as re

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"

_NAME_RE = re.compile(r"^\s*([^<]+?)\s*(?=<arg_key>|$)", re.DOTALL)
_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def _string_args(tool_name: str, tools: list[Any] | None) -> set[str]:
    if not tools:
        return set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        return {
            key
            for key, schema in properties.items()
            if isinstance(schema, dict) and schema.get("type") == "string"
        }
    return set()


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse the inner body of one Poolside Laguna tool call."""
    body = text.strip()
    if body.startswith(tool_call_start) and body.endswith(tool_call_end):
        body = body[len(tool_call_start) : -len(tool_call_end)].strip()

    name_match = _NAME_RE.match(body)
    if not name_match:
        return {"name": "unknown", "arguments": {"raw": text.strip()}}

    name = name_match.group(1).strip()
    string_args = _string_args(name, tools)
    arguments = {}
    for match in _ARG_RE.finditer(body):
        key = match.group(1).strip()
        value = match.group(2).strip()
        arguments[key] = value if key in string_args else _deserialize(value)
    return {"name": name, "arguments": arguments}
