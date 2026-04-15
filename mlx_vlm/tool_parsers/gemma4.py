import json

import regex as re

tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"

_ESCAPE = '<|"|>'


def _find_matching_brace(text, start):
    """Find the index of the closing '}' that matches the '{' at *start*,
    respecting nested braces/brackets and <|"|>-escaped strings."""
    depth = 0
    i = start
    while i < len(text):
        if text[i:].startswith(_ESCAPE):
            # Skip over escaped string content
            end_idx = text.find(_ESCAPE, i + len(_ESCAPE))
            if end_idx == -1:
                return len(text)
            i = end_idx + len(_ESCAPE)
            continue
        ch = text[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text)


def _parse_value(text):
    """Parse a single value from gemma tool-call syntax.

    Handles:
    - <|"|>...<|"|> escaped strings  -> str
    - {...}  nested objects           -> dict
    - [...]  arrays                   -> list
    - bare literals (numbers, booleans, null) via json.loads
    """
    text = text.strip()

    # Escaped string
    if text.startswith(_ESCAPE):
        inner = text[len(_ESCAPE) :]
        end = inner.find(_ESCAPE)
        if end == -1:
            return inner
        return inner[:end]

    # Nested object
    if text.startswith("{"):
        close = _find_matching_brace(text, 0)
        return _parse_object(text[1:close])

    # Array
    if text.startswith("["):
        close = _find_matching_brace(text, 0)
        return _parse_array(text[1:close])

    # Bare literal
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _split_top_level(text, delimiter=","):
    """Split *text* on *delimiter* only when not inside braces/brackets or
    <|"|>-escaped strings."""
    parts = []
    depth = 0
    current_start = 0
    i = 0
    while i < len(text):
        if text[i:].startswith(_ESCAPE):
            end_idx = text.find(_ESCAPE, i + len(_ESCAPE))
            if end_idx == -1:
                i = len(text)
            else:
                i = end_idx + len(_ESCAPE)
            continue
        ch = text[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == delimiter and depth == 0:
            parts.append(text[current_start:i])
            current_start = i + 1
        i += 1
    tail = text[current_start:]
    if tail.strip():
        parts.append(tail)
    return parts


def _parse_object(text):
    """Parse the interior of a ``{...}`` block into a dict."""
    result = {}
    entries = _split_top_level(text, ",")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # Find the first ':' that is not inside an escape sequence
        colon = _find_top_level_colon(entry)
        if colon == -1:
            continue
        key = entry[:colon].strip()
        value_str = entry[colon + 1 :]
        result[key] = _parse_value(value_str)
    return result


def _find_top_level_colon(text):
    """Return the index of the first ':' not inside an escape sequence."""
    i = 0
    while i < len(text):
        if text[i:].startswith(_ESCAPE):
            end_idx = text.find(_ESCAPE, i + len(_ESCAPE))
            if end_idx == -1:
                return -1
            i = end_idx + len(_ESCAPE)
            continue
        if text[i] == ":":
            return i
        i += 1
    return -1


def _parse_array(text):
    """Parse the interior of a ``[...]`` block into a list."""
    items = _split_top_level(text, ",")
    return [_parse_value(item) for item in items if item.strip()]


# Regex that captures the function name and uses a recursive pattern to
# match the balanced outer braces (handles arbitrary nesting).
# Function name uses ``[\w-]+`` (alphanumerics, underscore, hyphen) per the
# OpenAI tool name spec.
_tool_call_regex = re.compile(
    r"call:([\w-]+)(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})"
)


def parse_tool_call(text, tools=None):
    # Try recursive-group regex first for well-formed calls
    match = _tool_call_regex.search(text)
    if not match:
        # Fallback: find 'call:<name>{' and then balance braces manually
        m = re.search(r"call:([\w-]+)\{", text)
        if not m:
            raise ValueError("No function call found in tool call text.")
        func_name = m.group(1)
        brace_start = m.end() - 1  # index of the '{'
        brace_end = _find_matching_brace(text, brace_start)
        args_str = text[brace_start + 1 : brace_end]
    else:
        func_name = match.group(1)
        # Strip outer braces from the matched balanced group
        balanced = match.group(2)
        args_str = balanced[1:-1]

    arguments = _parse_object(args_str)
    return dict(name=func_name, arguments=arguments)
