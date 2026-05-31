import ast
import json
import re

tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"

_CALL_RE = re.compile(r"\s*([A-Za-z_][\w.-]*)\s*\(")


def _find_matching(text, start, open_ch, close_ch):
    depth = 0
    quote = None
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i

    return -1


def _split_top_level(text, delimiter=","):
    parts = []
    start = 0
    depth = 0
    quote = None
    escaped = False

    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == delimiter and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _find_top_level_equals(text):
    depth = 0
    quote = None
    escaped = False

    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return i

    return -1


def _parse_value(text):
    text = text.strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
            pass

    return text


def _strip_tool_markers(text):
    text = text.strip()
    if text.startswith(tool_call_start):
        text = text[len(tool_call_start) :].lstrip()
    if text.endswith(tool_call_end):
        text = text[: -len(tool_call_end)].rstrip()
    return text


def _strip_wrapping_list(text):
    text = text.strip()
    if not text.startswith("["):
        return text

    end = _find_matching(text, 0, "[", "]")
    if end == -1:
        raise ValueError("Unclosed tool-call list.")

    trailing = text[end + 1 :].strip()
    if trailing:
        raise ValueError(f"Unexpected text after tool-call list: {trailing}")

    return text[1:end]


def _parse_single_call(text):
    match = _CALL_RE.match(text)
    if not match:
        raise ValueError(f"No function provided: {text}")

    func_name = match.group(1)
    args_start = match.end() - 1
    args_end = _find_matching(text, args_start, "(", ")")
    if args_end == -1:
        raise ValueError(f"Unclosed function call: {text}")

    trailing = text[args_end + 1 :].strip()
    if trailing:
        raise ValueError(f"Unexpected text after function call: {trailing}")

    arguments = {}
    args_str = text[args_start + 1 : args_end].strip()
    if args_str:
        for arg in _split_top_level(args_str):
            split = _find_top_level_equals(arg)
            if split == -1:
                raise ValueError(f"Expected keyword argument in tool call: {arg}")

            key = arg[:split].strip()
            value = arg[split + 1 :].strip()
            arguments[key] = _parse_value(value)

    return dict(name=func_name, arguments=json.dumps(arguments, ensure_ascii=False))


def parse_tool_call(text, tools=None):
    text = _strip_wrapping_list(_strip_tool_markers(text))
    calls = [_parse_single_call(call) for call in _split_top_level(text)]
    if not calls:
        raise ValueError("No function provided.")
    if len(calls) == 1:
        return calls[0]
    return calls
