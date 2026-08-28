# Vendored from mlx-lm 0.31.3 (mlx_lm/tool_parsers/pythonic.py).
# Copyright © 2026 Apple Inc.

import ast
import re
from typing import Any

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


def _parse_keyword_arguments(arguments_text: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    position = 0

    while position < len(arguments_text):
        keyword = re.match(r"\s*([A-Za-z_]\w*)\s*=", arguments_text[position:])
        if keyword is None:
            raise ValueError("Tool calls must use keyword arguments.")

        key = keyword.group(1)
        if key in arguments:
            raise ValueError(f"Duplicate tool argument: {key!r}.")

        value_start = position + keyword.end()
        value_end = _find_argument_end(arguments_text, value_start)
        value_text = arguments_text[value_start:value_end].strip()
        if not value_text:
            raise ValueError(f"Tool argument {key!r} is missing a value.")

        try:
            value = ast.literal_eval(value_text)
        except (ValueError, TypeError, SyntaxError) as exc:
            if (
                len(value_text) >= 2
                and value_text[0] in {'"', "'"}
                and value_text[-1] == value_text[0]
            ):
                value = value_text[1:-1]
            else:
                raise ValueError(
                    f"Tool argument {key!r} must be a literal value."
                ) from exc

        arguments[key] = value
        position = value_end
        if position < len(arguments_text):
            position += 1

    return arguments


def _find_argument_end(text: str, start: int) -> int:
    quote = None
    escaped = False
    nesting = 0

    for index in range(start, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue

        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            nesting += 1
        elif character in ")]}":
            nesting -= 1
            if nesting < 0:
                raise ValueError("Invalid Pythonic tool call.")
        elif character == "," and nesting == 0:
            remainder = text[index + 1 :]
            if re.match(r"\s*[A-Za-z_]\w*\s*=", remainder):
                return index

    if quote is not None or nesting != 0:
        raise ValueError("Invalid Pythonic tool call.")
    return len(text)


def _parse_tolerant_tool_call(text: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"\s*\[\s*([A-Za-z_]\w*)\s*\((.*)\)\s*\]\s*",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("Invalid Pythonic tool call.")

    return {
        "name": match.group(1),
        "arguments": _parse_keyword_arguments(match.group(2)),
    }


def parse_tool_call(text: str, tools: Any | None = None):
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        try:
            return _parse_tolerant_tool_call(text)
        except ValueError as fallback_exc:
            raise fallback_exc from exc

    if not isinstance(expression, ast.List) or len(expression.elts) != 1:
        raise ValueError("Expected a single tool call inside a list.")

    call = expression.elts[0]
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise TypeError("Expected a named function call.")
    if call.args:
        raise ValueError("Tool calls must use keyword arguments.")

    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("Tool calls do not support unpacked arguments.")
        if keyword.arg in arguments:
            raise ValueError(f"Duplicate tool argument: {keyword.arg!r}.")
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError(
                f"Tool argument {keyword.arg!r} must be a literal value."
            ) from exc

    return {"name": call.func.id, "arguments": arguments}


tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
