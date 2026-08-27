# Vendored from mlx-lm 0.31.3 (mlx_lm/tool_parsers/pythonic.py).
# Copyright © 2026 Apple Inc.

import ast
from typing import Any

"""
Tool parser for Pythonic function call formats.

Parses assistant responses containing tool calls in formats like:
<|tool_call_start|>[function_name(arg1="value1", arg2=2)]<|tool_call_end|>
"""


def parse_tool_call(text: str, tools: Any | None = None):
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError("Invalid Pythonic tool call.") from exc

    if not isinstance(expression, ast.List) or len(expression.elts) != 1:
        raise ValueError("Expected a single tool call inside a list.")

    call = expression.elts[0]
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("Expected a named function call.")
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

    return dict(name=call.func.id, arguments=arguments)


tool_call_start = "<|tool_call_start|>"
tool_call_end = "<|tool_call_end|>"
