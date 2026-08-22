"""Parse ATEM tool calls emitted by Muse models."""

import json
import re

# Muse emits a self-reasoning turn before routing an ATEM call. Starting at the
# reasoning envelope lets the shared server remove the complete protocol block
# from assistant content after parsing the invocation.
tool_call_start = "to=self<|message|>"
tool_call_end = "</atem:function_calls>"
# Fallback block marker for tool calls emitted without the reasoning
# envelope (Muse sometimes skips the self-reasoning turn).
tool_call_block_start = "<atem:function_calls>"

_INVOKE_PATTERN = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">(?P<body>.*?)' r"</atem:invoke>",
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>'
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


def _parse_value(value: str):
    """Parse JSON values while preserving ATEM's unquoted string form."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_tool_call(text: str, tools=None):
    """Return one or more ATEM invocations in the common parser shape."""
    calls = []
    for invoke in _INVOKE_PATTERN.finditer(text):
        arguments = {
            parameter.group("name"): _parse_value(parameter.group("value"))
            for parameter in _PARAMETER_PATTERN.finditer(invoke.group("body"))
        }
        calls.append({"name": invoke.group("name"), "arguments": arguments})

    if not calls:
        raise ValueError("No ATEM function invocation found.")
    return calls[0] if len(calls) == 1 else calls
