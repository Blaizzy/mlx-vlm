"""Parse native XML tool calls from MiniCPM5 models.

Format: https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/chat_template.jinja
"""

import ast
import json
import xml.etree.ElementTree as ET

tool_call_start = "<function"
tool_call_end = "</function>"


def _argument_properties(name, tools):
    for tool in tools or []:
        function = tool.get("function") or {}
        if function.get("name") == name:
            return (function.get("parameters") or {}).get("properties", {})
    return {}


def _parse_value(value, schema):
    param_type = schema.get("type")
    if param_type == "string" or (
        isinstance(param_type, list) and "string" in param_type
    ):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # The chat template also renders Python literals for non-string arguments.
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_tool_call(text: str, tools=None):
    """Parse a complete function element or the shared server's stripped body."""
    text = text.strip()
    if not text.startswith(tool_call_start):
        # process_tool_calls removes both markers, leaving name="...">...</param>.
        text = f"{tool_call_start} {text}{tool_call_end}"

    try:
        function = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError("Invalid MiniCPM5 function call XML.") from exc

    name = (function.get("name") or "").strip()
    if function.tag != "function" or not name:
        raise ValueError("No MiniCPM5 function name provided.")

    properties = _argument_properties(name, tools)
    arguments = {}
    for param in function:
        key = param.get("name")
        if param.tag != "param" or not key or key in arguments or len(param):
            raise ValueError("Invalid MiniCPM5 function parameter.")
        # ElementTree decodes entities and CDATA without stripping text spaces.
        arguments[key] = _parse_value(param.text or "", properties.get(key, {}))

    return {"name": name, "arguments": arguments}
