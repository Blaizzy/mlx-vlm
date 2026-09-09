"""Tests for the GLM 4.7 tool-call parser."""

import json

import mlx_vlm.tool_parsers.glm47 as glm47
from mlx_vlm.server.responses_state import process_tool_calls
from mlx_vlm.tool_parsers.glm47 import parse_tool_call

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    }
]

# The name sits on its own line right after <tool_call>, so the regex that ends
# at the first <arg_key> captures the trailing newline with it.
MODEL_OUTPUT = (
    "<tool_call>get_weather\n"
    "<arg_key>zip</arg_key>\n"
    "<arg_value>10001</arg_value>\n"
    "<arg_key>days</arg_key>\n"
    "<arg_value>3</arg_value>\n"
    "</tool_call>"
)


def _tool_call_body(text: str) -> str:
    return text.removeprefix(glm47.tool_call_start).removesuffix(glm47.tool_call_end)


def test_tool_name_drops_the_trailing_newline():
    result = parse_tool_call(_tool_call_body(MODEL_OUTPUT), TOOLS)

    assert result["name"] == "get_weather"


def test_string_arguments_keep_their_declared_type():
    parsed = process_tool_calls(MODEL_OUTPUT, glm47, TOOLS)

    assert len(parsed["calls"]) == 1
    arguments = json.loads(parsed["calls"][0]["function"]["arguments"])
    assert arguments == {"zip": "10001", "days": 3}
