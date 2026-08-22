import json
from types import SimpleNamespace

from mlx_vlm.server.responses_state import process_tool_calls
from mlx_vlm.tool_parsers import (
    _infer_tool_parser,
    _infer_tool_parser_from_processor,
    load_tool_module,
)

POOLSIDE_TEMPLATE = """
{{ '<tool_call>' + tool_calls + '</tool_call>' }}
{{ '<arg_key>' + key + '</arg_key><arg_value>' + value + '</arg_value>' }}
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
]


def test_poolside_template_infers_dedicated_parser():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template=POOLSIDE_TEMPLATE)
    )

    assert _infer_tool_parser(POOLSIDE_TEMPLATE) == "poolside_v1"
    assert _infer_tool_parser_from_processor(processor) == "poolside_v1"


def test_poolside_output_parses_for_server_response():
    parser = load_tool_module("poolside_v1")
    result = process_tool_calls(
        "<tool_call>lookup<arg_key>repo</arg_key><arg_value>god-ai</arg_value>"
        "<arg_key>limit</arg_key><arg_value>3</arg_value></tool_call>",
        parser,
        tools=TOOLS,
    )

    assert result["remaining_text"] == ""
    assert len(result["calls"]) == 1
    call = result["calls"][0]["function"]
    assert call["name"] == "lookup"
    assert json.loads(call["arguments"]) == {"repo": "god-ai", "limit": 3}


def test_poolside_output_preserves_string_arguments():
    parser = load_tool_module("poolside_v1")
    result = parser.parse_tool_call(
        "lookup<arg_key>repo</arg_key><arg_value>00123</arg_value>",
        tools=TOOLS,
    )

    assert result == {"name": "lookup", "arguments": {"repo": "00123"}}
