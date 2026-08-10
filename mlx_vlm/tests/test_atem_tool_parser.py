import json

import pytest

from mlx_vlm.models.muse_glimmer import ModelConfig
from mlx_vlm.server.responses_state import process_tool_calls
from mlx_vlm.tool_parsers import _infer_tool_parser, atem, load_tool_module

ATEM_TEMPLATE = """
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
</atem:invoke>
</atem:function_calls>
"""


def test_infers_atem_parser_from_muse_chat_template():
    assert _infer_tool_parser(ATEM_TEMPLATE) == "atem"
    assert load_tool_module("atem") is atem


def test_muse_config_exposes_native_reasoning_boundaries():
    config = ModelConfig()

    assert config.thinking_start_token == "to=self<|message|>"
    assert config.thinking_end_token == "<|eom|>"


def test_parses_json_and_unquoted_atem_parameter_values():
    result = atem.parse_tool_call(
        """
<atem:invoke name="weather.get_weather">
<atem:parameter name="city">New York</atem:parameter>
<atem:parameter name="days">3</atem:parameter>
<atem:parameter name="detailed">true</atem:parameter>
<atem:parameter name="filters">{"units":"metric"}</atem:parameter>
<atem:parameter name="labels">["rain", "wind"]</atem:parameter>
<atem:parameter name="note">  keep surrounding spaces  </atem:parameter>
</atem:invoke>
"""
    )

    assert result == {
        "name": "weather.get_weather",
        "arguments": {
            "city": "New York",
            "days": 3,
            "detailed": True,
            "filters": {"units": "metric"},
            "labels": ["rain", "wind"],
            "note": "  keep surrounding spaces  ",
        },
    }


def test_parses_multiline_parameters_and_multiple_invocations():
    result = atem.parse_tool_call(
        """
<atem:invoke name="files.write">
<atem:parameter name="content">first line
second line</atem:parameter>
</atem:invoke>
<atem:invoke name="files.read">
<atem:parameter name="path">notes.txt</atem:parameter>
</atem:invoke>
"""
    )

    assert result == [
        {
            "name": "files.write",
            "arguments": {"content": "first line\nsecond line"},
        },
        {"name": "files.read", "arguments": {"path": "notes.txt"}},
    ]


def test_process_tool_calls_converts_atem_to_openai_shape():
    result = process_tool_calls(
        """to=self<|message|>I should check the weather.<|eom|><|start|>assistant to=weather.get_weather<|message|><atem:function_calls>
<atem:invoke name="weather.get_weather">
<atem:parameter name="city">Warsaw</atem:parameter>
</atem:invoke>
</atem:function_calls>""",
        atem,
        tools=None,
    )

    assert result["remaining_text"] == ""
    assert len(result["calls"]) == 1
    call = result["calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "weather.get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Warsaw"}


def test_rejects_text_without_an_atem_invocation():
    with pytest.raises(ValueError, match="No ATEM function invocation"):
        atem.parse_tool_call("not a tool call")
