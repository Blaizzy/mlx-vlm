"""Tests for the Mistral tool-call parser (both official formats)."""

import json

import pytest

from mlx_vlm.tools import process_tool_calls
from mlx_vlm.tools.parsers import mistral
from mlx_vlm.tools.parsers.mistral import parse_tool_call

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    }
]

# What Ministral-8B-Instruct-2410 actually generates (classic v3 JSON-array form).
V3_OUTPUT = (
    '[TOOL_CALLS] [{"name": "get_weather", '
    '"arguments": {"city": "Paris", "days": 3}}]'
)
# Newer v11 / tekken form (Mistral-Small-3.x, Devstral).
V11_OUTPUT = '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "days": 3}'


def test_v11_args_format_still_parses():
    result = parse_tool_call('get_weather[ARGS]{"city": "Paris", "days": 3}')
    assert result == {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}


def test_v3_json_array_format_parses():
    result = parse_tool_call(
        '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    )
    assert result == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_v3_multiple_calls_in_array():
    result = parse_tool_call(
        '[{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]'
    )
    assert [c["name"] for c in result] == ["a", "b"]


@pytest.mark.parametrize("output", [V3_OUTPUT, V11_OUTPUT])
def test_process_tool_calls_end_to_end(output):
    result = process_tool_calls(output, mistral, TOOLS)
    assert result.remaining_text == ""
    assert len(result.calls) == 1
    call = result.calls[0]["function"]
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": "Paris", "days": 3}


def test_unparseable_raises():
    with pytest.raises(ValueError):
        parse_tool_call("not a tool call at all")
