import json
from types import SimpleNamespace

import pytest

from mlx_vlm.server.responses_state import process_tool_calls
from mlx_vlm.tool_parsers import (
    _infer_tool_parser,
    _infer_tool_parser_from_processor,
    load_tool_module,
)
from mlx_vlm.tool_parsers.pythonic import parse_tool_call

LFM_TOOL_TEMPLATE = """
{{ '<|tool_call_start|>[' + tool_calls + ']<|tool_call_end|>' }}
"""


def test_lfm_tool_call_markers_infer_pythonic_parser():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template=LFM_TOOL_TEMPLATE)
    )

    assert _infer_tool_parser(LFM_TOOL_TEMPLATE) == "pythonic"
    assert _infer_tool_parser_from_processor(processor) == "pythonic"


def test_lfm_tool_call_output_parses_for_server_response():
    parser_type = _infer_tool_parser(LFM_TOOL_TEMPLATE)
    parser = load_tool_module(parser_type)

    result = process_tool_calls(
        "<|tool_call_start|>[get_weather(location='Warsaw', unit='celsius')]"
        "<|tool_call_end|>",
        parser,
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert result["remaining_text"] == ""
    assert len(result["calls"]) == 1
    assert result["calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(result["calls"][0]["function"]["arguments"]) == {
        "location": "Warsaw",
        "unit": "celsius",
    }


def test_single_quoted_string_preserves_embedded_commas():
    result = parse_tool_call(
        "[write_file(path='game.js', content='const player = { x: 0, y: 1 };')]"
    )

    assert result == {
        "name": "write_file",
        "arguments": {
            "path": "game.js",
            "content": "const player = { x: 0, y: 1 };",
        },
    }


def test_double_quoted_string_preserves_embedded_commas():
    result = parse_tool_call(
        '[write_file(content="width=device-width, initial-scale=1.0")]'
    )

    assert result == {
        "name": "write_file",
        "arguments": {"content": "width=device-width, initial-scale=1.0"},
    }


def test_multiline_string_is_preserved():
    result = parse_tool_call(
        "[write_file(path='game.html', content='<!doctype html>\n"
        "<canvas></canvas>\n</html>')]"
    )

    assert result == {
        "name": "write_file",
        "arguments": {
            "path": "game.html",
            "content": "<!doctype html>\n<canvas></canvas>\n</html>",
        },
    }


def test_string_with_unescaped_matching_quotes_is_preserved():
    result = parse_tool_call(
        "[write_file(path='game.html', "
        "content='<script>const label = 'Score';</script>')]"
    )

    assert result == {
        "name": "write_file",
        "arguments": {
            "path": "game.html",
            "content": "<script>const label = 'Score';</script>",
        },
    }


def test_multiline_double_quoted_html_is_preserved():
    result = parse_tool_call(
        '[write_file(path="game.html", content="<canvas id="game">\n</canvas>")]'
    )

    assert result == {
        "name": "write_file",
        "arguments": {
            "path": "game.html",
            "content": '<canvas id="game">\n</canvas>',
        },
    }


def test_multiline_write_file_output_parses_for_server_response():
    parser = load_tool_module("pythonic")
    result = process_tool_calls(
        "<|tool_call_start|>[write_file(path='game.html', "
        "content='<!doctype html>\n<script>const label = 'Score';</script>')]"
        "<|tool_call_end|>",
        parser,
        tools=[{"type": "function", "function": {"name": "write_file"}}],
    )

    assert result["remaining_text"] == ""
    assert len(result["calls"]) == 1
    assert result["calls"][0]["function"]["name"] == "write_file"
    assert json.loads(result["calls"][0]["function"]["arguments"]) == {
        "path": "game.html",
        "content": "<!doctype html>\n<script>const label = 'Score';</script>",
    }


def test_nested_literal_arguments_are_parsed_without_splitting():
    result = parse_tool_call(
        "[configure(options={'position': [0, 1], 'enabled': True})]"
    )

    assert result == {
        "name": "configure",
        "arguments": {"options": {"position": [0, 1], "enabled": True}},
    }


def test_malformed_quoted_argument_is_rejected():
    with pytest.raises(ValueError, match="Invalid Pythonic tool call"):
        parse_tool_call("[write_file(content='const player = { x: 0, y: 1 };)]")


def test_non_literal_argument_is_rejected():
    with pytest.raises(ValueError, match="must be a literal value"):
        parse_tool_call("[write_file(content=get_content())]")
