import pytest

from mlx_vlm.tools.parsers.pythonic import parse_tool_call

LFM_TOOL_TEMPLATE = """
{{ '<|tool_call_start|>[' + tool_calls + ']<|tool_call_end|>' }}
"""


def test_multiline_double_quoted_html_is_preserved():
    result = parse_tool_call(
        '[write_file(path="game.html", content="<canvas id="game">\n</canvas>")]'
    )

    assert result == {
        "name": "write_file",
        "arguments": {"path": "game.html", "content": '<canvas id="game">\n</canvas>'},
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
