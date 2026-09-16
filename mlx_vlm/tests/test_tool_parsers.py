"""Tool-call parsing and parser selection."""

import json
import unittest

import pytest

import mlx_vlm.tools.parsers.gemma4 as gemma4
import mlx_vlm.tools.parsers.glm47 as glm47
from mlx_vlm.tools import _infer_tool_parser, process_tool_calls
from mlx_vlm.tools.parsers import atem, cohere2_moe, mistral, pythonic

# ATEM

ATEM_TEMPLATE = """
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
</atem:invoke>
</atem:function_calls>
"""


def test_rejects_text_without_an_atem_invocation():
    with pytest.raises(ValueError, match="No ATEM function invocation"):
        atem.parse_tool_call("not a tool call")


# Cohere


def test_cohere_action_array_parses_to_openai_tool_calls():
    result = cohere2_moe.parse_tool_call("""
        [
          {"tool_call_id": "1", "tool_name": "grep", "parameters": {"pattern": "<\\|channel>"}},
          {"tool_call_id_id": "2", "tool_name": "read", "parameters": {"path": "file.py"}}
        ]
        """)

    assert [call["name"] for call in result] == ["grep", "read"]
    assert json.loads(result[0]["arguments"]) == {"pattern": "<|channel>"}
    assert json.loads(result[1]["arguments"]) == {"path": "file.py"}


def test_cohere_single_action_object_parses_to_openai_tool_call():
    result = cohere2_moe.parse_tool_call(
        '{"tool_call_id": "1", "tool_name": "grep", "parameters": {"pattern": "foo"}}'
    )

    assert result["name"] == "grep"
    assert json.loads(result["arguments"]) == {"pattern": "foo"}


# Gemma 4

# Wire-format helpers
TC_START = "<|tool_call>"
TC_END = "<tool_call|>"
ESC = '<|"|>'


def _gemma4_call(call_body: str) -> str:
    """Wrap a ``call:name{...}`` body in the Gemma 4 tool-call markers."""
    return f"{TC_START}{call_body}{TC_END}"


def _gemma4_string(value: str) -> str:
    """Render a string value with the Gemma 4 escape delimiter."""
    return f"{ESC}{value}{ESC}"


class TestGemma4ToolParser(unittest.TestCase):
    # ── snake_case (regression: must keep working) ────────────────────────

    # ── hyphenated names (the bug this PR fixes) ──────────────────────────

    def test_hyphenated_name_with_nested_args(self):
        text = _gemma4_call(
            f"call:edit-file{{path:{_gemma4_string('test.txt')},edits:[{{newText:{_gemma4_string('orange')},oldText:{_gemma4_string('apple')}}}]}}"
        )
        result = gemma4.parse_tool_call(text)
        self.assertEqual(result["name"], "edit-file")
        args = json.loads(result["arguments"])
        self.assertEqual(args["path"], "test.txt")
        self.assertEqual(args["edits"], [{"newText": "orange", "oldText": "apple"}])

    # ── arguments type ────────────────────────────────────────────────────

    # ── DiffusionGemma payload tolerance ──────────────────────────────────

    def test_bare_function_syntax(self):
        result = gemma4.parse_tool_call("get_weather{city:Austin}")
        self.assertEqual(result["name"], "get_weather")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Austin"})

    def test_process_tool_calls_ignores_non_call_prose(self):
        result = process_tool_calls("Like call: prince", gemma4, tools=None)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.remaining_text, "Like call: prince")

    # ── error path ────────────────────────────────────────────────────────

    def test_no_call_raises(self):
        with self.assertRaises(ValueError):
            gemma4.parse_tool_call("just a normal model response, no tool call here")


# GLM 4.7 and Laguna

GLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"zip": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]

# The name sits on its own line right after <tool_call>, so the regex that ends
# at the first <arg_key> captures the trailing newline with it.
GLM_OUTPUT = (
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
    result = glm47.parse_tool_call(_tool_call_body(GLM_OUTPUT), GLM_TOOLS)

    assert result["name"] == "get_weather"


# Poolside Laguna emits the same GLM shape inline (name and args on one line).
# glm47 handles it once the name is stripped, so no dedicated Laguna parser is
# needed; verified on real poolside/Laguna-XS-2.1 weights.
LAGUNA_OUTPUT = (
    "<tool_call>get_weather"
    "<arg_key>zip</arg_key><arg_value>10001</arg_value>"
    "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
)


# Mistral

MISTRAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
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


@pytest.mark.parametrize("output", [V3_OUTPUT, V11_OUTPUT])
def test_process_tool_calls_end_to_end(output):
    result = process_tool_calls(output, mistral, MISTRAL_TOOLS)
    assert result.remaining_text == ""
    assert len(result.calls) == 1
    call = result.calls[0]["function"]
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": "Paris", "days": 3}


def test_unparseable_raises():
    with pytest.raises(ValueError):
        mistral.parse_tool_call("not a tool call at all")


# Pythonic

LFM_TOOL_TEMPLATE = """
{{ '<|tool_call_start|>[' + tool_calls + ']<|tool_call_end|>' }}
"""


def test_multiline_double_quoted_html_is_preserved():
    result = pythonic.parse_tool_call(
        '[write_file(path="game.html", content="<canvas id="game">\n</canvas>")]'
    )

    assert result == {
        "name": "write_file",
        "arguments": {"path": "game.html", "content": '<canvas id="game">\n</canvas>'},
    }


def test_nested_literal_arguments_are_parsed_without_splitting():
    result = pythonic.parse_tool_call(
        "[configure(options={'position': [0, 1], 'enabled': True})]"
    )

    assert result == {
        "name": "configure",
        "arguments": {"options": {"position": [0, 1], "enabled": True}},
    }


def test_malformed_quoted_argument_is_rejected():
    with pytest.raises(ValueError, match="Invalid Pythonic tool call"):
        pythonic.parse_tool_call(
            "[write_file(content='const player = { x: 0, y: 1 };)]"
        )


def test_non_literal_argument_is_rejected():
    with pytest.raises(ValueError, match="must be a literal value"):
        pythonic.parse_tool_call("[write_file(content=get_content())]")


# Parser selection

# Minimal marker snippets that each parser's ParserSpec matches.
COHERE = "prefix <|START_ACTION|> suffix"
MISTRAL = "please [TOOL_CALLS] now"
GLM = "arg <arg_key> key"
JSON_TOOLS = "<tool_call> ... tool_call.name ..."
QWEN3_CODER = "<tool_call>\n<function=foo tool_call.name"


# --- string templates: unchanged behavior -------------------------------------


# --- multi-variant templates (the fix) ----------------------------------------


def test_dict_without_markers_returns_none():
    assert _infer_tool_parser({"default": "x", "tool_use": "y"}) is None


@pytest.mark.parametrize("template", [None, {}, [], 123, {"tool_use": None}])
def test_non_routable_inputs_return_none(template):
    assert _infer_tool_parser(template) is None


# --- through a processor (the runtime entry point) ----------------------------


def test_override_bypasses_inference():
    assert _infer_tool_parser("anything", override="mistral") == "mistral"
    with pytest.raises(ValueError):
        _infer_tool_parser("anything", override="does_not_exist")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
