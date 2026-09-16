"""Tests for tool-parser selection, including multi-variant chat templates."""

import pytest

from mlx_vlm.tools import _infer_tool_parser

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
