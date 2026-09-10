"""Tests for tool-parser selection, including multi-variant chat templates."""

from types import SimpleNamespace

import pytest

from mlx_vlm.tools import _infer_tool_parser, _infer_tool_parser_from_processor

# Minimal marker snippets that each parser's ParserSpec matches.
COHERE = "prefix <|START_ACTION|> suffix"
MISTRAL = "please [TOOL_CALLS] now"
GLM = "arg <arg_key> key"
JSON_TOOLS = "<tool_call> ... tool_call.name ..."
QWEN3_CODER = "<tool_call>\n<function=foo tool_call.name"


# --- string templates: unchanged behavior -------------------------------------


def test_string_template_routes():
    assert _infer_tool_parser(JSON_TOOLS) == "json_tools"
    assert _infer_tool_parser(COHERE) == "cohere2_moe"


def test_string_no_markers_returns_none():
    assert _infer_tool_parser("a plain template with no tool markers") is None


def test_priority_specific_beats_generic_json_tools():
    # matches both qwen3_coder (priority 100) and json_tools (priority 0)
    assert _infer_tool_parser(QWEN3_CODER) == "qwen3_coder"


# --- multi-variant templates (the fix) ----------------------------------------


def test_dict_template_routes_on_tool_use_variant():
    template = {"default": "no markers here", "tool_use": COHERE, "rag": "x"}
    assert _infer_tool_parser(template) == "cohere2_moe"


def test_dict_prefers_tool_use_over_default():
    template = {"default": GLM, "tool_use": MISTRAL}
    assert _infer_tool_parser(template) == "mistral"


def test_dict_falls_back_to_default_then_any():
    assert _infer_tool_parser({"default": COHERE}) == "cohere2_moe"
    assert _infer_tool_parser({"weird_only": MISTRAL}) == "mistral"


def test_list_of_named_templates_routes():
    template = [
        {"name": "default", "template": "nothing"},
        {"name": "tool_use", "template": JSON_TOOLS},
    ]
    assert _infer_tool_parser(template) == "json_tools"


def test_real_cohere_shape_all_variants_have_marker():
    # c4ai-command-r7b: default/tool_use/rag all carry <|START_ACTION|>
    template = {"default": COHERE, "tool_use": COHERE, "rag": COHERE}
    assert _infer_tool_parser(template) == "cohere2_moe"


def test_dict_without_markers_returns_none():
    assert _infer_tool_parser({"default": "x", "tool_use": "y"}) is None


@pytest.mark.parametrize("template", [None, {}, [], 123, {"tool_use": None}])
def test_non_routable_inputs_return_none(template):
    assert _infer_tool_parser(template) is None


# --- through a processor (the runtime entry point) ----------------------------


def test_from_processor_with_dict_template():
    tok = SimpleNamespace(chat_template={"default": "x", "tool_use": COHERE})
    processor = SimpleNamespace(tokenizer=tok)
    assert _infer_tool_parser_from_processor(processor) == "cohere2_moe"


def test_override_bypasses_inference():
    assert _infer_tool_parser("anything", override="mistral") == "mistral"
    with pytest.raises(ValueError):
        _infer_tool_parser("anything", override="does_not_exist")
