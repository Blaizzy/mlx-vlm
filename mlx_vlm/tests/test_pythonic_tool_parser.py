import json
from types import SimpleNamespace

from mlx_vlm.server.responses_state import process_tool_calls
from mlx_vlm.tool_parsers import (
    _infer_tool_parser,
    _infer_tool_parser_from_processor,
    load_tool_module,
)

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
