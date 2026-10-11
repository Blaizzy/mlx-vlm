"""ATEM must parse the post-response-template content seen by agent clients."""

import json
import pytest
from mlx_vlm.tools import load_tool_module, process_tool_calls
from mlx_vlm.server.responses_state import ToolCallStreamState

CALL = '<atem:function_calls><atem:invoke name="read"><atem:parameter name="path">/tmp/synthetic.txt</atem:parameter></atem:invoke></atem:function_calls>'
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]


@pytest.mark.parametrize("prefix", ["", "to=read<|message|>"])
def test_atem_content_after_recipient_parsing_is_executable(prefix):
    parsed = process_tool_calls(prefix + CALL, load_tool_module("atem"), TOOLS)
    assert len(parsed.calls) == 1
    assert parsed.calls[0]["function"]["name"] == "read"
    assert json.loads(parsed.calls[0]["function"]["arguments"]) == {
        "path": "/tmp/synthetic.txt"
    }


@pytest.mark.parametrize("size", [1, 2, 7, len(CALL)])
def test_atem_stream_hides_only_closed_call_block(size):
    parser = load_tool_module("atem")
    state = ToolCallStreamState(parser.tool_call_start, parser.tool_call_end)
    text = "Before. " + CALL + " After."
    visible = []
    for i in range(0, len(text), size):
        visible.append(state.feed(text[i : i + size], last=i + size >= len(text)) or "")
    assert "".join(visible) == "Before.   After."


def test_reasoning_envelope_without_call_keeps_final_answer():
    parser = load_tool_module("atem")
    state = ToolCallStreamState(parser.tool_call_start, parser.tool_call_end)
    assert state.feed("ROGUE 42", last=True) == "ROGUE 42"
