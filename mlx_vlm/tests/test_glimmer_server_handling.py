"""Tests for Muse Glimmer final-parse handling in the server.

The streamed deltas are handled by ``MuseGlimmerStreamState`` (see
``test_glimmer_stream.py``); these tests cover the final-parse functions that
recompute the authoritative response from the full text:
``_split_thinking`` (reasoning routing) and ``process_tool_calls`` (ATEM).
"""

from mlx_vlm.server.responses_state import _split_thinking, process_tool_calls
from mlx_vlm.tool_parsers import atem

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


class FakeGlimmerProcessor:
    pass


FakeGlimmerProcessor.__module__ = "mlx_vlm.models.muse_glimmer.processing_muse_glimmer"


def _atem_block(name, cmd):
    return (
        "<atem:function_calls>\n"
        f'<atem:invoke name="{name}">\n'
        f'<atem:parameter name="command">{cmd}</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )


# --- _split_thinking ------------------------------------------------------

def test_glimmer_reasoning_routed_to_reasoning():
    reasoning, content = _split_thinking(
        "to=self<|message|>The user wants a color. Blue is typical.<|eom|>"
        "to=user<|message|>blue",
        processor=FakeGlimmerProcessor(),
    )
    assert "The user wants a color" in reasoning
    assert content.strip() == "blue"


def test_glimmer_answer_only_stays_in_content():
    reasoning, content = _split_thinking(
        "to=user<|message|>The file was created.<|eom|>",
        processor=FakeGlimmerProcessor(),
    )
    assert reasoning is None
    assert "The file was created." in content
    assert "to=user" not in content


def test_glimmer_multi_reasoning_blocks():
    reasoning, content = _split_thinking(
        "to=self<|message|>think one<|eom|>to=self<|message|>think two<|eom|>"
        "to=user<|message|>answer",
        processor=FakeGlimmerProcessor(),
    )
    assert "think one" in reasoning and "think two" in reasoning
    assert content.strip() == "answer"


def test_glimmer_configured_tokens_answer_only():
    """Answer-only output with the Glimmer pair configured via thinking
    tokens must stay in content (no loose-branch misrouting)."""
    reasoning, content = _split_thinking(
        "to=user<|message|>The file was created.<|eom|>",
        thinking_start_token="to=self<|message|>",
        thinking_end_token="<|eom|>",
        processor=FakeGlimmerProcessor(),
    )
    assert reasoning is None
    assert "The file was created." in content


def test_non_glimmer_standard_markers_unaffected():
    reasoning, content = _split_thinking(
        "<think>inner</think>final",
        processor=FakeGlimmerProcessor(),
    )
    assert "inner" in (reasoning or "")
    assert content.strip() == "final"


def test_non_glimmer_answer_only_not_misrouted():
    reasoning, content = _split_thinking(
        "to=user<|message|>The file was created.<|eom|>",
        processor=object(),
    )
    # Non-Glimmer processor: no glimmer handling, no configured glimmer
    # tokens — the text is plain content.
    assert reasoning is None


# --- process_tool_calls ----------------------------------------------------

def test_process_tool_calls_bare_block_with_header():
    out = (
        "to=bash<|message|>"
        + _atem_block("bash", "echo hi")
    )
    tc = process_tool_calls(out, atem, TOOLS)
    assert len(tc["calls"]) == 1
    assert tc["calls"][0]["function"]["name"] == "bash"
    assert "echo hi" in tc["calls"][0]["function"]["arguments"]
    assert "to=bash<|message|>" not in tc["remaining_text"]


def test_process_tool_calls_multiple_blocks():
    out = (
        "to=read<|message|>"
        + _atem_block("bash", "one")
        + "<|eom|>"
        + _atem_block("bash", "two")
    )
    tc = process_tool_calls(out, atem, TOOLS)
    assert len(tc["calls"]) == 2, f"expected 2 calls, got {len(tc['calls'])}"
    assert "one" in tc["calls"][0]["function"]["arguments"]
    assert "two" in tc["calls"][1]["function"]["arguments"]
    assert "<|eom|>" not in tc["remaining_text"]


def test_process_tool_calls_envelope_form():
    out = (
        "to=self<|message|>I need to run a command.<|eom|>"
        "to=bash<|message|>"
        + _atem_block("bash", "date")
    )
    tc = process_tool_calls(out, atem, TOOLS)
    assert len(tc["calls"]) == 1
    assert "date" in tc["calls"][0]["function"]["arguments"]
    assert "to=self" not in tc["remaining_text"]
    assert "I need to run" not in tc["remaining_text"]


def test_process_tool_calls_full_header_form():
    out = (
        "to=self<|message|>I should check the weather.<|eom|>"
        "<|start|>assistant to=weather.get_weather<|message|>"
        "<atem:function_calls>\n"
        '<atem:invoke name="weather.get_weather">\n'
        '<atem:parameter name="city">Warsaw</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    tc = process_tool_calls(out, atem, tools=None)
    assert len(tc["calls"]) == 1
    assert tc["remaining_text"] == ""


def test_process_tool_calls_truncated_envelope():
    """A reasoning envelope cut off before <|eom|> must not leak to=self."""
    out = "to=self<|message|>planning cut off to=user<|message|>answer"
    tc = process_tool_calls(out, atem, TOOLS)
    assert "to=self" not in tc["remaining_text"]
