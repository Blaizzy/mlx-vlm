"""Golden tests for the vendored DeepSeek-V4 encoding and the processor/parser built on it.

Fixtures are the official DeepSeek-V4 encoding test cases (see fixtures/deepseek_v4_encoding).
"""

import json
from pathlib import Path

import pytest

from mlx_vlm.models.deepseek_v4 import encoding_dsv4
from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import encode_deepseek_v4_prompt
from mlx_vlm.tools import load_tool_module, process_tool_calls

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v4_encoding"
ASSISTANT_THINK = "<｜Assistant｜><think>"


def _load(case):
    data = json.loads(
        (FIXTURES / f"test_input_{case}.json").read_text(encoding="utf-8")
    )
    gold = (FIXTURES / f"test_output_{case}.txt").read_text(encoding="utf-8")
    return data, gold


@pytest.mark.parametrize("case,mode", [(2, "thinking"), (3, "thinking"), (4, "chat")])
def test_reference_encoding_matches_gold(case, mode):
    messages, gold = _load(case)
    assert encoding_dsv4.encode_messages(messages, thinking_mode=mode) == gold


def test_case_1_reference_and_processor_paths_match_gold():
    data, gold = _load(1)
    messages = [dict(m) for m in data["messages"]]
    reference = [dict(m) for m in messages]
    reference[0]["tools"] = data["tools"]
    assert encoding_dsv4.encode_messages(reference, thinking_mode="thinking") == gold
    # The processor receives tools as a separate argument (OpenAI style).
    assert (
        encode_deepseek_v4_prompt(messages, tools=data["tools"], enable_thinking=True)
        == gold
    )


def test_case_1_dsml_parser_matches_reference_parser():
    _, gold = _load(1)
    start = gold.find(ASSISTANT_THINK) + len(ASSISTANT_THINK)
    turn = gold[start : gold.find("<｜User｜>", start)]
    reference = encoding_dsv4.parse_message_from_completion_text(
        turn, thinking_mode="thinking"
    )
    output = turn.split(encoding_dsv4.thinking_end_token, 1)[1]
    result = process_tool_calls(output, load_tool_module("deepseek_v4"), tools=None)
    assert [c["function"]["name"] for c in result.calls] == [
        c["function"]["name"] for c in reference["tool_calls"]
    ]
    assert [json.loads(c["function"]["arguments"]) for c in result.calls] == [
        json.loads(c["function"]["arguments"]) for c in reference["tool_calls"]
    ]
