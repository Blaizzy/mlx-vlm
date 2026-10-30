"""Streaming arguments remain prefixes of validated, complete tool calls."""

import json

import pytest

from mlx_vlm.tools import load_tool_module, process_tool_calls
from mlx_vlm.tools.streaming import ToolCallDeltaState

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "path": {"type": "string"},
                },
            },
        },
    }
]


def encoded_call(parser, content):
    args = {"path": "demo.py", "content": content}
    if parser == "json_tools":
        return (
            "<tool_call>"
            + json.dumps({"name": "write_file", "arguments": args}, ensure_ascii=False)
            + "</tool_call>"
        )
    if parser == "cohere2_moe":
        return (
            "<|START_ACTION|>"
            + json.dumps(
                [{"tool_name": "write_file", "parameters": args}], ensure_ascii=False
            )
            + "<|END_ACTION|>"
        )
    if parser == "qwen3_coder":
        return f"<tool_call><function=write_file><parameter=path>demo.py</parameter><parameter=content>\n{content}\n</parameter></function></tool_call>"
    return f'<|tool_call>call:write_file{{path:<|"|>demo.py<|"|>,content:<|"|>{content}<|"|>}}<tool_call|>'


@pytest.mark.parametrize(
    "parser", ["json_tools", "cohere2_moe", "qwen3_coder", "gemma4"]
)
@pytest.mark.parametrize("chunk_size", [1, 7, 31, 8192])
def test_argument_prefixes_and_final_call_match(parser, chunk_size):
    content = 'print("hello \\ 世界")\n' * 50
    text = encoded_call(parser, content)
    module = load_tool_module(parser)
    state = ToolCallDeltaState(module, TOOLS)
    received = {}
    early = False
    for offset in range(0, len(text), chunk_size):
        for delta in state.feed(text[offset : offset + chunk_size]):
            call = received.setdefault(delta["index"], {"arguments": ""})
            call["id"] = delta.get("id", call.get("id"))
            call["name"] = delta["function"].get("name", call.get("name"))
            call["arguments"] += delta["function"].get("arguments", "")
            early |= (
                len(call["arguments"]) > 100 and offset + chunk_size < len(text) - 100
            )
    final = process_tool_calls(text, module, TOOLS).calls
    assert state.finish(final) == []
    assert received[0]["name"] == "write_file"
    assert received[0]["id"] == final[0]["id"]
    assert received[0]["arguments"] == final[0]["function"]["arguments"]
    assert json.loads(received[0]["arguments"]) == {
        "path": "demo.py",
        "content": content,
    }
    assert early or chunk_size >= len(text)


@pytest.mark.parametrize(
    "parser", ["json_tools", "cohere2_moe", "qwen3_coder", "gemma4"]
)
def test_multiple_calls_keep_separate_ids(parser):
    module = load_tool_module(parser)
    state = ToolCallDeltaState(module, TOOLS)
    text = encoded_call(parser, "first") + encoded_call(parser, "second")
    deltas = [delta for ch in text for delta in state.feed(ch)]
    final = process_tool_calls(text, module, TOOLS).calls
    state.finish(final)
    ids = [delta["id"] for delta in deltas if "id" in delta]
    assert len(ids) == len(set(ids)) == 2
    assert [call["id"] for call in final] == ids
    assert [json.loads(call["function"]["arguments"])["content"] for call in final] == [
        "first",
        "second",
    ]


INT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "content": {"type": "string"},
                },
            },
        },
    }
]

NO_PARAM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.mark.parametrize("chunk_size", [1, 8192])
@pytest.mark.parametrize(
    "parser, tools, text, kept",
    [
        ("json_tools", TOOLS, '<tool_call>{"name": "get_time"}</tool_call>', []),
        (
            "json_tools",
            TOOLS,
            '<tool_call>{"name": 5, "arguments": {}}</tool_call>',
            [],
        ),
        ("json_tools", TOOLS, "<tool_call>5</tool_call>", []),
        ("json_tools", TOOLS, '<tool_call>"get_time"</tool_call>', []),
        (
            "json_tools",
            TOOLS,
            '<tool_call>[{"name": "write_file", "arguments": {"path": "x"}},'
            ' "oops"]</tool_call>',
            ["write_file"],
        ),
        (
            "qwen3_coder",
            INT_TOOLS,
            "<tool_call><function=write_file><parameter=count>twenty</parameter>"
            "<parameter=content>x</parameter></function></tool_call>",
            [],
        ),
    ],
)
def test_streaming_tolerates_what_the_final_parser_drops(
    parser, tools, text, kept, chunk_size
):
    """A malformed call the final parser drops must not abort the stream."""
    module = load_tool_module(parser)
    state = ToolCallDeltaState(module, tools)
    received = {}
    for offset in range(0, len(text), chunk_size):
        for delta in state.feed(text[offset : offset + chunk_size]):
            call = received.setdefault(delta["index"], {"arguments": ""})
            call["name"] = delta["function"].get("name", call.get("name"))
            call["arguments"] += delta["function"].get("arguments", "")
    final = process_tool_calls(text, module, tools).calls
    assert [call["function"]["name"] for call in final] == kept
    state.finish(final)
    assert [call["name"] for call in received.values()] == kept


def test_trailing_whitespace_before_tool_call_end_keeps_the_call():
    """A blank line before </tool_call> is not grounds to drop the call."""
    module = load_tool_module("qwen3_coder")
    text = "<tool_call><function=get_time></function>\n\n</tool_call>"
    final = process_tool_calls(text, module, NO_PARAM_TOOLS).calls
    assert [call["function"]["name"] for call in final] == ["get_time"]
    state = ToolCallDeltaState(module, NO_PARAM_TOOLS)
    deltas = [delta for ch in text for delta in state.feed(ch)]
    assert state.finish(final) == []
    assert [delta["function"]["name"] for delta in deltas if "id" in delta] == [
        "get_time"
    ]


def test_truncated_call_cannot_finish_successfully():
    module = load_tool_module("json_tools")
    state = ToolCallDeltaState(module, TOOLS)
    assert state.feed(
        '<tool_call>{"name":"write_file","arguments":{"content":"unfinished'
    )
    with pytest.raises(ValueError, match="incomplete tool call"):
        state.finish([])


def test_cohere_parallel_action_and_escape_repair():
    module = load_tool_module("cohere2_moe")
    text = '<|START_ACTION|>[{"tool_name":"write_file","parameters":{"content":"a\\|b"}},{"tool_name":"write_file","parameters":{"content":"c"}}]<|END_ACTION|>'
    state = ToolCallDeltaState(module, TOOLS)
    for char in text:
        state.feed(char)
    calls = process_tool_calls(text, module, TOOLS).calls
    assert state.finish(calls) == []
    assert [json.loads(call["function"]["arguments"])["content"] for call in calls] == [
        "a|b",
        "c",
    ]


@pytest.mark.parametrize(
    "value", ["null", "number", "", "\ntext\n", "\n\n", "</param", '"\\value']
)
def test_xml_strings_preserve_final_parser_semantics(value):
    module = load_tool_module("qwen3_coder")
    text = encoded_call("qwen3_coder", value)
    state = ToolCallDeltaState(module, TOOLS)
    for char in text:
        state.feed(char)
    expected = process_tool_calls(text, module, TOOLS).calls
    assert state.finish(expected) == []
