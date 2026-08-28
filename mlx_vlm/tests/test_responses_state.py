import json

import pytest
from fastapi import HTTPException

from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server.responses_state import ToolCallStreamState, _response_items_to_chat


def test_function_output_image_stays_after_tool_result():
    image_url = "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    items = [
        {
            "type": "function_call",
            "name": "view_image",
            "arguments": "{}",
            "call_id": "call_view_image",
        },
        {
            "type": "function_call_output",
            "call_id": "call_view_image",
            "output": [
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": "high",
                }
            ],
        },
    ]

    messages, images = _response_items_to_chat(items)

    assert images == [image_url]
    assert messages[-2:] == [
        {
            "role": "tool",
            "tool_call_id": "call_view_image",
            "content": "[Image output attached in the next message]",
        },
        {"role": "user", "content": [{"type": "image"}]},
    ]

    prompt = apply_chat_template(
        None,
        {"model_type": "qwen2_vl"},
        messages,
        num_images=len(images),
    )
    assert prompt.index("Tool:") < prompt.index("<image>")
    assert image_url not in prompt


def test_function_output_preserves_text_alongside_visual_input():
    image_url = "https://example.com/result.png"
    items = [
        {
            "type": "function_call_output",
            "call_id": "call_analyze_image",
            "output": [
                {"type": "input_text", "text": "Rendered result"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    messages, images = _response_items_to_chat(items)

    assert images == [image_url]
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_analyze_image",
            "content": "Rendered result\n[Image output attached in the next message]",
        },
        {"role": "user", "content": [{"type": "image"}]},
    ]


def test_function_call_none_content_is_normalized_for_chat_templates():
    items = [
        {
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"location":"SF"}',
            "call_id": "call_get_weather",
        }
    ]

    messages, images = _response_items_to_chat(items)
    normalized = apply_chat_template(
        None,
        {"model_type": "qwen3_vl"},
        messages,
        return_messages=True,
    )

    assert images == []
    assert messages[0]["content"] is None
    assert normalized[0]["content"] == ""


def test_message_image_stays_on_its_original_user_turn():
    image_url = "https://example.com/first-turn.png"
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "First turn"},
                {"type": "input_image", "image_url": image_url},
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I see it."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Second turn"}],
        },
    ]

    messages, images = _response_items_to_chat(items)
    normalized = apply_chat_template(
        None,
        {"model_type": "qwen2_vl"},
        messages,
        num_images=len(images),
        return_messages=True,
    )

    assert images == [image_url]
    assert any(part["type"] == "image" for part in normalized[0]["content"])
    assert normalized[-1]["content"] == [
        {"type": "text", "text": "Second turn", "content": "Second turn"}
    ]


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "file_id": "file-image"}],
        },
        {
            "type": "function_call_output",
            "call_id": "call_view_image",
            "output": [
                {
                    "type": "input_image",
                    "file_id": "file-image",
                    "image_url": "data:image/png;base64,ZmFrZQ==",
                }
            ],
        },
    ],
)
def test_response_input_rejects_image_file_id(item):
    with pytest.raises(HTTPException, match=r"input_image\.file_id is not supported"):
        _response_items_to_chat([item])


def test_unknown_function_output_blocks_remain_text():
    unknown = {"type": "custom_output", "value": {"answer": 42}}
    messages, images = _response_items_to_chat(
        [
            {
                "type": "function_call_output",
                "call_id": "call_custom",
                "output": [unknown],
            }
        ]
    )

    assert images == []
    assert json.loads(messages[0]["content"]) == [unknown]


def _stream_tool_content(chunks, tc_start="<tool_call>", tc_end="</tool_call>"):
    state = ToolCallStreamState(tc_start, tc_end)
    visible = []
    for index, chunk in enumerate(chunks):
        delta = state.feed(chunk, last=index == len(chunks) - 1)
        if delta:
            visible.append(delta)
    return "".join(visible)


def test_tool_content_resumes_after_completed_call():
    content = _stream_tool_content(
        [
            "Let me look. ",
            "<tool_call>",
            '{"name": "get_weather"}',
            "</tool_call>",
            " The weather is sunny.",
        ]
    )

    assert content == "Let me look.  The weather is sunny."


def test_tool_content_preserves_both_sides_of_coalesced_call():
    content = _stream_tool_content(
        ['Before <tool_call>{"name": "a"}</tool_call> after.']
    )

    assert content == "Before  after."


def test_tool_content_preserves_text_between_coalesced_calls():
    content = _stream_tool_content(
        [
            '<tool_call>{"name": "a"}</tool_call>'
            " between "
            '<tool_call>{"name": "b"}</tool_call>'
            " done."
        ]
    )

    assert content == " between  done."


def test_tool_content_handles_markers_split_across_chunks():
    content = _stream_tool_content(
        [
            "Before <tool",
            '_call>{"name": "a"}</tool',
            "_call> after.",
        ]
    )

    assert content == "Before  after."


def test_tool_content_is_invariant_to_chunk_boundaries():
    source = (
        'Before <tool_call>{"name": "a"}</tool_call>'
        ' between <tool_call>{"name": "b"}</tool_call> after.'
    )
    expected = "Before  between  after."

    assert _stream_tool_content(list(source)) == expected
    for split_at in range(len(source) + 1):
        assert _stream_tool_content([source[:split_at], source[split_at:]]) == expected


def test_tool_content_suppresses_unfinished_call():
    content = _stream_tool_content(["Before ", "<tool_call>", '{"name": "a"}'])

    assert content == "Before "


def test_tool_content_without_end_marker_keeps_latching_behavior():
    content = _stream_tool_content(
        ["Before ", "<tool_call>", '{"name": "a"}', " trailing"],
        tc_end="",
    )

    assert content == "Before "


def test_unfinished_start_marker_is_released_when_stream_ends():
    content = _stream_tool_content(["A literal <tool"])

    assert content == "A literal <tool"
