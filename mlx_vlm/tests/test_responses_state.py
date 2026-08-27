import json

import pytest
from fastapi import HTTPException

from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server.responses_state import (
    _response_items_to_chat,
    suppress_tool_call_content,
)


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


def _stream(chunks, tc_start, tc_end):
    """Feed chunks through suppress_tool_call_content, return visible content."""
    full = ""
    in_tool_call = False
    content = ""
    for chunk in chunks:
        full += chunk
        in_tool_call, delta = suppress_tool_call_content(
            full, in_tool_call, tc_start, chunk, tc_end
        )
        if delta:
            content += delta
    return content


def test_streamed_content_resumes_after_a_tool_call_ends():
    """Text after a completed tool call must not be swallowed.

    Suppression used to latch on for the rest of the stream, because
    `tc_start in full_output` stays true once the call has been seen.
    """
    content = _stream(
        [
            "Let me look. ",
            "<tool_call>",
            '{"name": "get_weather"}',
            "</tool_call>",
            " The weather is sunny.",
        ],
        "<tool_call>",
        "</tool_call>",
    )

    assert "get_weather" not in content
    assert content == "Let me look.  The weather is sunny."


def test_streamed_content_resumes_between_consecutive_tool_calls():
    content = _stream(
        [
            "<tool_call>",
            '{"name": "a"}',
            "</tool_call>",
            " then ",
            "<tool_call>",
            '{"name": "b"}',
            "</tool_call>",
            " done.",
        ],
        "<tool_call>",
        "</tool_call>",
    )

    assert content == " then  done."


def test_streamed_tool_call_markup_is_still_suppressed():
    content = _stream(
        ["Hello ", "<tool_call>", '{"name": "a"}'],
        "<tool_call>",
        "</tool_call>",
    )

    assert content == "Hello "


def test_suppression_without_an_end_marker_is_unchanged():
    """Parsers with an empty tool_call_end keep the latching behavior."""
    content = _stream(
        ["Hello ", "<tool_call>", '{"name": "a"}', " trailing"],
        "<tool_call>",
        "",
    )

    assert content == "Hello "


def test_tool_call_markup_inside_a_single_chunk_is_still_suppressed():
    """A chunk that carries a whole call, or the end marker plus trailing
    text, must not leak the markup itself into content."""
    whole_call = _stream(
        ['<tool_call>{"name": "shell", "command": "pwd"}</tool_call>'],
        "<tool_call>",
        "</tool_call>",
    )
    assert whole_call == ""

    end_and_tail = _stream(
        ["<tool_call>", '{"name": "shell"}', "</tool_call> done."],
        "<tool_call>",
        "</tool_call>",
    )
    assert end_and_tail == " done."
