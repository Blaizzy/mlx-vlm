import json

import pytest
from fastapi import HTTPException

from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server.responses_state import _response_items_to_chat


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
