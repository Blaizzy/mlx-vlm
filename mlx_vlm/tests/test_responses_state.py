import json

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
                {"type": "input_image", "image_url": image_url, "detail": "high"}
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
        None, {"model_type": "qwen2_vl"}, messages, num_images=len(images)
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


def test_tool_content_without_end_marker_keeps_latching_behavior():
    content = _stream_tool_content(
        ["Before ", "<tool_call>", '{"name": "a"}', " trailing"], tc_end=""
    )

    assert content == "Before "


def test_unfinished_start_marker_is_released_when_stream_ends():
    content = _stream_tool_content(["A literal <tool"])

    assert content == "A literal <tool"
