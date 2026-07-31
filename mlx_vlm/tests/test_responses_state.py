from mlx_vlm.server.responses_state import _response_items_to_chat


def test_function_output_image_becomes_visual_input():
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
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_view_image",
        "content": "[Image output attached]",
    }
    assert image_url not in messages[-1]["content"]


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
            "content": "Rendered result",
        }
    ]
