"""Tests for prompt_utils module, specifically multimodal content handling."""

from mlx_vlm.prompt_utils import apply_chat_template, extract_text_from_content


def _assistant_tool_call(content):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {}},
            }
        ],
    }


class TestExtractTextFromContent:
    """Tests for the extract_text_from_content function."""

    def test_none_content(self):
        """None should return empty string."""
        result = extract_text_from_content(None)
        assert result == ""


class TestApplyChatTemplateIntegration:
    """Integration tests for apply_chat_template with multimodal content.

    These tests verify the actual bug fix works end-to-end, not just the helper.
    Uses return_messages=True to inspect intermediate messages without mocking.
    """

    def test_nemotron_omni_formats_image_and_audio_messages(self):
        """Nemotron Omni should use typed multimodal content for HF templates."""
        from mlx_vlm.prompt_utils import apply_chat_template

        for model_type in ("nemotron_h_nano_omni", "nemotronh_nano_omni_reasoning_v3"):
            result = apply_chat_template(
                None,
                {"model_type": model_type},
                "Describe the inputs.",
                return_messages=True,
                num_images=1,
                num_audios=1,
            )

            assert result == [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Describe the inputs.",
                            "content": "Describe the inputs.",
                        },
                        {"type": "audio"},
                    ],
                }
            ]

    def test_gemma4_unified_formats_video_and_audio_messages(self):
        """Video prompts should retain audio placeholders when audio is present."""
        from mlx_vlm.prompt_utils import apply_chat_template

        result = apply_chat_template(
            None,
            {"model_type": "gemma4_unified"},
            "Describe the video and audio.",
            return_messages=True,
            video=["clip.mp4"],
            fps=1,
            num_audios=1,
        )

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "clip.mp4",
                        "max_pixels": 224 * 224,
                        "fps": 1,
                    },
                    {"type": "audio"},
                    {
                        "type": "text",
                        "text": "Describe the video and audio.",
                        "content": "Describe the video and audio.",
                    },
                ],
            }
        ]

    def test_step3p7_formats_image_patch_token(self):
        """Step-3.7 prompts should include the placeholder its processor expands."""
        from mlx_vlm.prompt_utils import apply_chat_template

        result = apply_chat_template(
            None,
            {"model_type": "step3p7"},
            "What do you see?",
            return_messages=True,
            num_images=1,
        )

        assert result == [{"role": "user", "content": "<im_patch>What do you see?"}]

    def test_assistant_tool_call_content_is_preserved(self):
        """Existing text and structured content should remain unchanged."""
        for content in ["I will check.", [{"type": "text", "text": "I will check."}]]:
            result = apply_chat_template(
                None,
                {"model_type": "qwen3_vl"},
                _assistant_tool_call(content),
                return_messages=True,
            )

            assert result[0]["content"] == content

    def test_pydantic_basemodel_content_extraction(self):
        """Test that BaseModel message objects are handled correctly."""
        from pydantic import BaseModel

        from mlx_vlm.prompt_utils import apply_chat_template

        class ChatMessage(BaseModel):
            role: str
            content: list

        config = {"model_type": "qwen2_vl"}

        # BaseModel with multimodal content
        message = ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ABC123XYZ"},
                },
            ],
        )

        result = apply_chat_template(
            None, config, [message], return_messages=True, num_images=1
        )

        # Should extract text, not include base64
        assert isinstance(result, list)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert "ABC123" not in content, "Base64 leaked from BaseModel content!"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        assert "ABC123" not in str(
                            text
                        ), "Base64 leaked from BaseModel content!"

    def test_single_dict_prompt_multimodal(self):
        """Single dict prompt with multimodal content should not include base64.

        This tests the isinstance(prompt, dict) code path, which is different
        from isinstance(prompt, list) where we pass a list of message dicts.
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        config = {"model_type": "qwen2_vl"}

        # Single dict prompt (NOT a list of dicts)
        single_prompt = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this single prompt image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,SINGLEBASE64DATA"},
                },
            ],
        }

        result = apply_chat_template(
            None,
            config,
            single_prompt,  # Note: dict, not [dict]
            return_messages=True,
            num_images=1,
        )

        assert isinstance(result, list)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert (
                    "SINGLEBASE64" not in content
                ), "Base64 leaked from single dict prompt!"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        assert "SINGLEBASE64" not in str(text), "Base64 leaked!"


class TestExtractTextFromContentEdgeCases:
    """Edge case tests for extract_text_from_content."""

    def test_content_with_non_dict_items(self):
        """Non-dict items in list should be skipped."""
        content = ["just a string", {"type": "text", "text": "Valid item"}, 123, None]
        result = extract_text_from_content(content)
        assert result == "Valid item"

    def test_text_item_with_empty_text(self):
        """Text items with empty text should not add extra spaces."""
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Actual content"},
            {"type": "text", "text": ""},
        ]
        result = extract_text_from_content(content)
        assert result == "Actual content"


def test_apply_chat_template_uses_generic_text_model_fallback():
    class TextProcessor:
        chat_template = "{{ messages }}"

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "Hello"}]
            return "templated"

    result = apply_chat_template(TextProcessor(), {"model_type": "llama"}, "Hello")

    assert result == "templated"


def test_apply_chat_template_preserves_explicit_thinking_enabled():
    class ThinkingProcessor:
        chat_template = "{{ messages }}"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            self.kwargs = kwargs
            if kwargs.get("enable_thinking") is False:
                return "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            return "<|im_start|>assistant\n<think>\n"

    processor = ThinkingProcessor()
    result = apply_chat_template(
        processor,
        {"model_type": "qwen3_5_moe"},
        "Describe this image.",
        num_images=1,
        enable_thinking=True,
    )

    assert processor.kwargs["enable_thinking"] is True
    assert result.endswith("<think>\n")


def test_apply_chat_template_maps_enable_thinking_for_thinking_mode_templates():
    class ThinkingModeProcessor:
        chat_template = "{% if thinking_mode == 'enabled' %}<mm:think>{% endif %}"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            del messages, tokenize, add_generation_prompt
            self.kwargs = kwargs
            return "prompt"

    processor = ThinkingModeProcessor()
    result = apply_chat_template(
        processor,
        {"model_type": "minimax_m3_vl"},
        "Describe this image.",
        num_images=1,
        enable_thinking=True,
    )

    assert result == "prompt"
    assert processor.kwargs["enable_thinking"] is True
    assert processor.kwargs["thinking_mode"] == "enabled"


class TestModelSpecificPromptContracts:
    """Guard model-specific multimodal message formats from regressions."""

    def test_ernie4_5_vl_uses_image_url_before_text(self):
        from mlx_vlm.prompt_utils import apply_chat_template

        result = apply_chat_template(
            None,
            {"model_type": "ernie4_5_moe_vl"},
            "Describe this image.",
            return_messages=True,
            num_images=1,
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert [item["type"] for item in result[0]["content"]] == ["image_url", "text"]
        assert result[0]["content"][1]["text"] == "Describe this image."
