"""Tests for prompt_utils module, specifically multimodal content handling."""

from mlx_vlm.prompt_utils import extract_text_from_content


class TestExtractTextFromContent:
    """Tests for the extract_text_from_content function."""

    def test_string_content_passthrough(self):
        """String content should be returned as-is."""
        content = "Hello, describe this image."
        result = extract_text_from_content(content)
        assert result == "Hello, describe this image."

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = extract_text_from_content("")
        assert result == ""

    def test_multimodal_content_with_text_and_image(self):
        """Should extract only text from multimodal content, skipping image_url."""
        content = [
            {"type": "text", "text": "Describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS..."},
            },
        ]
        result = extract_text_from_content(content)
        assert result == "Describe this image in detail."

    def test_multimodal_content_with_input_image(self):
        """Should handle input_image type (alternative format)."""
        content = [
            {"type": "text", "text": "What do you see?"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQ..."},
        ]
        result = extract_text_from_content(content)
        assert result == "What do you see?"

    def test_multimodal_content_with_multiple_text_parts(self):
        """Should concatenate multiple text parts."""
        content = [
            {"type": "text", "text": "First part."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Second part."},
        ]
        result = extract_text_from_content(content)
        assert result == "First part. Second part."

    def test_multimodal_content_with_input_text(self):
        """Should handle input_text type."""
        content = [
            {"type": "input_text", "text": "Alternative text format."},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "Alternative text format."

    def test_multimodal_content_with_content_field(self):
        """Should handle text items with 'content' instead of 'text' field."""
        content = [
            {"type": "text", "content": "Using content field."},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "Using content field."

    def test_multimodal_content_with_audio(self):
        """Should skip audio content."""
        content = [
            {"type": "text", "text": "Transcribe this audio."},
            {"type": "input_audio", "input_audio": {"data": "base64audiodata..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "Transcribe this audio."

    def test_empty_list(self):
        """Empty list should return empty string."""
        result = extract_text_from_content([])
        assert result == ""

    def test_list_with_only_images(self):
        """List with only images should return empty string."""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]
        result = extract_text_from_content(content)
        assert result == ""

    def test_none_content(self):
        """None should return empty string."""
        result = extract_text_from_content(None)
        assert result == ""

    def test_large_base64_not_included(self):
        """Ensure large base64 strings are NOT included in output.

        This is the critical test case for the bug fix.
        A 428x1000 pixel image encoded as base64 is ~570KB.
        If this were tokenized as text, it would produce ~422k tokens.
        """
        # Simulate a realistic multimodal message with large base64
        large_base64 = "iVBOR" + "A" * 570000  # ~570KB of base64 data
        content = [
            {"type": "text", "text": "Extract product information from this image."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{large_base64}"},
            },
        ]
        result = extract_text_from_content(content)

        # Result should be the text only, not the base64
        assert result == "Extract product information from this image."
        # Result should be short, not hundreds of KB
        assert len(result) < 1000

    def test_real_world_openai_format(self):
        """Test with exact format sent by OpenAI-compatible clients."""
        content = [
            {
                "type": "text",
                "text": "이미지에서 상품명, 가격, 설명을 추출해주세요.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                },
            },
        ]
        result = extract_text_from_content(content)
        assert result == "이미지에서 상품명, 가격, 설명을 추출해주세요."


class TestApplyChatTemplateIntegration:
    """Integration tests for apply_chat_template with multimodal content.

    These tests verify the actual bug fix works end-to-end, not just the helper.
    Uses return_messages=True to inspect intermediate messages without mocking.
    """

    def test_multimodal_message_does_not_include_base64_in_prompt(self):
        """Critical regression test: base64 should NOT appear in formatted messages.

        This test reproduces the exact bug scenario:
        - OpenAI-compatible multimodal message with base64 image
        - apply_chat_template should extract only text for tokenization
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        config = {"model_type": "qwen2_vl"}

        # Multimodal message with base64 (the bug trigger)
        large_base64 = "iVBOR" + "A" * 10000  # Smaller for test speed
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{large_base64}"},
                    },
                ],
            }
        ]

        # Use return_messages=True to get intermediate messages without needing processor
        result = apply_chat_template(
            None,  # processor not needed with return_messages=True
            config,
            messages,
            return_messages=True,
            num_images=1,
        )

        # Verify: the messages should NOT contain base64
        assert isinstance(result, list)
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, str):
                assert "iVBOR" not in content, "Base64 data leaked into text content!"
                assert (
                    len(content) < 1000
                ), f"Content too long ({len(content)}), likely contains base64"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "") or item.get("content", "")
                        assert "iVBOR" not in str(
                            text
                        ), "Base64 data leaked into text content!"

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
            None,
            config,
            [message],
            return_messages=True,
            num_images=1,
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

    def test_text_content_preserved_correctly(self):
        """Verify that text content is preserved correctly after extraction."""
        from mlx_vlm.prompt_utils import apply_chat_template

        config = {"model_type": "qwen2_vl"}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this product image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]

        result = apply_chat_template(
            None,
            config,
            messages,
            return_messages=True,
            num_images=1,
        )

        # Find the text content in the result
        found_text = False
        for msg in result:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "") or item.get("content", "")
                        if "analyze this product" in text:
                            found_text = True
            elif isinstance(content, str) and "analyze this product" in content:
                found_text = True

        assert found_text, "Original text content was not preserved!"


class TestExtractTextFromContentEdgeCases:
    """Edge case tests for extract_text_from_content."""

    def test_malformed_content_item_no_type(self):
        """Items without 'type' should be skipped."""
        content = [
            {"text": "No type field"},
            {"type": "text", "text": "Has type field"},
        ]
        result = extract_text_from_content(content)
        assert result == "Has type field"

    def test_content_with_non_dict_items(self):
        """Non-dict items in list should be skipped."""
        content = [
            "just a string",
            {"type": "text", "text": "Valid item"},
            123,
            None,
        ]
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
