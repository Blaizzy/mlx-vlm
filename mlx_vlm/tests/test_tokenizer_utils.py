"""Tests for tokenizer_utils module."""

import pytest

from mlx_vlm.tokenizer_utils import (
    REPLACEMENT_CHAR,
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    StreamingDetokenizer,
    TokenizerWrapper,
    _is_bpe_decoder,
    _is_spm_decoder,
    _is_spm_decoder_no_space,
    _match,
)

# ============================================================================
# Mock Classes
# ============================================================================


class MockTokenizer:
    """Mock tokenizer for testing detokenizers."""

    def __init__(self, vocab=None):
        self.vocab = vocab or {
            "hello": 0,
            "world": 1,
            "▁hello": 2,
            "▁world": 3,
            "<0xE5>": 4,
            "<0xA4>": 5,
            "<0xA2>": 6,
            "<0xE7>": 7,
            "<0xA1>": 8,
            "<0xAF>": 9,
            "test": 10,
            "▁test": 11,
        }

    def decode(self, tokens):
        """Simple decode for testing."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return "".join(inv_vocab.get(t, "") for t in tokens)


class MockBPETokenizer:
    """Mock tokenizer with BPE-style vocabulary."""

    def __init__(self):
        # BPE uses special unicode characters for bytes
        # See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.vocab = {
            "Ġhello": 0,  # Ġ represents space in GPT-2 BPE
            "Ġworld": 1,
            "hello": 2,
            "world": 3,
            "test": 4,
        }

    def decode(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        text = "".join(inv_vocab.get(t, "") for t in tokens)
        return text.replace("Ġ", " ")


# ============================================================================
# Tests for Helper Functions
# ============================================================================


class TestMatch:
    """Tests for _match helper function."""

    def test_match_different_types(self):
        assert _match(1, "1") is False
        assert _match([], {}) is False


class TestIsSpmDecoder:
    """Tests for _is_spm_decoder function."""

    def test_invalid_spm_decoder(self):
        decoder = {"type": "ByteLevel"}
        assert _is_spm_decoder(decoder) is False


class TestIsSpmDecoderNoSpace:
    """Tests for _is_spm_decoder_no_space function."""

    def test_valid_spm_decoder_no_space(self):
        decoder = {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        }
        assert _is_spm_decoder_no_space(decoder) is True


class TestIsBpeDecoder:
    """Tests for _is_bpe_decoder function."""

    def test_valid_bpe_decoder(self):
        decoder = {"type": "ByteLevel"}
        assert _is_bpe_decoder(decoder) is True


# ============================================================================
# Tests for NaiveStreamingDetokenizer
# ============================================================================


class TestNaiveStreamingDetokenizer:
    """Tests for NaiveStreamingDetokenizer class."""

    def test_skip_special_tokens(self):
        tokenizer = MockTokenizer()
        detokenizer = NaiveStreamingDetokenizer(tokenizer)

        detokenizer.add_token(0)
        detokenizer.add_token(1, skip_special_token_ids=[1])
        detokenizer.finalize()

        assert "world" not in detokenizer.text


# ============================================================================
# Tests for SPMStreamingDetokenizer
# ============================================================================


class TestSPMStreamingDetokenizer:
    """Tests for SPMStreamingDetokenizer class."""

    def test_skip_special_tokens(self):
        tokenizer = MockTokenizer()
        detokenizer = SPMStreamingDetokenizer(tokenizer)

        detokenizer.add_token(2)  # "▁hello"
        detokenizer.add_token(3, skip_special_token_ids=[3])  # "▁world" - skipped
        detokenizer.finalize()

        assert "hello" in detokenizer.text
        assert "world" not in detokenizer.text


class TestSPMStreamingDetokenizerUTF8:
    """Tests for UTF-8 byte token handling in SPMStreamingDetokenizer."""

    def test_english_with_byte_tokens(self):
        """Test English text mixed with byte-encoded special chars."""
        # é = C3 A9 in UTF-8
        vocab = {"▁caf": 0, "<0xC3>": 1, "<0xA9>": 2, "▁is": 3, "▁great": 4}
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁caf
        detokenizer.add_token(1)  # <0xC3>
        detokenizer.add_token(2)  # <0xA9>
        detokenizer.add_token(3)  # ▁is
        detokenizer.add_token(4)  # ▁great
        detokenizer.finalize()

        assert detokenizer.text == "café is great"

    def test_utf8_invalid_sequence(self):
        """Test handling of invalid UTF-8 byte sequences."""
        vocab = {
            "test": 0,
            "<0xFF>": 1,  # Invalid UTF-8 byte
            "<0xFE>": 2,  # Invalid UTF-8 byte
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        detokenizer.add_token(0)  # test
        detokenizer.add_token(1)  # <0xFF>
        detokenizer.add_token(2)  # <0xFE>
        detokenizer.finalize()

        # Should use replacement character for invalid sequences
        assert "test" in detokenizer.text
        assert REPLACEMENT_CHAR in detokenizer.text


class TestSPMStreamingDetokenizerLastSegment:
    """Tests for last_segment property in SPMStreamingDetokenizer."""

    def test_last_segment_basic(self):
        vocab = {"▁hello": 0, "▁world": 1}
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁hello
        # First access to last_segment
        segment1 = detokenizer.last_segment

        detokenizer.add_token(1)  # ▁world
        detokenizer.finalize()
        segment2 = detokenizer.last_segment

        # Segments should be different parts of the text
        assert "hello" in segment1 or "hello" in segment2
        assert "world" in segment2 or "world" in detokenizer.text


# ============================================================================
# Tests for BPEStreamingDetokenizer
# ============================================================================


class TestBPEStreamingDetokenizer:
    """Tests for BPEStreamingDetokenizer class."""

    def test_initialization(self):
        tokenizer = MockBPETokenizer()
        detokenizer = BPEStreamingDetokenizer(tokenizer)

        assert detokenizer._byte_decoder is not None
        assert detokenizer.text == ""


# ============================================================================
# Tests for TokenizerWrapper
# ============================================================================


class TestTokenizerWrapper:
    """Tests for TokenizerWrapper class."""

    def test_attribute_forwarding(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)

        # Should forward vocab attribute to underlying tokenizer
        assert wrapper.vocab == tokenizer.vocab

    def test_custom_detokenizer_class(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer, SPMStreamingDetokenizer)

        assert isinstance(wrapper.detokenizer, SPMStreamingDetokenizer)


# ============================================================================
# Tests for StreamingDetokenizer Base Class
# ============================================================================


class TestStreamingDetokenizerBase:
    """Tests for StreamingDetokenizer base class."""

    def test_not_implemented_methods(self):
        class TestDetokenizer(StreamingDetokenizer):
            pass

        detokenizer = TestDetokenizer()

        with pytest.raises(NotImplementedError):
            detokenizer.reset()

        with pytest.raises(NotImplementedError):
            detokenizer.add_token(0)

        with pytest.raises(NotImplementedError):
            detokenizer.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
