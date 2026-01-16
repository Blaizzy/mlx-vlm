"""Tests for tokenizer_utils module."""

import pytest

from mlx_vlm.tokenizer_utils import (
    REPLACEMENT_CHAR,
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    StreamingDetokenizer,
    TokenizerWrapper,
    _has_bytelevel_pretokenizer,
    _is_bpe_decoder,
    _is_spm_decoder,
    _is_spm_decoder_no_space,
    _match,
    _remove_space,
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


class TestRemoveSpace:
    """Tests for _remove_space function."""

    def test_removes_leading_space(self):
        assert _remove_space(" hello") == "hello"

    def test_no_leading_space(self):
        assert _remove_space("hello") == "hello"

    def test_empty_string(self):
        assert _remove_space("") == ""

    def test_only_space(self):
        assert _remove_space(" ") == ""

    def test_multiple_leading_spaces(self):
        # Only removes first space
        assert _remove_space("  hello") == " hello"

    def test_none_input(self):
        # Empty/falsy input returns as-is
        assert _remove_space(None) is None


class TestMatch:
    """Tests for _match helper function."""

    def test_match_simple_values(self):
        assert _match(1, 1) is True
        assert _match("a", "a") is True
        assert _match(1, 2) is False

    def test_match_different_types(self):
        assert _match(1, "1") is False
        assert _match([], {}) is False

    def test_match_lists(self):
        assert _match([1, 2, 3], [1, 2, 3]) is True
        assert _match([1, 2], [1, 2, 3]) is False
        assert _match([1, 2, 3], [1, 2]) is False

    def test_match_dicts(self):
        assert _match({"a": 1}, {"a": 1}) is True
        assert _match({"a": 1}, {"a": 2}) is False
        assert _match({"a": 1}, {"b": 1}) is False
        assert _match({"a": 1}, {"a": 1, "b": 2}) is False

    def test_match_nested_structures(self):
        nested1 = {"type": "Sequence", "items": [1, 2, {"nested": True}]}
        nested2 = {"type": "Sequence", "items": [1, 2, {"nested": True}]}
        nested3 = {"type": "Sequence", "items": [1, 2, {"nested": False}]}

        assert _match(nested1, nested2) is True
        assert _match(nested1, nested3) is False


class TestIsSpmDecoder:
    """Tests for _is_spm_decoder function."""

    def test_valid_spm_decoder(self):
        decoder = {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ],
        }
        assert _is_spm_decoder(decoder) is True

    def test_invalid_spm_decoder(self):
        decoder = {"type": "ByteLevel"}
        assert _is_spm_decoder(decoder) is False

    def test_partial_spm_decoder(self):
        decoder = {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            ],
        }
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

    def test_with_strip(self):
        # Should return False if Strip is present
        decoder = {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ],
        }
        assert _is_spm_decoder_no_space(decoder) is False


class TestIsBpeDecoder:
    """Tests for _is_bpe_decoder function."""

    def test_valid_bpe_decoder(self):
        decoder = {"type": "ByteLevel"}
        assert _is_bpe_decoder(decoder) is True

    def test_invalid_bpe_decoder(self):
        decoder = {"type": "Sequence"}
        assert _is_bpe_decoder(decoder) is False

    def test_non_dict(self):
        assert _is_bpe_decoder("ByteLevel") is False
        assert _is_bpe_decoder(None) is False


class TestHasByteLevelPretokenizer:
    """Tests for _has_bytelevel_pretokenizer function."""

    def test_direct_bytelevel(self):
        pre_tokenizer = {"type": "ByteLevel"}
        assert _has_bytelevel_pretokenizer(pre_tokenizer) is True

    def test_sequence_with_bytelevel(self):
        pre_tokenizer = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split"},
                {"type": "ByteLevel"},
            ],
        }
        assert _has_bytelevel_pretokenizer(pre_tokenizer) is True

    def test_sequence_without_bytelevel(self):
        pre_tokenizer = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split"},
                {"type": "Whitespace"},
            ],
        }
        assert _has_bytelevel_pretokenizer(pre_tokenizer) is False

    def test_non_dict(self):
        assert _has_bytelevel_pretokenizer(None) is False
        assert _has_bytelevel_pretokenizer("ByteLevel") is False


# ============================================================================
# Tests for NaiveStreamingDetokenizer
# ============================================================================


class TestNaiveStreamingDetokenizer:
    """Tests for NaiveStreamingDetokenizer class."""

    def test_basic_detokenization(self):
        tokenizer = MockTokenizer()
        detokenizer = NaiveStreamingDetokenizer(tokenizer)

        detokenizer.add_token(0)  # "hello"
        detokenizer.add_token(1)  # "world"
        detokenizer.finalize()

        assert "hello" in detokenizer.text
        assert "world" in detokenizer.text

    def test_reset(self):
        tokenizer = MockTokenizer()
        detokenizer = NaiveStreamingDetokenizer(tokenizer)

        detokenizer.add_token(0)
        detokenizer.finalize()
        detokenizer.reset()

        assert detokenizer.text == "" or detokenizer._text == ""
        assert detokenizer._current_tokens == []

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

    def test_basic_detokenization(self):
        tokenizer = MockTokenizer()
        detokenizer = SPMStreamingDetokenizer(tokenizer)

        detokenizer.add_token(2)  # "▁hello"
        detokenizer.add_token(3)  # "▁world"
        detokenizer.finalize()

        assert "hello" in detokenizer.text
        assert "world" in detokenizer.text

    def test_trim_space(self):
        tokenizer = MockTokenizer()
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(2)  # "▁hello"
        detokenizer.finalize()

        # Should not start with space when trim_space=True
        assert not detokenizer.text.startswith(" ")

    def test_no_trim_space(self):
        tokenizer = MockTokenizer()
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        detokenizer.add_token(2)  # "▁hello"
        detokenizer.finalize()

        # Should preserve leading space when trim_space=False
        assert "hello" in detokenizer.text

    def test_reset(self):
        tokenizer = MockTokenizer()
        detokenizer = SPMStreamingDetokenizer(tokenizer)

        detokenizer.add_token(2)
        detokenizer.finalize()
        detokenizer.reset()

        assert detokenizer.text == ""
        assert detokenizer._unflushed == ""
        assert detokenizer._byte_buffer == bytearray()

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

    def test_ascii_english_text(self):
        """Test basic English ASCII text."""
        vocab = {
            "▁Hello": 0,
            "▁world": 1,
            "!": 2,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁Hello
        detokenizer.add_token(1)  # ▁world
        detokenizer.add_token(2)  # !
        detokenizer.finalize()

        assert detokenizer.text == "Hello world!"

    def test_english_with_byte_tokens(self):
        """Test English text mixed with byte-encoded special chars."""
        # é = C3 A9 in UTF-8
        vocab = {
            "▁caf": 0,
            "<0xC3>": 1,
            "<0xA9>": 2,
            "▁is": 3,
            "▁great": 4,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁caf
        detokenizer.add_token(1)  # <0xC3>
        detokenizer.add_token(2)  # <0xA9>
        detokenizer.add_token(3)  # ▁is
        detokenizer.add_token(4)  # ▁great
        detokenizer.finalize()

        assert detokenizer.text == "café is great"

    def test_english_subword_tokens(self):
        """Test English with subword tokenization."""
        vocab = {
            "▁un": 0,
            "believ": 1,
            "able": 2,
            "▁story": 3,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁un
        detokenizer.add_token(1)  # believ
        detokenizer.add_token(2)  # able
        detokenizer.add_token(3)  # ▁story
        detokenizer.finalize()

        assert detokenizer.text == "unbelievable story"

    def test_english_punctuation(self):
        """Test English with various punctuation."""
        vocab = {
            "▁Hello": 0,
            ",": 1,
            "▁how": 2,
            "▁are": 3,
            "▁you": 4,
            "?": 5,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        tokens = [0, 1, 2, 3, 4, 5]
        for token in tokens:
            detokenizer.add_token(token)
        detokenizer.finalize()

        assert detokenizer.text == "Hello, how are you?"

    def test_english_numbers(self):
        """Test English with numbers."""
        vocab = {
            "▁The": 0,
            "▁price": 1,
            "▁is": 2,
            "▁$": 3,
            "19": 4,
            ".": 5,
            "99": 6,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        tokens = [0, 1, 2, 3, 4, 5, 6]
        for token in tokens:
            detokenizer.add_token(token)
        detokenizer.finalize()

        assert detokenizer.text == "The price is $19.99"

    def test_mixed_english_chinese(self):
        """Test mixed English and Chinese text."""
        vocab = {
            "▁Hello": 0,
            "▁": 1,
            "世": 2,
            "界": 3,
            "!": 4,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁Hello
        detokenizer.add_token(1)  # ▁
        detokenizer.add_token(2)  # 世
        detokenizer.add_token(3)  # 界
        detokenizer.add_token(4)  # !
        detokenizer.finalize()

        assert "Hello" in detokenizer.text
        assert "世界" in detokenizer.text

    def test_utf8_chinese_character_meng(self):
        """Test decoding of 夢 (E5 A4 A2) from byte tokens."""
        vocab = {
            "相": 0,
            "思": 1,
            "<0xE5>": 2,
            "<0xA4>": 3,
            "<0xA2>": 4,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        # Add tokens: 相思夢 (where 夢 is encoded as 3 bytes)
        detokenizer.add_token(0)  # 相
        detokenizer.add_token(1)  # 思
        detokenizer.add_token(2)  # <0xE5>
        detokenizer.add_token(3)  # <0xA4>
        detokenizer.add_token(4)  # <0xA2>
        detokenizer.finalize()

        assert "夢" in detokenizer.text
        assert detokenizer.text == "相思夢"

    def test_utf8_chinese_character_yan(self):
        """Test decoding of 硯 (E7 A1 AF) from byte tokens."""
        vocab = {
            "來": 0,
            "<0xE7>": 1,
            "<0xA1>": 2,
            "<0xAF>": 3,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        detokenizer.add_token(1)  # <0xE7>
        detokenizer.add_token(2)  # <0xA1>
        detokenizer.add_token(3)  # <0xAF>
        detokenizer.add_token(0)  # 來
        detokenizer.finalize()

        assert "硯" in detokenizer.text
        assert "來" in detokenizer.text

    def test_utf8_full_sentence(self):
        """Test full sentence: 相思那得夢硯來。"""
        vocab = {
            "相": 0,
            "思": 1,
            "那": 2,
            "得": 3,
            "<0xE5>": 4,
            "<0xA4>": 5,
            "<0xA2>": 6,  # E5 A4 A2 = 夢
            "<0xE7>": 7,
            "<0xA1>": 8,
            "<0xAF>": 9,  # E7 A1 AF = 硯
            "來": 10,
            "。": 11,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        # Token sequence for: 相思那得夢硯來。
        tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for token in tokens:
            detokenizer.add_token(token)
        detokenizer.finalize()

        assert detokenizer.text == "相思那得夢硯來。"

    def test_utf8_mixed_with_spm_markers(self):
        """Test UTF-8 bytes mixed with SPM word markers."""
        vocab = {
            "▁hello": 0,
            "<0xE5>": 1,
            "<0xA4>": 2,
            "<0xA2>": 3,
            "▁world": 4,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=True)

        detokenizer.add_token(0)  # ▁hello
        detokenizer.add_token(1)  # <0xE5>
        detokenizer.add_token(2)  # <0xA4>
        detokenizer.add_token(3)  # <0xA2>
        detokenizer.add_token(4)  # ▁world
        detokenizer.finalize()

        assert "hello" in detokenizer.text
        assert "夢" in detokenizer.text
        assert "world" in detokenizer.text

    def test_utf8_bytes_at_end(self):
        """Test that byte tokens at the end are properly flushed."""
        vocab = {
            "test": 0,
            "<0xE5>": 1,
            "<0xA4>": 2,
            "<0xA2>": 3,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        detokenizer.add_token(0)  # test
        detokenizer.add_token(1)  # <0xE5>
        detokenizer.add_token(2)  # <0xA4>
        detokenizer.add_token(3)  # <0xA2>
        detokenizer.finalize()

        assert "test" in detokenizer.text
        assert "夢" in detokenizer.text

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

    def test_utf8_emoji(self):
        """Test decoding of emoji (4-byte UTF-8)."""
        # 😀 = F0 9F 98 80
        vocab = {
            "hi": 0,
            "<0xF0>": 1,
            "<0x9F>": 2,
            "<0x98>": 3,
            "<0x80>": 4,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        detokenizer.add_token(0)  # hi
        detokenizer.add_token(1)  # <0xF0>
        detokenizer.add_token(2)  # <0x9F>
        detokenizer.add_token(3)  # <0x98>
        detokenizer.add_token(4)  # <0x80>
        detokenizer.finalize()

        assert "hi" in detokenizer.text
        assert "😀" in detokenizer.text

    def test_utf8_consecutive_multibyte_chars(self):
        """Test consecutive multi-byte characters."""
        # 日本 = E6 97 A5 (日) + E6 9C AC (本)
        vocab = {
            "<0xE6>": 0,
            "<0x97>": 1,
            "<0xA5>": 2,
            "<0x9C>": 3,
            "<0xAC>": 4,
        }
        tokenizer = MockTokenizer(vocab)
        detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=False)

        # 日 = E6 97 A5
        detokenizer.add_token(0)  # <0xE6>
        detokenizer.add_token(1)  # <0x97>
        detokenizer.add_token(2)  # <0xA5>
        # 本 = E6 9C AC
        detokenizer.add_token(0)  # <0xE6>
        detokenizer.add_token(3)  # <0x9C>
        detokenizer.add_token(4)  # <0xAC>
        detokenizer.finalize()

        assert detokenizer.text == "日本"


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

    def test_reset(self):
        tokenizer = MockBPETokenizer()
        detokenizer = BPEStreamingDetokenizer(tokenizer)

        detokenizer.add_token(0)
        detokenizer.finalize()
        detokenizer.reset()

        assert detokenizer.text == ""
        assert detokenizer._unflushed == ""

    def test_byte_decoder_singleton(self):
        """Test that byte decoder is created only once (class-level)."""
        tokenizer1 = MockBPETokenizer()
        tokenizer2 = MockBPETokenizer()

        det1 = BPEStreamingDetokenizer(tokenizer1)
        det2 = BPEStreamingDetokenizer(tokenizer2)

        # Should share the same byte decoder
        assert det1._byte_decoder is det2._byte_decoder


# ============================================================================
# Tests for TokenizerWrapper
# ============================================================================


class TestTokenizerWrapper:
    """Tests for TokenizerWrapper class."""

    def test_detokenizer_access(self):
        tokenizer = MockTokenizer()
        wrapper = TokenizerWrapper(tokenizer)

        assert wrapper.detokenizer is not None
        assert isinstance(wrapper.detokenizer, NaiveStreamingDetokenizer)

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
