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


def test_match_different_types():
    assert _match(1, "1") is False
    assert _match([], {}) is False


def test_invalid_spm_decoder():
    decoder = {"type": "ByteLevel"}
    assert _is_spm_decoder(decoder) is False


def test_valid_spm_decoder_no_space():
    decoder = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    assert _is_spm_decoder_no_space(decoder) is True


def test_valid_bpe_decoder():
    decoder = {"type": "ByteLevel"}
    assert _is_bpe_decoder(decoder) is True


@pytest.mark.parametrize(
    "factory, tokens",
    [(NaiveStreamingDetokenizer, [0, 1]), (SPMStreamingDetokenizer, [2, 3])],
)
def test_skip_special_tokens(factory, tokens):
    detokenizer = factory(MockTokenizer())
    detokenizer.add_token(tokens[0])
    detokenizer.add_token(tokens[1], skip_special_token_ids=[tokens[1]])
    detokenizer.finalize()
    assert "hello" in detokenizer.text
    assert "world" not in detokenizer.text


@pytest.mark.parametrize(
    "vocab, trim, expected",
    [
        (["▁caf", "<0xC3>", "<0xA9>", "▁is", "▁great"], True, "café is great"),
        (["test", "<0xFF>", "<0xFE>"], False, None),
    ],
    ids=["utf8", "invalid-utf8"],
)
def test_spm_byte_tokens(vocab, trim, expected):
    tokenizer = MockTokenizer(dict(zip(vocab, range(len(vocab)))))
    detokenizer = SPMStreamingDetokenizer(tokenizer, trim_space=trim)
    for token in range(len(vocab)):
        detokenizer.add_token(token)
    detokenizer.finalize()
    if expected is not None:
        assert detokenizer.text == expected
    else:
        assert "test" in detokenizer.text
        assert REPLACEMENT_CHAR in detokenizer.text


def test_last_segment_basic():
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


def test_initialization():
    tokenizer = MockBPETokenizer()
    detokenizer = BPEStreamingDetokenizer(tokenizer)

    assert detokenizer._byte_decoder is not None
    assert detokenizer.text == ""


@pytest.mark.parametrize("factory", [None, SPMStreamingDetokenizer])
def test_tokenizer_wrapper(factory):
    tokenizer = MockTokenizer()
    wrapper = (
        TokenizerWrapper(tokenizer, factory) if factory else TokenizerWrapper(tokenizer)
    )
    assert wrapper.vocab == tokenizer.vocab
    if factory:
        assert isinstance(wrapper.detokenizer, factory)


def test_not_implemented_methods():
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
