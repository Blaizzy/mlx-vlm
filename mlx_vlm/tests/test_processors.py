"""Tokenizers, processors, media loading, prompts, and tool parser contracts."""

from __future__ import annotations

import base64
import importlib
import json
import pkgutil
import re
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from functools import wraps
from io import BytesIO
from operator import attrgetter
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoProcessor, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from mlx_vlm.generate import GenerationResult, generate
from mlx_vlm.generate.video import processor_handles_video
from mlx_vlm.prompt_utils import (
    apply_chat_template,
    extract_text_from_content,
    get_chat_template,
)
from mlx_vlm.server.generation import GenerationArguments
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
    _ServerTokenStreamer,
    load_tokenizer,
    make_streaming_detokenizer,
)
from mlx_vlm.tools import (
    SPECS,
    _infer_tool_parser,
    _infer_tool_parser_from_processor,
    load_tool_module,
    parsers,
    process_tool_calls,
)
from mlx_vlm.utils import (
    StoppingCriteria,
    VideoMetadata,
    estimate_num_image_tokens,
    load_image,
    load_image_processor,
    load_processor,
    load_video,
    prepare_inputs,
    process_image,
    resolve_video_sampling,
    should_add_special_tokens,
)

# Tokenizers and streaming detokenizers


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


@pytest.mark.parametrize("server_stream", [False, True], ids=["cli", "server"])
@pytest.mark.parametrize("trim_space", [False, True], ids=["keep-space", "trim-space"])
@pytest.mark.parametrize(
    "pieces, expected_segments, final_segment",
    [
        (["Ġhello", "Ġworld"], [" hello", " world"], ""),
        (["hello", "\n\n", "world"], ["hello", "\n\n", "world"], ""),
        (["高", "ა", "🙂"], ["高", "ა", "🙂"], ""),
        (["Ġé\n高", "Ġdone"], ["Ġé\n高", " done"], ""),
        ([" hello", "\t", "world"], [" hello", "\t", "world"], ""),
        (["Ã", "©", "\n\n", "高"], ["", "é", "\n\n", "高"], ""),
        (["Ã", "高", "©", "Ġdone"], ["", "\ufffd高", "\ufffd", " done"], ""),
        (["Ã", "高", "©"], ["", "\ufffd高", "\ufffd"], ""),
        (["hello", "Ã"], ["hello", ""], "\ufffd"),
        (["ðŁ", "ĻĤ"], ["", "🙂"], ""),
        (["ðŁ"], [""], "\ufffd"),
        (["\ufffd"], ["\ufffd"], ""),
        (["Ġ", "Ġ", "hello"], [" ", " ", "hello"], ""),
    ],
    ids=[
        "byte-alphabet",
        "literal-newlines",
        "literal-unicode",
        "whole-token-fallback",
        "literal-whitespace",
        "split-utf8",
        "interrupted-utf8",
        "invalid-utf8-at-end",
        "incomplete-utf8-at-end",
        "split-emoji",
        "incomplete-emoji-at-end",
        "literal-replacement-character",
        "trim-only-first-space",
    ],
)
def test_bpe_streaming_matches_native_bytelevel(
    tmp_path, pieces, expected_segments, final_segment, trim_space, server_stream
):
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.decoder = ByteLevelDecoder()
    backend.add_tokens(list(dict.fromkeys(pieces)))
    backend.save(str(tmp_path / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, clean_up_tokenization_spaces=False
    )
    tokens = [backend.token_to_id(piece) for piece in pieces]
    assert tokenizer.decode(tokens) == "".join(expected_segments) + final_segment

    factory = load_tokenizer(tmp_path, return_tokenizer=False)
    assert factory is BPEStreamingDetokenizer
    detokenizer = factory(tokenizer, trim_space=trim_space)
    expected_segments = expected_segments.copy()
    if trim_space:
        for i, segment in enumerate(expected_segments):
            if segment:
                expected_segments[i] = segment.removeprefix(" ")
                break
    expected = "".join(expected_segments) + final_segment

    # Resetting an existing detokenizer must preserve its normalized token map.
    for _ in range(2):
        detokenizer.reset()
        segments = []
        if server_stream:
            streamer = _ServerTokenStreamer(tokenizer, detokenizer)
            for token in tokens:
                segments.append(streamer.advance(token, finish_reason=None))
            assert segments == expected_segments
            assert streamer.finalize() == final_segment
            assert streamer.finalize() == ""
        else:
            for token in tokens:
                detokenizer.add_token(token)
                segments.append(detokenizer.last_segment)
                assert detokenizer.last_segment == ""
            assert segments == expected_segments
            detokenizer.finalize()
            assert detokenizer.last_segment == final_segment
            detokenizer.finalize()
            assert detokenizer.last_segment == ""
            assert detokenizer.text == expected
        assert "".join(segments) + final_segment == expected


@pytest.mark.parametrize("server_stream", [False, True], ids=["cli", "server"])
@pytest.mark.parametrize("normalized", [False, True], ids=["literal", "normalized"])
@pytest.mark.parametrize(
    "piece, canonical",
    [("ŉ", "ʼn"), ("ำ", "ํา"), ("ﭏ", "אל"), ("ﬃ", "ffi"), ("ͺ", " ͅ")],
    ids=["apostrophe-n", "thai", "hebrew", "ligature", "leading-space"],
)
def test_bpe_streaming_uses_canonical_added_tokens(
    tmp_path, piece, canonical, normalized, server_stream
):
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.normalizer = NFKC()
    backend.decoder = ByteLevelDecoder()
    backend.add_tokens([AddedToken(piece, normalized=normalized)])
    backend.save(str(tmp_path / "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, clean_up_tokenization_spaces=False
    )
    token = tokenizer.vocab[piece]
    expected = tokenizer.convert_ids_to_tokens(token)
    assert expected in {piece, canonical}
    assert tokenizer.decode([token]) == expected

    detokenizer = load_tokenizer(tmp_path, return_tokenizer=False)(tokenizer)
    if server_stream:
        streamer = _ServerTokenStreamer(tokenizer, detokenizer)
        assert streamer.advance(token, finish_reason=None) == expected
        assert streamer.finalize() == ""
    else:
        detokenizer.add_token(token)
        assert detokenizer.last_segment == expected
        detokenizer.finalize()
        assert detokenizer.last_segment == ""
        assert detokenizer.text == expected


@pytest.mark.parametrize("server_stream", [False, True], ids=["cli", "server"])
@pytest.mark.parametrize("stride", [1, 2], ids=["dense-vocab", "sparse-vocab"])
@pytest.mark.parametrize(
    "pieces, expected",
    [
        ([None], ""),
        ([None, "Ġhello"], "hello"),
        (["Ã", None, "©"], "é"),
        (["Ã", None], "\ufffd"),
        (["z", None, "z"], "zz"),
    ],
    ids=["only-unmapped", "leading-space", "split-utf8", "truncated-utf8", "ascii"],
)
def test_bpe_streaming_skips_unmapped_ids(stride, pieces, expected, server_stream):
    vocab = {piece: i * stride for i, piece in enumerate(["Ã", "©", "Ġhello", "z"])}
    backend = Tokenizer(WordLevel(vocab))
    backend.decoder = ByteLevelDecoder()
    tokenizer = NS(
        vocab=backend.get_vocab(),
        convert_ids_to_tokens=backend.id_to_token,
        decode=backend.decode,
    )
    unmapped = [
        i for i in range(max(vocab.values()) + 2) if backend.id_to_token(i) is None
    ]
    unmapped.append(2**32 - 1)
    tokens = [
        token
        for piece in pieces
        for token in (unmapped if piece is None else [vocab[piece]])
    ]
    assert tokenizer.decode(tokens).removeprefix(" ") == expected

    detokenizer = BPEStreamingDetokenizer(tokenizer, trim_space=True)
    streamer = _ServerTokenStreamer(tokenizer, detokenizer) if server_stream else None
    segments = []
    for token in tokens:
        if server_stream:
            segment = streamer.advance(token, finish_reason=None)
        else:
            detokenizer.add_token(token)
            segment = detokenizer.last_segment
        if token in unmapped:
            assert segment == ""
        segments.append(segment)
    if server_stream:
        segments.append(streamer.finalize())
    else:
        detokenizer.finalize()
        segments.append(detokenizer.last_segment)
        assert detokenizer.text == expected
    assert "".join(segments) == expected


@pytest.mark.parametrize("server_stream", [False, True], ids=["cli", "server"])
def test_bpe_streaming_empty_vocab(server_stream):
    backend = Tokenizer(WordLevel({}))
    tokenizer = NS(vocab=backend.get_vocab(), decode=backend.decode)
    assert tokenizer.decode([0]) == ""
    detokenizer = BPEStreamingDetokenizer(tokenizer)
    if server_stream:
        streamer = _ServerTokenStreamer(tokenizer, detokenizer)
        assert streamer.advance(0, finish_reason=None) == ""
        assert streamer.finalize() == ""
    else:
        detokenizer.add_token(0)
        detokenizer.finalize()
        assert detokenizer.text == ""


def test_bpe_streaming_copies_isolate_partial_utf8_and_skip_tokens():
    source = BPEStreamingDetokenizer(
        MockTokenizer({"Ã": 0, "©": 1, "高": 2, "<skip>": 3})
    )
    source.add_token(0)
    processor = NS(detokenizer=source)
    left = make_streaming_detokenizer(processor)
    right = make_streaming_detokenizer(processor)
    left.add_token(0)
    assert left.last_segment == ""
    right.add_token(2)
    assert right.last_segment == "高"
    left.add_token(3, skip_special_token_ids=[3])
    left.add_token(1)
    assert left.last_segment == "é"
    source.add_token(1)
    assert source.last_segment == "é"


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


TOKENIZER_PROCESSORS = {
    "step3p7": ("step3p7.processing_step3p7", "Step3VLProcessor"),
    "laguna": ("laguna.processing_laguna", "LagunaProcessor"),
    "molmo_point": ("molmo_point.processing_molmo_point", "MolmoPointProcessor"),
    "kimi_k3": ("kimi_k3.processing_kimi_k3", "KimiK3Processor"),
}


def _processor_module(name):
    return importlib.import_module("mlx_vlm.models." + TOKENIZER_PROCESSORS[name][0])


class RecordingTokenizer(PreTrainedTokenizerFast):
    def __call__(self, text, **kwargs):
        self.last_text = text
        return super().__call__(text, **kwargs)


class KimiTokenizer(PreTrainedTokenizerBase):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__()
        self.encode_calls = []

    def convert_tokens_to_ids(self, token):
        return 0

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return [1, 2, 3]

    def apply_chat_template(
        self, conversation, tokenize=False, add_generation_prompt=True
    ):
        return "rendered"


def fast_tokenizer(
    tokens=("[UNK]", "[PAD]", "hello"),
    *,
    tokenizer_class=PreTrainedTokenizerFast,
    split=True,
    **kwargs,
):
    backend = Tokenizer(
        WordLevel({t: i for i, t in enumerate(tokens)}, unk_token=tokens[0])
    )
    if split:
        backend.pre_tokenizer = Whitespace()
    return tokenizer_class(
        tokenizer_object=backend, unk_token=tokens[0], pad_token=tokens[1], **kwargs
    )


class ProcessorTokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    bos_token, eos_token, pad_token = "<bos>", "<eos>", "<pad>"
    pad_token_id = unk_token_id = 0
    image_token, image_token_id = "<image>", 100
    audio_token, audio_token_id = "<audio>", 101
    video_token, video_token_id = "<video>", 102
    boi_token, eoi_token = "<boi>", "<eoi>"
    boa_token, eoa_token = "<boa>", "<eoa>"

    def __init__(self, tokens=None, skip_spaces=False, pad=True, **attrs):
        self.init_kwargs = {}
        self.__dict__.update(attrs)
        self.special_tokens = None if tokens is None else dict(tokens)
        self.skip_spaces = skip_spaces
        self.pad = pad

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        return (self.special_tokens or {}).get(token, self.unk_token_id)

    def encode(self, text, **kwargs):
        if self.special_tokens is None:
            return list(range(10))
        pattern = "|".join(
            re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
        )
        chunks = re.findall((pattern + "|" if pattern else "") + r"[\s\S]", text)
        return [
            self.special_tokens.get(t, 1)
            for t in chunks
            if t in self.special_tokens or not self.skip_spaces or not t.isspace()
        ]

    def __call__(self, text, text_pair=None, return_token_type_ids=False, **kwargs):
        self.last_text, self.last_kwargs = (text, kwargs)
        rows = [self.encode(t) for t in ([text] if isinstance(text, str) else text)]
        width = max(map(len, rows)) if self.pad else 0
        result = {
            "input_ids": [
                row + [self.pad_token_id] * (width - len(row)) for row in rows
            ],
            "attention_mask": [
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ],
        }
        if return_token_type_ids:
            result["token_type_ids"] = [[0] * width for row in rows]
        return result

    def add_special_tokens(self, tokens):
        pass

    def build_inputs_with_special_tokens(self, ids):
        return ids


class GemmaTokenizer(ProcessorTokenizer):
    def __init__(self):
        super().__init__(
            {"<|image|>": 100, "<|audio|>": 101, "<|video|>": 102},
            image_token="<|image|>",
            audio_token="<|audio|>",
            video_token="<|video|>",
            chat_template="mock",
        )

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        parts = ["<bos>"]
        for message in messages:
            for item in message["content"]:
                parts.append(
                    item["text"]
                    if item["type"] == "text"
                    else getattr(self, item["type"] + "_token")
                )
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "".join(parts)
        return self(rendered) if tokenize else rendered


class TinyDiffusionGemma4Tokenizer(ProcessorTokenizer):
    TOOL_ATTRS = ("stc_token", "etc_token", "escape_token", "soc_token", "eoc_token")

    def __init__(self, profile):
        super().__init__(**profile)
        self.additional_special_tokens = []
        for key in self.TOOL_ATTRS:
            setattr(self, key, None)

    @property
    def all_special_ids(self):
        tokens = self.additional_special_tokens + [
            getattr(self, key)
            for key in self.TOOL_ATTRS
            if getattr(self, key) is not None
        ]
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def add_special_tokens(self, tokens):
        for token in tokens.get("additional_special_tokens", []):
            self.special_tokens[token] = self.video_token_id


@pytest.mark.parametrize("name", ["step3p7", "laguna", "molmo_point"])
def test_tokenizer_loader_options(name):
    module = _processor_module(name)
    cls = getattr(module, TOKENIZER_PROCESSORS[name][1])
    tok = ProcessorTokenizer(
        chat_template="template",
        vocab={"Got": 0, "Ġit": 1},
        backend_tokenizer=NS(decoder="bad"),
    )
    target = "transformers.AutoTokenizer.from_pretrained"
    kwargs = {"trust_remote_code": name != "molmo_point"}
    if name == "laguna":
        target = "transformers.PreTrainedTokenizerFast.from_pretrained"
        tok = fast_tokenizer(
            ("<unk>", "<pad>", "<eos>", "prompt"),
            eos_token="<eos>",
            chat_template="template",
        )
        kwargs.update(
            processor_kwargs={"local_files_only": True}, quantize_activations=True
        )
    with (
        patch(target, return_value=tok) as loader,
        (
            patch.object(module, "load_chat_template")
            if name == "molmo_point"
            else nullcontext()
        ),
        patch.object(
            cls, "check_argument_for_proper_class", return_value=None, create=True
        ),
    ):
        p = cls.from_pretrained("/tmp/model", **kwargs)
    assert p.tokenizer is tok
    expected = {"trust_remote_code": name != "molmo_point"}
    if name != "molmo_point":
        expected["fix_mistral_regex"] = True
    if name == "laguna":
        expected["local_files_only"] = True
    if name == "molmo_point":
        expected["padding_side"] = "left"
    loader.assert_called_once_with("/tmp/model", **expected)
    if name == "step3p7":
        assert p.detokenizer_class is BPEStreamingDetokenizer
        assert "ByteLevel" in repr(tok.backend_tokenizer.decoder)
        p.detokenizer = object()
        for token in (0, 1):
            p.detokenizer.add_token(token)
        p.detokenizer.finalize()
        assert p.detokenizer.text == "Got it"
    if name == "laguna":
        assert not should_add_special_tokens("laguna", p)
        assert should_add_special_tokens("llama", p)


class TestKimiTokenizer:
    @pytest.fixture
    def module(self):
        return _processor_module("kimi_k3")

    def test_fast_tokenizer_round_trip(self, module, tmp_path):
        module.KimiK3Processor(tokenizer=fast_tokenizer()).save_pretrained(tmp_path)
        assert (tmp_path / "tokenizer.json").is_file()
        with patch.object(module, "_convert_kimi_k3_tiktoken") as convert:
            loaded = module.KimiK3Processor.from_pretrained(tmp_path)
        convert.assert_not_called()
        assert loaded.tokenizer.encode("hello", add_special_tokens=False) == [2]

    @pytest.mark.parametrize(
        "remote", [False, True], ids=["local-tiktoken", "remote-fast"]
    )
    def test_tokenizer_loading(self, module, tmp_path, remote):
        tokenizer = KimiTokenizer()
        configs = dict(
            tokenizer={},
            tokenizer_config={"added_tokens_decoder": {}},
            preprocessor_config=(
                {"media_proc_cfg": {"patch_size": 18}} if remote else {}
            ),
        )
        for name, config in configs.items():
            (tmp_path / f"{name}.json").write_text(json.dumps(config))
        vocab = tmp_path / "tiktoken.model"
        vocab.write_text("tiktoken ranks")
        if not remote:
            (tmp_path / "tokenizer.json").unlink()

        def download(repo_id, filename, **kwargs):
            assert repo_id == "moonshotai/Kimi-K3"
            assert kwargs["revision"] == "model-revision"
            assert filename in ("tokenizer.json", "preprocessor_config.json")
            return tmp_path / filename

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=download),
            patch.object(
                PreTrainedTokenizerFast, "from_pretrained", return_value=tokenizer
            ) as fast,
            patch.object(
                module,
                "_convert_kimi_k3_tiktoken",
                return_value=tokenizer,
            ) as convert,
        ):
            source = "moonshotai/Kimi-K3" if remote else tmp_path
            kwargs = {"revision": "model-revision"} if remote else {}
            loaded = module.KimiK3Processor.from_pretrained(source, **kwargs)
        assert isinstance(loaded, module.KimiK3Processor)
        if remote:
            assert fast.call_args.args[0] == "moonshotai/Kimi-K3"
            assert fast.call_args.kwargs["revision"] == "model-revision"
            assert not fast.call_args.kwargs["trust_remote_code"]
            assert loaded.image_processor.patch_size == 18
            convert.assert_not_called()
        else:
            convert.assert_called_once_with(vocab, tmp_path / "tokenizer_config.json")
            fast.assert_not_called()

    def test_conversion_requires_tiktoken(self, module):
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(ImportError, match="Install `tiktoken`.*tokenizer.json"),
        ):
            module._convert_kimi_k3_tiktoken("tiktoken.model", "tokenizer_config.json")

    def test_restores_all_control_slots(self, module):
        supplied = {100: "[BOS]", 103: "<|open|>", 355: "[PAD]"}
        config = {
            "added_tokens_decoder": {
                str(i): {"content": t} for i, t in supplied.items()
            }
        }
        tokens = module._kimi_k3_control_tokens(config, base_vocab_size=100)
        assert len(tokens) == 256 and tokens[1] == "<|reserved_token_101|>"
        assert {i: tokens[i - 100] for i in supplied} == supplied


def test_demotes_tool_parser_tokens():
    from pathlib import Path

    profiles = json.loads(Path(__file__).with_name("processor_cases.json").read_text())[
        "profiles"
    ]
    module = importlib.import_module(
        "mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma"
    )
    base = importlib.import_module("mlx_vlm.models.gemma4.processing_gemma4")
    tok = TinyDiffusionGemma4Tokenizer(profiles["diffusion_tokenizer"])
    tok.special_tokens.update(
        {t: 70 + i for i, t in enumerate(module._TOOL_PARSER_TOKENS)}
    )
    tok.special_tokens["<extra_special>"] = 90
    tokens = ("<|tool_call>", "<tool_call|>", '<|"|>', "<|channel>", "<channel|>")
    for attr, token in zip(tok.TOOL_ATTRS, tokens):
        setattr(tok, attr, token)
    tok.additional_special_tokens = ["<extra_special>"]
    with patch.object(
        base.Gemma4Processor, "from_pretrained", return_value=NS(tokenizer=tok)
    ):
        p = module.DiffusionGemma4Processor.from_pretrained("demo")
    assert p.tokenizer is tok
    assert tok.additional_special_tokens == [
        "<extra_special>"
    ] and tok.all_special_ids == [90]
    assert all(getattr(tok, attr) is None for attr in tok.TOOL_ATTRS)
    assert all(
        tok.convert_tokens_to_ids(t) != tok.unk_token_id
        for t in module._TOOL_PARSER_TOKENS
    )


# Processor contracts

equal = np.testing.assert_array_equal
DATA = json.loads(Path(__file__).with_name("processor_cases.json").read_text())
PROFILES = DATA["profiles"]
# Importing the registry also installs each family's AutoProcessor routing patch.
m, c = NS(), NS()
for name, (path, cls) in DATA["modules"].items():
    if "." not in path:
        path += f".processing_{path}"
    module = importlib.import_module("mlx_vlm.models." + path)
    setattr(m, name, module)
    if cls:
        setattr(c, name, getattr(module, cls))


def _ignore_types(cls):
    return patch.object(
        cls, "check_argument_for_proper_class", return_value=None, create=True
    )


def _assert_shapes(data, **shapes):
    assert {key: data[key].shape for key in shapes} == shapes


def _message(*items):
    return [{"role": "user", "content": list(items)}]


def _gemma_image(cls=m.g4u.Gemma4UnifiedImageProcessor, max_soft_tokens=4):
    return cls(**PROFILES["gemma_image"], max_soft_tokens=max_soft_tokens)


def _lfm_processor(image_processor=None):
    return NS(
        image_processor=image_processor or m.lfm.Lfm2VlNumpyImageProcessor(),
        tokenizer=ProcessorTokenizer(),
        **PROFILES["lfm_tokens"],
        _merge_kwargs=lambda *a, **kw: {"text_kwargs": {}, "images_kwargs": {}},
    )


def _write_configs(path, **configs):
    for name, config in configs.items():
        (Path(path) / f"{name}.json").write_text(json.dumps(config))


def _assert_attrs(obj, **expected):
    assert {key: attrgetter(key)(obj) for key in expected} == expected


@pytest.mark.parametrize("model_type,module_path,class_name", DATA["routes"])
def test_auto_processor_routes_to_custom_loader(
    tmp_path, model_type, module_path, class_name
):
    cls = getattr(importlib.import_module("mlx_vlm.models." + module_path), class_name)
    _write_configs(tmp_path, config={"model_type": model_type})
    sentinel = object()
    with patch.object(cls, "from_pretrained", return_value=sentinel) as loader:
        assert (
            AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False) is sentinel
        )
        loader.assert_called_once_with(tmp_path, trust_remote_code=False)
        loader.reset_mock()
        assert AutoProcessor.from_pretrained(tmp_path) is sentinel
        loader.assert_called_once_with(tmp_path, trust_remote_code=True)


def test_flint_tokenizer_prompt_and_stop_tokens(tmp_path):
    importlib.import_module("mlx_vlm.models.flint")
    raw = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "hello": 1,
                "<|end_of_text|>": 2,
                "<|user|>": 3,
                "<|assistant|>": 4,
            },
            unk_token="[UNK]",
        )
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        eos_token="<|end_of_text|>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
    )
    tokenizer.chat_template = (
        "{% for m in messages %}<|{{ m.role }}|>"
        "{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}"
        "{{ m.content }}{% for tc in m.tool_calls %}<tool_call>{{ tc.function.name }}"
        "{% for k, v in tc.function.arguments.items() %}"
        "<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value>{% endfor %}"
        "</tool_call>{% endfor %}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path)
    _write_configs(
        tmp_path,
        config={
            "model_type": "flint",
            "auto_map": {"AutoConfig": "must_not_execute.FlintConfig"},
        },
    )
    processor = load_processor(tmp_path, eos_token_ids=[2, 4], trust_remote_code=False)
    assert isinstance(processor, PreTrainedTokenizerBase)
    assert processor.backend_tokenizer.to_str() == tokenizer.backend_tokenizer.to_str()
    prompt = apply_chat_template(processor, {"model_type": "flint"}, "hello")
    assert prompt == "<|user|>hello<|assistant|>"
    assert processor.encode(prompt) == [3, 1, 4]
    assert processor.stopping_criteria(2)
    assert processor.stopping_criteria(4)
    assert not processor.stopping_criteria(1)
    assert _infer_tool_parser_from_processor(processor) == "glm47"
    previous_answer = dict(
        role="assistant", content="hello", reasoning_content="Remember this."
    )
    previous_call = dict(
        role="assistant",
        content=None,
        reasoning_content="Look it up.",
        tool_calls=[dict(function=dict(name="lookup", arguments='{"query": "hello"}'))],
    )
    for messages in (previous_call, [previous_answer, previous_call]):
        original = deepcopy(messages)
        rendered = apply_chat_template(processor, {"model_type": "flint"}, messages)
        assert "<think>Look it up.</think><tool_call>lookup" in rendered
        assert "<arg_key>query</arg_key><arg_value>hello</arg_value>" in rendered
        if isinstance(messages, list):
            assert "<think>Remember this.</think>hello" in rendered
        assert messages == original


def test_qwen3_5_moe_text_stale_vl_processor_loads_tokenizer(tmp_path):
    from transformers import Qwen2Tokenizer

    importlib.import_module("mlx_vlm.models.qwen3_5_moe_text")
    _write_configs(tmp_path, config={"model_type": "qwen3_5_moe_text"})

    Qwen2Tokenizer(
        vocab={"<|endoftext|>": 0, "t": 1, "3": 2, "Ġ": 3, "4": 4}, merges=[]
    ).save_pretrained(tmp_path)
    tokenizer_config = tmp_path / "tokenizer_config.json"
    data = json.loads(tokenizer_config.read_text())
    data["processor_class"] = "Qwen3VLProcessor"
    tokenizer_config.write_text(json.dumps(data))

    processor = load_processor(tmp_path, eos_token_ids=[0])

    assert not hasattr(processor, "image_processor")
    assert processor.encode("t3 t4", add_special_tokens=False) == [1, 2, 3, 1, 4]


class _ImageStub:
    model_input_names = ["pixel_values"]
    merge_size, max_image_tiles = 2, 4
    do_image_splitting = False

    def __init__(self, outputs=None, **attrs):
        self.__dict__.update(attrs)
        self.outputs = outputs or {
            "pixel_values": np.zeros((1, 3, 224, 224), np.float32)
        }

    def __call__(self, images=None, **kwargs):
        self.last_kwargs = kwargs
        return self.outputs

    def fetch_images(self, images):
        return images if isinstance(images, list) else [images]


def _mock_ip(**extra):
    return _ImageStub({"pixel_values": np.zeros((1, 3, 224, 224), np.float32), **extra})


def _stub(name):
    case = deepcopy(DATA["stubs"][name])
    outputs = {
        k: np.zeros(shape, np.float32) for k, shape in case.get("shapes", {}).items()
    }
    outputs.update(
        {k: np.array(v, np.int64) for k, v in case.get("arrays", {}).items()}
    )
    return _ImageStub({**outputs, **case.get("values", {})}, **case.get("attrs", {}))


def _make_image(height=224, width=224):
    return Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )


def _assert_all_mx(result, *media):
    assert {"input_ids", "attention_mask", *media} <= result.keys()
    for key, value in result.items():
        if value is not None:
            assert isinstance(value, mx.array), f"{key}: {type(value).__name__}"


SMOKE_PROCESSORS = DATA["smoke"]


def _make_processor(name):
    case = SMOKE_PROCESSORS[name]
    defaults = DATA["smoke_defaults"].get(case.get("family"), {})
    case = deepcopy({**defaults, **case})
    module = importlib.import_module(f"mlx_vlm.models.{name}.processing_{name}")
    cls = getattr(module, case["class"])
    ip = _stub(case["stub"]) if "stub" in case else _mock_ip(**case.get("outputs", {}))
    tok = ProcessorTokenizer(**case.get("tokenizer", {}))
    kwargs = dict(case.get("kwargs", {}))
    if name == "paligemma":
        ip.image_seq_length, tok.add_tokens = 4, lambda *a, **kw: None
    if name.startswith("qwen") or name == "ernie4_5_moe_vl":
        ip = _mock_ip(image_grid_thw=np.array([[1, 16, 16]], np.int64))
    if name == "qwen3_omni_moe":
        kwargs.update(
            video_processor=NS(model_input_names=[], merge_size=2),
            feature_extractor=NS(model_input_names=[]),
        )
    with _ignore_types(cls):
        return cls(image_processor=ip, tokenizer=tok, **kwargs)


@pytest.mark.parametrize(
    "name,with_image",
    [
        (name, mode == "image")
        for name, case in SMOKE_PROCESSORS.items()
        for mode in case.get("modes", ["image", "text"])
    ],
)
def test_processor_mlx_outputs(name, with_image):
    p = _make_processor(name)
    kwargs = dict(text=["Hello world"])
    if with_image:
        token = {
            "gemma3": "<boi>",
            "ernie4_5_moe_vl": "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>",
        }.get(name, getattr(p, "image_token", "<image>"))
        images = [_make_image()]
        if name in ("smolvlm", "mllama", "idefics2", "idefics3", "pixtral", "mistral3"):
            images = [images]
        kwargs.update(text=[token + " Describe"], images=images)
        if name == "paligemma":
            kwargs["text"] = "describe"
    if name == "qwen3_omni_moe":
        kwargs["text"] = kwargs["text"][0]
    result = p(**kwargs)
    _assert_all_mx(result, *(["pixel_values"] if with_image else []))
    if name == "mllama" and with_image:
        assert "cross_attention_mask" in result


def test_gemma3n_batches_images_and_audio():
    processor = _make_processor("gemma3n")
    processor.feature_extractor = Mock(return_value={"input_features": [[0.0]] * 2})
    images = [_make_image(), _make_image()]

    result = processor(
        text=["<image><audio>one", "<image><audio>two"],
        images=images,
        audio=[[0.0], [0.0]],
        padding=True,
    )

    assert result["input_ids"].shape[0] == 2
    assert processor.tokenizer.last_kwargs["padding"] is True


def test_nemotron_parse_packed_page_geometry_and_patch_order():
    image = Image.fromarray(np.arange(17 * 65 * 3, dtype=np.uint8).reshape(17, 65, 3))
    inputs = m.nemotron_parse_v3_5.ImageProcessor()(image)
    assert inputs["image_sizes"].tolist() == [[32, 128]]
    assert inputs["content_sizes"].tolist() == [[17, 65]]
    assert inputs["pixel_values"].shape == (16, 768)
    expected = (
        np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 255
        - np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    ) / np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    equal(inputs["pixel_values"][0], expected[:, :16, :16].reshape(-1))


def test_unlimited_ocr_default_chat_template_omits_trailing_space():
    Template = pytest.importorskip("jinja2").Template
    p = object.__new__(c.ocr)
    rendered = Template(p.default_chat_template).render(
        messages=DATA["ocr_messages"], add_generation_prompt=True
    )
    assert rendered == "<image>document parsing. partial continue"


@pytest.mark.parametrize("name,start,end", DATA["cleanup"])
def test_output_control_tokens(name, start, end):
    cls = (
        m.aya_vision.AyaVisionOutputProcessor
        if name == "aya_output"
        else getattr(c, name)
    )
    if name == "aya_output":
        native = importlib.import_module(
            "transformers.models.aya_vision.processing_aya_vision"
        )
        assert issubclass(cls, native.AyaVisionProcessor)
    text = "Title: A calm river cruise\nKeywords: boat, river"
    assert cls.__new__(cls).clean_output(start + text + end) == text


@pytest.mark.parametrize("name,token,token_id", DATA["eos_tokens"])
def test_additional_eos_tokens(name, token, token_id):
    cls = getattr(c, name)
    p = _make_processor(name) if name == "idefics3" else cls.__new__(cls)
    p.tokenizer = ProcessorTokenizer({token: token_id})
    assert p.additional_eos_token_ids == [token_id]


class TestGemma4UnifiedProcessor:
    def _processor(self, video=False):
        video_cls = m.g4u.Gemma4UnifiedVideoProcessor
        video_processor = _gemma_image(video_cls, 70) if video else None
        p = c.g4u(
            image_processor=_gemma_image(),
            tokenizer=GemmaTokenizer(),
            video_processor=video_processor,
            image_seq_length=4,
        )
        assert "video_processor" in p.get_attributes()
        assert isinstance(p.video_processor, m.g4u.Gemma4UnifiedVideoProcessor)
        return p

    def test_merged_image_patches_and_positions(self):
        data, soft_tokens = _gemma_image()(Image.new("RGB", (8, 8)))
        _assert_shapes(data, pixel_values=(1, 4, 48), image_position_ids=(1, 4, 2))
        assert soft_tokens == [4]
        assert data["image_position_ids"][0].tolist() == [
            [x, y] for y in range(2) for x in range(2)
        ]

    @pytest.mark.parametrize("height", [1, 3], ids=["1px-tall", "3px-tall"])
    def test_short_pil_images_stay_channels_last(self, height):
        pixels = np.arange(height * 4 * 3, dtype=np.uint8).reshape(height, 4, 3)
        p = m.g4.Gemma4ImageProcessor(do_resize=False, do_rescale=False)
        data, _ = p(Image.fromarray(pixels))
        equal(data["pixel_values"][0], pixels.transpose(2, 0, 1))

        p = _gemma_image()
        p.do_resize = True
        data, _ = p(Image.new("RGB", (900, height)))
        assert data["image_position_ids"][0].max(axis=0).tolist() == [3, 0]

    def test_video_padding_and_positions(self):
        video = np.zeros((2, 3, 4, 8), np.uint8)
        p = _gemma_image(m.g4.Gemma4VideoProcessor, max_soft_tokens=70)
        assert p.video_sampling_defaults() == {"max_frames": p.num_frames}
        data = p([video], fps=[1.0])
        _assert_shapes(
            data, pixel_values_videos=(1, 2, 280, 12), video_position_ids=(1, 2, 280, 2)
        )
        assert data["num_frames_per_video"] == data["num_soft_tokens_per_frame"] == [2]
        assert data["frame_timestamps"] == [[0.0, 1.0]]
        positions = data["video_position_ids"][0, 0]
        assert positions[:8].tolist() == [[x, y] for y in range(2) for x in range(4)]
        assert np.all(positions[8:] == -1)

    def test_audio_chunk_masks(self):
        extractor = m.g4u.Gemma4UnifiedAudioFeatureExtractor(
            audio_samples_per_token=4, feature_size=4
        )
        result = extractor([np.arange(n, dtype=np.float32) for n in (6, 9)])
        assert result["input_features"].shape == (2, 3, 4)
        assert result["input_features_mask"].tolist() == [
            [True, True, False],
            [True] * 3,
        ]

    @pytest.mark.parametrize("video", [False, True], ids=["image-chat", "video-call"])
    def test_multimodal_inputs(self, video):
        p = self._processor(video)
        if video:
            result = p(
                text=[p.video_token + "describe"],
                videos=[np.zeros((2, 3, 4, 8), np.uint8)],
                fps=[1.0],
            )
        else:
            messages = _message(
                {"type": "image", "image": Image.new("RGB", (8, 8))},
                {"type": "text", "text": "Describe this image in detail."},
            )
            result = p.apply_chat_template(messages, **PROFILES["chat_kwargs"])
            assert isinstance(result["input_ids"], mx.array)
        case = DATA["gemma_multimodal"][video]
        _assert_shapes(result, **{k: tuple(v) for k, v in case["shapes"].items()})
        assert isinstance(result[case["pixels"]], mx.array)
        assert case["marker"] in p.tokenizer.last_text[0]
        assert mx.sum(result["mm_token_type_ids"] == (2 if video else 1)).item() == 4

    def test_video_placeholder_without_tokenizing(self):
        rendered = self._processor().apply_chat_template(
            _message(
                {"type": "video", "video": "clip.mp4"},
                {"type": "text", "text": "Describe this video."},
            ),
            tokenize=False,
            enable_thinking=False,
        )
        assert "<|video|>" in rendered


class TestLlavaOnevisionProcessor:
    def test_video_expansion(self):
        p = _make_processor("llava_onevision")
        video = np.zeros((4, 3, 384, 384), np.float32)
        expanded = p._expand_placeholders(
            "<video> Describe", iter(()), iter([video]), (384, 384)
        )
        assert expanded.count("<video>") == 4 * 14 * 14 + 1
        with pytest.raises(ValueError):
            p._expand_placeholders(
                "<image> <image>", iter([[384, 384]]), iter(()), (384, 384)
            )

    @pytest.mark.parametrize(
        "channels_first", [False, True], ids=["hwc-white", "chw-red"]
    )
    def test_video_normalization(self, channels_first):
        frames = np.full((2, 40, 60, 3), 255, np.uint8)
        if channels_first:
            frames = np.zeros((2, 3, 40, 60), np.uint8)
            frames[:, 0] = 255
        pixels = _make_processor("llava_onevision").video_processor([frames])
        assert pixels.shape == (1, 2, 3, 384, 384)
        if channels_first:
            assert np.allclose(pixels[:, :, 0], 1.0, atol=1e-5)
            assert np.allclose(pixels[:, :, 1], -1.0, atol=1e-5)
        else:
            assert np.allclose(pixels, 1.0, atol=1e-5)


def test_pali_gemma_tokenizer_kwargs_do_not_leak_into_image_processor():
    p = _make_processor("paligemma")
    p.image_processor.valid_kwargs = type(
        "ImageKwargs", (), {"__annotations__": {"do_resize": bool}}
    )
    result = p(
        text="describe",
        images=[_make_image()],
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        do_resize=False,
    )
    _assert_all_mx(result)
    assert p.image_processor.last_kwargs == {"do_resize": False}
    assert p.tokenizer.last_kwargs == dict(
        padding=True, padding_side="left", add_special_tokens=False
    )


def test_minicpmv_video_marker_expands_to_frame_bounds():
    tokenizer = ProcessorTokenizer(**PROFILES["minicpm_tokenizer"])
    options = dict(**PROFILES["minicpm_image"])
    with _ignore_types(c.minicpm):
        p = c.minicpm(
            tokenizer=tokenizer,
            image_processor=m.minicpm.MiniCPMVImageProcessor(**options),
            video_processor=m.minicpm.MiniCPMVVideoProcessor(**options),
        )
    result = p(
        text=["<|video_pad|> Describe this."],
        videos=[np.zeros((2, 3, 16, 16), np.uint8)],
        slice_mode=False,
        max_num_frames=2,
        padding=False,
    )
    vp = p.video_processor
    assert vp.video_sampling_defaults() == {"max_frames": vp.max_num_frames}
    assert len(result["pixel_values"][0]) == 2
    assert result["tgt_sizes"][0].shape == result["image_bound"][0].shape == (2, 2)
    assert result["num_frames_per_video"] == [[2]]
    assert result["num_patches_per_frame"] == [[1, 1]]
    for start, end in result["image_bound"][0]:
        assert np.all(result["input_ids"][0, start:end] == 102)


class TestGlmOcrProcessor:
    GEOMETRY = dict(patch_size=14, temporal_patch_size=2, merge_size=2)

    @pytest.mark.parametrize("reference", [False, True], ids=["shape", "torch-parity"])
    def test_image_patches(self, reference):
        p = m.glm_ocr.Glm46VImageProcessor(
            **self.GEOMETRY, min_pixels=784, max_pixels=50176
        )
        image = Image.new("RGB", (56, 28))
        if reference:
            try:
                import torch
                from transformers.models.glm46v.image_processing_glm46v import (
                    Glm46VImageProcessor as HFProcessor,
                )
            except Exception as exc:
                pytest.skip(f"Transformers torch image backend unavailable: {exc}")
            expected = HFProcessor(
                **self.GEOMETRY, size={"shortest_edge": 784, "longest_edge": 50176}
            )(images=image)
        actual = p(images=image)
        if reference:
            for key in ("pixel_values", "image_grid_thw"):
                value = expected[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                equal(value, actual[key])
        else:
            assert actual["image_grid_thw"].tolist() == [[1, 2, 4]]
            assert actual["pixel_values"].shape == (8, 1176)


@pytest.mark.parametrize(
    "module,rows,cols,length", [(m.smolvlm, 3, 4, 81), (m.idefics3, 2, 2, 4)]
)
def test_tiled_image_prompt(module, rows, cols, length):
    for h, w in [(0, 0), (rows, cols)]:
        text = module.get_image_prompt_string(h, w, length, "<F>", "<image>", "<G>")
        assert text.count("<image>") == (h * w + 1) * length
        assert "<G>" in text
        if h:
            assert "<row_1_col_1>" in text
            assert f"<row_{h}_col_{w}>" in text


class TestMllamaProcessor:
    def test_cross_attention_mask_helpers(self):
        mask = m.mllama.get_cross_attention_token_mask(
            [1, 2, 128256, 3, 4, 128256, 5, 6], 128256
        )
        assert mask == [[2, 5], [5, 8]]
        assert m.mllama.get_cross_attention_token_mask([1, 2, 3], 128256) == []
        dense = m.mllama.convert_sparse_cross_attention_mask_to_dense(
            [mask], [[2, 3]], 4, 8
        )
        assert dense.shape == (1, 8, 2, 4)
        assert dense[0, 2, 0, 0] == 1 and dense[0, 0, 0, 0] == 0

    @pytest.mark.parametrize("text,expected", DATA["mllama_prompts"])
    def test_build_string_from_input(self, text, expected):
        assert m.mllama.build_string_from_input(text, "<bos>", "<|image|>") == expected


class TestQwen3VLProcessor:
    def _capturing_processor(self, grids):
        p = _make_processor("qwen3_vl")
        p.image_processor = _mock_ip(
            pixel_values=np.zeros((len(grids), 3, 224, 224), np.float32),
            image_grid_thw=np.array(grids, np.int64),
        )
        p.tokenizer = ProcessorTokenizer({}, pad=False)
        return p

    def test_surplus_tokens_follow_prompt_and_batch_order(self):
        block = "<|vs|><|image_pad|><|ve|>"
        kwargs = dict(
            image_token="<|image_pad|>",
            vision_start_token="<|vs|>",
            vision_end_token="<|ve|>",
            count=1,
        )
        assert (
            m.qwen3._drop_surplus_image_tokens(
                "old <|image_pad|> new " + block, **kwargs
            )
            == "old  new " + block
        )
        p = self._capturing_processor([[1, 4, 4], [1, 4, 8]])
        p(
            text=["first " + block, "stale " + block + " current " + block],
            images=[_make_image(), _make_image()],
        )
        assert p.tokenizer.last_text == [
            prefix + "<|vs|>" + "<|image_pad|>" * n + "<|ve|>"
            for prefix, n in [("first ", 4), ("stale  current ", 8)]
        ]

    @pytest.mark.parametrize(
        "case", DATA["qwen_image_errors"], ids=["ambiguous-flat", "extra-grouped"]
    )
    def test_invalid_image_counts(self, case):
        p = self._capturing_processor([[1, 4, 4], [1, 4, 8], [1, 4, 12]])
        images = [_make_image() for _ in range(3)]
        if case["grouped"]:
            images = [images[:2], images[2:]]
        with pytest.raises(ValueError, match=case["error"]):
            p(text=case["text"], images=images)

    @pytest.mark.parametrize("layout", ["nested-pil", "flat-pil", "hwc-array"])
    def test_video_frame_layouts(self, layout):
        frames = [Image.new("RGB", (224, 224), (i * 40, 128, 128)) for i in range(4)]
        video = {
            "nested-pil": [frames],
            "flat-pil": frames,
            "hwc-array": [np.zeros((4, 224, 224, 3), np.uint8)],
        }[layout]
        p = m.qwen3.Qwen3VLVideoProcessor(**PROFILES["qwen_video"])
        output = p(videos=video)
        equal(output["video_grid_thw"], [[2, 16, 16]])
        assert output["pixel_values_videos"].shape == (512, 1176)


def test_pixtral_image_preprocess_resizes_to_patch_multiple_and_pads():
    image_processor = m.pixtral_ip.PixtralImageProcessor(**PROFILES["pixtral_image"])
    wide = Image.fromarray(np.zeros((31, 55, 3), dtype=np.uint8))
    square = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    output = image_processor([[wide, square]])
    assert output["image_sizes"] == [(28, 42), (28, 28)]
    assert output["pixel_values"].shape == (2, 3, 28, 42)


class TestErnie4_5VLProcessor:
    def test_helper_functions(self):
        for fn, inputs, expected in [
            (m.ernie.round_by_factor, [100, 56, 42], [112, 56, 56]),
            (m.ernie.ceil_by_factor, [100, 56, 57], [112, 56, 84]),
            (m.ernie.floor_by_factor, [100, 56, 55], [84, 56, 28]),
        ]:
            assert [fn(x, 28) for x in inputs] == expected
        h, w = m.ernie.smart_resize(224, 224, factor=28)
        assert h % 28 == w % 28 == 0
        h, w = m.ernie.smart_resize(10, 10, factor=28, min_pixels=56 * 56)
        assert h * w >= 56 * 56
        h, w = m.ernie.smart_resize(10000, 10000, factor=28, max_pixels=28 * 28 * 1280)
        assert h * w <= 28 * 28 * 1280

    def test_image_processor(self):
        p = m.ernie.ImageProcessor()
        _assert_attrs(p, patch_size=14, merge_size=2, factor=28)
        (h, w), (gh, gw) = p.get_smart_resize(224, 224)
        assert h % 28 == w % 28 == 0 and (gh, gw) == (h // 14, w // 14)
        image = Image.new("RGB", (224, 224), "red")
        for inputs, count in [
            (image, 1),
            ([image, Image.new("RGB", (448, 448), "blue")], 2),
        ]:
            result = p.preprocess(inputs)
            assert {"pixel_values", "image_grid_thw"} <= result.keys()
            assert result["image_grid_thw"].shape[0] == count
            if count == 1:
                assert result["image_grid_thw"][0, 0] == 1
        pixels = np.random.rand(3, 224, 224).astype(np.float32)
        assert p._extract_patches(pixels, 16, 16).shape == (256, 588)
        assert {"pixel_values", "image_grid_thw"} <= p(images=image).keys()


class TestLfm2VlProcessorPatch:
    @pytest.mark.parametrize(
        "batched", [False, True], ids=["tiles", "flat-image-batch"]
    )
    def test_marker_expansion(self, batched):
        ip = m.lfm.Lfm2VlNumpyImageProcessor(max_num_patches=256, do_resize=False)
        assert ip.max_num_patches == 1024 and ip.do_resize
        p = _lfm_processor(ip)
        images = (
            [_make_image(540, 960) for _ in range(3)]
            if batched
            else [_make_image(1440, 2560)]
        )
        suffixes = ["First", "Second", "Third"] if batched else ["Describe this image"]
        prompts = ["<image>" + text for text in suffixes]
        result = m.lfm._patched_call(
            p, images=images, text=prompts if batched else prompts[0]
        )
        assert result["pixel_values"].shape == (3 if batched else 9, 1024, 768)
        expanded = p.tokenizer.last_text
        assert len(expanded) == len(suffixes)
        for text, suffix in zip(expanded, suffixes):
            assert text.count("<|image_start|>") == 1 and text.endswith(suffix)
            assert text.count("<image>") == (252 if batched else 8 * 256 + 252)
        if not batched:
            markers = [f"<|img_row_{r}_col_{c}|>" for r in (1, 2) for c in (1, 2, 3, 4)]
            assert text.startswith("<|image_start|>" + markers[0])
            assert all(marker in text for marker in markers)
            assert text.index(markers[-1]) < text.index("<|img_thumbnail|>")
            assert "<|img_thumbnail|>" + "<image>" * 252 + "<|image_end|>" in text
        raw = ip(images, return_tensors="np", do_resize=False)
        if batched:
            assert int(raw["pixel_attention_mask"].sum()) == sum(
                r * c for r, c in raw["spatial_shapes"]
            )
        else:
            assert raw["spatial_shapes"].tolist() == [[32, 32]] * 8 + [[24, 42]]

    def test_scalar_image_rows_and_cols_are_supported(self):
        result = m.lfm._patched_call(
            _lfm_processor(_stub("lfm_scalar")),
            images=_make_image(),
            text="<image>Describe this image",
        )
        assert {"input_ids", "attention_mask"} <= result.keys()

    def test_resample_filter_follows_the_checkpoint(self):
        for kwargs, expected in [
            ({}, 3),
            ({"resample": 3}, 3),
            ({"resample": 2}, 2),
            ({"resample": Image.Resampling.LANCZOS}, 1),
            ({"resample": "nonsense"}, 3),
        ]:
            assert m.lfm.Lfm2VlNumpyImageProcessor(
                **kwargs
            ).resample is Image.Resampling(expected)
        image = _make_image(540, 960)
        outputs = [
            m.lfm.Lfm2VlNumpyImageProcessor(resample=r)([image], return_tensors="np")[
                "pixel_values"
            ]
            for r in (3, 2)
        ]
        assert not np.allclose(*outputs)

    @pytest.mark.parametrize("layout", ["lists", "tuples", "arrays"])
    def test_mismatched_nested_image_groups(self, layout):
        images = [Image.new("RGB", (64, 64), (i * 60, 0, 0)) for i in range(4)]
        groups = [images[:1], images[1:]]
        if layout == "tuples":
            groups = tuple(tuple(group) for group in groups)
        elif layout == "arrays":
            groups = [np.stack(group) for group in groups]
        with pytest.raises(ValueError, match="text \\[2, 2\\] and images \\[1, 3\\]"):
            m.lfm._patched_call(
                _lfm_processor(),
                images=groups,
                text=["<image><image>Prompt A", "<image><image>Prompt B"],
            )


def test_molmo_point_processor_uses_image_processor_for_images():
    fields = _stub("molmo").outputs
    fields["image_token_pooling"] = fields["image_token_pooling"].astype(np.int64)
    p = c.molmo_point(
        ProcessorTokenizer(bos_token_id=1, eos_token_id=2),
        image_processor=NS(preprocess=lambda images: fields),
    )
    assert (
        fields.keys() <= p(text=m.molmo_point.IMAGE_PROMPT, images=_make_image()).keys()
    )


def test_kimi_chat_tokens_and_media():
    p = c.kimi_k3(tokenizer=KimiTokenizer())
    text = "literal <|end_of_msg|> marker"
    p.apply_chat_template([{"role": "user", "content": text}], tokenize=True)
    for source, split in [(text, True), ("<|end_of_msg|>", False)]:
        matches = [
            kwargs for value, kwargs in p.tokenizer.encode_calls if value == source
        ]
        assert matches and matches[0]["split_special_tokens"] is split
    result = apply_chat_template(
        p, {"model_type": "kimi_k3"}, "Describe this image.", num_images=1
    )
    assert "Describe this image.<|kimi_image_placeholder|>" in result
    assert '<|open|>message role="assistant"' in result
    with pytest.raises(ValueError, match="unsupported media type: video"):
        p(videos=[np.zeros((2, 16, 24, 3), np.uint8)], text="Describe this video.")


def test_locate_anything_save_pretrained_round_trips_custom_config(tmp_path):
    template = "{{ messages }}"
    geometry = dict(patch_size=28, merge_kernel_size=[2, 4], in_token_limit=1234)
    tokenizer = fast_tokenizer(chat_template=template)
    p = c.locateanything(
        image_processor=m.locate_ip.LocateAnythingImageProcessor(**geometry),
        tokenizer=tokenizer,
        chat_template=template,
    )
    saved = p.save_pretrained(tmp_path)
    configs = {
        name: json.loads((tmp_path / f"{name}.json").read_text())
        for name in ("processor_config", "preprocessor_config", "chat_template")
    }
    assert str(tmp_path / "processor_config.json") in saved
    assert configs["processor_config"]["processor_class"] == "LocateAnythingProcessor"
    for key in ("processor_config", "chat_template"):
        assert configs[key]["chat_template"] == template
    assert {k: configs["preprocessor_config"][k] for k in geometry} == geometry
    with patch.object(
        m.locateanything.AutoTokenizer, "from_pretrained", return_value=fast_tokenizer()
    ):
        loaded = c.locateanything.from_pretrained(tmp_path)
    _assert_attrs(loaded.image_processor, **geometry)
    assert loaded.chat_template == loaded.tokenizer.chat_template == template


def test_qwen3_vl_video_timestamp_video_prompt_falls_back_to_processor_fps():
    p = _make_processor("qwen3_vl")
    p.tokenizer = ProcessorTokenizer(
        {"<|video_pad|>": 102}, video_token="<|video_pad|>"
    )
    p.image_processor = None
    p.video_processor = _stub("qwen_video")
    p.vision_start_token, p.vision_end_token = ("<|vision_start|>", "<|vision_end|>")
    p.vision_start_token_id, p.vision_end_token_id = (58, 59)
    p(
        text=["<|vision_start|><|video_pad|><|vision_end|>Describe the clip."],
        videos=["clip.mp4"],
    )
    rendered = p.tokenizer.last_text[0]
    assert rendered.count(" seconds>") == 2
    assert "<0.2 seconds>" in rendered and "<1.2 seconds>" in rendered


class TestQwen3VLVideoTimestamps:
    """Qwen3-VL markers follow the frames that were actually sampled."""

    VIDEO_BLOCK = "<|vision_start|><|video_pad|><|vision_end|>"

    @pytest.fixture
    def p(self):
        tokens = [
            "[UNK]",
            "[PAD]",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ]
        tokenizer = fast_tokenizer(
            tokens,
            tokenizer_class=RecordingTokenizer,
            split=False,
            additional_special_tokens=tokens[2:],
        )
        return c.qwen3(
            image_processor=m.qwen3.Qwen3VLImageProcessor(),
            video_processor=m.qwen3.Qwen3VLVideoProcessor(**PROFILES["qwen_video"]),
            tokenizer=tokenizer,
        )

    @staticmethod
    def markers(p):
        return re.findall(r"<(\d+\.\d) seconds>", p.tokenizer.last_text[0])

    @staticmethod
    def frames(count=4):
        return np.zeros((count, 3, 56, 56), dtype=np.uint8)

    @pytest.mark.parametrize("prepare", [False, True], ids=["direct", "prepare-inputs"])
    @pytest.mark.parametrize(
        "indices,expected",
        [([0, 30, 60, 90], ["0.5", "2.5"]), ([0, 30, 60], ["0.5", "2.0"])],
        ids=["even", "odd"],
    )
    def test_metadata_sets_the_timestamps(self, p, prepare, indices, expected):
        metadata = dict(total_num_frames=91, fps=30, frames_indices=indices)
        if prepare:
            prepare_inputs(
                p,
                prompts=self.VIDEO_BLOCK,
                videos=[self.frames(len(indices))],
                video_metadata=[metadata],
            )
        else:
            p(
                text=[self.VIDEO_BLOCK],
                videos=[self.frames(len(indices))],
                video_metadata=[VideoMetadata(**metadata)],
            )
        assert self.markers(p) == expected

    @pytest.mark.parametrize("prepare", [False, True], ids=["direct", "prepare-inputs"])
    @pytest.mark.parametrize(
        "frame_count,indices",
        [(4, [0, 30, 60]), (3, [0, 30, 60, 90])],
        ids=["missing-index", "extra-index"],
    )
    def test_metadata_frame_count_is_checked_before_padding(
        self, p, prepare, frame_count, indices
    ):
        metadata = dict(total_num_frames=91, fps=30, frames_indices=indices)
        with pytest.raises(ValueError, match="frame indices must match"):
            if prepare:
                prepare_inputs(
                    p,
                    prompts=self.VIDEO_BLOCK,
                    videos=[self.frames(frame_count)],
                    video_metadata=[metadata],
                )
            else:
                p(
                    text=[self.VIDEO_BLOCK],
                    videos=[self.frames(frame_count)],
                    video_metadata=[VideoMetadata(**metadata)],
                )

    def test_fps_without_metadata_spaces_frames_evenly(self, p):
        p(text=[self.VIDEO_BLOCK], videos=[self.frames()], fps=[1.0])
        assert self.markers(p) == ["0.5", "2.5"]

    def test_clamped_clip_keeps_real_timestamps(self, p, synthetic_video):
        # 20 s at 30 fps; four frames are far below the default 2 fps.
        prepare_inputs(
            p, prompts=self.VIDEO_BLOCK, videos=[synthetic_video], max_frames=4
        )
        assert self.markers(p) == ["3.3", "16.6"]

    @pytest.mark.parametrize(
        "metadata,error",
        [
            ([dict(fps=30, frames_indices=[0, 30])] * 2, "one video_metadata"),
            ([dict(fps=30, frames_indices=[0, 30])], "frame indices must match"),
            ([dict(fps=0, frames_indices=[0, 1, 2, 3])], "positive and finite"),
        ],
        ids=["count", "frames", "fps"],
    )
    def test_invalid_metadata(self, p, metadata, error):
        with pytest.raises(ValueError, match=error):
            p(text=[self.VIDEO_BLOCK], videos=[self.frames()], video_metadata=metadata)


class TestMageVLProcessor:
    """Mage VL image/video processing and processor-to-model compatibility."""

    VIDEO_BLOCK = "<|vision_start|><|video_pad|><|vision_end|>"
    IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
    CHAT_TEMPLATE = DATA["mage_chat_template"]
    IMAGE_CONFIG = DATA["checkpoints"]["mage"]["preprocessor_config"]

    def processor(self):
        tokens = [
            "[UNK]",
            "[PAD]",
            m.mage.IMAGE_PAD,
            m.mage.VIDEO_PAD,
            m.mage.VISION_START,
            m.mage.VISION_END,
        ]
        tokenizer = fast_tokenizer(
            tokens,
            tokenizer_class=RecordingTokenizer,
            split=False,
            additional_special_tokens=tokens[2:],
            chat_template=self.CHAT_TEMPLATE,
        )
        return c.mage(
            image_processor=m.qwen3.Qwen3VLImageProcessor(**self.IMAGE_CONFIG),
            tokenizer=tokenizer,
        )

    @pytest.fixture
    def p(self):
        return self.processor()

    @staticmethod
    def frames(count=3, width=32, value=0):
        return np.full((count, 3, 32, width), value, dtype=np.uint8)

    @staticmethod
    def image_counts(output):
        return (np.array(output["input_ids"]) == 2).sum(axis=1).tolist()

    def test_image_batch_keeps_distinct_counts(self, p):
        images = [Image.new("RGB", (32, 32)), Image.new("RGB", (64, 32))]
        output = p(text=[self.IMAGE_BLOCK, self.IMAGE_BLOCK], images=images)
        assert self.image_counts(output) == [1, 2]
        assert "patch_positions" not in output
        expected = p.image_processor(images)
        equal(output["pixel_values"], expected["pixel_values"])

    @pytest.mark.parametrize("prepare", [False, True], ids=["direct", "prepare-inputs"])
    def test_video_timestamps_positions_and_attention(self, p, prepare):
        indices, fps = ([0, 20, 59], 10) if prepare else ([0, 15, 60], 30)
        metadata = dict(
            total_num_frames=indices[-1] + 1, fps=fps, frames_indices=indices
        )
        call = prepare_inputs if prepare else p
        kwargs = {
            "prompts" if prepare else "text": self.VIDEO_BLOCK,
            "videos": [self.frames()],
            "video_metadata": [metadata if prepare else VideoMetadata(**metadata)],
        }
        output = call(*([p] if prepare else []), **kwargs)
        assert p.tokenizer.last_text == [
            "".join(f"<{i / fps:.1f} seconds>{self.IMAGE_BLOCK}" for i in indices)
        ]
        assert self.image_counts(output) == [3]
        assert not {"pixel_values_videos", "video_grid_thw"} & output.keys()
        assert output["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
        positions = np.array(output["patch_positions"]).reshape(3, 4, 3)
        equal(positions[:, :, 0], [[i] * 4 for i in indices])
        equal(positions[1, :, 1:], [[0, 0], [0, 1], [1, 0], [1, 1]])
        assert m.mage_vision.build_cu_seqlens(
            output["image_grid_thw"].tolist(), 12, 4
        ) == [0, 4, 8, 12]

    @pytest.mark.parametrize("prompts,videos,kwargs,error", DATA["mage_errors"])
    def test_invalid_video(self, p, prompts, videos, kwargs, error):
        if "video_metadata" in kwargs and kwargs["video_metadata"]:
            kwargs = {
                "video_metadata": [VideoMetadata(**v) for v in kwargs["video_metadata"]]
            }
        with pytest.raises(ValueError, match=error):
            p(
                text=self.VIDEO_BLOCK * prompts,
                videos=[self.frames()] * videos,
                **kwargs,
            )

    def test_shared_decoder_supports_odd_frame_counts(self, p, tmp_path):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32)
        )
        if not writer.isOpened():
            pytest.skip("mp4v encoder unavailable")
        for value in (0, 100, 200):
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        writer.release()
        output = prepare_inputs(p, videos=[path], prompts=self.VIDEO_BLOCK, nframes=3)
        assert output["image_grid_thw"].shape == (3, 3)
        assert np.array(output["patch_positions"])[-1, 0] == 2

    def test_mixed_video_pixels_affect_only_their_visual_embeddings(self, p):
        config = m.mage_config.ModelConfig(
            text_config=m.mage_config.TextConfig(**PROFILES["mage_text"]),
            vision_config=m.mage_config.VisionConfig(**PROFILES["mage_vision"]),
            image_token_id=2,
            video_token_id=3,
        )
        model = m.mage_model.Model(config)
        kwargs = {
            "text": self.IMAGE_BLOCK + self.VIDEO_BLOCK,
            "images": [Image.new("RGB", (32, 32), "red")],
        }
        first = p(**kwargs, videos=[self.frames(2, value=0)])
        second = p(**kwargs, videos=[self.frames(2, value=255)])
        a = model.get_input_embeddings(**first).inputs_embeds
        b = model.get_input_embeddings(**second).inputs_embeds
        mx.eval(a, b)
        assert mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item()
        a, b = np.array(a)[0], np.array(b)[0]
        visual_indices = np.flatnonzero(np.array(first["input_ids"])[0] == 2)
        equal(a[visual_indices[0]], b[visual_indices[0]])
        assert not np.allclose(a[visual_indices[1:]], b[visual_indices[1:]])
        assert m.mage_model._as_grid_list(np.array([1, 4, 4])) == [(1, 4, 4)]


def tiny_diffusion_gemma_processor(image_processor=None):
    return c.diffusion(
        image_processor=image_processor,
        tokenizer=TinyDiffusionGemma4Tokenizer(PROFILES["diffusion_tokenizer"]),
        video_processor=_stub("diffusion_video"),
    )


class TestDiffusionGemma4Processor:
    def test_generate_strips_channel_scaffolding(self):
        dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
        model = NS(config=NS(model_type="diffusion_gemma", eos_token_id=999999))
        p = tiny_diffusion_gemma_processor()
        p.tokenizer.stopping_criteria = StoppingCriteria([999999], p.tokenizer)
        chunk = GenerationResult(**PROFILES["generation_chunk"])
        with patch.object(dispatch, "stream_generate", return_value=iter([chunk])):
            assert generate(model, p, "").text == "Title: A calm river cruise"

    def test_video_chat_cross_thread(self):
        def produce():
            p = tiny_diffusion_gemma_processor()
            rendered = apply_chat_template(
                p,
                NS(model_type="diffusion_gemma"),
                "Describe this video.",
                video=["clip.mp4"],
            )
            assert p.video_token in rendered
            return p(text=rendered, videos=[np.zeros((2, 3, 4, 4), np.uint8)])

        def consume(result):
            mx.eval(result)
            return (
                result["pixel_values"].shape,
                mx.sum(result["mm_token_type_ids"] == 2).item(),
            )

        with ThreadPoolExecutor(1) as producer, ThreadPoolExecutor(1) as consumer:
            result = producer.submit(produce).result(timeout=5)
            assert consumer.submit(consume, result).result(timeout=5) == (
                (2, 3, 4, 4),
                2,
            )

    def test_mixed_media_pixel_order(self):
        image = np.ones((3, 4, 4), np.float32)
        ip = _ImageStub(({"pixel_values": image[None]}, [1]))
        p = tiny_diffusion_gemma_processor(image_processor=ip)
        result = p(
            text="<video> then <image>",
            images=[image],
            videos=[np.zeros((2, 3, 4, 4), np.uint8)],
        )
        assert "pixel_values_videos" not in result
        pixels = result["pixel_values"]
        assert pixels.shape == (3, 3, 4, 4)
        assert mx.all(pixels[:2] == 0).item() and mx.all(pixels[2] == 1).item()


@pytest.mark.parametrize("name", DATA["checkpoints"])
def test_checkpoint_loading(tmp_path, name):
    _write_configs(tmp_path, **DATA["checkpoints"][name])
    family = "lfm" if name == "lfm_override" else name
    cls = getattr(c, family, None)
    tok = ProcessorTokenizer(
        **DATA["loader_tokenizers"].get(name, {"chat_template": "{{ messages }}"})
    )
    kwargs = DATA["loader_kwargs"].get(name, {})
    if name == "paddleocr_vl":
        assert load_image_processor(tmp_path) is None
    if name == "mage":
        tok = TestMageVLProcessor().processor().tokenizer
    elif name == "diffusion":
        tok = TinyDiffusionGemma4Tokenizer(PROFILES["diffusion_tokenizer"])
        tok.chat_template = None
    elif name == "deepseek":
        tok = ProcessorTokenizer(
            chat_template=None, apply_chat_template=lambda *a, **kw: "templated"
        )
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch("transformers.AutoTokenizer.from_pretrained", return_value=tok))
        if name not in ("glm_ocr", "omni", "mage", "nemotron"):
            base = c.g4 if name == "diffusion" else cls
            enter(_ignore_types(base))
        if name == "glm_ocr":
            enter(patch("mlx_vlm.models.base.load_chat_template"))
        elif family == "lfm":
            enter(patch.object(m.lfm, "Siglip2ImageProcessor", NS, create=True))
            enter(patch.object(m.lfm, "_SLOW_PROCESSOR_AVAILABLE", True))
        elif name == "dots_ocr":
            image_loader = Mock(return_value=_mock_ip())
            enter(
                patch.dict(
                    "transformers.__dict__",
                    {"AutoImageProcessor": NS(from_pretrained=image_loader)},
                )
            )
        elif name == "omni":
            fe = NS(model_input_names=["input_features"], sampling_rate=16000)
            enter(
                patch(
                    "transformers.AutoFeatureExtractor.from_pretrained", return_value=fe
                )
            )
        if name == "nemotron":
            p = load_processor(str(tmp_path), add_detokenizer=False)
        elif name in ("omni", "mage", "diffusion", "deepseek"):
            p = AutoProcessor.from_pretrained(tmp_path)
        else:
            p = cls.from_pretrained(tmp_path, **kwargs)
        if name == "deepseek":
            explicit = cls.from_pretrained("repo/name", chat_template="{{ explicit }}")
            assert explicit.chat_template == "{{ explicit }}"
            template = "{{ messages[0]['content'] }}"
            (tmp_path / "chat_template.jinja").write_text(template)
            assert m.deepseek.load_deepseek_v4_chat_template(tmp_path) == template
    assert isinstance(p, cls)
    _assert_attrs(p, **DATA["loader_attributes"].get(name, {}))
    for attr, expected in DATA["loader_components"].get(name, {}).items():
        assert type(getattr(p, attr)).__name__ == expected
    if name == "dots_ocr":
        image_loader.assert_called_once()
        assert image_loader.call_args.kwargs["use_fast"] is False
    elif name == "diffusion":
        assert p.get_attributes() == ["image_processor", "tokenizer", "video_processor"]
    elif name == "mistral3":
        out = p(text=["[IMG]Describe"], images=[[_make_image()]])
        _assert_all_mx(out, "pixel_values")
        assert out["pixel_values"].shape[:2] == (1, 3)
        assert all(int(size) % 28 == 0 for size in out["image_sizes"][0].tolist())
    elif name == "nemotron":
        out = prepare_inputs(
            p,
            images=[_make_image()],
            prompts="<image>\nDescribe this image.",
            image_token_index=p.image_token_id,
        )
        assert "pixel_values" in out and int(out["num_tokens"][0].item()) > 0
    elif name == "mage":
        assert processor_handles_video(p)
        _assert_attrs(
            resolve_video_sampling(p, {}), min_frames=1, frame_factor=1, max_frames=32
        )
        assert TestMageVLProcessor.image_counts(
            p(text=m.mage.VIDEO_PAD, videos=[TestMageVLProcessor.frames()])
        ) == [3]


def test_nemotron_processor_modalities_and_metadata(tmp_path):
    metadata = deepcopy(DATA["checkpoints"]["nemotron"])
    config = metadata["config"]
    config["model_type"] = "nemotron_h_omni"
    metadata["processor_config"][
        "processor_class"
    ] = "NemotronH_Omni_Reasoning_V3Processor"
    metadata["preprocessor_config"].update(
        norm_mean=[0.5] * 3,
        norm_std=[0.5] * 3,
        min_num_patches=4,
        max_num_patches=16,
        video_target_num_patches=8,
    )
    _write_configs(tmp_path, **metadata)
    vocab = {str(i): i for i in range(21)}
    for index, token in [
        (0, "[UNK]"),
        (5, "seconds"),
        (18, "<image>"),
        (19, "<img>"),
        (20, "</img>"),
    ]:
        del vocab[str(index)]
        vocab[token] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[UNK]",
        pad_token="[UNK]",
        additional_special_tokens=["<image>", "<img>", "</img>"],
    )
    tokenizer.save_pretrained(tmp_path)
    processor = load_processor(tmp_path, add_detokenizer=False, trust_remote_code=False)
    assert processor_handles_video(processor)
    messages = apply_chat_template(
        processor, config, "Describe.", num_images=1, return_messages=True
    )
    assert messages[0]["content"][0] == {"type": "image"}
    messages = apply_chat_template(
        processor, config, "Describe.", video="clip.mp4", return_messages=True
    )
    assert messages[0]["content"][0]["type"] == "video"
    image = Image.new("RGB", (32, 32), "red")
    inputs = prepare_inputs(processor, prompts="<image>Describe.", images=[image])
    assert int(mx.sum(inputs["input_ids"] == 18)) == int(inputs["num_tokens"].sum())
    frames = np.zeros((5, 3, 32, 64), dtype=np.uint8)
    inputs = prepare_inputs(processor, prompts="<video>Describe.", videos=[frames])
    assert inputs["pixel_values_videos"].shape[0] == 5
    assert int(mx.sum(inputs["input_ids"] == 18)) == 3 * int(inputs["num_tokens"][0])
    assert (
        int(mx.sum(inputs["input_ids"] == 5)) == 5
    )  # Per-frame timestamps survived the loader.
    assert processor.image_processor._is_video_mode is False
    with pytest.raises(ValueError, match="one video"):
        prepare_inputs(processor, prompts="<video><video>", videos=[frames, frames])

    waveform = np.linspace(-1, 1, 1600, dtype=np.float32)
    keyword = processor(text="<so_embedding>", audio=waveform)
    positional = processor(None, "<so_embedding>", None, waveform)
    for inputs in (keyword, positional):
        assert len(inputs["sound_clips"]) == 1
        np.testing.assert_array_equal(inputs["sound_clips"][0], waveform)
    np.testing.assert_array_equal(positional["input_ids"], keyword["input_ids"])


# Prompt construction


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


@pytest.mark.parametrize(
    "content,expected",
    [
        (None, ""),
        (
            ["just a string", {"type": "text", "text": "Valid item"}, 123, None],
            "Valid item",
        ),
        (
            [{"type": "text", "text": text} for text in ("", "Actual content", "")],
            "Actual content",
        ),
    ],
)
def test_extract_text_from_content(content, expected):
    assert extract_text_from_content(content) == expected


@pytest.mark.parametrize(
    "family,kind",
    [
        ("nemotron_h_nano_omni", "image-audio"),
        ("nemotronh_nano_omni_reasoning_v3", "image-audio"),
        ("qwen3_omni_moe", "qwen-image-audio"),
        ("gemma4_unified", "video-audio"),
        ("prism_hadamard_qwen35", "image-video"),
        ("step3p7", "patch"),
    ],
)
def test_prompt_media_format(family, kind):
    text, options = "Describe the inputs.", dict(num_images=1, num_audios=1)
    if kind == "video-audio":
        text = "Describe the video and audio."
        options = dict(video=["clip.mp4"], fps=1, num_audios=1)
    elif kind == "patch":
        text, options = "What do you see?", dict(num_images=1)
    elif kind == "image-video":
        options = dict(num_images=2, video="clip.mp4")
    text_part = dict(type="text", text=text, content=text)
    expected = {
        "image-audio": [dict(type="image"), text_part, dict(type="audio")],
        "qwen-image-audio": [dict(type="audio"), dict(type="image"), text_part],
        "video-audio": [
            dict(type="video", video="clip.mp4", max_pixels=224 * 224, fps=1),
            dict(type="audio"),
            text_part,
        ],
        "image-video": [
            dict(type="image"),
            dict(type="image"),
            dict(type="video", video="clip.mp4", max_pixels=224 * 224, fps=1),
            text_part,
        ],
        "patch": "<im_patch>" + text,
    }[kind]
    result = apply_chat_template(
        None, {"model_type": family}, text, return_messages=True, **options
    )
    assert result == [dict(role="user", content=expected)]


@pytest.mark.parametrize("representation", ["pydantic-list", "single-dict"])
def test_prompt_does_not_leak_image_payload(representation):
    from pydantic import BaseModel

    class ChatMessage(BaseModel):
        role: str
        content: list

    pydantic = representation == "pydantic-list"
    marker = "ABC123" if pydantic else "SINGLEBASE64"
    payload = "ABC123XYZ" if pydantic else "SINGLEBASE64DATA"
    text = "What is in this image?" if pydantic else "Analyze this single prompt image."
    message = dict(
        role="user",
        content=[
            dict(type="text", text=text),
            dict(
                type="image_url", image_url=dict(url="data:image/png;base64," + payload)
            ),
        ],
    )
    prompt = [ChatMessage(**message)] if pydantic else message
    result = apply_chat_template(
        None, {"model_type": "qwen2_vl"}, prompt, return_messages=True, num_images=1
    )
    assert isinstance(result, list)
    for message in result:
        content = message.get("content", "")
        if isinstance(content, str):
            assert marker not in content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    assert marker not in str(
                        item.get("text", "") or item.get("content", "")
                    )


class TestApplyChatTemplateIntegration:
    """Integration tests for apply_chat_template with multimodal content.

    These tests verify the actual bug fix works end-to-end, not just the helper.
    Uses return_messages=True to inspect intermediate messages without mocking.
    """

    @pytest.mark.parametrize(
        "family,markers",
        [
            ("deepseek_v4", ("<image>", "<image>")),
            ("qwen3_vl", ("<image>", "<image>")),
            ("ernie4_5_moe_vl", ("<image>", "<image>")),
            ("internvl_chat", ("<image>", "<image>")),
            ("gemma4", ("<image>", "<image>")),
            ("step3p7", ("<im_patch>", "<im_patch>")),
            ("gemma3", ("<start_of_image>", "<start_of_image>")),
            ("phi4mm", ("<|image_1|>", "<|image_2|>")),
        ],
    )
    @pytest.mark.parametrize("representation", ["dict", "list", "pydantic"])
    def test_interleaved_images_reach_renderer(self, family, markers, representation):
        from pydantic import BaseModel

        class Message(BaseModel):
            role: str
            content: list

        message = dict(
            role="user",
            content=[
                dict(type="input_text", text="before "),
                dict(
                    type="image_url", image_url=dict(url="data:image/png;base64,FIRST")
                ),
                dict(type="text", text=" between "),
                dict(type="input_image", image_url="data:image/png;base64,SECOND"),
                dict(type="text", text=" after"),
            ],
        )
        original = deepcopy(message)
        prompt = (
            message
            if representation == "dict"
            else [Message(**message)] if representation == "pydantic" else [message]
        )
        normalized = apply_chat_template(
            None, dict(model_type=family), prompt, num_images=2, return_messages=True
        )
        assert "base64" not in str(normalized)
        rendered = get_chat_template(None, normalized, add_generation_prompt=True)
        assert rendered == f"before {markers[0]} between {markers[1]} after"
        assert message == original

    def test_explicit_images_without_side_channel_count(self):
        message = dict(
            role="user", content=[dict(type="text", text="before "), dict(type="image")]
        )
        rendered = apply_chat_template(None, dict(model_type="qwen3_vl"), [message])
        assert rendered == "before <image>"

    @pytest.mark.parametrize(
        "family,separator", [("deepseek", ""), ("deepseek41", "\n\n")]
    )
    def test_deepseek_processor_preserves_inline_image_position(
        self, family, separator
    ):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            unk_token="[UNK]",
        )
        processor = getattr(c, family)(tokenizer)
        message = dict(
            role="user",
            content=[
                dict(type="text", text="before"),
                dict(type="image_url", image_url=dict(url="x")),
                dict(type="text", text="after"),
            ],
        )
        model_type = "deepseek_v4" if family == "deepseek" else "deepseek_v41"
        rendered = apply_chat_template(
            processor, dict(model_type=model_type), message, num_images=1
        )
        assert f"before{separator}<｜deepseek_image｜>{separator}after" in rendered

    @pytest.mark.parametrize(
        "family,expected",
        [
            ("qwen3_vl", "<image> before <image> after"),
            ("qwen2_vl", "before <image> after<image>"),
            ("phi4mm", "<|image_1|>before <|image_2|> after"),
        ],
    )
    def test_extra_side_channel_images_keep_default_placement(self, family, expected):
        message = dict(
            role="user",
            content=[
                dict(type="text", text="before "),
                dict(type="image"),
                dict(type="text", text=" after"),
            ],
        )
        assert (
            apply_chat_template(None, dict(model_type=family), message, num_images=2)
            == expected
        )

    def test_interleaved_images_keep_side_channel_audio_and_video(self):
        message = dict(
            role="user",
            content=[
                dict(type="text", text="before"),
                dict(type="image"),
                dict(type="text", text="after"),
            ],
        )
        result = apply_chat_template(
            None,
            dict(model_type="qwen3_vl"),
            message,
            num_images=1,
            num_audios=1,
            video="clip.mp4",
            return_messages=True,
        )
        parts = result[0]["content"]
        assert [part["type"] for part in parts] == [
            "video",
            "audio",
            "text",
            "image",
            "text",
        ]
        assert parts[2]["text"] == "before" and parts[4]["text"] == "after"

    def test_explicit_images_do_not_bypass_single_image_limit(self):
        message = dict(role="user", content=[dict(type="image"), dict(type="image")])
        with pytest.raises(ValueError, match="multi-image"):
            apply_chat_template(None, dict(model_type="mllama"), message)

    def test_tool_image_does_not_add_another_image_to_user_turn(self):
        messages = [
            dict(role="user", content="Inspect the result."),
            dict(role="tool", tool_call_id="image", content=[dict(type="image")]),
            dict(role="user", content="What changed?"),
        ]
        rendered = apply_chat_template(
            None, dict(model_type="qwen3_vl"), messages, num_images=1
        )
        assert rendered.count("<image>") == 1
        assert (
            rendered.index("Tool:")
            < rendered.index("<image>")
            < rendered.index("What changed?")
        )

    def test_image_stays_on_its_original_user_turn(self):
        messages = [
            dict(
                role="user",
                content=[dict(type="text", text="First turn"), dict(type="image")],
            ),
            dict(role="assistant", content="I see it."),
            dict(role="user", content="Second turn"),
        ]
        normalized = apply_chat_template(
            None,
            {"model_type": "qwen2_vl"},
            messages,
            num_images=1,
            return_messages=True,
        )
        assert any(part["type"] == "image" for part in normalized[0]["content"])
        assert normalized[-1]["content"] == [
            dict(type="text", text="Second turn", content="Second turn")
        ]
        prompt = get_chat_template(None, normalized, add_generation_prompt=True)
        assert prompt.count("<image>") == 1
        assert prompt.index("<image>") < prompt.index("Second turn")

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


@pytest.mark.parametrize("model_type", ["llama", "qwen3_5"])
@pytest.mark.parametrize(
    "template",
    [
        "{{ tools | tojson }}",
        "{% for tool in tools %}{% for key, value in tool.items() %}"
        "{{ key }}={{ value | tojson }};{% endfor %}{% endfor %}",
    ],
)
def test_tool_schema_key_order_does_not_change_prompt(model_type, template):
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "enum": ["src", "tests"]},
            "options": {
                "type": "object",
                "properties": {"recursive": {"type": "boolean"}},
            },
        },
        "required": ["path", "options"],
    }
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": parameters},
        }
        for name in ["search_files", "read_file"]
    ]

    def reverse_keys(value):
        if isinstance(value, dict):
            return {key: reverse_keys(value[key]) for key in reversed(value)}
        if isinstance(value, list):
            return [reverse_keys(item) for item in value]
        return value

    tokenizer = fast_tokenizer(chat_template=template)

    def render(value):
        return apply_chat_template(
            tokenizer, {"model_type": model_type}, "Find files.", tools=value
        )

    reordered = reverse_keys(tools)
    original_json = json.dumps(reordered)
    prompt = render(tools)
    assert render(reordered) == prompt
    assert json.dumps(reordered) == original_json
    assert render(list(reversed(tools))) != prompt

    changed = deepcopy(tools)
    changed[0]["function"]["parameters"]["properties"]["path"]["type"] = "integer"
    assert render(changed) != prompt
    if template == "{{ tools | tojson }}":
        assert json.loads(prompt) == tools


@pytest.mark.parametrize("tools", [None, [], (lambda: None,)])
def test_template_preserves_empty_and_callable_tools(tools):
    processor = MagicMock(chat_template="{{ tools }}")
    get_chat_template(processor, [{"role": "user", "content": "Hi"}], True, tools=tools)
    assert processor.apply_chat_template.call_args.kwargs["tools"] == tools


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


def test_qwen3_omni_enables_thinking_by_default():
    processor = MagicMock(chat_template="{{ messages }}")
    processor.apply_chat_template.return_value = "prompt"

    result = apply_chat_template(
        processor, {"model_type": "qwen3_omni_moe"}, "Describe this image."
    )

    assert result == "prompt"
    assert processor.apply_chat_template.call_args.kwargs["enable_thinking"] is True


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


# Reasoning-template arguments


def test_no_reasoning_strength_when_effort_unset():
    kw = GenerationArguments(enable_thinking=True, reasoning=True).to_template_kwargs()

    assert "reasoning_strength" not in kw
    assert "reasoning_effort" not in kw


# Tool parsing

PARSER_NAMES = sorted(
    module.name
    for module in pkgutil.iter_modules(parsers.__path__)
    if not module.ispkg and not module.name.startswith("_")
)
WEATHER_ARGS = {"city": "Paris", "days": 3}


def _weather_tools(**fields):
    properties = {name: dict(type=kind) for name, kind in fields.items()}
    return [
        dict(
            type="function",
            function=dict(
                name="get_weather",
                parameters=dict(type="object", properties=properties),
            ),
        )
    ]


WEATHER_TOOLS = _weather_tools(city="string", days="integer")
# Literal wire examples are independent of the parser's marker constants.
WIRE_CALLS = {
    "atem": 'to=self<|message|><atem:function_calls><atem:invoke name="get_weather">'
    '<atem:parameter name="city">Paris</atem:parameter>'
    '<atem:parameter name="days">3</atem:parameter></atem:invoke></atem:function_calls>',
    "cohere2_moe": '<|START_ACTION|>{"tool_name":"get_weather",'
    '"parameters":{"city":"Paris","days":3}}<|END_ACTION|>',
    "gemma4": '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>',
    "glm47": "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
    "json_tools": '<tool_call>{"name":"get_weather",'
    '"arguments":{"city":"Paris","days":3}}</tool_call>',
    "kimi_k2": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
    '<|tool_call_argument_begin|>{"city":"Paris","days":3}'
    "<|tool_call_end|><|tool_calls_section_end|>",
    "longcat": "<longcat_tool_call>get_weather<longcat_arg_key>city</longcat_arg_key>"
    "<longcat_arg_value>Paris</longcat_arg_value><longcat_arg_key>days</longcat_arg_key>"
    "<longcat_arg_value>3</longcat_arg_value></longcat_tool_call>",
    "minicpm5": '<function name="get_weather"><param name="city">Paris</param>'
    '<param name="days">3</param></function>',
    "minimax_m2": '<minimax:tool_call><invoke name="get_weather">'
    '<parameter name="city">Paris</parameter><parameter name="days">3</parameter>'
    "</invoke></minimax:tool_call>",
    "minimax_m3": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="get_weather">'
    "]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[<days>3"
    "]<]minimax[>[</days>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
    "mistral": '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "days": 3}',
    "harmony": "<|channel|>commentary to=functions.get_weather <|constrain|>json"
    '<|message|>{"city": "Paris", "days": 3}<|call|>',
    "pythonic": '<|tool_call_start|>[get_weather(city="Paris", days=3)]<|tool_call_end|>',
    "qwen3_coder": "<tool_call>\n<function=get_weather><parameter=city>Paris</parameter>"
    "<parameter=days>3</parameter></function></tool_call>",
}


WIRE_VARIANTS = {
    "mistral": [
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]'
    ],
}


def _parse(name, text, tools=None):
    return load_tool_module(name).parse_tool_call(text, tools)


def _call(name, **arguments):
    return dict(name=name, arguments=arguments)


def _arguments(call):
    value = call["arguments"]
    return json.loads(value) if isinstance(value, str) else value


def test_every_parser_has_registration_and_wire_example():
    assert set(PARSER_NAMES) == {spec.name for spec in SPECS} == set(WIRE_CALLS)
    assert set(WIRE_VARIANTS) <= set(PARSER_NAMES)


@pytest.mark.parametrize("name", PARSER_NAMES)
def test_parser_contract(name):
    parser = load_tool_module(name)
    for wire in [WIRE_CALLS[name], *WIRE_VARIANTS.get(name, ())]:
        body = wire.removeprefix(parser.tool_call_start).removesuffix(
            parser.tool_call_end
        )
        parsed = parser.parse_tool_call(body, WEATHER_TOOLS)
        calls = parsed if isinstance(parsed, list) else [parsed]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert _arguments(calls[0]) == WEATHER_ARGS
        for count, with_prose in [(1, False), (1, True), (2, True)]:
            # A bare call exercises EOF; newlines delimit Mistral's repeated calls.
            output = "\n".join([wire] * count)
            if with_prose:
                output = f"Before\n{output}\nAfter"
            result = process_tool_calls(output, parser, WEATHER_TOOLS)
            if with_prose:
                assert result.remaining_text.split() == ["Before", "After"]
            else:
                assert result.remaining_text == ""
            assert len(result.calls) == count
            assert len({call["id"] for call in result.calls}) == count
            for index, call in enumerate(result.calls):
                assert call["id"]
                assert call["type"] == "function"
                assert call["index"] == index
                assert call["function"]["name"] == "get_weather"
                assert json.loads(call["function"]["arguments"]) == WEATHER_ARGS
    for text in ("Ordinary assistant prose.", "Like call: prince"):
        result = process_tool_calls(text, parser, tools=None)
        assert result.calls == []
        assert result.remaining_text == text


@pytest.mark.parametrize("name", PARSER_NAMES)
def test_parser_selection(name):
    # Specific formats must outrank the generic JSON fallback.
    generic = "<tool_call> tool_call.name"
    template = WIRE_CALLS[name] + generic
    for value in (
        template,
        {"default": generic, "tool_use": template},
        [
            {"name": "default", "template": generic},
            {"name": "tool_use", "template": template},
        ],
    ):
        assert _infer_tool_parser(value) == name
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))
    assert _infer_tool_parser_from_processor(processor) == name
    assert _infer_tool_parser("anything", override=name) == name


@pytest.mark.parametrize(
    "name,text,error",
    [
        ("atem", "not a tool call", "No ATEM function invocation"),
        ("minicpm5", '<function name="lookup"><param name="value">unfinished', None),
        ("minicpm5", '<function name=""></function>', None),
        ("minicpm5", '<function name="lookup"><param>3</param></function>', None),
        ("gemma4", "just a normal model response, no tool call here", None),
        ("mistral", "not a tool call at all", None),
        (
            "pythonic",
            "[write_file(content='const player = { x: 0, y: 1 };)]",
            "Invalid Pythonic tool call",
        ),
        ("pythonic", "[write_file(content=get_content())]", "must be a literal value"),
    ],
)
def test_invalid_calls(name, text, error):
    with pytest.raises(ValueError, match=error):
        _parse(name, text)


@pytest.mark.parametrize(
    "parser,argument_type,text,expected,tools",
    [
        (
            "gemma4",
            str,
            '<|tool_call>call:edit-file{path:<|"|>test.txt<|"|>,edits:[{newText:<|"|>orange<|"|>,oldText:<|"|>apple<|"|>}]}<tool_call|>',
            _call(
                "edit-file",
                path="test.txt",
                edits=[{"newText": "orange", "oldText": "apple"}],
            ),
            None,
        ),
        (
            "gemma4",
            str,
            "get_weather{city:Austin}",
            _call("get_weather", city="Austin"),
            None,
        ),
        (
            "pythonic",
            dict,
            '[write_file(path="game.html", content="<canvas id="game">\n</canvas>")]',
            _call(
                "write_file", path="game.html", content='<canvas id="game">\n</canvas>'
            ),
            None,
        ),
        (
            "pythonic",
            dict,
            "[configure(options={'position': [0, 1], 'enabled': True})]",
            _call("configure", options={"position": [0, 1], "enabled": True}),
            None,
        ),
        (
            "cohere2_moe",
            str,
            '{"tool_call_id":"1","tool_name":"grep","parameters":{"pattern":"foo"}}',
            _call("grep", pattern="foo"),
            None,
        ),
        (
            "cohere2_moe",
            str,
            r'[{"tool_call_id":"1","tool_name":"grep","parameters":{"pattern":"<\|channel>"}},'
            '{"tool_call_id_id":"2","tool_name":"read","parameters":{"path":"file.py"}}]',
            [_call("grep", pattern="<|channel>"), _call("read", path="file.py")],
            None,
        ),
        (
            "glm47",
            dict,
            "get_weather\n<arg_key>zip</arg_key>\n<arg_value>10001</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n",
            _call("get_weather", zip="10001", days=3),
            _weather_tools(zip="string", days="integer"),
        ),
        (
            "qwen3_coder",
            dict,
            '<function=configure>\n<parameter=options>\n{"depth": 2}\n</parameter>\n'
            "<parameter=ids>\n[1, 2]\n</parameter>\n"
            "<parameter=tag>\n123\n</parameter>\n</function>",
            _call("configure", options={"depth": 2}, ids=[1, 2], tag="123"),
            [
                dict(
                    type="function",
                    function=dict(
                        name="configure",
                        parameters=dict(
                            type="object",
                            properties=dict(
                                options=dict(description="untyped"),
                                ids=dict(description="untyped"),
                                tag=dict(description="untyped"),
                            ),
                        ),
                    ),
                )
            ],
        ),
        (
            "qwen3_coder",
            dict,
            "<function=write>\n<parameter=content>\nfirst\n"
            "<parameter=name>\nlast\n</parameter>\n</function>",
            _call("write", content="first\n<parameter=name>\nlast"),
            None,
        ),
        (
            "qwen3_coder",
            dict,
            "<function=write>\n<parameter=path>\na.txt\n</parameter>\n"
            "<parameter=content>\nhello\n</function>",
            _call("write", path="a.txt", content="hello"),
            None,
        ),
        (
            "qwen3_coder",
            dict,
            "<function=write>\n<parameter=path>\na.txt\n</parameter>\n"
            "<parameter=content\n</function>",
            _call("write", path="a.txt"),
            None,
        ),
        (
            "qwen3_coder",
            dict,
            "<function=get_weather>\n<parameter=zip>\n10001\n</parameter>\n"
            "<parameter=days>\nthree\n</function>",
            _call("get_weather", zip="10001"),
            _weather_tools(zip="string", days="integer"),
        ),
        (
            "qwen3_coder",
            dict,
            "<function=write><parameter=content><parameter=</parameter></function>",
            _call("write", content="<parameter="),
            None,
        ),
        (
            "qwen3_coder",
            dict,
            "<function=write><parameter=content>"
            "Use <parameter=name> in the template.</parameter></function>",
            _call("write", content="Use <parameter=name> in the template."),
            None,
        ),
        (
            "qwen3_coder",
            dict,
            '<function=configure><parameter=options>{"depth": 2}</parameter>'
            "</function>",
            _call("configure", options={"depth": 2}),
            [
                dict(
                    type="function",
                    function=dict(
                        name="configure",
                        parameters=dict(type="object", properties=dict(options={})),
                    ),
                )
            ],
        ),
    ],
    ids=[
        "gemma-nested",
        "gemma-bare",
        "pythonic-html",
        "pythonic-nested",
        "cohere-object",
        "cohere-array-escape",
        "glm-newline",
        "qwen-untyped",
        "qwen-line-start-parameter-tag",
        "qwen-unclosed-last",
        "qwen-truncated-parameter-tag",
        "qwen-unclosed-last-invalid",
        "qwen-literal-parameter-prefix",
        "qwen-literal-parameter-tag",
        "qwen-empty-property-schema",
    ],
)
def test_parser_syntax(parser, argument_type, text, expected, tools):
    result = _parse(parser, text, tools)
    assert isinstance(result, type(expected))
    calls = result if isinstance(result, list) else [result]
    expected_calls = expected if isinstance(expected, list) else [expected]
    assert all(isinstance(call["arguments"], argument_type) for call in calls)
    assert [dict(call, arguments=_arguments(call)) for call in calls] == expected_calls


class TestGemma4StringArguments:
    @pytest.mark.parametrize(
        "content",
        [
            'def strip_closing_braces(text):\n    return text.replace("}", "")\n',
            "a { b",
            "a } b { c",
            '{"suffix": "}", "text": "ação 日本 🐦"}\n',
        ],
        ids=["python-source", "opening-brace", "closing-before-opening", "json-source"],
    )
    def test_literal_braces_preserve_content_and_following_arguments(self, content):
        text = (
            f'<|tool_call>call:write_file{{content:<|"|>{content}<|"|>,'
            'path:<|"|>helpers.py<|"|>}<tool_call|>'
        )
        parsed = _parse("gemma4", text)
        assert parsed["name"] == "write_file"
        assert _arguments(parsed) == {"content": content, "path": "helpers.py"}
        result = process_tool_calls(text, load_tool_module("gemma4"), None)
        assert result.remaining_text == ""
        assert len(result.calls) == 1
        assert result.calls[0]["function"] == {
            "name": "write_file",
            "arguments": parsed["arguments"],
        }

    def test_braces_in_nested_array_strings_preserve_following_arguments(self):
        text = (
            'call:edit-file{edits:[{oldText:<|"|>a } b<|"|>,'
            'newText:<|"|>c<|"|>}],path:<|"|>helpers.py<|"|>}'
        )
        parsed = _parse("gemma4", text)
        assert parsed["name"] == "edit-file"
        assert _arguments(parsed) == {
            "edits": [{"oldText": "a } b", "newText": "c"}],
            "path": "helpers.py",
        }

    def test_deep_string_call_syntax_is_not_selected_as_the_function(self):
        parsed = _parse(
            "gemma4",
            'call:mcp.files:write{a:{b:{c:{d:{text:<|"|>call:fake{value:1}'
            '<|"|>}}}},path:<|"|>helpers.py<|"|>}',
        )
        assert parsed["name"] == "mcp.files:write"
        assert _arguments(parsed) == {
            "a": {"b": {"c": {"d": {"text": "call:fake{value:1}"}}}},
            "path": "helpers.py",
        }


@pytest.mark.parametrize("name", PARSER_NAMES)
def test_parser_accepts_boolean_property_schemas(name):
    # ``true`` is a valid JSON Schema for a property that admits any value.
    tools = [
        dict(
            type="function",
            function=dict(
                name="get_weather",
                parameters=dict(type="object", properties=dict(city=True, days=True)),
            ),
        )
    ]
    result = process_tool_calls(WIRE_CALLS[name], load_tool_module(name), tools)
    assert [call["function"]["name"] for call in result.calls] == ["get_weather"]
    assert json.loads(result.calls[0]["function"]["arguments"])["city"] == "Paris"


@pytest.mark.parametrize(
    "template",
    [None, {}, [], 123, {"tool_use": None}, {"default": "x", "tool_use": "y"}],
)
def test_non_routable_inputs_return_none(template):
    assert _infer_tool_parser(template) is None


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError):
        _infer_tool_parser("anything", override="does_not_exist")


MINICPM_CDATA_CALL = (
    '<function name="write_file"><param name="content">'
    "<![CDATA[  <html>\nA & B\n</html>  ]]></param>"
    '<param name="version">123</param><param name="count">3</param>'
    '<param name="enabled">True</param></function>'
)
MINICPM_MULTICALL = (
    f'Before{MINICPM_CDATA_CALL}Between<function name="get_time"></function>After'
)


def test_minicpm5_cdata_and_argument_types():
    tools = [
        dict(
            function=dict(
                name="write_file",
                parameters={"properties": {"version": {"type": "string"}}},
            )
        )
    ]
    assert _parse("minicpm5", MINICPM_CDATA_CALL, tools) == _call(
        "write_file",
        content="  <html>\nA & B\n</html>  ",
        version="123",
        count=3,
        enabled=True,
    )
    result = process_tool_calls(MINICPM_MULTICALL, load_tool_module("minicpm5"), None)
    assert result.remaining_text == "Before Between After"
    assert [call["function"]["name"] for call in result.calls] == [
        "write_file",
        "get_time",
    ]
    assert json.loads(result.calls[1]["function"]["arguments"]) == {}


HARMONY_ANALYSIS_THEN_CALL = (
    "<|channel|>analysis<|message|>The user wants the weather. Call get_weather."
    "<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"city": "Paris", "days": 3}<|call|>'
)


def test_harmony_extracts_commentary_tool_call_past_analysis():
    result = process_tool_calls(
        HARMONY_ANALYSIS_THEN_CALL, load_tool_module("harmony"), WEATHER_TOOLS
    )
    assert len(result.calls) == 1
    assert result.calls[0]["function"]["name"] == "get_weather"
    assert json.loads(result.calls[0]["function"]["arguments"]) == WEATHER_ARGS
    assert "to=functions" not in result.remaining_text


def test_harmony_ignores_plain_commentary_preamble():
    preamble = "<|channel|>commentary<|message|>Let me look that up.<|end|>"
    result = process_tool_calls(preamble, load_tool_module("harmony"), None)
    assert result.calls == []
    assert result.remaining_text == preamble


# Loading and utility contracts


class MockProcessor:
    def __init__(self, tokenizer_return_value=None):
        self.image_token = "<image>"
        result = tokenizer_return_value or SimpleNamespace(
            input_ids=mx.array([[1, 2, 3]]), attention_mask=mx.array([[7, 8, 9]])
        )
        self.tokenizer = MagicMock(
            pad_token=None, eos_token="[EOS]", return_value=result
        )

    def __call__(
        self, text=None, images=None, audio=None, padding=None, return_tensors="mlx"
    ):
        assert return_tensors == "mlx"
        images = images if isinstance(images, list) else [images]
        images = [image for image in images if image is not None]
        count = text.count("<image>") if text else 0
        if count != len(images):
            raise ValueError(
                f"Number of image tokens in prompt_token_ids ({count}) "
                f"does not match number of images ({len(images)})"
            )
        return dict(
            input_ids=mx.array([1, 2, 3]),
            attention_mask=mx.array([7, 8, 9]),
            pixel_values=mx.zeros((4, 5, 6)) if images else [],
        )


@pytest.mark.parametrize(
    "prompt,with_image,error,pad_token",
    [
        ("test", False, False, None),
        ("<image>", True, False, "[EOS]"),
        ("test <image>", True, False, "[EOS]"),
        ("test without image token", True, True, "[EOS]"),
        ("test with <image> token", False, False, "[EOS]"),
    ],
)
def test_prepare_inputs(prompt, with_image, error, pad_token):
    processor = MockProcessor(
        SimpleNamespace(input_ids=[[1, 2, 3]], attention_mask=[7, 8, 9])
    )
    processor.tokenizer.pad_token = pad_token
    image = mx.zeros((3, 224, 224)) if with_image else None
    if error:
        with pytest.raises(
            ValueError,
            match="Number of image tokens in prompt_token_ids.*does not match number of images",
        ):
            prepare_inputs(
                processor, prompts=prompt, images=image, image_token_index=None
            )
        return
    inputs = prepare_inputs(
        processor, prompts=prompt, images=image, image_token_index=None
    )
    assert processor.tokenizer.pad_token == "[EOS]"
    expected = [1, 2, 3] if with_image else [[1, 2, 3]]
    assert mx.array_equal(inputs["input_ids"], mx.array(expected))
    if prompt == "test <image>":
        assert mx.array_equal(inputs["pixel_values"], mx.zeros((4, 5, 6)))
        assert mx.array_equal(inputs["attention_mask"], mx.array([7, 8, 9]))


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize(
    "options,expected_length",
    [
        ({}, 10),
        ({"truncation": True, "max_length": 4}, 4),
        ({"truncation": True}, 6),
        ({"truncation": False, "max_length": 4}, 10),
        ({"padding": "max_length", "max_length": 12}, 12),
    ],
    ids=["default", "truncate", "model-limit", "no-truncation", "fixed-padding"],
)
def test_prepare_inputs_text_length_options(
    wrapped, padding_side, options, expected_length
):
    tokens = "[UNK] [PAD] hello world one two three four five six".split()
    tokenizer = fast_tokenizer(tokens, model_max_length=6)
    prompts = ["hello world one two three four five six hello world", "hello world"]
    expected = tokenizer(
        prompts,
        **{
            "add_special_tokens": False,
            "padding": True,
            "padding_side": padding_side,
            "return_tensors": "np",
            **options,
        },
    )
    processor = NS(tokenizer=tokenizer) if wrapped else tokenizer
    actual = prepare_inputs(
        processor, prompts=prompts, padding_side=padding_side, **options
    )

    assert actual["input_ids"].shape == (2, expected_length)
    for key in ("input_ids", "attention_mask"):
        assert mx.array_equal(actual[key], mx.array(expected[key]))


def test_prepare_inputs_text_options_do_not_forward_media_or_generation_options():
    tokenizer = fast_tokenizer()
    spy = Mock(wraps=tokenizer, pad_token=tokenizer.pad_token)
    unrelated = dict(
        min_pixels=1024,
        max_pixels=4096,
        fps=2.0,
        nframes=4,
        temperature=0.7,
        max_tokens=16,
    )
    actual = prepare_inputs(
        NS(tokenizer=spy),
        prompts="hello hello hello hello hello hello",
        truncation=True,
        max_length=4,
        **unrelated,
    )

    assert actual["input_ids"].shape == (1, 4)
    assert unrelated.keys().isdisjoint(spy.call_args.kwargs)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "call,expected,error",
    [
        (lambda: (), (), None),
        (lambda *, max_length=None: (max_length,), (2,), None),
        (
            lambda truncation=False, max_length=None, /: (truncation, max_length),
            (False, None),
            None,
        ),
        (lambda: (), (), TypeError),
        (lambda: (), (), ValueError),
    ],
    ids=["strict", "max-length", "positional", "signature-type", "signature-value"],
)
def test_prepare_inputs_text_custom_length_parameters(wrapped, call, expected, error):
    result = NS(input_ids=[[1, 2]], attention_mask=[[1, 1]])

    class Tokenizer:
        pad_token = "[PAD]"

        # Inspect and enforce the delegated length-option signature.
        @wraps(call)
        def __call__(self, text, **kwargs):
            for key in (
                "add_special_tokens",
                "padding",
                "padding_side",
                "return_tensors",
            ):
                kwargs.pop(key)
            assert call(**kwargs) == expected
            return result

    tokenizer = Tokenizer()
    context = (
        patch("mlx_vlm.utils.inspect.signature", side_effect=error)
        if error
        else nullcontext()
    )
    with context:
        actual = prepare_inputs(
            NS(tokenizer=tokenizer) if wrapped else tokenizer,
            prompts="hello hello",
            truncation=True,
            max_length=2,
        )
    assert actual["input_ids"].tolist() == result.input_ids
    assert actual["attention_mask"].tolist() == result.attention_mask


@pytest.mark.parametrize(
    "options,expected_length",
    [
        ({}, 2),
        ({"truncation": False, "max_length": 4}, 6),
        ({"truncation": None, "max_length": None}, 6),
    ],
    ids=["custom-defaults", "disabled", "explicit-none"],
)
def test_prepare_inputs_text_preserves_custom_length_defaults(options, expected_length):
    class DefaultLengthTokenizer(PreTrainedTokenizerFast):
        def __call__(self, text, truncation=True, max_length=2, **kwargs):
            return super().__call__(
                text, truncation=truncation, max_length=max_length, **kwargs
            )

    tokenizer = fast_tokenizer(tokenizer_class=DefaultLengthTokenizer)
    actual = prepare_inputs(tokenizer, prompts="hello " * 6, **options)
    assert actual["input_ids"].shape == (1, expected_length)


def test_prepare_inputs_preserves_mlx_attention_mask_for_thread_handoff():
    attention_mask = mx.array([[1, 1]], dtype=mx.int32)

    class Processor:
        tokenizer = SimpleNamespace(pad_token="[PAD]", eos_token="[EOS]")

        def __call__(self, text=None, images=None, padding=None, return_tensors="mlx"):
            return {
                "input_ids": mx.array([[1, 2]], dtype=mx.int32),
                "attention_mask": attention_mask,
                "pixel_values": mx.zeros((1, 2), dtype=mx.float32),
            }

    inputs = prepare_inputs(
        Processor(), prompts="test <image>", images=mx.zeros((3, 8, 8))
    )
    consumed = []

    def consume_attention_mask():
        consumed.append(inputs["attention_mask"].tolist())

    worker = Thread(target=consume_attention_mask)
    worker.start()
    worker.join(timeout=1)

    assert inputs["attention_mask"] is attention_mask
    assert consumed == [[[1, 1]]]


@pytest.mark.parametrize("name", ["qwen3_vl", "qwen2_5_vl"])
def test_prepare_inputs_image_limits_are_per_request(name):
    processor = _make_processor(name)
    processor.image_processor = m.qwen3.Qwen3VLImageProcessor(
        min_pixels=1024, max_pixels=1048576, patch_size=16, merge_size=2
    )
    processor.tokenizer.special_tokens = {processor.image_token: 100}
    for options, expected_tokens in [
        ({}, 4),
        (dict(min_pixels=16384), 16),
        (dict(max_pixels=1024), 1),
        ({}, 4),
    ]:
        inputs = prepare_inputs(
            processor,
            prompts=processor.image_token,
            images=[Image.new("RGB", (64, 64))],
            temperature=0.7,
            max_tokens=8,
            **options,
        )
        assert inputs["input_ids"].shape[-1] == expected_tokens
        assert int(inputs["image_grid_thw"][0].prod()) // 4 == expected_tokens
        assert not (
            {"min_pixels", "max_pixels", "temperature", "max_tokens"}
            & processor.tokenizer.last_kwargs.keys()
        )
    assert processor.image_processor.min_pixels == 1024
    assert processor.image_processor.max_pixels == 1048576


def _make_test_image_bytes():
    """Create a small valid PNG in memory."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (4, 4), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.mark.parametrize(
    "kind", ["pil", "data-uri", "http", "bad-data-uri", "missing-path"]
)
def test_load_image(kind):
    source = Image.new("RGBA", (4, 4), color="red")
    payload = _make_test_image_bytes().read()
    if kind == "data-uri":
        source = "data:image/png;base64," + base64.b64encode(payload).decode()
    elif kind == "http":
        source = "https://example.com/image.png"
    elif kind in ("bad-data-uri", "missing-path"):
        source, error = (
            ("data:image/png;base64NOCOMMA", "missing comma separator")
            if kind == "bad-data-uri"
            else (Path("/nonexistent/path/image.png"), "Failed to load image")
        )
        with pytest.raises(ValueError, match=error):
            load_image(source)
        return
    response = MagicMock(content=payload)
    response.__enter__.return_value = response
    with patch("mlx_vlm.utils.requests.get", return_value=response):
        image = load_image(source)
    assert image.mode == "RGB" and image.size == (4, 4)


class TestProcessImage:
    def _image(self, width=640, height=480):
        from PIL import Image

        return Image.new("RGB", (width, height), color=(120, 40, 200))

    def test_resize_shape_applied_without_custom_processor(self):
        img = process_image(self._image(), (320, 320), None)
        assert max(img.size) <= 320

    def test_resize_shape_ignored_with_custom_processor_warns(self):
        from mlx_vlm.models.base import BaseImageProcessor

        class DummyProcessor(BaseImageProcessor):
            def preprocess(self, images):
                return images

        original = self._image()
        with pytest.warns(UserWarning, match="resize_shape.*DummyProcessor"):
            img = process_image(original, (320, 320), DummyProcessor())

        assert img.size == original.size


class TestEstimateNumImageTokens:
    def _processor(self):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        return Qwen3VLImageProcessor()

    def _actual_tokens(self, processor, width, height, **kwargs):
        import numpy as np
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(9, 30, 51))
        grid = processor([img], **kwargs)["image_grid_thw"][0]
        return int(np.prod(grid)) // processor.merge_size**2

    @pytest.mark.parametrize(
        "width,height", [(64, 64), (640, 480), (1000, 1400), (2500, 1200), (333, 517)]
    )
    def test_estimate_matches_actual_processing(self, width, height):
        processor = self._processor()
        estimate = estimate_num_image_tokens(processor, height, width)
        assert estimate == self._actual_tokens(processor, width, height)

    def test_estimate_matches_actual_with_resized_dimensions(self):
        processor = self._processor()
        estimate = estimate_num_image_tokens(
            processor, 1400, 1000, resized_height=448, resized_width=448
        )
        assert estimate == self._actual_tokens(
            processor, 1000, 1400, resized_height=448, resized_width=448
        )

    def test_unsupported_processor_raises(self):
        with pytest.raises(NotImplementedError, match="num_image_tokens"):
            estimate_num_image_tokens(SimpleNamespace(), 480, 640)


class TestMiMoV2Processor:
    def test_processor_attributes(self):
        from mlx_vlm.models.mimo_v2.processing import MiMoV2Processor

        assert MiMoV2Processor.get_attributes() == [
            "image_processor",
            "tokenizer",
            "video_processor",
        ]

    def test_audio_codes_expand_placeholders_by_grouped_length(self):
        from mlx_vlm.models.mimo_v2.processing import MiMoV2Processor
        from mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

        processor = object.__new__(MiMoV2Processor)
        processor.audio_token = "<|audio_pad|>"
        processor._audio_tokenizer = SimpleNamespace(
            encode=lambda *args, **kwargs: mx.zeros((20, 5), dtype=mx.int32)
        )
        captured = {}

        def process(*args, **kwargs):
            captured.update(kwargs)
            return {}

        with patch.object(Qwen2_5_VLProcessor, "__call__", side_effect=process):
            result = processor(
                text=["before<|audio_pad|>after"],
                audio=np.zeros(1600, dtype=np.float32),
            )

        assert captured["text"] == ["before<|audio_pad|><|audio_pad|>after"]
        assert result["audio_codes"].shape == (5, 20)
        assert result["audio_code_lengths"] == [5]

    def test_audio_codes_preserve_batch_boundaries(self):
        from mlx_vlm.models.mimo_v2.processing import MiMoV2Processor
        from mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

        processor = object.__new__(MiMoV2Processor)
        processor.audio_token = "<|audio_pad|>"

        def encode(item, **kwargs):
            length = 5 if item == "first" else 3
            offset = 0 if item == "first" else 100
            return mx.arange(20 * length).reshape(20, length) + offset

        processor._audio_tokenizer = SimpleNamespace(encode=encode)
        captured = {}

        def process(*args, **kwargs):
            captured.update(kwargs)
            return {}

        with patch.object(Qwen2_5_VLProcessor, "__call__", side_effect=process):
            result = processor(
                text=["a<|audio_pad|>", "b<|audio_pad|>"],
                audio=["first", "second"],
            )

        assert captured["text"] == [
            "a<|audio_pad|><|audio_pad|>",
            "b<|audio_pad|>",
        ]
        assert result["audio_codes"].shape == (8, 20)
        assert result["audio_code_lengths"] == [5, 3]
        assert result["audio_codes"][5, 0].item() == 100


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """A deterministic 600-frame 64x64 clip at 30 fps, i.e. 20 seconds."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    for i in range(600):
        writer.write(np.full((64, 64, 3), i % 256, np.uint8))
    writer.release()
    return str(path)


class _AttributeVideoProcessor:
    """A video processor naming its knobs the way load_video does."""

    fps = 1.0
    min_frames = 8
    max_frames = 100


class TestLoadVideo:
    def test_unknown_keyword_is_rejected(self, synthetic_video):
        with pytest.raises(TypeError, match="fpss"):
            load_video(synthetic_video, fpss=1.0)

    def test_timestamps_span_the_clip_at_the_source_frame_rate(self, synthetic_video):
        _, metadata = load_video(synthetic_video, fps=1.0)
        assert metadata.timestamps[0] == pytest.approx(0.0)
        assert metadata.timestamps[-1] == pytest.approx(20.0, abs=0.05)


class TestResolveVideoSampling:
    def test_processor_beats_defaults(self):
        processor = SimpleNamespace(video_processor=_AttributeVideoProcessor())
        resolved = resolve_video_sampling(processor, {})
        assert (resolved.fps, resolved.min_frames, resolved.max_frames) == (1.0, 8, 100)


class TestVideoMetadataForwarding:
    def test_each_video_uses_its_own_sampling_rate(self):
        class Processor:
            tokenizer = SimpleNamespace(pad_token="<pad>")

            def __call__(self, text, images=None, videos=None, fps=None, **kwargs):
                self.fps = fps
                self.kwargs = kwargs
                return {
                    "input_ids": np.array([[1], [2]]),
                    "attention_mask": np.array([[1], [1]]),
                }

        processor = Processor()
        video = np.zeros((2, 3, 8, 8), dtype=np.uint8)
        samplings = []

        def load(path, sampling, frame_sampler=None):
            samplings.append(sampling)
            return video, VideoMetadata(
                total_num_frames=30,
                fps=30,
                frames_indices=[0, 29],
            )

        with patch("mlx_vlm.utils.load_video", side_effect=load):
            prepare_inputs(
                processor,
                videos=["first.mp4", "second.mp4"],
                prompts=["first", "second"],
                fps=[1, 2],
                nframes=2,
            )

        assert [sampling.fps for sampling in samplings] == [1, 2]
        assert [sampling.nframes for sampling in samplings] == [2, 2]
        assert "nframes" not in processor.kwargs

    def test_metadata_is_only_forwarded_to_declaring_processors(self):
        class Processor:
            tokenizer = SimpleNamespace(pad_token="<pad>")

            def __call__(self, text, images=None, videos=None, fps=None, **kwargs):
                self.kwargs = kwargs
                self.fps = fps
                return {"input_ids": np.array([[1]]), "attention_mask": np.array([[1]])}

        processor = Processor()
        metadata = VideoMetadata(total_num_frames=30, fps=30, frames_indices=[0, 29])
        video = np.zeros((2, 3, 32, 32), dtype=np.uint8)
        with patch("mlx_vlm.utils.load_video", return_value=(video, metadata)):
            prepare_inputs(processor, videos=["clip.mp4"], prompts="Describe this.")
        assert "video_metadata" not in processor.kwargs
        assert processor.fps == [metadata.sampled_fps]


@pytest.fixture
def embedding_gemma2_processor():
    from transformers.models.gemma4.feature_extraction_gemma4 import (
        Gemma4AudioFeatureExtractor,
    )

    from mlx_vlm.models.embedding_gemma2 import EmbeddingGemma2Processor
    from mlx_vlm.models.embedding_gemma2.image_processing_embedding_gemma2 import (
        EmbeddingGemma2ImageProcessor,
    )
    from mlx_vlm.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )

    tokens = dict(
        image_token="<|image|>",
        boi_token="<|image>",
        eoi_token="<image|>",
        audio_token="<|audio|>",
        boa_token="<|audio>",
        eoa_token="<audio|>",
        video_token="<|video|>",
    )
    tokenizer = fast_tokenizer(extra_special_tokens=tokens)
    template = (
        "{% for msg in messages %}{% for item in msg['content'] %}"
        "{% if item['type'] == 'text' %}{{ item['text'] }}"
        "{% else %}{{ '<|' + item['type'] + '|>' }}{% endif %}"
        "{% endfor %}{% endfor %}"
    )
    return EmbeddingGemma2Processor(
        tokenizer=tokenizer,
        feature_extractor=Gemma4AudioFeatureExtractor(
            feature_size=32, frame_length_ms=24, hop_length_ms=8, preemphasis=0.5
        ),
        image_processor=EmbeddingGemma2ImageProcessor(
            patch_size=4, pooling_kernel_size=2, max_soft_tokens=70
        ),
        video_processor=EmbeddingGemma2VideoProcessor(
            patch_size=4,
            pooling_kernel_size=2,
            max_soft_tokens=70,
            fps=2,
            max_frames=3,
            overflow_strategy="truncate",
            add_timestamps=False,
        ),
        chat_template=template,
        image_seq_length=70,
        audio_seq_length=96,
        audio_ms_per_token=32,
    )


def _embedding_gemma2_media():
    rng = np.random.default_rng(42)
    return (
        Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)),
        rng.normal(0, 0.1, 3200).astype(np.float32),
        rng.integers(0, 256, (2, 24, 32, 3), dtype=np.uint8),
    )


def test_embedding_gemma2_without_torch(
    tmp_path, embedding_gemma2_processor, synthetic_video
):
    """Guard file decoding, chat templates, and offline reload against Torch imports."""
    import wave

    embedding_gemma2_processor.save_pretrained(tmp_path)
    _write_configs(tmp_path, config={"model_type": "embedding_gemma2"})
    image, audio, _ = _embedding_gemma2_media()
    image.save(tmp_path / "image.png")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes((audio * 32767).astype("<i2").tobytes())
    script = textwrap.dedent("""
        import importlib.util
        import sys
        from pathlib import Path

        blocked = {"torch", "torchvision", "torchaudio", "torchcodec", "av"}
        original_find_spec = importlib.util.find_spec
        def find_spec(name, *args, **kwargs):
            return None if name.split(".")[0] in blocked else original_find_spec(name, *args, **kwargs)
        importlib.util.find_spec = find_spec
        class RejectTorch:
            def find_spec(self, fullname, path=None, target=None):
                assert fullname.split(".")[0] not in blocked, fullname
        sys.meta_path.insert(0, RejectTorch())

        import mlx.core as mx
        import numpy as np
        from mlx_vlm.models.embedding_gemma2 import EmbeddingGemma2Processor

        folder = Path(sys.argv[1])
        processor = EmbeddingGemma2Processor.from_pretrained(folder, local_files_only=True)
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": str(folder / "image.png")},
            {"type": "audio", "url": str(folder / "audio.wav")},
            {"type": "video", "url": sys.argv[2]},
        ]}]
        expected = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="np"
        )
        assert {"input_ids", "pixel_values", "input_features", "pixel_values_videos"} <= set(expected)
        processor._get_num_multimodal_tokens(image_sizes=[(24, 32)], audio_lengths=[3200])
        processor.save_pretrained(folder / "resaved")
        reloaded = type(processor).from_pretrained(folder / "resaved", local_files_only=True)
        assert processor.to_dict() == reloaded.to_dict()
        actual = reloaded.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="mlx"
        )
        for key in expected:
            assert isinstance(actual[key], mx.array), key
            np.testing.assert_array_equal(np.asarray(actual[key]), expected[key])
        assert not any(name.split(".")[0] in blocked for name in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), synthetic_video],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("floating", [False, True], ids=["uint8", "float32"])
def test_embedding_gemma2_image_options(embedding_gemma2_processor, floating):
    processor = embedding_gemma2_processor.image_processor
    image = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    if floating:
        image = image.astype(np.float32) / 255
    config = processor.to_dict()
    output = processor(
        [image],
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        return_tensors="np",
    )
    expected = image.reshape(4, 4, 6, 4, 3).transpose(0, 2, 1, 3, 4).reshape(24, 48)
    equal(output["pixel_values"][0, :24], expected)
    equal(output["pixel_values"][0, 24:], 0)
    equal(output["image_position_ids"][0, :6], [[x, 0] for x in range(6)])
    assert output["num_soft_tokens_per_image"][0] == 6
    assert processor.to_dict() == config
    assert np.isfinite(
        processor([image], do_rescale=not floating)["pixel_values"]
    ).all()


@pytest.mark.parametrize("height", [1, 3], ids=["1px-tall", "3px-tall"])
def test_embedding_gemma2_short_pil_images_stay_channels_last(height):
    # A 3-pixel-tall RGB PIL image becomes (3, W, 3) after conversion, which
    # shape-based channel inference reads as channels-first; a 1-pixel-tall
    # one becomes (1, W, 3) and loses two of its three channels. PIL pixels
    # are always HWC, so the layout must not be guessed for PIL inputs.
    from mlx_vlm.models.embedding_gemma2.image_processing_embedding_gemma2 import (
        EmbeddingGemma2ImageProcessor,
    )

    processor = EmbeddingGemma2ImageProcessor(
        patch_size=16, pooling_kernel_size=4, max_soft_tokens=280
    )
    output = processor.preprocess(
        [Image.new("RGB", (900, height))], return_tensors="np"
    )

    # Every patch keeps all 3 channels: 16 * 16 * 3, not 16 * 16 * 1.
    assert output["pixel_values"].shape[-1] == 768
    # A landscape image spans the x axis, not y (position ids are [x, y]).
    equal(output["image_position_ids"][0].max(axis=0).tolist(), [1119, 3])

    portrait = processor.preprocess(
        [Image.new("RGB", (height, 900))], return_tensors="np"
    )
    equal(portrait["image_position_ids"][0].max(axis=0).tolist(), [3, 1119])


def test_embedding_gemma2_pil_frames_stay_channels_last():
    # The video processor stacks PIL frames into (T, H, W, C) before
    # inferring the layout, so a 3-frame batch of short frames is mistaken
    # for channels-first and reaches the vision tower transposed.
    from transformers.image_utils import ChannelDimension

    from mlx_vlm.models.embedding_gemma2.image_processing_embedding_gemma2 import (
        EmbeddingGemma2ImageProcessor,
    )
    from mlx_vlm.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )

    image_processor = EmbeddingGemma2ImageProcessor(
        patch_size=16, pooling_kernel_size=4, max_soft_tokens=280
    )
    video_processor = EmbeddingGemma2VideoProcessor(
        patch_size=16, pooling_kernel_size=4, max_soft_tokens=280
    )

    height = 3
    content = np.zeros((height, 900, 3), np.uint8)
    content[:, :450, 0] = 255
    content[height // 2 :, :, 1] = 255
    frames = [Image.fromarray(content) for _ in range(3)]

    video = video_processor.preprocess(
        [frames], return_tensors="np", do_sample_frames=False
    )
    reference = image_processor.preprocess(
        frames,
        return_tensors="np",
        input_data_format=ChannelDimension.LAST,
    )
    equal(video["pixel_values_videos"], reference["pixel_values"])
    equal(video["video_position_ids"], reference["image_position_ids"])

    # Unambiguous numpy frame stacks (height not in (1, 3, 4)) infer HWC and
    # match the PIL result.
    tall = [Image.fromarray(np.zeros((40, 900, 3), np.uint8)) for _ in range(3)]
    raw_video = video_processor.preprocess(
        [np.stack([np.asarray(frame) for frame in tall])],
        return_tensors="np",
        do_sample_frames=False,
    )
    pil_video = video_processor.preprocess(
        [tall], return_tensors="np", do_sample_frames=False
    )
    equal(raw_video["pixel_values_videos"], pil_video["pixel_values_videos"])

    # An ambiguous *array* stack is still resolved by shape inference: raw
    # arrays carry no layout guarantee, so their behaviour is unchanged.
    ambiguous = np.zeros((3, 3, 900, 3), np.uint8)
    forced_first = video_processor.preprocess(
        [ambiguous],
        return_tensors="np",
        do_sample_frames=False,
        input_data_format=ChannelDimension.FIRST,
    )
    inferred = video_processor.preprocess(
        [ambiguous], return_tensors="np", do_sample_frames=False
    )
    equal(inferred["pixel_values_videos"], forced_first["pixel_values_videos"])


def test_embedding_gemma2_chat_audio_from_video(
    embedding_gemma2_processor, synthetic_video, monkeypatch
):
    processor = embedding_gemma2_processor
    assert "<|audio|>" in processor.apply_chat_template(
        _message({"type": "audio", "url": "unopened.wav"})
    )
    _, audio, _ = _embedding_gemma2_media()
    requested = []

    def load_audio(value, sampling_rate):
        if isinstance(value, str) and value == synthetic_video:
            requested.append((value, sampling_rate))
            return audio
        return value

    monkeypatch.setattr(processor, "_load_audio", load_audio)
    conversation = _message({"type": "video", "url": synthetic_video})
    before = deepcopy(conversation)
    result = processor.apply_chat_template(
        conversation,
        load_audio_from_video=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )
    assert requested == [(synthetic_video, 16000)]
    assert {"input_features", "pixel_values_videos"} <= set(result)
    assert conversation == before


@pytest.mark.parametrize(
    "modality", ["text", "image", "audio", "video", "mixed", "nested"]
)
def test_embedding_gemma2_processor_round_trip(
    tmp_path, embedding_gemma2_processor, modality
):
    p = embedding_gemma2_processor
    image, audio, video = _embedding_gemma2_media()
    inputs = {
        "text": dict(text=["hello", "hello hello"]),
        "image": dict(images=[image]),
        "audio": dict(audio=[audio]),
        "video": dict(videos=[video]),
        "mixed": dict(images=[[image]], audio=[[audio]], videos=[[video]]),
        "nested": dict(
            images=[[image, image], [image]], audio=[[audio], [audio, audio]]
        ),
    }[modality]
    expected = p(**inputs, return_tensors="np")
    p.save_pretrained(tmp_path)
    _write_configs(tmp_path, config={"model_type": "embedding_gemma2"})
    loaded = load_processor(tmp_path, add_detokenizer=False, local_files_only=True)
    assert type(loaded) is type(p)
    assert loaded.to_dict() == p.to_dict()
    assert loaded.chat_template == p.chat_template
    for backend in ("np", "mlx"):
        actual = loaded(**inputs, return_tensors=backend)
        assert set(actual) == set(expected)
        for key in expected:
            equal(np.asarray(actual[key]), expected[key])
            if backend == "mlx":
                assert isinstance(actual[key], mx.array)


@pytest.mark.parametrize(
    "fps,strategy,expected",
    [
        (2, "uniform", [0, 4, 8]),
        (2, "truncate", [0, 2, 4]),
        (None, "uniform", [0, 5, 10]),
        (None, "truncate", [0, 1, 2]),
    ],
)
def test_embedding_gemma2_video_sampling(
    embedding_gemma2_processor, fps, strategy, expected
):
    from transformers.video_utils import VideoMetadata as HFVideoMetadata

    metadata = HFVideoMetadata(total_num_frames=11, fps=4, duration=2.75)
    actual = embedding_gemma2_processor.video_processor.sample_frames(
        metadata, fps=fps, max_frames=3, overflow_strategy=strategy
    )
    equal(actual, expected)


def test_embedding_gemma2_mixed_chat_and_metadata(embedding_gemma2_processor):
    from transformers.video_utils import VideoMetadata as HFVideoMetadata

    p = embedding_gemma2_processor
    image, audio, video = _embedding_gemma2_media()
    conversation = _message(
        {"type": "audio", "audio": audio},
        {"type": "text", "text": " hello "},
        {"type": "image", "image": image},
        {"type": "video", "video": video},
    )
    templated = p.apply_chat_template(
        conversation, tokenize=True, return_dict=True, return_tensors="mlx", fps=None
    )
    direct = p(
        text="<|audio|> hello <|image|><|video|>",
        images=[[image]],
        audio=[audio],
        videos=[video],
        return_tensors="mlx",
        fps=None,
    )
    for key in direct:
        equal(np.asarray(templated[key]), np.asarray(direct[key]))
    metadata = HFVideoMetadata(total_num_frames=2, fps=2, duration=1)
    result = p(
        videos=[video],
        video_metadata=[metadata],
        add_timestamps=True,
        return_metadata=True,
        return_text_replacement_offsets=True,
        return_tensors="mlx",
    )
    assert isinstance(result["input_ids"], mx.array)
    equal(result["video_metadata"][0].frames_indices, [0, 1])
    assert result["text_replacement_offsets"][0][0]["type"] == "video"


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_embedding_gemma2_missing_media_is_rejected(
    embedding_gemma2_processor, modality
):
    with pytest.raises(ValueError, match="no .* passed"):
        embedding_gemma2_processor(text=f"<|{modality}|>", return_tensors="mlx")


@pytest.mark.parametrize("quantize", [False, True])
def test_embedding_gemma2_conversion_saves_processor(
    tmp_path, embedding_gemma2_processor, quantize, synthetic_video
):
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from mlx_vlm import convert, load
    from mlx_vlm.models import embedding_gemma2
    from mlx_vlm.tests.test_models import DATA as MODEL_DATA
    from mlx_vlm.tests.test_models import build_config

    case = next(c for c in MODEL_DATA["cases"] if c["module"] == "embedding_gemma2")
    config = build_config(embedding_gemma2, case["config"])
    config.text_config.hidden_size = 32
    config.text_config.intermediate_size = 64
    config.text_config.hidden_size_per_layer_input = 32
    config.text_config.head_dim = 16
    config.text_config.per_layer_config = {"01": {"head_dim": 32}}
    model = embedding_gemma2.Model(config)
    source, destination = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    _write_configs(source, config=asdict(model.config))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    p = embedding_gemma2_processor
    p.save_pretrained(source)
    convert(
        str(source),
        str(destination),
        dtype="bfloat16",
        quantize=quantize,
        q_bits=8,
        q_group_size=32,
    )
    loaded_model, loaded_processor = load(destination, local_files_only=True)
    assert type(loaded_processor) is type(p)
    assert loaded_processor.to_dict() == p.to_dict()
    assert loaded_processor.chat_template == p.chat_template
    loaded_processor.save_pretrained(tmp_path / "resaved")
    image, audio, video = _embedding_gemma2_media()
    inputs = dict(
        images=[[image]], audio=[[audio]], videos=[[video]], return_tensors="np"
    )
    actual = loaded_processor(**inputs)
    for key, expected in p(**inputs).items():
        equal(actual[key], expected)
    file_inputs = dict(
        videos=[synthetic_video], add_timestamps=True, return_tensors="np"
    )
    file_actual = loaded_processor(**file_inputs)
    for key, expected in p(**file_inputs).items():
        equal(file_actual[key], expected)
    text = loaded_processor(text="hello", return_tensors="mlx")
    output = loaded_model(**text).text_embeds
    assert output.shape == (1, 24)
    assert mx.all(mx.isfinite(output)).item()


@pytest.fixture
def embedding_gemma2_decoder():
    from mlx_vlm.models.embedding_gemma2 import _video_decoder as decoder

    try:
        decoder._load_ffmpeg()
    except decoder.VideoDecodeUnavailable as exc:
        pytest.skip(str(exc))
    return decoder


@pytest.mark.parametrize("library", ["avformat", "avcodec", "avutil"])
def test_embedding_gemma2_rejects_unknown_ffmpeg_abi(tmp_path, monkeypatch, library):
    from mlx_vlm.models.embedding_gemma2 import _video_decoder as decoder

    libraries = {}
    for stem, major in (("avformat", 61), ("avcodec", 61), ("avutil", 59)):
        path = tmp_path / f"lib{stem}.dylib"
        path.touch()
        lib = Mock()
        getattr(lib, stem + "_version").return_value = (major + (stem == library)) << 16
        libraries[str(path)] = lib
    monkeypatch.setattr(decoder.C, "CDLL", lambda path: libraries[path])
    with patch.object(decoder._FFmpeg, "bind") as bind:
        with pytest.raises(decoder.VideoDecodeUnavailable, match="Unsupported .* ABI"):
            decoder._FFmpeg(tmp_path)
        bind.assert_not_called()  # Reject before any native decoder operation.


@pytest.mark.parametrize("reason", ["libraries", "metal", "platform"])
def test_embedding_gemma2_missing_native_backend(tmp_path, monkeypatch, reason):
    import cv2

    from mlx_vlm.models.embedding_gemma2 import _video_decoder as decoder

    monkeypatch.setattr(decoder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(decoder.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mx.metal, "is_available", lambda: reason != "metal")
    monkeypatch.setattr(cv2, "__file__", str(tmp_path / "cv2" / "__init__.py"))
    if reason == "platform":
        monkeypatch.setattr(decoder.platform, "machine", lambda: "x86_64")
    # Bypass the cache so a previously loaded bundle cannot hide missing support.
    with pytest.raises(decoder.VideoDecodeUnavailable):
        decoder._load_ffmpeg.__wrapped__()


@pytest.mark.parametrize("failure", ["environment", "format", "kernel"])
def test_embedding_gemma2_video_fallback(
    embedding_gemma2_processor, synthetic_video, monkeypatch, failure
):
    from mlx_vlm.models.embedding_gemma2 import _video_decoder as decoder
    from mlx_vlm.models.embedding_gemma2 import _video_kernel as kernel

    def unavailable(*args, **kwargs):
        raise decoder.VideoDecodeUnavailable("Unsupported test backend")

    if failure == "environment":
        monkeypatch.setattr(decoder, "_load_ffmpeg", unavailable)
    else:
        try:
            decoder._load_ffmpeg()
        except decoder.VideoDecodeUnavailable as exc:
            pytest.skip(str(exc))
        if failure == "format":
            original = decoder.Decoder.copy_frame
            calls = 0

            def copy_frame(self, frame):
                nonlocal calls
                calls += 1
                if calls > 1:
                    unavailable()
                return original(self, frame)

            monkeypatch.setattr(decoder.Decoder, "copy_frame", copy_frame)
        else:

            def kernel_error(*args):
                raise RuntimeError("Metal compiler failure")

            monkeypatch.setattr(kernel, "yuv_to_rgb", kernel_error)
    sampler = lambda metadata, **kw: np.array([0, 59, 599])
    expected, info = load_video(synthetic_video, frame_sampler=sampler)
    actual, metadata = embedding_gemma2_processor.video_processor._decode_video(
        synthetic_video, sampler
    )
    equal(actual, expected.transpose(0, 2, 3, 1))
    equal(metadata.frames_indices, info.frames_indices)
    assert metadata.video_backend == "opencv"
    assert metadata.fps == info.fps


def test_embedding_gemma2_native_seek_and_cleanup(
    embedding_gemma2_decoder, synthetic_video
):
    module = embedding_gemma2_decoder
    with module.Decoder(synthetic_video) as decoder:
        last, _ = decoder.get(599)
        first, _ = decoder.get(0)
        repeated, _ = decoder.get(599)
        for actual, expected in zip(repeated, last):
            equal(actual, expected)
        assert not np.array_equal(first[0], last[0])
        assert decoder.metadata == dict(total_num_frames=600, fps=30, duration=20)
    decoder.close()  # Closing twice is harmless, including after partial failure.
    assert not any(
        (decoder.context, decoder.codec_context, decoder.frame, decoder.packet)
    )
    failed = module.Decoder.__new__(module.Decoder)
    with patch.object(module._load_ffmpeg(), "avcodec_open2", return_value=-22):
        with pytest.raises(module.VideoDecodeUnavailable):
            failed.__init__(synthetic_video)
    assert not any((failed.context, failed.codec_context, failed.frame, failed.packet))


@pytest.mark.parametrize("indices", [[], [-1], [600]])
def test_embedding_gemma2_native_invalid_sampling(
    embedding_gemma2_decoder, synthetic_video, indices
):
    with pytest.raises(ValueError, match="Frame indices"):
        embedding_gemma2_decoder.decode_video(synthetic_video, lambda metadata: indices)


@pytest.mark.parametrize("layout", ["rgb", "hdr", "interlaced", "stride", "odd"])
def test_embedding_gemma2_native_unsupported_frame(
    embedding_gemma2_decoder, synthetic_video, layout
):
    module = embedding_gemma2_decoder
    with module.Decoder(synthetic_video) as decoder:
        frame = module.Frame(width=16, height=8)
        frame.format = decoder.api.av_get_pix_fmt(
            b"rgb24" if layout == "rgb" else b"yuv420p"
        )
        frame.color_trc = 16 if layout == "hdr" else 2
        frame.flags = 8 if layout == "interlaced" else 0
        if layout == "odd":
            frame.width = 17
        with pytest.raises(module.VideoDecodeUnavailable):
            decoder.copy_frame(frame)


@pytest.mark.parametrize("mov", [False, True])
def test_embedding_gemma2_variable_packet_timing(mov):
    from mlx_vlm.models.embedding_gemma2 import _video_decoder as module

    decoder = module.Decoder.__new__(module.Decoder)
    storage = module.Packet()
    decoder.packet = module.C.pointer(storage)
    decoder.context = None
    decoder.index = 0
    decoder.time_base = (1, 10)
    decoder.is_mov = mov
    # Decode order differs from presentation order, as in a video with B-frames.
    packets = iter([(0, 2, -2), (9, 2, 0), (2, 2, 3), (5, 2, 7)])

    def read_packet(context, packet):
        try:
            storage.pts, storage.duration, storage.dts = next(packets)
        except StopIteration:
            return module.EOF
        return 0

    decoder.api = SimpleNamespace(
        av_read_frame=read_packet,
        av_packet_unref=lambda p: None,
        check=lambda result: result,
    )
    timeline = decoder.scan()
    assert [p[0] for p in timeline] == [0, 2, 5, 9]
    assert [p[1] for p in timeline] == ([2, 4, 2, 3] if mov else [2, 2, 2, 2])
    assert decoder.metadata["duration"] == pytest.approx(1.2 if mov else 1.1)


@pytest.mark.parametrize(
    "fmt,depth,chroma_y,width,expected",
    [
        ("yuv420p", 8, 1, 32, [254, 0, 0]),
        ("yuv420p", 8, 1, 18, [252, 0, 0]),
        ("yuvj420p", 8, 1, 32, [238, 14, 14]),
        ("yuvj420p", 8, 1, 18, [238, 15, 14]),
        ("yuv420p10le", 10, 1, 32, [252, 0, 0]),
        ("yuv422p10le", 10, 0, 18, [252, 0, 0]),
    ],
)
def test_embedding_gemma2_metal_color_rounding(fmt, depth, chroma_y, width, expected):
    """Golden RGB from FFmpeg 8.1.2, rawvideo -> rgb24 with sws_flags=0."""
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    from mlx_vlm.models.embedding_gemma2._video_kernel import yuv_to_rgb

    dtype = np.uint8 if depth == 8 else np.uint16
    planes = [np.full((6, width), 81 << (depth - 8), dtype)] + [
        np.full((6 >> chroma_y, width // 2), value << (depth - 8), dtype)
        for value in (90, 240)
    ]
    actual = np.asarray(
        yuv_to_rgb(
            planes,
            dict(
                colorspace=2,
                format=fmt,
                depth=depth,
                log2_chroma_w=1,
                log2_chroma_h=chroma_y,
            ),
        )
    )
    equal(actual, np.broadcast_to(np.array(expected, dtype=np.uint8), (6, width, 3)))


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_embedding_gemma2_prepare_inputs_routes_media(
    embedding_gemma2_processor, modality
):
    p = embedding_gemma2_processor
    image, audio, _ = _embedding_gemma2_media()
    media = {"images": [image]} if modality == "image" else {"audio": [audio]}
    prompt = f"<|{modality}|> hello"
    expected = p(text=[prompt], **media, return_tensors="mlx")
    actual = prepare_inputs(p, prompts=[prompt], **media, return_tensors="mlx")
    for key in expected:
        equal(np.asarray(actual[key]), np.asarray(expected[key]))


class TestMiniCPMV47Processor:
    @staticmethod
    def processor():
        from mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7 import (
            MiniCPMV4_7Processor,
            MiniCPMVImageProcessor,
            MiniCPMVVideoProcessor,
        )

        options = dict(PROFILES["minicpm_image"])
        return MiniCPMV4_7Processor(
            tokenizer=ProcessorTokenizer(**PROFILES["minicpm_tokenizer"]),
            image_processor=MiniCPMVImageProcessor(**options),
            video_processor=MiniCPMVVideoProcessor(**options),
        )

    @pytest.mark.parametrize("mode,count", [("16x", 1), ("4x", 4)])
    def test_mixed_media_and_mode(self, mode, count):
        processor = self.processor()
        processor.image_processor.use_image_id = True
        inputs = processor(
            text="<|video_pad|> then <|image_pad|>",
            images=[Image.new("RGB", (56, 56), "red")],
            videos=[np.zeros((2, 3, 56, 56), dtype=np.uint8)],
            downsample_mode=mode,
        )
        assert inputs["downsample_mode"] == mode
        assert np.count_nonzero(inputs["mm_token_type_ids"] == 2) == 2 * count
        assert np.count_nonzero(inputs["mm_token_type_ids"] == 1) == count
        assert len(inputs["pixel_values"][0]) == 3
        assert inputs["input_ids"][0, 0] == processor.tokenizer.im_start_id
        assert (
            np.count_nonzero(inputs["input_ids"] == processor.tokenizer.im_id_start_id)
            == 1
        )
        assert processor.image_processor.downsample_mode == "16x"
        np.testing.assert_array_equal(inputs["pixel_values"][0][0], -1)
        assert np.any(inputs["pixel_values"][0][-1] == 1)

    def test_placeholder_validation_and_thinking(self):
        processor = self.processor()
        with pytest.raises(ValueError, match="placeholders"):
            processor(text="Describe", images=[Image.new("RGB", (56, 56))])
        with pytest.raises(ValueError, match="placeholders"):
            processor(text="<|image_pad|>")
        prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        assert processor._strip_think_prefix(prompt) == prompt
        processor.tokenizer.apply_chat_template = Mock(return_value=prompt)
        assert processor.apply_chat_template([], enable_thinking=False) == prompt
        processor.tokenizer.apply_chat_template.assert_called_once_with(
            [], enable_thinking=False
        )

    def test_save_load_roundtrip(self, tmp_path):
        from mlx_vlm.models.minicpmv4_7 import MiniCPMV4_7Processor

        processor = self.processor()
        tokens = processor.tokenizer.special_tokens
        backend = Tokenizer(
            WordLevel(
                vocab={token: i for i, token in enumerate(tokens)}, unk_token="<unk>"
            )
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            eos_token="</image>",
            pad_token="<unk>",
            additional_special_tokens=list(tokens),
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message.content }}{% endfor %}"
        )
        processor.tokenizer = tokenizer
        processor.video_processor.max_num_frames = 7
        processor.image_processor.downsample_mode = "4x"
        processor.save_pretrained(tmp_path)
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "minicpmv4_7", "downsample_mode": "16x"})
        )
        loaded = AutoProcessor.from_pretrained(tmp_path)
        assert isinstance(loaded, MiniCPMV4_7Processor)
        assert loaded.image_processor.downsample_mode == "4x"
        assert loaded.video_processor.max_num_frames == 7
        assert loaded.tokenizer.chat_template == tokenizer.chat_template
        result = loaded(text="<|image_pad|>", images=[Image.new("RGB", (56, 56))])
        assert np.count_nonzero(result["mm_token_type_ids"] == 1) == 4
