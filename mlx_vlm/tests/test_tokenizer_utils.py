"""Tests for tokenizer_utils module."""

import importlib
import json
import re
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

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
from mlx_vlm.utils import should_add_special_tokens


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
