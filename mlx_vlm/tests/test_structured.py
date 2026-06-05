import sys
import types

import mlx.core as mx

from mlx_vlm import structured
from mlx_vlm.structured import ThinkingAwareLogitsProcessor


class TinyThinkingTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {
            "<think>": [10],
            "</think>": [20],
        }[text]


class RecordingProcessor:
    def __init__(self):
        self.calls = []

    def clone(self):
        clone = RecordingProcessor()
        clone.calls.append(("cloned", None))
        return clone

    def process_last_token(self, token, logits):
        self.calls.append(("process_last_token", token))
        return logits + 1

    def __call__(self, input_ids, logits):
        self.calls.append(("call", input_ids.tolist()))
        return logits + 2


def test_thinking_aware_processor_passes_logits_until_thinking_ends():
    inner = RecordingProcessor()
    processor = ThinkingAwareLogitsProcessor(
        inner,
        TinyThinkingTokenizer(),
        enable_thinking=True,
    )
    logits = mx.zeros((1, 3), dtype=mx.float32)

    out = processor.process_last_token(11, logits)
    mx.eval(out)
    assert out.tolist() == logits.tolist()
    assert inner.calls == []

    out = processor.process_last_token(20, logits)
    mx.eval(out)
    assert out.tolist() == (logits + 1).tolist()
    assert inner.calls == [("process_last_token", 20)]

    out = processor.process_last_token(3, logits)
    mx.eval(out)
    assert out.tolist() == (logits + 1).tolist()
    assert inner.calls[-1] == ("process_last_token", 3)


def test_thinking_aware_processor_delegates_immediately_without_thinking():
    inner = RecordingProcessor()
    processor = ThinkingAwareLogitsProcessor(
        inner,
        TinyThinkingTokenizer(),
        enable_thinking=False,
    )
    logits = mx.zeros((1, 3), dtype=mx.float32)

    out = processor(mx.array([1, 2, 3]), logits)
    mx.eval(out)

    assert out.tolist() == (logits + 2).tolist()
    assert inner.calls == [("call", [1, 2, 3])]


def test_thinking_aware_processor_clone_resets_phase_state():
    inner = RecordingProcessor()
    processor = ThinkingAwareLogitsProcessor(
        inner,
        TinyThinkingTokenizer(),
        enable_thinking=True,
    )
    processor.process_last_token(20, mx.zeros((1, 3), dtype=mx.float32))

    clone = processor.clone()
    out = clone.process_last_token(11, mx.zeros((1, 3), dtype=mx.float32))
    mx.eval(out)

    assert isinstance(clone.processor, RecordingProcessor)
    assert clone.processor is not inner
    assert clone.processor.calls == [("cloned", None)]
    assert out.tolist() == [[0.0, 0.0, 0.0]]


def test_json_schema_processor_uses_compact_whitespace_pattern(monkeypatch):
    observed = {}
    fake_llguidance = types.ModuleType("llguidance")
    fake_llguidance.__path__ = []
    fake_hf = types.ModuleType("llguidance.hf")

    class FakeJsonCompiler:
        def __init__(self, **kwargs):
            observed["compiler_kwargs"] = kwargs

        def compile(self, schema_text):
            observed["schema_text"] = schema_text
            return "compiled grammar"

    fake_llguidance.JsonCompiler = FakeJsonCompiler
    fake_llguidance.grammar_from = lambda *_args: "fallback grammar"
    fake_hf.from_tokenizer = lambda tokenizer: "llg-tokenizer"
    fake_llguidance.hf = fake_hf
    monkeypatch.setitem(sys.modules, "llguidance", fake_llguidance)
    monkeypatch.setitem(sys.modules, "llguidance.hf", fake_hf)
    structured._llg_tokenizer_cache.clear()

    processor = structured.build_json_schema_logits_processor(
        object(),
        {"type": "object"},
    )

    assert processor.grammar == "compiled grammar"
    assert observed["compiler_kwargs"] == {
        "separators": (", ", ": "),
        "whitespace_pattern": "",
    }
    assert observed["schema_text"] == '{"type": "object"}'
