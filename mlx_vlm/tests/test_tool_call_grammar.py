"""Schema-constrained tool calls for the GLM-4.7 / Flint ``<arg_key>`` format."""

import pytest

pytest.importorskip("llguidance")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

import llguidance  # noqa: E402
import llguidance.hf  # noqa: E402
import llguidance.numpy  # noqa: E402

from mlx_vlm.structured import (  # noqa: E402
    LLGuidanceLogitsProcessor,
    build_glm47_tool_call_grammar,
    build_tool_call_logits_processor,
)

MARKERS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "number"},
                },
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "background": {"type": "boolean"},
                },
            },
        },
    },
]


@pytest.fixture(scope="module")
def tokenizer():
    """Tiny offline byte-level tokenizer with the Flint/GLM tool markers."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|end_of_text|>"
    )
    fast.add_tokens(MARKERS)
    fast.add_special_tokens({"additional_special_tokens": ["<|observation|>"]})
    return fast


def _matcher(tokenizer, grammar):
    llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
    assert llguidance.LLMatcher.validate_grammar(grammar, llg_tokenizer) == ""
    return llguidance.LLMatcher(llg_tokenizer, grammar), llg_tokenizer


def _accepts(tokenizer, grammar, text):
    matcher, _ = _matcher(tokenizer, grammar)
    for token in tokenizer.encode(text, add_special_tokens=False):
        matcher.consume_token(token)
        if matcher.is_error():
            return False
    return matcher.is_accepting()


def _call(name, *args):
    body = "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args)
    return f"<tool_call>{name}{body}</tool_call>"


class TestGlm47ToolCallGrammar:
    @pytest.mark.parametrize(
        "text",
        [
            "Plain answer, no tools.",
            "<think>plan the edit</think>Done.",
            _call("read", ("filePath", "/repo/a.py")),
            "<think>look first</think>Reading." + _call("read", ("filePath", "/a")),
            _call("bash", ("command", "ls -la\npwd"))
            + _call("read", ("filePath", "/b")),
            _call("read", ("filePath", "/a"), ("offset", "-3"), ("limit", "2.5e1")),
            _call("bash", ("command", "sleep 1"), ("background", "true")),
            _call("bash"),
        ],
    )
    def test_accepts_valid_output(self, tokenizer, text):
        grammar = build_glm47_tool_call_grammar(tokenizer, TOOLS)
        assert _accepts(tokenizer, grammar, text)

    @pytest.mark.parametrize(
        "text",
        [
            _call("write_file", ("filePath", "/a")),
            _call("read", ("path", "/a")),
            _call("bash", ("filePath", "/a")),
            _call("read", ("offset", "ten")),
            _call("read", ("limit", "1.2.3")),
            _call("bash", ("background", "yes")),
            _call("read", ("offset", "1")),
            _call("read"),
            _call("read", ("filePath", "/a"), ("filePath", "/b")),
        ],
    )
    def test_rejects_invalid_calls(self, tokenizer, text):
        grammar = build_glm47_tool_call_grammar(tokenizer, TOOLS)
        assert not _accepts(tokenizer, grammar, text)

    def test_arguments_follow_schema_order(self, tokenizer):
        grammar = build_glm47_tool_call_grammar(tokenizer, TOOLS)
        in_order = _call("read", ("filePath", "/a"), ("limit", "5"))
        swapped = _call("read", ("limit", "5"), ("filePath", "/a"))
        assert _accepts(tokenizer, grammar, in_order)
        assert not _accepts(tokenizer, grammar, swapped)

    def test_only_tool_keys_can_follow_arg_key(self, tokenizer):
        grammar = build_glm47_tool_call_grammar(tokenizer, TOOLS)
        matcher, llg_tokenizer = _matcher(tokenizer, grammar)
        for token in tokenizer.encode(
            "<tool_call>read<arg_key>", add_special_tokens=False
        ):
            matcher.consume_token(token)
        mask = llguidance.numpy.allocate_token_bitmask(1, llg_tokenizer.vocab_size)
        llguidance.numpy.fill_next_token_bitmask(matcher, mask, 0)
        allowed = {
            tokenizer.decode([i])
            for i in range(llg_tokenizer.vocab_size)
            if (int(mask[0][i >> 5]) >> (i & 31)) & 1
        }
        assert allowed == {"f"}  # filePath is required and comes first

    def test_markers_use_token_ids(self, tokenizer):
        grammar = build_glm47_tool_call_grammar(tokenizer, TOOLS)
        tool_call_id = tokenizer.convert_tokens_to_ids("<tool_call>")
        assert f"<[{tool_call_id}]>" in grammar

    def test_tool_names_and_keys_are_quoted(self, tokenizer):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": 'say"hi',
                    "parameters": {"properties": {'a"b\\c': {"type": "string"}}},
                },
            }
        ]
        grammar = build_glm47_tool_call_grammar(tokenizer, tools)
        assert _accepts(tokenizer, grammar, _call('say"hi', ('a"b\\c', "x")))


def _allowed_after(processor, tokenizer, text):
    """Token ids the processor allows after `text`."""
    import mlx.core as mx

    prompt = [0]  # the first call sees the prompt, not a generated token
    ids = tokenizer.encode(text, add_special_tokens=False)
    logits = mx.zeros((1, len(tokenizer)))
    out = processor(mx.array([prompt]), logits)
    for i in range(len(ids)):
        out = processor(mx.array([prompt + ids[: i + 1]]), logits)
    return {i for i, v in enumerate(out[0].tolist()) if v != float("-inf")}


class TestStopTokens:
    """Models that end turns on several tokens must be able to stop."""

    def test_all_stopping_criteria_tokens_can_end_output(self, tokenizer):
        from types import SimpleNamespace

        observation = tokenizer.convert_tokens_to_ids("<|observation|>")
        tokenizer.stopping_criteria = SimpleNamespace(
            eos_token_ids=[tokenizer.eos_token_id, observation]
        )
        try:
            processor = build_tool_call_logits_processor(tokenizer, "glm47", TOOLS)
            allowed = _allowed_after(
                processor, tokenizer, _call("read", ("filePath", "/a"))
            )
        finally:
            del tokenizer.stopping_criteria
        assert observation in allowed

    def test_json_schema_processor_uses_stopping_criteria(self, tokenizer):
        from types import SimpleNamespace

        from mlx_vlm.structured import build_json_schema_logits_processor

        observation = tokenizer.convert_tokens_to_ids("<|observation|>")
        tokenizer.stopping_criteria = SimpleNamespace(
            eos_token_ids=[tokenizer.eos_token_id, observation]
        )
        try:
            processor = build_json_schema_logits_processor(
                tokenizer, {"type": "object", "properties": {}}
            )
            allowed = _allowed_after(processor, tokenizer, "{}")
        finally:
            del tokenizer.stopping_criteria
        assert observation in allowed


class TestBuildToolCallLogitsProcessor:
    def test_glm47_returns_processor(self, tokenizer):
        processor = build_tool_call_logits_processor(tokenizer, "glm47", TOOLS)
        assert isinstance(processor, LLGuidanceLogitsProcessor)

    @pytest.mark.parametrize("parser", [None, "json_tools", "qwen3_coder"])
    def test_unsupported_parser_returns_none(self, tokenizer, parser):
        assert build_tool_call_logits_processor(tokenizer, parser, TOOLS) is None

    @pytest.mark.parametrize("tools", [None, []])
    def test_no_tools_returns_none(self, tokenizer, tools):
        assert build_tool_call_logits_processor(tokenizer, "glm47", tools) is None

    @pytest.mark.parametrize(
        "tools",
        [
            [
                {
                    "type": "function",
                    "function": {"name": "x", "parameters": {"properties": {"": {}}}},
                }
            ],
            [{"type": "function", "function": {"name": ""}}],
            [
                {
                    "type": "function",
                    "function": {"name": "x", "parameters": {"properties": "bad"}},
                }
            ],
        ],
    )
    def test_odd_schemas_fall_back_to_none(self, tokenizer, tools):
        assert build_tool_call_logits_processor(tokenizer, "glm47", tools) is None

    def test_unwraps_processor_tokenizer(self, tokenizer):
        class Processor:
            pass

        processor = Processor()
        processor.tokenizer = tokenizer
        result = build_tool_call_logits_processor(processor, "glm47", TOOLS)
        assert isinstance(result, LLGuidanceLogitsProcessor)


class TestServerToolCallGrammar:
    @pytest.fixture
    def enabled(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_CONSTRAIN_TOOL_CALLS", "1")

    def _build(
        self,
        processor="proc",
        parser="glm47",
        tools=TOOLS,
        tool_choice=None,
        existing=None,
    ):
        from types import SimpleNamespace

        from mlx_vlm.server.request_normalization import (
            _build_tool_call_logits_processors,
        )

        calls = []

        def factory(processor, tool_parser_type, tools):
            calls.append((processor, tool_parser_type, tools))
            return "tool-processor"

        gen_args = SimpleNamespace(logits_processors=existing)
        result = _build_tool_call_logits_processors(
            gen_args,
            processor,
            parser,
            tools,
            tool_choice,
            logits_processor_factory=factory,
        )
        return result, calls

    def test_off_by_default(self, monkeypatch):
        from mlx_vlm.server.generation import get_server_constrain_tool_calls

        monkeypatch.delenv("MLX_VLM_CONSTRAIN_TOOL_CALLS", raising=False)
        assert get_server_constrain_tool_calls() is False
        assert self._build()[0] is None

    def test_enabled_builds_processor(self, enabled):
        result, calls = self._build()
        assert result == ["tool-processor"]
        assert calls == [("proc", "glm47", TOOLS)]

    def test_response_format_keeps_priority(self, enabled):
        result, calls = self._build(existing=["json-schema"])
        assert result is None and calls == []

    @pytest.mark.parametrize("tools", [None, []])
    def test_no_tools(self, enabled, tools):
        assert self._build(tools=tools)[0] is None

    def test_tool_choice_none(self, enabled):
        assert self._build(tool_choice="none")[0] is None

    def test_named_tool_choice_limits_tools(self, enabled):
        choice = {"type": "function", "function": {"name": "bash"}}
        result, calls = self._build(tool_choice=choice)
        assert result == ["tool-processor"]
        assert [t["function"]["name"] for t in calls[0][2]] == ["bash"]

    def test_unsupported_format_returns_none(self, enabled):
        from types import SimpleNamespace

        from mlx_vlm.server.request_normalization import (
            _build_tool_call_logits_processors,
        )

        gen_args = SimpleNamespace(logits_processors=None)
        assert (
            _build_tool_call_logits_processors(
                gen_args, "proc", "json_tools", TOOLS, None
            )
            is None
        )
