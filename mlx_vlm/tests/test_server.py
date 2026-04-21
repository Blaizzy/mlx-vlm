from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.mark.parametrize("value", [224, "22", [1.0], [1.5], [True], [1, 2, 3]])
def test_chat_completions_endpoint_rejects_invalid_resize_shape(client, value):
    response = client.post(
        "/chat/completions",
        json={
            "model": "demo",
            "messages": [{"role": "user", "content": "Hello"}],
            "resize_shape": value,
        },
    )

    assert response.status_code == 422


def test_chat_request_schema_allows_one_or_two_resize_shape_values():
    resize_shape = server.ChatRequest.model_json_schema()["properties"]["resize_shape"]
    lengths = {
        (item["minItems"], item["maxItems"])
        for item in resize_shape["anyOf"]
        if item.get("type") == "array"
    }

    assert lengths == {(1, 1), (2, 2)}


def test_responses_endpoint_forwards_new_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "enable_thinking": False,
                "thinking_budget": 24,
                "thinking_start_token": "<think>",
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["enable_thinking"] is False
    assert mock_template.call_args.kwargs["thinking_budget"] == 24
    assert mock_template.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["thinking_budget"] == 24
    assert mock_generate.call_args.kwargs["thinking_start_token"] == "<think>"


def test_chat_completions_endpoint_forwards_explicit_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
                "resize_shape": [512],
            },
        )

    assert response.status_code == 200
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["resize_shape"] == (512, 512)


# ---------------------------------------------------------------------------
# Prompt cache TTL tests
# ---------------------------------------------------------------------------


def test_get_prompt_cache_ttl_default(monkeypatch):
    monkeypatch.delenv("PROMPT_CACHE_TTL", raising=False)
    assert server.get_prompt_cache_ttl() == 300


def test_get_prompt_cache_ttl_from_env(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "600")
    assert server.get_prompt_cache_ttl() == 600


def test_prompt_cache_state_touch():
    from mlx_vlm.generate import PromptCacheState

    state = PromptCacheState()
    old_time = state.last_used

    import time

    time.sleep(0.01)
    state.touch()
    assert state.last_used > old_time


def test_evict_stale_prompt_caches(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "1")
    server._prompt_cache_states.clear()
    state = server.get_prompt_cache_state("test-model")
    state.last_used = 0
    assert len(server._prompt_cache_states) == 1
    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert len(server._prompt_cache_states) == 0


def test_evict_skips_fresh_caches(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "9999")
    server._prompt_cache_states.clear()
    server.get_prompt_cache_state("fresh-model")
    evicted = server.evict_stale_prompt_caches()
    assert evicted == 0
    assert len(server._prompt_cache_states) == 1
    server._prompt_cache_states.clear()


def test_evict_disabled_when_ttl_zero(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "0")
    server._prompt_cache_states.clear()
    state = server.get_prompt_cache_state("test-model")
    state.last_used = 0
    evicted = server.evict_stale_prompt_caches()
    assert evicted == 0
    server._prompt_cache_states.clear()


# ---------------------------------------------------------------------------
# Prompt cache TTL - scenario tests
# ---------------------------------------------------------------------------


def test_telegram_idle_13_hours_evicted(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time

    now = time.time()
    state = server.get_prompt_cache_state("qwen3.5-35b")
    state.token_ids = list(range(12000))
    state.cache = ["fake_kv_layer"]
    state.last_used = now - (13 * 3600)
    state.created_at = now - (13 * 3600)

    assert len(server._prompt_cache_states) == 1
    assert state.token_count == 12000

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert len(server._prompt_cache_states) == 0

    fresh = server.get_prompt_cache_state("qwen3.5-35b")
    assert fresh.cache is None
    assert fresh.token_ids is None
    server._prompt_cache_states.clear()


def test_active_conversation_not_evicted(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time

    now = time.time()
    state = server.get_prompt_cache_state("qwen3.5-35b")
    state.token_ids = list(range(8000))
    state.cache = ["fake_kv"]

    for i in range(10):
        state.last_used = now - (30 * (10 - i))

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 0
    assert state.token_count == 8000
    server._prompt_cache_states.clear()


def test_multiple_users_only_stale_evicted(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time

    now = time.time()

    active = server.get_prompt_cache_state("model", cache_key="user-a")
    active.token_ids = list(range(5000))
    active.cache = ["kv_a"]
    active.last_used = now - 60

    stale = server.get_prompt_cache_state("model", cache_key="user-b")
    stale.token_ids = list(range(9000))
    stale.cache = ["kv_b"]
    stale.last_used = now - 600

    assert len(server._prompt_cache_states) == 2

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert "model::user-a" in server._prompt_cache_states
    assert "model::user-b" not in server._prompt_cache_states
    assert server._prompt_cache_states["model::user-a"].token_count == 5000
    server._prompt_cache_states.clear()


def test_cache_just_under_ttl_not_evicted(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time

    now = time.time()
    state = server.get_prompt_cache_state("model")
    state.token_ids = list(range(1000))
    state.cache = ["kv"]
    state.last_used = now - 299

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 0
    server._prompt_cache_states.clear()


def test_invalidated_cache_cleared_on_eviction(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "60")
    server._prompt_cache_states.clear()

    import time

    state = server.get_prompt_cache_state("model")
    state.token_ids = list(range(20000))
    state.cache = ["big_kv_layer_1", "big_kv_layer_2"]
    state.last_used = time.time() - 120

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert state.cache is None
    assert state.token_ids is None
    server._prompt_cache_states.clear()


def test_short_ttl_evicts_between_requests(monkeypatch):
    monkeypatch.setenv("PROMPT_CACHE_TTL", "5")
    server._prompt_cache_states.clear()

    import time

    state = server.get_prompt_cache_state("model")
    state.token_ids = list(range(3000))
    state.cache = ["kv"]
    state.last_used = time.time() - 10

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert len(server._prompt_cache_states) == 0
    server._prompt_cache_states.clear()


# ── Continuous batching / ResponseGenerator tests ─────────────────────


class TestResponseGenerator:
    """Tests for the ResponseGenerator continuous batching engine."""

    def test_generate_arguments_defaults(self):
        args = server.GenerationArguments()
        assert args.max_tokens == server.DEFAULT_MAX_TOKENS
        assert args.temperature == server.DEFAULT_TEMPERATURE
        assert args.enable_thinking is True
        assert args.logit_bias is None

    def test_generate_arguments_to_generate_kwargs(self):
        args = server.GenerationArguments(
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            logit_bias={3: -0.5},
            enable_thinking=False,
            thinking_budget=100,
        )
        kw = args.to_generate_kwargs()
        assert kw["max_tokens"] == 50
        assert kw["top_k"] == 40
        assert kw["min_p"] == 0.05
        assert kw["repetition_penalty"] == 1.15
        assert kw["logit_bias"] == {3: -0.5}
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 100

    def test_generate_arguments_to_template_kwargs(self):
        args = server.GenerationArguments(enable_thinking=False, thinking_budget=50)
        kw = args.to_template_kwargs()
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 50

    def test_generate_arguments_omits_none_optionals(self):
        args = server.GenerationArguments()
        kw = args.to_generate_kwargs()
        assert "repetition_penalty" not in kw
        assert "logit_bias" not in kw
        assert "thinking_budget" not in kw

    def test_build_gen_args_from_openai_request(self):
        req = SimpleNamespace(
            max_output_tokens=128,
            temperature=0.5,
            top_p=0.9,
            top_k=32,
            min_p=0.1,
            repetition_penalty=1.2,
            logit_bias={"5": -1.0},
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 128
        assert args.top_k == 32
        assert args.logit_bias == {5: -1.0}  # string keys converted to int

    def test_build_gen_args_from_chat_request(self):
        req = SimpleNamespace(
            max_tokens=256,
            max_output_tokens=None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=None,
            logit_bias=None,
            enable_thinking=True,
            thinking_budget=None,
            thinking_start_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True


class TestSplitThinking:
    """Tests for thinking tag parsing."""

    def test_channel_tags(self):
        text = "<|channel>thought\nReasoning here.<channel|>The answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Reasoning here."
        assert content == "The answer."

    def test_think_tags(self):
        text = "<think>Thinking.</think>Answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking."
        assert content == "Answer."

    def test_partial_close_tag_only(self):
        text = "Thinking text\n</think>\nAnswer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking text"
        assert content == "Answer."

    def test_no_thinking(self):
        text = "Just plain text."
        reasoning, content = server._split_thinking(text)
        assert reasoning is None
        assert content == "Just plain text."

    def test_empty_content_after_thinking(self):
        text = "<|channel>thought\nOnly thinking.<channel|>"
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Only thinking."
        assert content == ""


class TestChatMessageSchema:
    """Tests for ChatMessage accepting tool-calling roles and fields."""

    def test_accepts_tool_role(self):
        msg = server.ChatMessage(role="tool", content="result", tool_call_id="tc_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_1"

    def test_accepts_assistant_with_tool_calls(self):
        msg = server.ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_reasoning_field(self):
        msg = server.ChatMessage(
            role="assistant", content="answer", reasoning="thought"
        )
        assert msg.reasoning == "thought"


class TestSuppressToolCallContent:
    """Tests for tool-call markup suppression in streaming."""

    def test_no_tool_module(self):
        in_tc, content = server.suppress_tool_call_content(
            "Hello world", False, None, "world"
        )
        assert in_tc is False
        assert content == "world"

    def test_normal_text_before_tool_call(self):
        in_tc, content = server.suppress_tool_call_content(
            "I will call", False, "<tool_call>", "call"
        )
        assert in_tc is False
        assert content == "call"

    def test_suppresses_on_start_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>", False, "<tool_call>", ">"
        )
        assert in_tc is True
        assert content is None

    def test_suppresses_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool", False, "<tool_call>", "<tool"
        )
        assert in_tc is False
        assert content is None

    def test_stays_suppressed_after_entering(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>get_weather", True, "<tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool_call>call:get_weather", False, "<|tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool", False, "<|tool_call>", "<|tool"
        )
        assert in_tc is False
        assert content is None


class TestProcessToolCalls:
    """Tests for tool call parsing from model output."""

    def test_no_tool_calls(self):
        # Minimal tool module mock
        module = SimpleNamespace(tool_call_start="<tc>", tool_call_end="</tc>")
        result = server.process_tool_calls("Just text.", module, [])
        assert result["calls"] == []
        assert result["remaining_text"] == "Just text."


class TestCountThinkingTagTokens:
    """Tests for thinking tag token counting."""

    def test_channel_tags(self):
        assert (
            server._count_thinking_tag_tokens("<|channel>thought\ntext<channel|>answer")
            == 4
        )

    def test_think_tags(self):
        assert server._count_thinking_tag_tokens("<think>text</think>answer") == 2

    def test_no_tags(self):
        assert server._count_thinking_tag_tokens("plain text") == 0


def test_chat_completions_stop_passed_as_eos_tokens(client):
    model = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=""))
    config = SimpleNamespace(model_type="test")
    result = SimpleNamespace(
        text="Hello",
        prompt_tokens=5,
        generation_tokens=1,
        total_tokens=6,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=1.0,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_gen,
    ):
        resp = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hello"}],
                "stop": ["\n\n", "</s>"],
            },
        )
    assert resp.status_code == 200
    assert mock_gen.call_args.kwargs["eos_tokens"] == ["\n\n", "</s>"]


def test_responses_stop_passed_as_eos_tokens(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="test")
    result = SimpleNamespace(
        text="Hello",
        prompt_tokens=5,
        generation_tokens=1,
        total_tokens=6,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_gen,
    ):
        resp = client.post(
            "/responses",
            json={"model": "demo", "input": "hello", "stop": ["DONE"]},
        )
    assert resp.status_code == 200
    assert mock_gen.call_args.kwargs["eos_tokens"] == ["DONE"]


def test_chat_completions_tool_choice_required_injects_system_instruction(client):
    model = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template="template"))
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt") as mock_tpl,
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser", return_value=None),
    ):
        resp = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
                "tool_choice": "required",
            },
        )
    assert resp.status_code == 200
    messages = mock_tpl.call_args.args[2]
    assert messages[0]["role"] == "system"
    assert "must call one of the available tools" in messages[0]["content"]


def test_responses_json_mode_injects_system_instruction(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text='{"ok": true}',
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt") as mock_tpl,
        patch.object(server, "generate", return_value=result),
    ):
        resp = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": "hello",
                "response_format": {"type": "json_object"},
            },
        )
    assert resp.status_code == 200
    messages = mock_tpl.call_args.args[2]
    assert messages[0]["role"] == "system"
    assert "valid JSON only" in messages[0]["content"]
