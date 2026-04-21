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


# ---------------------------------------------------------------------------
# Stop sequences tests
# ---------------------------------------------------------------------------


def test_chat_completions_stop_passed_as_eos_tokens(client):
    """stop parameter should be passed as eos_tokens strings to generate."""
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
    assert "eos_tokens" in mock_gen.call_args.kwargs
    assert mock_gen.call_args.kwargs["eos_tokens"] == ["\n\n", "</s>"]


def test_chat_completions_no_stop_no_eos_tokens(client):
    """Without stop parameter, eos_tokens should not be in kwargs."""
    model = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=""))
    config = SimpleNamespace(model_type="test")
    result = SimpleNamespace(
        text="Hi",
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
            json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert "eos_tokens" not in mock_gen.call_args.kwargs


def test_responses_stop_passed_as_eos_tokens(client):
    """stop parameter on /responses should pass strings as eos_tokens."""
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
            "/responses",
            json={"model": "demo", "input": "hi", "stop": "STOP"},
        )
    assert resp.status_code == 200
    assert "eos_tokens" in mock_gen.call_args.kwargs
    assert mock_gen.call_args.kwargs["eos_tokens"] == ["STOP"]


def test_resolve_stop_sequences_single_string():
    """resolve_stop_sequences should normalize a single string to a list."""
    result = server.resolve_stop_sequences("hello")
    assert result == ["hello"]


def test_resolve_stop_sequences_list():
    """resolve_stop_sequences should pass through a list."""
    result = server.resolve_stop_sequences(["a", "b"])
    assert result == ["a", "b"]


def test_resolve_stop_sequences_none():
    """resolve_stop_sequences should return None for None input."""
    assert server.resolve_stop_sequences(None) is None


def test_resolve_stop_sequences_limits_to_four():
    """resolve_stop_sequences should process at most 4 sequences."""
    result = server.resolve_stop_sequences(["a", "b", "c", "d", "e", "f"])
    assert len(result) == 4
