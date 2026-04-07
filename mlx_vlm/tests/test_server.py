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
    # Clear and populate
    server._prompt_cache_states.clear()
    state = server.get_prompt_cache_state("test-model")
    state.last_used = 0  # Force stale (epoch = very old)
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
    assert evicted == 0  # TTL=0 means no expiry
    server._prompt_cache_states.clear()


# ---------------------------------------------------------------------------
# Prompt cache TTL — real-world scenario tests
# ---------------------------------------------------------------------------


def test_telegram_idle_13_hours_evicted(monkeypatch):
    """Simulate: user sends image at 7:42 AM, bot responds, then 13 hours
    of silence. The stale KV cache should be evicted before the next request."""
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")  # 5 min TTL
    server._prompt_cache_states.clear()

    import time
    now = time.time()

    # Simulate the 7:42 AM image analysis — cache populated with tokens
    state = server.get_prompt_cache_state("qwen3.5-35b")
    state.token_ids = list(range(12000))  # 12K tokens from image + response
    state.cache = ["fake_kv_layer"]  # placeholder for KV cache
    state.last_used = now - (13 * 3600)  # 13 hours ago
    state.created_at = now - (13 * 3600)

    assert len(server._prompt_cache_states) == 1
    assert state.token_count == 12000

    # Cleanup runs — should evict the 13-hour-old entry
    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert len(server._prompt_cache_states) == 0

    # Next request creates a fresh cache — no stale KV to corrupt
    fresh = server.get_prompt_cache_state("qwen3.5-35b")
    assert fresh.cache is None
    assert fresh.token_ids is None
    server._prompt_cache_states.clear()


def test_active_conversation_not_evicted(monkeypatch):
    """Simulate: user is actively chatting every 30 seconds.
    Cache should never be evicted during an active conversation."""
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time
    now = time.time()

    state = server.get_prompt_cache_state("qwen3.5-35b")
    state.token_ids = list(range(8000))
    state.cache = ["fake_kv"]

    # Simulate 10 messages, 30s apart — each touches the cache
    for i in range(10):
        state.last_used = now - (30 * (10 - i))  # most recent was 30s ago

    # Last used 30s ago — well within 300s TTL
    evicted = server.evict_stale_prompt_caches()
    assert evicted == 0
    assert state.token_count == 8000  # cache intact
    server._prompt_cache_states.clear()


def test_multiple_users_only_stale_evicted(monkeypatch):
    """Simulate: two users with different cache keys. One idle 10 min,
    one active 1 min ago. Only the stale one should be evicted."""
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time
    now = time.time()

    # User A: active 1 min ago
    active = server.get_prompt_cache_state("model", cache_key="user-a")
    active.token_ids = list(range(5000))
    active.cache = ["kv_a"]
    active.last_used = now - 60

    # User B: idle 10 min
    stale = server.get_prompt_cache_state("model", cache_key="user-b")
    stale.token_ids = list(range(9000))
    stale.cache = ["kv_b"]
    stale.last_used = now - 600

    assert len(server._prompt_cache_states) == 2

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert "model::user-a" in server._prompt_cache_states
    assert "model::user-b" not in server._prompt_cache_states
    # Active user's cache untouched
    assert server._prompt_cache_states["model::user-a"].token_count == 5000
    server._prompt_cache_states.clear()


def test_cache_just_under_ttl_not_evicted(monkeypatch):
    """Cache idle for just under TTL should NOT be evicted."""
    monkeypatch.setenv("PROMPT_CACHE_TTL", "300")
    server._prompt_cache_states.clear()

    import time
    now = time.time()

    state = server.get_prompt_cache_state("model")
    state.token_ids = list(range(1000))
    state.cache = ["kv"]
    state.last_used = now - 299  # 1 second under TTL

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 0
    server._prompt_cache_states.clear()


def test_invalidated_cache_cleared_on_eviction(monkeypatch):
    """Evicted entries should have their cache and token_ids set to None."""
    monkeypatch.setenv("PROMPT_CACHE_TTL", "60")
    server._prompt_cache_states.clear()

    import time

    state = server.get_prompt_cache_state("model")
    state.token_ids = list(range(20000))  # 20K tokens of KV cache
    state.cache = ["big_kv_layer_1", "big_kv_layer_2"]
    state.last_used = time.time() - 120  # 2 min idle, TTL is 1 min

    # Keep a reference to verify invalidation
    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    # The state object should be invalidated
    assert state.cache is None
    assert state.token_ids is None
    server._prompt_cache_states.clear()


def test_short_ttl_evicts_between_requests(monkeypatch):
    """With a very short TTL (e.g., 5s), cache should be evicted if user
    pauses for just a few seconds — useful for testing/dev."""
    monkeypatch.setenv("PROMPT_CACHE_TTL", "5")
    server._prompt_cache_states.clear()

    import time

    state = server.get_prompt_cache_state("model")
    state.token_ids = list(range(3000))
    state.cache = ["kv"]
    state.last_used = time.time() - 10  # 10s ago, TTL is 5s

    evicted = server.evict_stale_prompt_caches()
    assert evicted == 1
    assert len(server._prompt_cache_states) == 0
    server._prompt_cache_states.clear()
