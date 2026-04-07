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
