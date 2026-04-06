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
# Stop sequences tests
# ---------------------------------------------------------------------------


def test_chat_completions_stop_string_passed_as_eos_tokens(client):
    """stop parameter should be resolved to eos_tokens in generate kwargs."""
    model = SimpleNamespace()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            chat_template="",
            encode=lambda s, add_special_tokens=False: [42],
        ),
    )
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
        patch.object(server, "get_cached_model", return_value=(model, processor, config)),
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
        patch.object(server, "get_cached_model", return_value=(model, processor, config)),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result) as mock_gen,
    ):
        resp = client.post(
            "/chat/completions",
            json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert "eos_tokens" not in mock_gen.call_args.kwargs


def test_responses_stop_string_passed_as_eos_tokens(client):
    """stop parameter on /responses should also resolve to eos_tokens."""
    model = SimpleNamespace()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            chat_template="",
            encode=lambda s, add_special_tokens=False: [99],
        ),
    )
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
        patch.object(server, "get_cached_model", return_value=(model, processor, config)),
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
    """resolve_stop_sequences should handle a single string."""
    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            encode=lambda s, add_special_tokens=False: [10, 20],
        ),
    )
    result = server.resolve_stop_sequences("hello")
    assert result == ["hello"]


def test_resolve_stop_sequences_list():
    """resolve_stop_sequences should handle a list of strings."""
    call_count = [0]
    token_map = {0: [10], 1: [20, 30]}

    def fake_encode(s, add_special_tokens=False):
        idx = call_count[0]
        call_count[0] += 1
        return token_map.get(idx, [])

    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=fake_encode),
    )
    result = server.resolve_stop_sequences(["a", "b"])
    assert result == ["a", "b"]


def test_resolve_stop_sequences_none():
    """resolve_stop_sequences should return None for None input."""
    assert server.resolve_stop_sequences(None) is None


def test_resolve_stop_sequences_limits_to_four():
    """resolve_stop_sequences should process at most 4 sequences."""
    call_count = [0]

    def fake_encode(s, add_special_tokens=False):
        call_count[0] += 1
        return [call_count[0]]

    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=fake_encode),
    )
    result = server.resolve_stop_sequences(["a", "b", "c", "d", "e", "f"])
    assert len(result) == 4
