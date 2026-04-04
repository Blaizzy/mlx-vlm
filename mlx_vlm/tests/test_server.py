from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server


def make_response_result():
    return SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
    )


def make_chat_result():
    return SimpleNamespace(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )


@pytest.fixture(autouse=True)
def reset_server_state(monkeypatch):
    monkeypatch.delenv("PRELOAD_MODEL", raising=False)
    monkeypatch.delenv("PRELOAD_ADAPTER", raising=False)
    monkeypatch.delenv("MLX_TRUST_REMOTE_CODE", raising=False)
    for env_name, _ in server._SERVER_GENERATION_ENV.values():
        monkeypatch.delenv(env_name, raising=False)
    server.model_cache = {}
    yield
    server.model_cache = {}


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


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/responses", {"input": "Hello"}),
        ("/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
    ],
)
def test_generation_endpoints_require_model_when_no_server_default(client, path, payload):
    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert "model" in response.text.lower()


def test_responses_endpoint_uses_server_generation_defaults_in_metadata(
    client, monkeypatch
):
    monkeypatch.setenv("PRELOAD_MODEL", "server-model")
    monkeypatch.setenv("MLX_VLM_SERVER_MAX_TOKENS", "64")
    monkeypatch.setenv("MLX_VLM_SERVER_TOP_P", "0.85")
    monkeypatch.setenv("MLX_VLM_SERVER_TOP_K", "20")
    monkeypatch.setenv("MLX_VLM_SERVER_MIN_P", "0.05")
    monkeypatch.setenv("MLX_VLM_SERVER_REPETITION_PENALTY", "1.1")
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ) as mock_get_cached_model,
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(
            server, "generate", return_value=make_response_result()
        ) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "input": "Hello",
                "max_output_tokens": 12,
                "temperature": 0.9,
            },
        )

    assert response.status_code == 200
    mock_get_cached_model.assert_called_once_with("server-model", None)
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["temperature"] == pytest.approx(0.9)
    assert mock_generate.call_args.kwargs["top_p"] == pytest.approx(0.85)
    assert mock_generate.call_args.kwargs["top_k"] == 20
    assert mock_generate.call_args.kwargs["min_p"] == pytest.approx(0.05)
    assert mock_generate.call_args.kwargs["repetition_penalty"] == pytest.approx(1.1)
    assert response.json()["model"] == "server-model"
    assert response.json()["max_output_tokens"] == 12
    assert response.json()["temperature"] == pytest.approx(0.9)
    assert response.json()["top_p"] == pytest.approx(0.85)


def test_responses_endpoint_forwards_new_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(
            server, "generate", return_value=make_response_result()
        ) as mock_generate,
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

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(
            server, "generate", return_value=make_chat_result()
        ) as mock_generate,
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
