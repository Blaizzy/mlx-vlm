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
# finish_reason tests
# ---------------------------------------------------------------------------


def _mock_result(text="Hello!", **overrides):
    defaults = dict(
        text=text,
        prompt_tokens=10,
        generation_tokens=5,
        total_tokens=15,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=1.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _base_mocks(result=None):
    """Return context managers for model, template, and generate mocks."""
    if result is None:
        result = _mock_result()
    model = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=""))
    config = SimpleNamespace(model_type="test")
    return (
        patch.object(server, "get_cached_model", return_value=(model, processor, config)),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    )


def test_chat_completions_finish_reason_stop_without_tools(client):
    """finish_reason should be 'stop' for plain text responses."""
    m1, m2, m3 = _base_mocks()
    with m1, m2, m3:
        resp = client.post(
            "/chat/completions",
            json={"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_finish_reason_tool_calls(client):
    """finish_reason should be 'tool_calls' when tool calls are detected."""
    result = _mock_result()
    m1, m2, m3 = _base_mocks(result)

    fake_tool_calls = {
        "calls": [{"type": "function", "id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
        "remaining_text": "",
    }

    with m1, m2, m3, \
         patch.object(server, "_infer_tool_parser", return_value="qwen3_coder"), \
         patch.object(server, "load_tool_module", return_value=SimpleNamespace()), \
         patch.object(server, "process_tool_calls", return_value=fake_tool_calls):
        resp = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "search for test"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            },
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_completions_finish_reason_stop_with_tools_no_calls(client):
    """finish_reason should be 'stop' when tools defined but model doesn't call any."""
    result = _mock_result(text="I don't need tools for this.")
    m1, m2, m3 = _base_mocks(result)

    no_calls = {"calls": [], "remaining_text": "I don't need tools for this."}

    with m1, m2, m3, \
         patch.object(server, "_infer_tool_parser", return_value="qwen3_coder"), \
         patch.object(server, "load_tool_module", return_value=SimpleNamespace()), \
         patch.object(server, "process_tool_calls", return_value=no_calls):
        resp = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            },
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["finish_reason"] == "stop"
