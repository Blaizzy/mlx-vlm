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
# Context tracking tests
# ---------------------------------------------------------------------------


def test_check_context_length_within_limit():
    """Should not raise when within limit."""
    fake_proc = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda s, add_special_tokens=False: list(range(10))),
    )
    server.check_context_length("short", fake_proc, 100)  # No exception


def test_check_context_length_exceeds_limit():
    """Should raise HTTPException when exceeding limit."""
    from fastapi import HTTPException as _HTTPException

    fake_proc = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda s, add_special_tokens=False: list(range(200))),
    )
    with pytest.raises(_HTTPException) as exc_info:
        server.check_context_length("long", fake_proc, 100)
    assert exc_info.value.status_code == 400
    assert "200 tokens" in exc_info.value.detail


def test_check_context_length_zero_unlimited():
    """max_context=0 should skip check entirely."""
    server.check_context_length("anything", None, 0)  # No exception


def test_get_max_context_tokens_default():
    """Default should be 0 (unlimited)."""
    import os
    os.environ.pop("MAX_CONTEXT_TOKENS", None)
    assert server.get_max_context_tokens() == 0


def test_get_max_context_tokens_from_env():
    """Should read from MAX_CONTEXT_TOKENS env var."""
    import os
    os.environ["MAX_CONTEXT_TOKENS"] = "16384"
    assert server.get_max_context_tokens() == 16384
    os.environ.pop("MAX_CONTEXT_TOKENS")


# ---------------------------------------------------------------------------
# JSON mode / response_format tests
# ---------------------------------------------------------------------------


def test_resolve_response_format_json_adds_instruction():
    msgs = [{"role": "user", "content": "hi"}]
    result = server.resolve_response_format(msgs, {"type": "json_object"})
    assert result[0]["role"] == "system"
    assert "json" in result[0]["content"].lower()
    assert len(result) == 2


def test_resolve_response_format_text_no_change():
    msgs = [{"role": "user", "content": "hi"}]
    result = server.resolve_response_format(msgs, {"type": "text"})
    assert len(result) == 1


def test_resolve_response_format_none_no_change():
    msgs = [{"role": "user", "content": "hi"}]
    result = server.resolve_response_format(msgs, None)
    assert len(result) == 1


def test_chat_completions_json_mode_accepted(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text='{"answer": 42}',
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
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Give me JSON"}],
                "response_format": {"type": "json_object"},
            },
        )

    assert response.status_code == 200
    # The first message passed to apply_chat_template should be the injected system msg
    chat_messages = mock_template.call_args.args[2]
    assert chat_messages[0]["role"] == "system"
    assert "json" in chat_messages[0]["content"].lower()


def test_responses_json_mode_accepted(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = SimpleNamespace(
        text='{"answer": 42}',
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
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": "Give me JSON",
                "response_format": {"type": "json_object"},
            },
        )

    assert response.status_code == 200
    # The first message passed to apply_chat_template should be the injected system msg
    chat_messages = mock_template.call_args.args[2]
    assert chat_messages[0]["role"] == "system"
    assert "json" in chat_messages[0]["content"].lower()
