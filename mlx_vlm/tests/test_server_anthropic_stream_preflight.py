"""Regression test for the /v1/messages streaming context-budget preflight.

``_preflight_stream_context_budget`` raises ``HTTPException(400)`` for an
over-budget prompt (and records the failure itself). The streaming branch of
``anthropic_messages_endpoint`` used to call it without an ``except
HTTPException`` handler, so the client error fell through to the endpoint's
outer ``except Exception`` and was reported as a ``500`` ``api_error`` with a
``"400: ..."`` message. The OpenAI routers already convert such errors; this
test pins the same behaviour for the Anthropic-compatible endpoint.

Model-free: the request path up to the preflight is patched out.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.anthropic as server_anthropic
from mlx_vlm.server.generation import GenerationArguments


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _patched_request_path(preflight):
    """Patch every step between the endpoint entry and the preflight call so
    the request reaches the preflight without loading a model."""
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace()
    return (
        patch.object(
            server_anthropic,
            "get_cached_model",
            return_value=(model, processor, config),
        ),
        patch.object(
            server_anthropic,
            "_anthropic_messages_to_internal",
            return_value=(["msg"], [], None, None),
        ),
        patch.object(
            server_anthropic, "_infer_tool_parser_from_processor", return_value=None
        ),
        patch.object(
            server_anthropic,
            "_build_gen_args",
            return_value=GenerationArguments(temperature=0.7),
        ),
        patch.object(server_anthropic, "apply_chat_template", return_value="prompt"),
        patch.object(server_anthropic, "_preflight_stream_context_budget", preflight),
    )


def test_stream_over_budget_returns_anthropic_400_not_500(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", SimpleNamespace())

    async def _reject(**_kwargs):
        raise HTTPException(status_code=400, detail="prompt is too long")

    patches = _patched_request_path(_reject)
    for p in patches:
        p.start()
    try:
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        for p in patches:
            p.stop()

    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == "prompt is too long"
    # The old behaviour leaked the HTTPException's "400: " prefix into a 500.
    assert not payload["error"]["message"].startswith("400:")


def test_stream_within_budget_starts_stream(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", SimpleNamespace())

    async def _ok(**_kwargs):
        return None

    patches = _patched_request_path(_ok)
    # When the preflight passes it must not short-circuit: the endpoint returns
    # a streaming 200 response instead of an error. The generated SSE body is
    # out of scope here.
    for p in patches:
        p.start()
    try:
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "demo",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
    finally:
        for p in patches:
            p.stop()
