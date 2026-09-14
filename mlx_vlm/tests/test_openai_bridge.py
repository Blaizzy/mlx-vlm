import json
from io import BytesIO

import pytest

from mlx_vlm.openai_bridge import (
    BridgeConfig,
    BridgeConfigurationError,
    _response_text,
    build_request,
    call_responses,
    load_config,
)


def test_load_config_requires_key():
    with pytest.raises(BridgeConfigurationError, match="OPENAI_API_KEY"):
        load_config({})


def test_build_request_is_parameterized():
    request = build_request(
        "hello",
        BridgeConfig("secret", "https://example.test/v1", "test-model"),
        system="be concise",
    )
    body = json.loads(request.data)
    assert request.full_url == "https://example.test/v1/responses"
    assert request.headers["Authorization"] == "Bearer secret"
    assert body == {
        "model": "test-model",
        "instructions": "be concise",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
    }


def test_call_responses_and_extracts_output(monkeypatch):
    payload = {"output_text": "answer", "output": []}

    class FakeResponse(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        assert request.method == "POST"
        assert timeout == 3
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr("mlx_vlm.openai_bridge.urlopen", fake_urlopen)
    result = call_responses(
        "hello", BridgeConfig("secret", "https://example.test/v1", timeout=3)
    )
    assert _response_text(result) == "answer"
