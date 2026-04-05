import json
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


# --- Logprobs helpers ---

def _make_logprobs_array(chosen_token_id, logprob_value, vocab_size=100):
    """Create a fake logprobs object that supports indexing by token id."""
    arr = [float("-inf")] * vocab_size
    arr[chosen_token_id] = logprob_value
    return arr


def _make_stream_chunks():
    """Return a list of SimpleNamespace chunks mimicking stream_generate output."""
    return [
        SimpleNamespace(
            text="Hello",
            token=15,
            logprobs=_make_logprobs_array(15, -0.5),
            prompt_tokens=8,
            generation_tokens=1,
            total_tokens=9,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        ),
        SimpleNamespace(
            text=" world",
            token=42,
            logprobs=_make_logprobs_array(42, -0.3),
            prompt_tokens=8,
            generation_tokens=2,
            total_tokens=10,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        ),
    ]


def _mock_tokenizer():
    """Return a mock tokenizer with a decode method."""
    class _Tok:
        def decode(self, ids):
            mapping = {15: "Hello", 42: " world"}
            return "".join(mapping.get(i, "?") for i in ids)
    return _Tok()


def _patch_for_logprobs(client, *, logprobs, stream=False):
    """Helper that patches the server and posts a chat request."""
    model = SimpleNamespace()
    tokenizer = _mock_tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer)
    config = SimpleNamespace(model_type="qwen2_vl")

    patches = [
        patch.object(server, "get_cached_model", return_value=(model, processor, config)),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(_make_stream_chunks())),
    ]
    if not logprobs:
        # When logprobs is not requested, the non-streaming path uses generate()
        result = SimpleNamespace(
            text="Hello world",
            token=42,
            logprobs=None,
            prompt_tokens=8,
            generation_tokens=2,
            total_tokens=10,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        )
        patches.append(patch.object(server, "generate", return_value=result))

    body = {
        "model": "demo",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
    }
    if logprobs:
        body["logprobs"] = True

    import contextlib
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        response = client.post("/chat/completions", json=body)
    return response


def test_chat_completions_logprobs_returned_when_requested(client):
    response = _patch_for_logprobs(client, logprobs=True)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["logprobs"] is not None
    assert data["choices"][0]["logprobs"]["content"] is not None
    assert len(data["choices"][0]["logprobs"]["content"]) == 2


def test_chat_completions_logprobs_absent_by_default(client):
    response = _patch_for_logprobs(client, logprobs=False)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["logprobs"] is None


def test_chat_completions_logprobs_format(client):
    response = _patch_for_logprobs(client, logprobs=True)
    data = response.json()
    content = data["choices"][0]["logprobs"]["content"]

    first = content[0]
    assert first["token"] == "Hello"
    assert abs(first["logprob"] - (-0.5)) < 1e-6
    assert first["bytes"] == list("Hello".encode("utf-8"))

    second = content[1]
    assert second["token"] == " world"
    assert abs(second["logprob"] - (-0.3)) < 1e-6
    assert second["bytes"] == list(" world".encode("utf-8"))


def test_chat_completions_streaming_logprobs(client):
    response = _patch_for_logprobs(client, logprobs=True, stream=True)
    assert response.status_code == 200

    chunks = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[len("data: "):]))

    # At least one content chunk should have logprobs
    logprob_chunks = [
        c for c in chunks
        if c["choices"][0].get("logprobs") is not None
    ]
    assert len(logprob_chunks) >= 1

    first_lp = logprob_chunks[0]["choices"][0]["logprobs"]["content"][0]
    assert "token" in first_lp
    assert "logprob" in first_lp
    assert "bytes" in first_lp
