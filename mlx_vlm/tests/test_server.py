import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server


def _parse_sse(response):
    """Parse the ``data:`` events out of a TestClient streaming response."""
    events = []
    for raw in response.iter_lines():
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def _fake_chunk(text, *, output_tokens):
    return SimpleNamespace(
        text=text,
        prompt_tokens=10,
        generation_tokens=output_tokens,
        total_tokens=10 + output_tokens,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )


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


def test_chat_completions_stream_strips_tool_call_markup(client):
    """Gemma 4's raw ``<|tool_call>…<tool_call|>`` markup must not leak
    into any ``delta.content`` of the streamed response. The structured
    ``tool_calls`` array is still emitted in the final delta.
    """
    model = SimpleNamespace()
    tokenizer = SimpleNamespace(
        chat_template="{{ '<|tool_call>' }}{{ '<tool_call|>' }}"
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    config = SimpleNamespace(model_type="gemma3n")

    def fake_stream_generate(*args, **kwargs):
        # mlx-vlm's detokenizer buffers special-token sequences and
        # flushes the whole tool-call markup as a single chunk — this
        # mirrors the real behaviour observed with Gemma 4.
        yield _fake_chunk("\n\n", output_tokens=1)
        yield _fake_chunk(
            "<|tool_call>call:read" '{filePath:<|"|>README.md<|"|>}<tool_call|>',
            output_tokens=2,
        )
        yield _fake_chunk("\n", output_tokens=3)

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", side_effect=fake_stream_generate),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "read file"}],
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "filePath": {"type": "string"},
                                },
                                "required": ["filePath"],
                            },
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    events = _parse_sse(response)
    assert events, "expected at least one SSE event"

    # The intermediate deltas must not contain the raw tool-call markup.
    # The final delta (with tool_calls populated) is allowed to have an
    # empty content.
    intermediate = events[:-1]
    for event in intermediate:
        content = event["choices"][0]["delta"].get("content") or ""
        assert (
            "<|tool_call>" not in content
        ), f"raw tool-call markup leaked into delta.content: {content!r}"
        assert (
            "<tool_call|>" not in content
        ), f"raw tool-call end marker leaked into delta.content: {content!r}"

    # The final delta carries the structured tool_calls array and empty
    # content, per the OpenAI Chat Completions streaming spec.
    final = events[-1]
    final_delta = final["choices"][0]["delta"]
    assert final_delta.get("content") in ("", None)
    tool_calls = final_delta.get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "filePath": "README.md"
    }


def test_chat_completions_stream_passes_through_plain_content(client):
    """Regression guard: when the model does not emit any tool-call
    markup, content must flow through unchanged, including when the
    chat template advertises a tool parser.
    """
    model = SimpleNamespace()
    tokenizer = SimpleNamespace(
        chat_template="{{ '<|tool_call>' }}{{ '<tool_call|>' }}"
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    config = SimpleNamespace(model_type="gemma3n")

    def fake_stream_generate(*args, **kwargs):
        yield _fake_chunk("Hello", output_tokens=1)
        yield _fake_chunk(" world", output_tokens=2)

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", side_effect=fake_stream_generate),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    events = _parse_sse(response)
    assert events

    streamed = "".join(
        (event["choices"][0]["delta"].get("content") or "") for event in events
    )
    assert streamed == "Hello world"

    final = events[-1]
    final_delta = final["choices"][0]["delta"]
    assert final_delta.get("tool_calls") in ([], None)


def _run_stream_with_chunks(client, chunks, *, tools=None):
    """Helper: drive ``/chat/completions`` with a synthetic chunk sequence
    and return the parsed SSE events."""
    model = SimpleNamespace()
    tokenizer = SimpleNamespace(
        chat_template="{{ '<|tool_call>' }}{{ '<tool_call|>' }}"
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    config = SimpleNamespace(model_type="gemma3n")

    def fake_stream_generate(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    request_body = {
        "model": "demo",
        "messages": [{"role": "user", "content": "go"}],
        "stream": True,
    }
    if tools is not None:
        request_body["tools"] = tools

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", side_effect=fake_stream_generate),
    ):
        response = client.post("/chat/completions", json=request_body)

    assert response.status_code == 200
    return _parse_sse(response)


_READ_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    }
]


def test_chat_completions_stream_handles_markup_split_across_chunks(client):
    """Two-chunk split: the start marker arrives in chunk 1, the end
    marker in chunk 2. Neither intermediate delta may contain any piece
    of the markup, and the final delta must still carry the structured
    tool call."""
    events = _run_stream_with_chunks(
        client,
        chunks=[
            _fake_chunk("hello <|tool_call>call:read{file", output_tokens=1),
            _fake_chunk('Path:<|"|>README.md<|"|>}<tool_call|>', output_tokens=2),
        ],
        tools=_READ_TOOL,
    )

    for event in events[:-1]:
        content = event["choices"][0]["delta"].get("content") or ""
        assert "<|tool_call>" not in content
        assert "<tool_call|>" not in content
        assert "call:read" not in content
        assert "filePath" not in content

    streamed = "".join(
        (event["choices"][0]["delta"].get("content") or "") for event in events[:-1]
    )
    assert streamed.startswith("hello")

    final_delta = events[-1]["choices"][0]["delta"]
    assert final_delta.get("content") in ("", None)
    tool_calls = final_delta.get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {
        "filePath": "README.md"
    }


def test_chat_completions_stream_holds_back_partial_start_marker(client):
    """The tail of a chunk is a partial prefix of the start marker
    (``<|t``). The next chunk completes it into a real tool call. The
    partial tokens must never be emitted as content."""
    events = _run_stream_with_chunks(
        client,
        chunks=[
            _fake_chunk("ready <|t", output_tokens=1),
            _fake_chunk(
                'ool_call>call:read{filePath:<|"|>README.md<|"|>}<tool_call|>',
                output_tokens=2,
            ),
        ],
        tools=_READ_TOOL,
    )

    for event in events[:-1]:
        content = event["choices"][0]["delta"].get("content") or ""
        assert "<|t" not in content
        assert "<|tool_call>" not in content

    streamed = "".join(
        (event["choices"][0]["delta"].get("content") or "") for event in events[:-1]
    )
    assert streamed.startswith("ready")

    final_delta = events[-1]["choices"][0]["delta"]
    tool_calls = final_delta.get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read"
