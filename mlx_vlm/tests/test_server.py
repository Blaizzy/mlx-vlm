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


def sse_json_events(response_text):
    events = []
    for block in response_text.split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                events.append(data)
            else:
                events.append(json.loads(data))
    return events


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


# ── Continuous batching / ResponseGenerator tests ─────────────────────


class TestResponseGenerator:
    """Tests for the ResponseGenerator continuous batching engine."""

    def test_generate_arguments_defaults(self):
        args = server.GenerationArguments()
        assert args.max_tokens == server.DEFAULT_MAX_TOKENS
        assert args.temperature == server.DEFAULT_TEMPERATURE
        assert args.enable_thinking is True
        assert args.logit_bias is None

    def test_generate_arguments_to_generate_kwargs(self):
        args = server.GenerationArguments(
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            logit_bias={3: -0.5},
            enable_thinking=False,
            thinking_budget=100,
        )
        kw = args.to_generate_kwargs()
        assert kw["max_tokens"] == 50
        assert kw["top_k"] == 40
        assert kw["min_p"] == 0.05
        assert kw["repetition_penalty"] == 1.15
        assert kw["logit_bias"] == {3: -0.5}
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 100

    def test_generate_arguments_to_template_kwargs(self):
        args = server.GenerationArguments(enable_thinking=False, thinking_budget=50)
        kw = args.to_template_kwargs()
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 50

    def test_generate_arguments_omits_none_optionals(self):
        args = server.GenerationArguments()
        kw = args.to_generate_kwargs()
        assert "repetition_penalty" not in kw
        assert "logit_bias" not in kw
        assert "thinking_budget" not in kw

    def test_build_gen_args_from_openai_request(self):
        req = SimpleNamespace(
            max_output_tokens=128,
            temperature=0.5,
            top_p=0.9,
            top_k=32,
            min_p=0.1,
            repetition_penalty=1.2,
            logit_bias={"5": -1.0},
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
            stop=["END"],
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 128
        assert args.top_k == 32
        assert args.logit_bias == {5: -1.0}  # string keys converted to int
        assert args.stop == ["END"]

    def test_build_gen_args_from_chat_request(self):
        req = SimpleNamespace(
            max_tokens=256,
            max_output_tokens=None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=None,
            logit_bias=None,
            enable_thinking=True,
            thinking_budget=None,
            thinking_start_token=None,
            stop="END",
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True
        assert args.stop == ["END"]

    def test_normalize_stop_rejects_more_than_four_values(self):
        with pytest.raises(server.HTTPException):
            server._normalize_stop(["a", "b", "c", "d", "e"])

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("END", ["END"]),
            (["END"], ["END"]),
            (["a", "b", "c", "d"], ["a", "b", "c", "d"]),
        ],
    )
    def test_normalize_stop_accepts_openai_shapes(self, raw, expected):
        assert server._normalize_stop(raw) == expected

    @pytest.mark.parametrize("raw", ["", [], ["END", ""], ["END", 1], {"stop": "END"}])
    def test_normalize_stop_rejects_invalid_values(self, raw):
        with pytest.raises(server.HTTPException):
            server._normalize_stop(raw)


class TestStopMatcher:
    """Tests for decoded text stop matching."""

    def test_trims_stop_sequence_across_chunks(self):
        matcher = server.StopMatcher(["END"])

        text, stopped = matcher.feed("hello E")
        assert text == "hello"
        assert stopped is False

        text, stopped = matcher.feed("ND tail")
        assert text == " "
        assert stopped is True
        assert matcher.flush() == ""

    def test_flushes_held_suffix_without_stop(self):
        matcher = server.StopMatcher(["END"])

        text, stopped = matcher.feed("hello E")
        assert text == "hello"
        assert stopped is False
        assert matcher.flush() == " E"

    def test_single_character_stop_does_not_buffer_unmatched_text(self):
        matcher = server.StopMatcher(["x"])

        text, stopped = matcher.feed("abc")
        assert text == "abc"
        assert stopped is False

    def test_trims_stop_sequence_in_single_chunk(self):
        matcher = server.StopMatcher(["END"])

        text, stopped = matcher.feed("hello END tail")

        assert text == "hello "
        assert stopped is True
        assert matcher.flush() == ""

    def test_stop_at_beginning_emits_no_text(self):
        matcher = server.StopMatcher(["END"])

        text, stopped = matcher.feed("END tail")

        assert text == ""
        assert stopped is True

    def test_uses_earliest_matching_stop_sequence(self):
        matcher = server.StopMatcher(["world", "END"])

        text, stopped = matcher.feed("hello world END")

        assert text == "hello "
        assert stopped is True

    def test_does_not_emit_text_after_stop(self):
        matcher = server.StopMatcher(["END"])

        assert matcher.feed("hello END") == ("hello ", True)
        assert matcher.feed("tail") == ("", True)

    def test_partial_prefix_is_released_when_it_cannot_match(self):
        matcher = server.StopMatcher(["END"])

        assert matcher.feed("hello E") == ("hello", False)
        assert matcher.feed("x") == (" ", False)
        assert matcher.flush() == "Ex"


def test_chat_completions_fallback_applies_stop_sequence(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(
            text="hello E",
            prompt_tokens=8,
            generation_tokens=1,
            peak_memory=0.1,
        ),
        SimpleNamespace(
            text="ND tail",
            prompt_tokens=8,
            generation_tokens=2,
            peak_memory=0.1,
        ),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": "END",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello "
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_fallback_applies_earliest_stop_sequence(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(
            text="hello world END tail",
            prompt_tokens=8,
            generation_tokens=1,
            peak_memory=0.1,
        ),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": ["END", "world"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello "
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_fallback_reports_length_when_stop_does_not_match(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(
            text="hello",
            prompt_tokens=8,
            generation_tokens=1,
            peak_memory=0.1,
        ),
        SimpleNamespace(
            text=" world",
            prompt_tokens=8,
            generation_tokens=2,
            peak_memory=0.1,
        ),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": "END",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_completions_streaming_fallback_applies_stop_sequence(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(
            text="hello E",
            prompt_tokens=8,
            generation_tokens=1,
            peak_memory=0.1,
        ),
        SimpleNamespace(
            text="ND tail",
            prompt_tokens=8,
            generation_tokens=2,
            peak_memory=0.1,
        ),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stop": "END",
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    events = sse_json_events(body)
    chunks = [event for event in events if event != "[DONE]"]
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == [
        "hello",
        " ",
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"


def test_chat_completions_streaming_fallback_reports_length_without_stop_match(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(
            text="hello",
            prompt_tokens=8,
            generation_tokens=1,
            peak_memory=0.1,
        ),
        SimpleNamespace(
            text=" world",
            prompt_tokens=8,
            generation_tokens=2,
            peak_memory=0.1,
        ),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stop": "END",
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    events = sse_json_events(body)
    chunks = [event for event in events if event != "[DONE]"]
    assert (
        "".join(
            chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks
        )
        == "hello world"
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert events[-1] == "[DONE]"


def test_responses_fallback_applies_stop_sequence(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(text="answer<", prompt_tokens=5, generation_tokens=1),
        SimpleNamespace(text="stop>extra", prompt_tokens=5, generation_tokens=2),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": "Hello",
                "stop": "<stop>",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == "answer"


def test_responses_streaming_fallback_applies_stop_sequence(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        SimpleNamespace(text="answer<", prompt_tokens=5, generation_tokens=1),
        SimpleNamespace(text="stop>extra", prompt_tokens=5, generation_tokens=2),
    ]

    with (
        patch.object(server, "response_generator", None),
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
    ):
        with client.stream(
            "POST",
            "/responses",
            json={
                "model": "demo",
                "input": "Hello",
                "stream": True,
                "stop": "<stop>",
            },
        ) as response:
            body = response.read().decode()

    assert response.status_code == 200
    events = sse_json_events(body)
    deltas = [
        event["delta"]
        for event in events
        if isinstance(event, dict) and event.get("type") == "response.output_text.delta"
    ]
    done = next(
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "response.output_text.done"
    )
    completed = next(
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "response.completed"
    )

    assert "".join(deltas) == "answer"
    assert done["text"] == "answer"
    assert completed["response"]["output_text"] == "answer"


class TestSplitThinking:
    """Tests for thinking tag parsing."""

    def test_channel_tags(self):
        text = "<|channel>thought\nReasoning here.<channel|>The answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Reasoning here."
        assert content == "The answer."

    def test_think_tags(self):
        text = "<think>Thinking.</think>Answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking."
        assert content == "Answer."

    def test_partial_close_tag_only(self):
        text = "Thinking text\n</think>\nAnswer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking text"
        assert content == "Answer."

    def test_no_thinking(self):
        text = "Just plain text."
        reasoning, content = server._split_thinking(text)
        assert reasoning is None
        assert content == "Just plain text."

    def test_empty_content_after_thinking(self):
        text = "<|channel>thought\nOnly thinking.<channel|>"
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Only thinking."
        assert content == ""


class TestChatMessageSchema:
    """Tests for ChatMessage accepting tool-calling roles and fields."""

    def test_accepts_tool_role(self):
        msg = server.ChatMessage(role="tool", content="result", tool_call_id="tc_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "tc_1"

    def test_accepts_assistant_with_tool_calls(self):
        msg = server.ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "tc_1", "function": {"name": "f", "arguments": "{}"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_reasoning_field(self):
        msg = server.ChatMessage(
            role="assistant", content="answer", reasoning="thought"
        )
        assert msg.reasoning == "thought"


class TestSuppressToolCallContent:
    """Tests for tool-call markup suppression in streaming."""

    def test_no_tool_module(self):
        in_tc, content = server.suppress_tool_call_content(
            "Hello world", False, None, "world"
        )
        assert in_tc is False
        assert content == "world"

    def test_normal_text_before_tool_call(self):
        in_tc, content = server.suppress_tool_call_content(
            "I will call", False, "<tool_call>", "call"
        )
        assert in_tc is False
        assert content == "call"

    def test_suppresses_on_start_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>", False, "<tool_call>", ">"
        )
        assert in_tc is True
        assert content is None

    def test_suppresses_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool", False, "<tool_call>", "<tool"
        )
        assert in_tc is False
        assert content is None

    def test_stays_suppressed_after_entering(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<tool_call>get_weather", True, "<tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool_call>call:get_weather", False, "<|tool_call>", "weather"
        )
        assert in_tc is True
        assert content is None

    def test_pipe_delimited_partial_marker(self):
        in_tc, content = server.suppress_tool_call_content(
            "text<|tool", False, "<|tool_call>", "<|tool"
        )
        assert in_tc is False
        assert content is None


class TestProcessToolCalls:
    """Tests for tool call parsing from model output."""

    def test_no_tool_calls(self):
        # Minimal tool module mock
        module = SimpleNamespace(tool_call_start="<tc>", tool_call_end="</tc>")
        result = server.process_tool_calls("Just text.", module, [])
        assert result["calls"] == []
        assert result["remaining_text"] == "Just text."


class TestCountThinkingTagTokens:
    """Tests for thinking tag token counting."""

    def test_channel_tags(self):
        assert (
            server._count_thinking_tag_tokens("<|channel>thought\ntext<channel|>answer")
            == 4
        )

    def test_think_tags(self):
        assert server._count_thinking_tag_tokens("<think>text</think>answer") == 2

    def test_no_tags(self):
        assert server._count_thinking_tag_tokens("plain text") == 0
