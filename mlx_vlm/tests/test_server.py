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
        processor = lambda tokens, logits: logits
        args = server.GenerationArguments(
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            logit_bias={3: -0.5},
            enable_thinking=False,
            thinking_budget=100,
            logits_processors=[processor],
        )
        kw = args.to_generate_kwargs()
        assert kw["max_tokens"] == 50
        assert kw["top_k"] == 40
        assert kw["min_p"] == 0.05
        assert kw["repetition_penalty"] == 1.15
        assert kw["logit_bias"] == {3: -0.5}
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 100
        assert kw["logits_processors"] == [processor]

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
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 128
        assert args.top_k == 32
        assert args.logit_bias == {5: -1.0}  # string keys converted to int

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
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True

    def test_extract_chat_response_format_json_schema(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal",
                    "schema": {
                        "type": "object",
                        "properties": {"animal": {"type": "string"}},
                        "required": ["animal"],
                    },
                },
            },
            text=None,
        )

        schema = server._extract_response_format_schema(req)

        assert schema["properties"]["animal"]["type"] == "string"

    def test_extract_responses_text_format_json_schema(self):
        req = SimpleNamespace(
            response_format=None,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "animal",
                    "schema": {
                        "type": "object",
                        "properties": {"animal": {"type": "string"}},
                        "required": ["animal"],
                    },
                }
            },
        )

        schema = server._extract_response_format_schema(req)

        assert schema["required"] == ["animal"]

    def test_build_structured_logits_processors_uses_tokenizer(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal",
                    "schema": {"type": "object"},
                },
            },
            text=None,
        )
        proc = SimpleNamespace(tokenizer=object())

        with patch.object(
            server, "build_json_schema_logits_processor", return_value="processor"
        ) as mock_build:
            processors = server._build_structured_logits_processors(req, proc)

        assert processors == ["processor"]
        assert mock_build.call_args.args[1] == {"type": "object"}


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


class TestResponsesToolsToChatFormat:
    """Tests for _responses_tools_to_chat_format()."""

    def test_none_passthrough(self):
        assert server._responses_tools_to_chat_format(None) is None

    def test_empty_list(self):
        assert server._responses_tools_to_chat_format([]) is None

    def test_responses_shape_converted_to_chat_shape(self):
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {}}},
            }
        ]
        result = server._responses_tools_to_chat_format(tools)
        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {}}},
                },
            }
        ]

    def test_chat_shape_passes_through(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        assert server._responses_tools_to_chat_format(tools) == tools

    def test_mixed_shapes(self):
        tools = [
            {"type": "function", "name": "a", "parameters": {}},
            {
                "type": "function",
                "function": {"name": "b", "parameters": {}},
            },
        ]
        result = server._responses_tools_to_chat_format(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_missing_parameters_defaults_to_empty_object(self):
        tools = [{"type": "function", "name": "x"}]
        result = server._responses_tools_to_chat_format(tools)
        assert result[0]["function"]["parameters"] == {}


class TestResponsesInputItemSchema:
    """Tests for the ResponsesInputItem Pydantic model."""

    def test_message_item(self):
        item = server.ResponsesInputItem(role="user", content="hello")
        assert item.role == "user"
        assert item.content == "hello"
        assert item.type is None

    def test_function_call_item(self):
        item = server.ResponsesInputItem(
            type="function_call",
            call_id="call_1",
            name="get_weather",
            arguments='{"city":"Paris"}',
        )
        assert item.type == "function_call"
        assert item.call_id == "call_1"
        assert item.name == "get_weather"
        assert item.arguments == '{"city":"Paris"}'
        assert item.role is None

    def test_function_call_output_item(self):
        item = server.ResponsesInputItem(
            type="function_call_output",
            call_id="call_1",
            output="Sunny",
        )
        assert item.type == "function_call_output"
        assert item.call_id == "call_1"
        assert item.output == "Sunny"


class TestResponsesEndpointTools:
    """End-to-end /responses tests covering the tools-on path (with mocks)."""

    @staticmethod
    def _build_mocks(model_output_text):
        model = SimpleNamespace()
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template="dummy template")
        )
        config = SimpleNamespace(model_type="qwen2_vl")
        gen_result = SimpleNamespace(
            text=model_output_text,
            prompt_tokens=12,
            generation_tokens=8,
            total_tokens=20,
        )
        tool_module = SimpleNamespace(
            tool_call_start="<tool_call>", tool_call_end="</tool_call>"
        )
        return {
            "model": model,
            "processor": processor,
            "config": config,
            "gen_result": gen_result,
            "tool_module": tool_module,
        }

    def test_tools_field_passed_to_apply_chat_template(self, client):
        """When tools are present, the converted chat-shape tools must be
        forwarded to apply_chat_template (otherwise the model never sees them)."""
        mocks = self._build_mocks("plain text")

        with (
            patch.object(
                server,
                "get_cached_model",
                return_value=(mocks["model"], mocks["processor"], mocks["config"]),
            ),
            patch.object(
                server, "apply_chat_template", return_value="prompt"
            ) as mock_template,
            patch.object(server, "_infer_tool_parser", return_value="qwen"),
            patch.object(server, "load_tool_module", return_value=mocks["tool_module"]),
            patch.object(server, "generate", return_value=mocks["gen_result"]),
            patch.object(server, "response_generator", new=None),
        ):
            response = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [{"role": "user", "content": "weather in Paris"}],
                    "max_output_tokens": 16,
                    "enable_thinking": False,
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                },
            )

        assert response.status_code == 200, response.text
        forwarded_tools = mock_template.call_args.kwargs.get("tools")
        assert forwarded_tools is not None
        # Must be the chat-shape (nested under 'function') after conversion.
        assert forwarded_tools[0]["function"]["name"] == "get_weather"

    def test_function_call_emitted_when_model_outputs_tool_call(self, client):
        """When the model output contains tool-call markers, /responses must
        produce a function_call output item rather than leaking the markers
        into a message."""
        mocks = self._build_mocks(
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        )

        with (
            patch.object(
                server,
                "get_cached_model",
                return_value=(mocks["model"], mocks["processor"], mocks["config"]),
            ),
            patch.object(server, "apply_chat_template", return_value="prompt"),
            patch.object(server, "_infer_tool_parser", return_value="qwen"),
            patch.object(server, "load_tool_module", return_value=mocks["tool_module"]),
            patch.object(
                server,
                "process_tool_calls",
                return_value={
                    "calls": [
                        {
                            "type": "function",
                            "index": 0,
                            "id": "call_xyz",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                    "remaining_text": "",
                },
            ),
            patch.object(server, "generate", return_value=mocks["gen_result"]),
            patch.object(server, "response_generator", new=None),
        ):
            response = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [{"role": "user", "content": "weather?"}],
                    "max_output_tokens": 32,
                    "enable_thinking": False,
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {},
                        }
                    ],
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        # Output should contain exactly one function_call item, no message.
        assert len(body["output"]) == 1
        item = body["output"][0]
        assert item["type"] == "function_call"
        assert item["name"] == "get_weather"
        assert item["arguments"] == '{"city": "Paris"}'
        assert item["call_id"] == "call_xyz"

    def test_function_call_input_item_translated_to_assistant_tool_call(self, client):
        """A function_call input item (replaying a prior assistant tool call)
        must become an assistant message with tool_calls in the prompt
        construction."""
        mocks = self._build_mocks("Sunny in Paris.")
        recorded = {}

        def capture_template(processor, config, chat_messages, **kwargs):
            recorded["chat_messages"] = list(chat_messages)
            return "prompt"

        with (
            patch.object(
                server,
                "get_cached_model",
                return_value=(mocks["model"], mocks["processor"], mocks["config"]),
            ),
            patch.object(server, "apply_chat_template", side_effect=capture_template),
            patch.object(server, "_infer_tool_parser", return_value=None),
            patch.object(server, "generate", return_value=mocks["gen_result"]),
            patch.object(server, "response_generator", new=None),
        ):
            response = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [
                        {"role": "user", "content": "weather?"},
                        {
                            "type": "function_call",
                            "call_id": "call_abc",
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_abc",
                            "output": "Sunny",
                        },
                    ],
                    "max_output_tokens": 16,
                    "enable_thinking": False,
                },
            )

        assert response.status_code == 200, response.text
        msgs = recorded["chat_messages"]
        # Sequence should be: user, assistant-with-tool_calls, tool-result.
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Paris"}'
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["content"] == "Sunny"
        assert msgs[2]["tool_call_id"] == "call_abc"
