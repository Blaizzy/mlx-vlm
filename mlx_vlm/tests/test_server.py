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
# tool_choice tests
# ---------------------------------------------------------------------------


def test_resolve_tool_choice_auto_passthrough():
    """tool_choice='auto' should return tools unchanged."""
    tools = [{"function": {"name": "search"}}]
    result_tools, instruction = server.resolve_tool_choice(tools, "auto")
    assert result_tools == tools
    assert instruction is None


def test_resolve_tool_choice_none_strips_tools():
    """tool_choice='none' should return None for tools."""
    tools = [{"function": {"name": "search"}}]
    result_tools, instruction = server.resolve_tool_choice(tools, "none")
    assert result_tools is None
    assert instruction is None


def test_resolve_tool_choice_required_adds_instruction():
    """tool_choice='required' should keep tools and add instruction."""
    tools = [{"function": {"name": "search"}}]
    result_tools, instruction = server.resolve_tool_choice(tools, "required")
    assert result_tools == tools
    assert instruction is not None
    assert "must call" in instruction.lower()


def test_resolve_tool_choice_specific_function():
    """Specific function tool_choice should filter to that tool."""
    tools = [
        {"function": {"name": "search"}},
        {"function": {"name": "fetch"}},
        {"function": {"name": "read"}},
    ]
    choice = {"type": "function", "function": {"name": "fetch"}}
    result_tools, instruction = server.resolve_tool_choice(tools, choice)
    assert len(result_tools) == 1
    assert result_tools[0]["function"]["name"] == "fetch"
    assert "fetch" in instruction


def test_resolve_tool_choice_unknown_tool_returns_400():
    """Unknown function name should raise InvalidToolChoiceError."""
    tools = [{"function": {"name": "search"}}]
    choice = {"type": "function", "function": {"name": "nonexistent"}}
    with pytest.raises(server.InvalidToolChoiceError, match="not found"):
        server.resolve_tool_choice(tools, choice)


def test_resolve_tool_choice_invalid_string_returns_error():
    """Invalid string tool_choice should raise InvalidToolChoiceError."""
    tools = [{"function": {"name": "search"}}]
    with pytest.raises(server.InvalidToolChoiceError, match="Invalid tool_choice"):
        server.resolve_tool_choice(tools, "bogus")


def test_resolve_tool_choice_none_value_passthrough():
    """tool_choice=None should return tools unchanged."""
    tools = [{"function": {"name": "search"}}]
    result_tools, instruction = server.resolve_tool_choice(tools, None)
    assert result_tools == tools
    assert instruction is None


def test_resolve_tool_choice_no_tools():
    """No tools should return None regardless of tool_choice."""
    result_tools, instruction = server.resolve_tool_choice(None, "required")
    assert result_tools is None
    assert instruction is None


def test_chat_completions_tool_choice_none_strips_tools(client):
    """tool_choice='none' should not pass tools to apply_chat_template."""
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
        patch.object(server, "apply_chat_template", return_value="prompt") as mock_tmpl,
        patch.object(server, "generate", return_value=result),
    ):
        resp = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
                "tool_choice": "none",
            },
        )
    assert resp.status_code == 200
    # tools should be None in the template call
    assert mock_tmpl.call_args.kwargs.get("tools") is None


def test_chat_completions_tool_choice_required_adds_system_msg(client):
    """tool_choice='required' should inject a system message."""
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
        patch.object(server, "apply_chat_template", return_value="prompt") as mock_tmpl,
        patch.object(server, "generate", return_value=result),
    ):
        resp = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "search for test"}],
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
                "tool_choice": "required",
            },
        )
    assert resp.status_code == 200
    # Check that messages passed to template include the system instruction
    messages_arg = mock_tmpl.call_args[0][2]  # 3rd positional arg
    system_msgs = [m for m in messages_arg if m.get("role") == "system"]
    assert any("must call" in m.get("content", "").lower() for m in system_msgs)
