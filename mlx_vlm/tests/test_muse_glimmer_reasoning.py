from types import SimpleNamespace

from mlx_vlm.server.generation import GenerationArguments
from mlx_vlm.server.request_normalization import _build_gen_args
from mlx_vlm.server.responses_state import (
    ThinkingStreamState,
    _split_thinking,
    _strip_content_markers,
)


def test_splits_muse_glimmer_reasoning_and_channel_header():
    reasoning, content = _split_thinking(
        "to=self<|message|>Say exactly: hello world ... No extra."
        "<|eom|><|start|>assistant to=user<|message|>hello world",
        "to=self<|message|>",
        "<|eom|>",
    )

    assert (reasoning, content) == (
        "Say exactly: hello world ... No extra.",
        "hello world",
    )


def test_streaming_thinking_strips_split_harmony_channel_header():
    state = ThinkingStreamState(
        thinking_start_token="to=self<|message|>", thinking_end_token="<|eom|>"
    )
    reasoning = []
    content = []

    for chunk in [
        "to=self<|message|>REASONING<|eom|><|start|>",
        "assistant to=user",
        "<|message|>hello world",
    ]:
        delta = state.feed(chunk)
        if delta.reasoning:
            reasoning.append(delta.reasoning)
        if delta.content:
            content.append(delta.content)

    assert "".join(reasoning) == "REASONING"
    assert "".join(content) == "hello world"


def test_strips_harmony_channel_headers_without_changing_ordinary_text():
    assert (
        _strip_content_markers("<|start|>assistant to=user<|message|>hello world")
        == "hello world"
    )
    assert (
        _strip_content_markers(
            "<|start|>assistant to=user<|message|>hello"
            "<|eom|><|start|>assistant to=user<|message|> world<|eot|>"
        )
        == "hello world"
    )
    assert _strip_content_markers("ordinary text") == "ordinary text"


def test_build_gen_args_prefers_request_and_environment_to_config(monkeypatch):
    monkeypatch.delenv("MLX_VLM_THINKING_START_TOKEN", raising=False)
    monkeypatch.delenv("MLX_VLM_THINKING_END_TOKEN", raising=False)
    config = SimpleNamespace(
        thinking_start_token="config-start",
        thinking_end_token="config-end",
    )

    config_args = _build_gen_args(SimpleNamespace(), config=config)
    assert config_args.thinking_start_token == "config-start"
    assert config_args.thinking_end_token == "config-end"

    no_marker_config_args = _build_gen_args(SimpleNamespace(), config=SimpleNamespace())
    assert no_marker_config_args.thinking_start_token is None
    assert no_marker_config_args.thinking_end_token is None

    monkeypatch.setenv("MLX_VLM_THINKING_START_TOKEN", "env-start")
    monkeypatch.setenv("MLX_VLM_THINKING_END_TOKEN", "env-end")
    env_args = _build_gen_args(SimpleNamespace(), config=config)
    assert env_args.thinking_start_token == "env-start"
    assert env_args.thinking_end_token == "env-end"

    request = SimpleNamespace(
        thinking_start_token="request-start",
        thinking_end_token="request-end",
    )
    request_args = _build_gen_args(request, config=config)
    assert request_args.thinking_start_token == "request-start"
    assert request_args.thinking_end_token == "request-end"


def test_template_kwargs_aliases_reasoning_effort_to_reasoning_strength():
    args = GenerationArguments(reasoning_effort="medium")
    kwargs = args.to_template_kwargs()

    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["reasoning_strength"] == "medium"

    kwargs_without_effort = GenerationArguments().to_template_kwargs()
    assert "reasoning_effort" not in kwargs_without_effort
    assert "reasoning_strength" not in kwargs_without_effort


# Captured from a live Muse Glimmer server: when the model answers without
# reasoning first, add_generation_prompt has already emitted "<|start|>assistant",
# so the header the model produces has no "<|start|>" prefix.
BARE_HEADER_OUTPUT = "to=user<|message|>391"
FULL_HEADER_OUTPUT = (
    "to=self<|message|>Reasoning here.<|eom|><|start|>assistant to=user<|message|>391"
)


def _drive_stream(chunks):
    state = ThinkingStreamState(
        thinking_start_token="to=self<|message|>", thinking_end_token="<|eom|>"
    )
    reasoning = []
    content = []
    for chunk in chunks:
        delta = state.feed(chunk)
        if delta.reasoning:
            reasoning.append(delta.reasoning)
        if delta.content:
            content.append(delta.content)
    return "".join(reasoning), "".join(content)


def test_strips_bare_harmony_channel_header():
    assert _strip_content_markers("to=user<|message|>391") == "391"


def test_streaming_strips_bare_header_at_every_split_point():
    for index in range(len(BARE_HEADER_OUTPUT) + 1):
        _, content = _drive_stream(
            [BARE_HEADER_OUTPUT[:index], BARE_HEADER_OUTPUT[index:]]
        )
        assert content == "391", f"split at {index}"


def test_streaming_strips_bare_header_one_character_at_a_time():
    _, content = _drive_stream(list(BARE_HEADER_OUTPUT))
    assert content == "391"


def test_streaming_splits_full_header_at_every_split_point():
    for index in range(len(FULL_HEADER_OUTPUT) + 1):
        reasoning, content = _drive_stream(
            [FULL_HEADER_OUTPUT[:index], FULL_HEADER_OUTPUT[index:]]
        )
        assert content == "391", f"split at {index}"
        assert reasoning == "Reasoning here.", f"split at {index}"


def test_keeps_harmony_like_text_once_content_has_started():
    text = "391 and the literal to=user<|message|> marker"
    assert _strip_content_markers(text) == text
    _, content = _drive_stream([text])
    assert content == text


def test_streaming_keeps_harmony_like_chunk_after_content_started():
    state = ThinkingStreamState(
        thinking_start_token="to=self<|message|>", thinking_end_token="<|eom|>"
    )

    first = state.feed("The answer is 391. ")
    second = state.feed("to=user<|message|> is the literal marker")

    assert first.content == "The answer is 391. "
    assert second.content == "to=user<|message|> is the literal marker"
