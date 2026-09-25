"""HTTP and realtime protocols, generation workers, model routing, and runtime state."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread, Timer
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from huggingface_hub import scan_cache_dir
from PIL import Image
from transformers.utils.chat_parsing import ResponseParser, parse_response

import mlx_vlm.reranker_loader as reranker_loader
import mlx_vlm.server as server
import mlx_vlm.server.anthropic as anthropic
import mlx_vlm.server.audio as server_audio
import mlx_vlm.server.cli as cli
import mlx_vlm.server.generation as generation
import mlx_vlm.server.openai as openai
import mlx_vlm.server.reranking as reranking
from mlx_vlm import apc
from mlx_vlm.apc import hash_image_payload
from mlx_vlm.generate import GenerationResult
from mlx_vlm.generate.image import ImageGenerationResult
from mlx_vlm.models.cache import KVCache
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server import GenerationArguments as Args
from mlx_vlm.server import ResponseGenerator as Generator
from mlx_vlm.server import realtime
from mlx_vlm.server.model_discovery import discover_models, is_model_directory
from mlx_vlm.server.responses_state import ToolCallStreamState, _response_items_to_chat
from mlx_vlm.server.runtime_config import RuntimeConfig
from mlx_vlm.tests.test_processors import MINICPM_MULTICALL
from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer, _ServerTokenStreamer
from mlx_vlm.tools import load_tool_module, process_tool_calls

_MUSE_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "close": ["<|eot|>", "<|eom|>"],
            "content": "text",
            "open_pattern": r"to=user<\|message\|>",
        },
        "reasoning_content": {
            "close": "<|eom|>",
            "content": "text",
            "open_pattern": r"to=self<\|message\|>",
        },
        "tool_calls": {
            "close": "</atem:invoke>",
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"'
                    r"[^>]*?>(?P<value>.*?)</atem:parameter>"
                ),
                "value_parser": {"args": {"allow_non_json": True}, "name": "json"},
            },
            "open_pattern": r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">',
            "repeats": True,
            "transform": {
                "function": {"arguments": "{content}", "name": "{name}"},
                "type": "function",
            },
        },
    },
    "start_anchor": "<|start|>assistant",
}


class _MuseResponseTemplateTokenizer:
    response_template = _MUSE_RESPONSE_TEMPLATE

    def parse_response(self, response, prefix=None):
        return parse_response(response, self.response_template, prefix=prefix)

    def get_response_parser(self, prefix=None):
        return ResponseParser(self.response_template, prefix=prefix)


def _msg(content="Hello", role="user", **extra):
    return dict(role=role, content=content, **extra)


def _input_message(text, role="user"):
    return _msg([dict(type="input_text", text=text)], role, type="message")


def _input_image(url, **options):
    return dict(type="input_image", image_url=url, **options)


def _function_result(output, name=None, call_id="call_view_image"):
    items = (
        [dict(type="function_call", name=name, arguments="{}", call_id=call_id)]
        if name
        else []
    )
    return [*items, dict(type="function_call_output", call_id=call_id, output=output)]


def _chat_request(**options):
    return server.ChatRequest(model="demo", messages=[_msg("hi")], **options)


def _post(client, api="chat", **payload):
    paths = dict(
        chat="/v1/chat/completions", responses="/v1/responses", messages="/v1/messages"
    )
    path = paths.get(api, api)
    body = {"model": "demo"}
    body["input" if "responses" in path else "messages"] = (
        "Hello" if "responses" in path else [_msg()]
    )
    if path == "/v1/messages":
        body["max_tokens"] = 4
    return client.post(path, json={**body, **payload})


def _result(text="done", **kwargs):
    return GenerationResult(
        **(
            dict(
                text=text,
                prompt_tokens=8,
                generation_tokens=4,
                total_tokens=12,
                prompt_tps=10.0,
                generation_tps=5.0,
                peak_memory=0.1,
            )
            | kwargs
        )
    )


def _token(text="", token=1, finish_reason=None, **kwargs):
    return server.StreamingToken(
        text=text, token=token, logprobs=0.0, finish_reason=finish_reason, **kwargs
    )


def _streaming(chunks, prompt_tokens=3):
    return NS(
        tokenizer=NS(decode=lambda tokens: ""),
        validate_context_budget=MagicMock(),
        generate=MagicMock(
            return_value=(
                server.GenerationContext(uid=1, prompt_tokens=prompt_tokens),
                iter(chunks),
            )
        ),
    )


def _tool(name="get_weather", api="chat"):
    if api == "messages":
        return dict(
            name=name, description="Get weather", input_schema={"type": "object"}
        )
    return dict(
        type="function", function=dict(name=name, parameters={"type": "object"})
    )


_JSON_TOOLS = NS(
    tool_call_start="<tool_call>",
    tool_call_end="</tool_call>",
    parse_tool_call=lambda call, tools: json.loads(call),
)
_MUSE_CALL = (
    "to=self<|message|>I need the weather tool.<|eom|>"
    "<|start|>assistant to=get_weather<|message|>"
    '<atem:function_calls><atem:invoke name="get_weather">'
    '<atem:parameter name="city">Warsaw</atem:parameter>'
    "</atem:invoke></atem:function_calls>"
)


@contextmanager
def _endpoint(
    *,
    model_type="qwen2_vl",
    processor=None,
    config=None,
    result=None,
    chunks=(),
    generator=None,
    template="prompt",
    parser=None,
):
    model, processor = NS(), processor or NS()
    config = config or NS(model_type=model_type)
    with ExitStack() as stack:

        def mock(name, **kwargs):
            return stack.enter_context(patch.object(server, name, **kwargs))

        cached = mock("get_cached_model", return_value=(model, processor, config))
        templating = mock("apply_chat_template", return_value=template)
        generation = mock("generate", return_value=result or _result())
        streaming = mock("stream_generate", side_effect=lambda *a, **kw: iter(chunks))
        stack.enter_context(
            patch.object(server.runtime, "response_generator", generator)
        )
        if parser:
            mock("_infer_tool_parser_from_processor", return_value="demo")
            mock("load_tool_module", return_value=parser)
        yield NS(
            cache=cached,
            template=templating,
            generate=generation,
            stream=streaming,
            config=config,
        )


def _stream_response(
    client,
    tokens,
    api="/chat/completions",
    *,
    prompt_tokens=3,
    endpoint=None,
    **payload,
):
    with _endpoint(generator=_streaming(tokens, prompt_tokens), **(endpoint or {})):
        return _post(client, api, stream=True, **payload)


def _chat_events(response, reason):
    chunks = _data(response)
    choices = [chunk for chunk in chunks if chunk.get("choices")]
    usage = next(chunk for chunk in chunks if chunk.get("usage") is not None)
    finish = next(
        chunk for chunk in choices if chunk["choices"][0]["finish_reason"] == reason
    )
    return choices, usage, finish


def _assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected


def _sse_events(body):
    for block in body.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "data" in fields and fields["data"] != "[DONE]":
            yield fields.get("event"), json.loads(fields["data"])


def _data(response):
    assert response.status_code == 200, response.text
    return [data for _, data in _sse_events(response.text)]


def _deltas(response, api="chat"):
    data = _data(response)
    if api == "chat":
        return [item["choices"][0]["delta"] for item in data if item.get("choices")]
    if api == "messages":
        return [
            item["delta"] for item in data if item.get("type") == "content_block_delta"
        ]
    return data


def _joined(deltas, key):
    return "".join(item.get(key) or "" for item in deltas)


def _thinking_text(response, api):
    deltas = _deltas(response, api)
    if api == "responses":
        return tuple(
            _joined(
                [d for d in deltas if d.get("type") == f"response.{kind}.delta"],
                "delta",
            )
            for kind in ("reasoning_text", "output_text")
        )
    keys = (
        ("thinking", "text") if api == "messages" else ("reasoning_content", "content")
    )
    return tuple(_joined(deltas, key) for key in keys)


def _gemma_thinking_channel_chunks():
    chunks = [
        (100, ""),
        (45518, ""),
        (107, ""),
        (101, ""),
        (236832, ""),
        (808, "<|channel>thought\n<channel|>7"),
        (236743, " *"),
        (236828, ""),
        (578, " 8"),
        (236743, " ="),
        (236810, ""),
        (236825, ""),
        (106, " 56"),
    ]
    return [
        _token(text, token, "stop" if i == len(chunks) - 1 else None)
        for i, (token, text) in enumerate(chunks)
    ]


_THINKING_CASES = {
    "gemma4": (_gemma_thinking_channel_chunks(), {}, ("", "7 * 8 = 56")),
    "cohere2_moe": (
        [
            _token("North reasoning."),
            _token("<|END_THINK"),
            _token("ING|><|START_"),
            _token("TEXT|>North answer.<|END_"),
            _token("TEXT|>", finish_reason="stop"),
        ],
        {},
        ("North reasoning.", "North answer."),
    ),
    "custom": (
        [
            _token(
                "<analysis>Custom reasoning.</analysis>Custom answer.",
                finish_reason="stop",
            )
        ],
        dict(thinking_start_token="<analysis>", thinking_end_token="</analysis>"),
        ("Custom reasoning.", "Custom answer."),
    ),
}


def _reset_runtime(monkeypatch, **overrides):
    state = dict(
        model_cache=server.ModelCacheRegistry(),
        response_generator=None,
        apc_manager=None,
    )
    for name, value in (state | overrides).items():
        monkeypatch.setattr(server.runtime, name, value)


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "input_value",
    ["", " \n\t ", [], [_msg("")], [_msg([{"type": "input_text", "text": " "}])]],
)
def test_responses_endpoint_rejects_empty_effective_input(client, input_value):
    with patch.object(openai, "get_cached_model") as load:
        response = _post(client, "responses", input=input_value)
    assert response.status_code == 400
    assert "non-empty message content" in response.json()["detail"]
    load.assert_not_called()


def test_chat_request_schema_requires_model():
    schema = server.ChatRequest.model_json_schema()
    assert "model" in schema["required"]
    assert {"tools", "tool_choice"} <= schema["properties"].keys()
    assert {
        (x["minItems"], x["maxItems"])
        for x in schema["properties"]["resize_shape"]["anyOf"]
        if x.get("type") == "array"
    } == {(1, 1), (2, 2)}


@pytest.mark.parametrize(
    "api,choice,normalized,names,instruction",
    [
        ("chat", "none", "none", [], None),
        (
            "chat",
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_weather"}},
            ["get_weather"],
            "must call the 'get_weather' function",
        ),
        ("messages", {"type": "none"}, "none", [], None),
        (
            "messages",
            {"type": "any"},
            "required",
            ["get_time", "get_weather"],
            "must call one or more",
        ),
    ],
    ids=["chat-disabled", "chat-forced", "anthropic-disabled", "anthropic-required"],
)
def test_tool_choice(client, api, choice, normalized, names, instruction):
    processor = NS(tokenizer=NS(chat_template="<tool_call>\n<function="))
    with _endpoint(
        model_type="qwen3_5" if api == "chat" else "qwen2_vl",
        processor=processor if api == "chat" else NS(),
    ) as fake:
        response = _post(
            client,
            api,
            tools=[_tool(n, api) for n in ("get_time", "get_weather")],
            tool_choice=choice,
            messages=[_msg("Be concise.", "system"), _msg("Say hello.")],
        )
    assert response.status_code == 200
    kwargs, messages = fake.template.call_args.kwargs, fake.template.call_args.args[2]
    assert kwargs["tool_choice"] == normalized
    assert [t["function"]["name"] for t in kwargs["tools"] or []] == names
    if instruction:
        assert instruction in messages[-1]["content"]
        if api == "chat":
            assert messages[0]["content"].startswith("Be concise.")
            assert instruction in messages[0]["content"]
    else:
        assert kwargs["tools"] is None
        if api == "chat":
            assert response.json()["choices"][0]["message"]["tool_calls"] is None


def test_chat_completions_tool_parser_override(client):
    processor = NS(tokenizer=NS(chat_template="plain template"))
    with _endpoint(
        processor=processor,
        result=_result(
            '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
        ),
    ):
        plain = _post(client, tools=[_tool()])
        overridden = _post(client, tools=[_tool()], tool_parser="json_tools")
        assert _post(client, tools=[_tool()], tool_parser="bogus").status_code == 422
    assert plain.status_code == overridden.status_code == 200
    assert plain.json()["choices"][0]["message"]["tool_calls"] is None
    assert (
        overridden.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_weather"
    )


@pytest.mark.parametrize(
    "api,choice,tools,detail",
    [
        ("chat", "required", [], "requires at least one tool"),
        (
            "chat",
            {"type": "function", "function": {"name": "missing"}},
            [_tool()],
            "unknown function 'missing'",
        ),
        ("chat", "sometimes", [], "Invalid tool_choice"),
        (
            "messages",
            {"type": "tool", "name": "missing"},
            [_tool(n, "messages") for n in ("get_time", "get_weather")],
            "unknown function 'missing'",
        ),
        ("messages", {"type": "any"}, [], "requires at least one tool"),
    ],
)
def test_invalid_tool_choice(client, api, choice, tools, detail):
    with _endpoint() as fake:
        response = _post(client, api, tools=tools, tool_choice=choice)
    assert response.status_code == 400
    payload = response.json()
    if api == "messages":
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
        assert detail in payload["error"]["message"]
    else:
        assert detail in payload["detail"]
        fake.cache.assert_not_called()


def test_speculative_server_reads_batch_coalesce_env(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", raising=False)
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.005)

    monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "2.5")
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.0025)

    monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "bad")
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.005)


def test_get_cached_model_omitted_adapter_inherits_loaded_adapter(monkeypatch):
    resources = NS(), NS(), NS(model_type="qwen2_vl")

    def make(model_path, adapter_path=None, **kwargs):
        return NS(
            model_path=model_path,
            adapter_path=adapter_path,
            model=resources[0],
            processor=resources[1],
            config=resources[2],
            wait_until_ready=lambda: resources,
            stop_and_join=lambda: None,
        )

    monkeypatch.setattr(server._app_module, "ResponseGenerator", make)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *a, **kw: None)
    _reset_runtime(monkeypatch, model_cache={})
    server.get_cached_model("demo-model", "adapter-a")
    server.get_cached_model("demo-model")
    cache = server.runtime.model_cache
    assert cache["cache_key"] == (
        "demo-model",
        "adapter-a",
        "text_generation",
        server.runtime.config.fingerprint(kinds={"text_generation"}),
    )
    assert cache["adapter_path"] == "adapter-a"


def test_unload_model_cache_group_resets_apc_around_generator_shutdown(monkeypatch):
    events = []

    class FakeAPCManager:
        def __init__(self):
            self.contents = ["old-model-prefix"]

        def clear(self):
            events.append(("clear", list(self.contents)))
            self.contents.clear()

    manager = FakeAPCManager()

    class FakeResponseGenerator:
        def stop_and_join(self):
            events.append(("stop", list(manager.contents)))
            # Simulate a store that was already in flight when shutdown began.
            manager.contents.append("draining-worker-prefix")

    response_generator = FakeResponseGenerator()
    registry = server.ModelCacheRegistry()
    registry.set(
        "text_generation",
        dict(
            model_path="old-model",
            adapter_path=None,
            response_generator=response_generator,
            apc_manager=manager,
        ),
    )
    _reset_runtime(
        monkeypatch,
        model_cache=registry,
        response_generator=response_generator,
        apc_manager=manager,
    )
    monkeypatch.setattr(server._app_module.gc, "collect", lambda: None)
    monkeypatch.setattr(server._app_module.mx, "clear_cache", lambda: None)

    assert server._app_module._unload_model_cache_group("text_generation") is True

    assert events == [
        ("clear", ["old-model-prefix"]),
        ("stop", []),
        ("clear", ["draining-worker-prefix"]),
    ]
    assert manager.contents == []
    assert registry.for_kind("text_generation") == {}
    assert server.runtime.response_generator is None
    assert server.runtime.apc_manager is None


def test_unsupported_model_request_does_not_crash_server(client, monkeypatch):
    def reject_model(*_args, **_kwargs):
        raise ValueError("Model type bert not supported.")

    monkeypatch.setattr(generation, "load", reject_model)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
    _reset_runtime(monkeypatch, model_cache={})

    response = client.post(
        "/v1/chat/completions",
        json=dict(
            model="google-bert/bert-base-multilingual-cased",
            messages=[dict(role="user", content="Hello")],
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Failed to load model: Model type bert not supported."
    )
    assert client.get("/health").status_code == 200


@pytest.fixture
def _audio_config(tmp_path, monkeypatch):
    import mlx_vlm.utils as mlx_utils

    def _make(model_type):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
        monkeypatch.setattr(mlx_utils, "get_model_path", lambda *a, **k: tmp_path)
        return str(tmp_path)

    return _make


def test_audio_endpoint_rejects_native_chat_model(_audio_config, monkeypatch):
    import mlx_audio.utils as mlx_audio_utils

    monkeypatch.setattr(mlx_audio_utils, "get_model_category", lambda *a, **k: None)
    path = _audio_config("qwen3_omni_moe")

    with pytest.raises(ValueError, match="/v1/chat/completions"):
        server._app_module.load_audio_model(path)


def test_audio_endpoint_loads_dedicated_stt_model(_audio_config, monkeypatch):
    import mlx_audio.utils as mlx_audio_utils

    sentinel = object()
    monkeypatch.setattr(mlx_audio_utils, "load_model", lambda *a, **k: sentinel)
    path = _audio_config("whisper")

    assert server._app_module.load_audio_model(path) is sentinel


def test_audio_endpoint_loads_audio_capable_native_model(_audio_config, monkeypatch):
    import mlx_audio.utils as mlx_audio_utils

    sentinel = object()
    monkeypatch.setattr(mlx_audio_utils, "get_model_category", lambda *a, **k: "sts")
    monkeypatch.setattr(mlx_audio_utils, "load_model", lambda *a, **k: sentinel)
    path = _audio_config("nemotron_voicechat")

    assert server._app_module.load_audio_model(path) is sentinel


def test_audio_stt_request_maps_native_chat_model_to_400(_audio_config, monkeypatch):
    import mlx_audio.utils as mlx_audio_utils

    monkeypatch.setattr(mlx_audio_utils, "get_model_category", lambda *a, **k: None)
    _reset_runtime(monkeypatch, model_cache={})
    path = _audio_config("qwen3_omni_moe")

    with pytest.raises(HTTPException) as exc_info:
        server.get_cached_model(path, model_kind="audio_stt")

    assert exc_info.value.status_code == 400
    assert "/v1/chat/completions" in exc_info.value.detail


def _generator(**overrides):
    gen = Generator.__new__(Generator)
    gen.__dict__.update(
        dict.fromkeys(
            (
                "adapter_path model processor config vision_cache draft_model draft_kind "
                "draft_model_path draft_kind_override kv_bits apc_manager apc_mode tokenizer _load_error"
            ).split()
        )
    )
    gen.__dict__.update(
        model_path="demo",
        stop_tokens=set(),
        kv_group_size=server.DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme=server.DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start=server.DEFAULT_QUANTIZED_KV_START,
        top_logprobs_k=0,
        requests=Queue(),
        _stop=False,
        _ready=Event(),
        _cancelled=set(),
        _cancel_lock=Lock(),
    )
    gen.__dict__.update(overrides)
    return gen


def test_server_caches_apc_mode_when_model_initializes(monkeypatch):
    config = NS(eos_token_id=[])
    language_model = NS()
    model = NS(language_model=language_model)
    processor = NS(tokenizer=NS())
    gen = _generator()
    gen.apc_manager = object()

    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    monkeypatch.setattr(
        generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, config),
    )
    apc_mode = MagicMock(return_value="exact")
    monkeypatch.setattr(apc, "model_apc_mode", apc_mode)

    gen._initialize_model()

    assert gen.apc_mode == "exact"
    apc_mode.assert_called_once_with(language_model)


def test_server_serves_ar_requests_after_drafter_mismatch(monkeypatch):
    config = NS(model_type="gemma4_text", hidden_size=5376, eos_token_id=[])
    model = NS(language_model=NS(config=config))
    drafter = NS(config=NS(model_type="gemma4_assistant", backbone_hidden_size=1536))
    gen, _ = _worker_setup(monkeypatch, initialize=False)
    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "assistant")
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    monkeypatch.setattr(
        generation,
        "load_model_resources",
        lambda *a, **kw: (model, NS(tokenizer=NS()), config),
    )
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.load_drafter", lambda *a, **kw: (drafter, "mtp")
    )
    queue = _enqueue(gen, max_tokens=1)
    with _running(gen):
        _, tokens = _drain(queue)
    assert (
        len(tokens) == 1
        and tokens[0].text == "10"
        and tokens[0].finish_reason == "length"
    )
    assert gen.draft_model is gen.draft_kind is None


def test_ar_thread_exception_reaches_pending_client_queue(monkeypatch):
    gen, _ = _worker_setup(monkeypatch, idle=True)
    error = RuntimeError("vision embedding failed")
    gen._gpu_embed = MagicMock(side_effect=error)
    queue = _enqueue(gen, max_tokens=2)
    with _running(gen) as worker:
        assert queue.get(timeout=1) is error and queue.get(timeout=1) is None
        assert worker.is_alive()


class TestModelDiscovery:
    @staticmethod
    def _model_directory(path):
        path.mkdir(parents=True)
        (path / "config.json").write_text('{"model_type": "qwen2_vl"}')
        (path / "model.safetensors").write_bytes(b"weights")
        return path

    @pytest.mark.parametrize(
        "config,valid",
        [
            ('{"model_type": "qwen2_vl"}', True),
            ('{"model_type": "custom", "model_file": "model.py"}', True),
            ("not json", False),
            ("{}", False),
        ],
        ids=["no-tokenizer", "custom-code", "malformed", "empty"],
    )
    def test_metadata(self, tmp_path, config, valid):
        model = self._model_directory(tmp_path / "model")
        (model / "config.json").write_text(config)
        (model / "model.py").write_text("raise RuntimeError('must not execute')")
        assert is_model_directory(model) is valid

    @pytest.mark.parametrize(
        "shard,valid",
        [(None, False), (b"", False), (b"weights", True)],
        ids=["missing", "empty", "complete"],
    )
    def test_shards(self, tmp_path, shard, valid):
        model = self._model_directory(tmp_path / "model")
        (model / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "model.safetensors", "b": "second.safetensors"}}'
        )
        if shard is not None:
            (model / "second.safetensors").write_bytes(shard)
        assert is_model_directory(model) is valid

    def test_rejects_adapters_and_broken_links(self, tmp_path):
        model = self._model_directory(tmp_path / "adapter")
        (model / "model.safetensors").rename(model / "adapter_model.safetensors")
        assert not is_model_directory(model)
        (model / "model.safetensors").symlink_to(model / "missing.safetensors")
        assert not is_model_directory(model)

    def test_pipeline_components(self, tmp_path):
        pipeline = tmp_path / "pipeline"
        component = self._model_directory(pipeline / "transformer")
        (pipeline / "model_index.json").write_text('{"_class_name": "FluxPipeline"}')
        (pipeline / "tokenizer").mkdir()
        assert is_model_directory(pipeline)
        (component / "model.safetensors").unlink()
        assert not is_model_directory(pipeline)
        self._model_directory(pipeline / "text_encoder")
        (component / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "missing.safetensors"}}'
        )
        assert not is_model_directory(pipeline)

    @pytest.mark.parametrize("main", ["absent", "complete", "incomplete"])
    def test_revisions_and_local_alias(self, tmp_path, main):
        repo = tmp_path / "models--local--vision"
        snapshots = [
            self._model_directory(repo / "snapshots" / (revision * 40))
            for revision in "ab"
        ]
        for modified, snapshot in zip((100, 200), snapshots):
            for path in (snapshot, *snapshot.iterdir()):
                os.utime(path, (modified, modified))
        if main != "absent":
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("a" * 40)
        if main == "incomplete":
            (snapshots[0] / "model.safetensors").unlink()
        selected = snapshots[0] if main == "complete" else snapshots[1]
        cache = scan_cache_dir(tmp_path)
        found = discover_models(cache)
        assert found == [
            dict(
                id="local/vision" if main == "complete" else str(selected),
                path=selected,
                created=100 if main == "complete" else 200,
            )
        ]
        assert discover_models(cache, [str(selected)]) == found

    @pytest.mark.parametrize("source", ["parent", "home", "model", "alias", "combined"])
    def test_custom_roots_and_aliases(self, tmp_path, source):
        root = tmp_path / "models"
        model = self._model_directory(root / "custom")
        alias = root / "alias"
        alias.symlink_to(model, target_is_directory=True)
        (root / "unrelated").mkdir()
        sources = dict(
            parent=str(root),
            home="~/" + os.path.relpath(root, Path.home()),
            model=str(model),
            alias=str(alias),
            missing=str(root / "missing"),
        )
        paths = list(sources.values()) if source == "combined" else [sources[source]]
        found = discover_models(NS(repos=[]), paths)
        assert (
            len(found) == 1
            and found[0]["id"] == str(model)
            and found[0]["path"] == model
        )

    @pytest.fixture
    def model_listing(self, client, monkeypatch, tmp_path):
        _reset_runtime(monkeypatch)
        monkeypatch.delenv("MLX_VLM_MODEL_PATHS", raising=False)
        cache_root = tmp_path / "cache"
        model = self._model_directory(
            cache_root / "models--local--vision" / "snapshots" / ("a" * 40)
        )
        refs = model.parent.parent / "refs"
        refs.mkdir()
        (refs / "main").write_text("a" * 40)
        scan = Mock(side_effect=lambda: scan_cache_dir(cache_root))
        monkeypatch.setattr(server, "scan_cache_dir", scan)

        def get(endpoint="/v1/models", **kwargs):
            response = client.get(endpoint, **kwargs)
            assert response.status_code == 200
            entries = response.json()["data"]
            ids = [m["id"] for m in entries]
            assert ids == sorted(set(ids), key=str.lower)
            return {m["id"]: m["loaded"] for m in entries}

        return NS(get=get, scan=scan, path=model, registry=server.runtime.model_cache)

    def test_endpoint_cache_and_loaded_status(self, model_listing):
        listing = model_listing
        for kind, model in (
            ("text_generation", "local/vision"),
            ("embedding", "/loaded/embedding"),
            ("tts", "/loaded/tts"),
        ):
            listing.registry.set(kind, {"model_path": model})
        expected = {
            "local/vision": True,
            "/loaded/embedding": True,
            "/loaded/tts": True,
        }
        assert listing.get("/models") == listing.get() == expected
        listing.registry.clear()
        assert listing.get() == {"local/vision": False}
        (listing.path / "model.safetensors").unlink()
        assert listing.get() == {}

    @pytest.mark.parametrize("cached", [False, True], ids=["missing-cache", "cached"])
    @pytest.mark.parametrize("source", ["environment", "query"])
    def test_endpoint_custom_paths(self, model_listing, monkeypatch, cached, source):
        listing = model_listing
        path = str(listing.path)
        if not cached:
            listing.scan.side_effect = server.CacheNotFound("missing cache", "/missing")
        params = {"model_dir": path} if source == "query" else {}
        if source == "environment":
            monkeypatch.setenv("MLX_VLM_MODEL_PATHS", path)
        listing.registry.set("text_generation", {"model_path": path})
        listing.registry.set("embedding", {"model_path": "/loaded/embedding"})
        assert listing.get(params=params) == {path: True, "/loaded/embedding": True}
        listing.registry.clear()
        assert listing.get(params=params) == {"local/vision" if cached else path: False}

    def test_endpoint_query_paths_are_additive_and_temporary(
        self, model_listing, monkeypatch, tmp_path
    ):
        configured = self._model_directory(tmp_path / "configured")
        requested = self._model_directory(
            tmp_path / "requested" / "model with spaces & symbols"
        )
        another = self._model_directory(tmp_path / "another")
        monkeypatch.setenv("MLX_VLM_MODEL_PATHS", str(configured))
        baseline = {"local/vision": False, str(configured): False}
        params = [("model_dir", str(path)) for path in (requested.parent, another, "")]
        assert model_listing.get(params=params) == {
            **baseline,
            str(requested): False,
            str(another): False,
        }
        assert os.environ["MLX_VLM_MODEL_PATHS"] == str(configured)
        assert model_listing.get() == baseline

    @pytest.mark.parametrize("use_cli_paths", [False, True])
    def test_cli_custom_model_paths(self, monkeypatch, tmp_path, use_cli_paths):
        monkeypatch.setattr(os, "environ", dict(os.environ))
        monkeypatch.setenv("MLX_VLM_MODEL_PATHS", "/existing/models")
        paths = [str(tmp_path / "model with spaces"), str(tmp_path / "other")]
        flags = (
            [arg for path in paths for arg in ("--model-dir", path)]
            if use_cli_paths
            else []
        )
        monkeypatch.setattr(sys, "argv", ["mlx_vlm.server", *flags])
        with patch.object(cli.uvicorn, "run") as run:
            cli.main()
        assert os.environ["MLX_VLM_MODEL_PATHS"] == (
            os.pathsep.join(paths) if use_cli_paths else "/existing/models"
        )
        run.assert_called_once()


def test_response_generator_diffusion_forwards_generation_options(monkeypatch):
    gen = _generator(
        model=NS(),
        processor=NS(),
        config=NS(eos_token_id=3),
        tokenizer=NS(all_special_ids=[0]),
        prefill_step_size=3072,
        apc_manager=NS(),
        apc_mode="exact",
    )
    calls = []

    def stream(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["on_result"](
            _result(
                "ok",
                token=7,
                prompt_tokens=2,
                generation_tokens=1,
                total_tokens=3,
                cached_tokens=1,
                finish_reason="length",
            )
        )
        yield from ()

    monkeypatch.setattr(generation, "stream_diffusion_generate_from_kwargs", stream)
    options = dict(
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=123,
        max_denoising_steps=7,
        block_length=16,
        num_to_transfer=3,
        max_transfer_per_step=2,
        editing_threshold=0.8,
        max_post_steps=5,
        stability_steps=1,
        diffusion_full_canvas=True,
        diffusion_min_canvas_length=4,
        diffusion_max_canvas_length=8,
        diffusion_sampler="entropy-bound",
        threshold=0.7,
        min_threshold=0.4,
    )
    queue = Queue()
    gen._generate_diffusion(
        uid=1,
        rqueue=queue,
        raw_inputs=dict(
            input_ids=mx.array([[11, 12]]),
            pixel_values="pixels",
            attention_mask="mask",
            mm_token_type_ids="types",
        ),
        args=Args(**options),
        cancelled=set(),
        apc_semantic_hash=73,
    )
    chunk = queue.get(timeout=1)
    assert (
        chunk.text,
        chunk.finish_reason,
        chunk.generation_tps,
        chunk.cached_tokens,
    ) == ("ok", "length", 5.0, 1)
    args, kwargs = calls[0]
    assert args[3].tolist() == [[11, 12]] and args[4:7] == ("pixels", "mask", {0})
    assert kwargs["skip_special_tokens"] is True
    assert args[7] == dict(
        options,
        mm_token_type_ids="types",
        prefill_step_size=3072,
        _apc_manager=gen.apc_manager,
        _apc_semantic_hash=73,
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/health"),
        ("get", "/metrics"),
        ("get", "/v1/metrics"),
        ("get", "/cache/stats"),
        ("get", "/v1/cache/stats"),
        ("post", "/cache/reset"),
        ("post", "/v1/cache/reset"),
        ("post", "/unload"),
    ],
)
def test_management_authentication(client, monkeypatch, method, path):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret-token")

    missing = getattr(client, method)(path)
    invalid = getattr(client, method)(
        path, headers={"Authorization": "Bearer wrong-token"}
    )
    valid = getattr(client, method)(
        path, headers={"Authorization": "Bearer secret-token"}
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200


def _fake_image_result(*, seed: int, output_path=None) -> ImageGenerationResult:
    image = Image.new("RGB", (16, 16), (seed % 255, 8, 16))
    data = ImageGenerationResult(
        array=mx.array(np.array(image)),
        seed=seed,
        width=16,
        height=16,
        steps=1,
        model="bonsai",
        family="bonsai",
        variant="ternary",
        guidance=1.0,
        peak_memory=0.0,
        prompt_tokens=5,
    )
    if output_path is not None:
        data.save(output_path)
    return data


@pytest.mark.parametrize(
    "method,format,seed,expand",
    [
        ("generations", "b64_json", 10, True),
        ("generations", "path", 20, False),
        ("edits", "b64_json", 30, False),
        ("edits", "path", 40, False),
    ],
)
def test_image_generation_and_editing(
    client, monkeypatch, tmp_path, method, format, seed, expand
):
    edit = method == "edits"
    model_type = "ideogram4" if expand else ("flux2" if edit else "bonsai")
    model_name = "black-forest-labs/FLUX.2-klein-9b-kv" if edit else "bonsai-ternary"
    calls = []

    def generate(model, request, **kwargs):
        calls.append(request)
        result = _fake_image_result(
            seed=request.seed, output_path=kwargs.get("output_path")
        )
        if expand:
            result.metadata["revised_prompt"] = '{"compositional_deconstruction":{}}'
        return result

    monkeypatch.setattr(openai, "edit_image" if edit else "generate_image", generate)
    extra = (
        dict(auto_json_caption=True, prompt_expansion_model="tiny-text-model")
        if expand
        else {}
    )
    if edit:
        extra["image"] = ["reference.png"] if format == "b64_json" else "reference.png"
    with _endpoint(model_type=model_type) as fake:
        response = client.post(
            f"/v1/images/{method}",
            json=dict(
                model=model_name,
                prompt="image",
                n=1 if expand else 2,
                seed=seed,
                size="256x256",
                steps=1,
                response_format=format,
                **({"output_dir": str(tmp_path)} if format == "path" else {}),
                **extra,
            ),
        )
    assert response.status_code == 200
    payload = response.json()
    if expand:
        assert calls[0].extra == extra
        assert (
            payload["data"][0]["revised_prompt"]
            == '{"compositional_deconstruction":{}}'
        )
    elif format == "path":
        paths = [Path(item["path"]) for item in payload["data"]]
        assert [p.name for p in paths] == [
            f"{'edit' if edit else 'image'}-{i}.png" for i in (seed, seed + 1)
        ]
        assert all(p.exists() for p in paths) and all(
            item["b64_json"] is None for item in payload["data"]
        )
    else:
        assert payload["size"] == "16x16" and [x["seed"] for x in payload["data"]] == [
            30,
            31,
        ]
        assert all(x["b64_json"] for x in payload["data"])
        assert [r.seed for r in calls] == [30, 31] and calls[0].image_paths == (
            "reference.png",
        )
        fake.cache.assert_called_once_with(model_name, model_kind="image_edit")


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"use_kv_cache": True, "output_resolution": 512},
        {"use_kv_cache": False, "output_resolution": 512, "negative_prompt": ""},
    ],
)
def test_image_editing_forwards_model_options(client, monkeypatch, options):
    edit = Mock(return_value=_fake_image_result(seed=7))
    monkeypatch.setattr(openai, "edit_image", edit)
    with _endpoint(model_type="qwen_image"):
        response = client.post(
            "/v1/images/edits",
            json=dict(
                model="Qwen/Qwen-Image-2.1",
                prompt="edit",
                image="reference.png",
                seed=7,
                size="512x512",
                steps=30,
                **options,
            ),
        )
    assert response.status_code == 200
    request = edit.call_args.args[1]
    assert request.extra == options
    assert request.width == request.height == 512
    assert request.steps == 30


@pytest.mark.parametrize("api", ["/responses", "/chat/completions"])
def test_responses_endpoint_forwards_new_sampling_args(client, api):
    options = dict(
        top_k=40, min_p=0.08, repetition_penalty=1.15, logit_bias={"12": -1.5}
    )
    thinking = dict(
        enable_thinking=False,
        thinking_budget=24,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
    )
    extra = (
        dict(max_output_tokens=12, **thinking)
        if api == "/responses"
        else dict(max_tokens=12, resize_shape=[512])
    )
    with _endpoint() as fake:
        response = _post(client, api, **options, **extra)
    assert response.status_code == 200
    _assert_fields(
        fake.generate.call_args.kwargs,
        **{**options, "logit_bias": {12: -1.5}},
        max_tokens=12,
    )
    if api == "/responses":
        _assert_fields(fake.template.call_args.kwargs, **thinking)
        _assert_fields(fake.generate.call_args.kwargs, **thinking)
    else:
        assert fake.generate.call_args.kwargs["resize_shape"] == (512, 512)


def _normalized(client, api, inputs, expected, **options):
    with _endpoint() as fake:
        response = _post(
            client,
            api,
            **{"input" if api == "responses" else "messages": inputs},
            **options,
        )
    assert response.status_code == 200
    assert fake.template.call_args.args[2] == expected
    return response, fake


def test_developer_message_normalization(client):
    _normalized(
        client,
        "responses",
        [
            _input_message("Developer instructions.", "developer"),
            _input_message("Hello"),
        ],
        [_msg("Top-level instructions.\n\nDeveloper instructions.", "system"), _msg()],
        instructions="Top-level instructions.",
    )


def test_assistant_reasoning_normalization(client):
    previous = _msg("Hello", "assistant", reasoning_content="Prior thought")
    _normalized(
        client,
        "chat",
        [_msg("Hi"), previous, _msg("Continue")],
        [_msg("Hi"), dict(previous, reasoning="Prior thought"), _msg("Continue")],
    )


def test_anthropic_image_normalization(client):
    url = "https://example.com/image.png"
    image = dict(type="image", source=dict(type="url", url=url))
    response, fake = _normalized(
        client,
        "messages",
        [_msg([dict(type="text", text="Describe it."), image])],
        [_msg("You are concise.", "system"), _msg("Describe it.")],
        system="You are concise.",
        max_tokens=12,
    )
    _assert_fields(fake.generate.call_args.kwargs, image=[url], max_tokens=12)
    _assert_fields(
        response.json(),
        type="message",
        role="assistant",
        content=[dict(type="text", text="done")],
        stop_reason="end_turn",
        usage=dict(
            input_tokens=8,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=4,
        ),
    )


def test_anthropic_system_normalization(client):
    last = _msg("Introduce the project.")
    _normalized(
        client,
        "messages",
        [_msg(), _msg([dict(type="text", text="Be precise.")], "system"), last],
        [_msg("Use short answers.", "system"), _msg(), _msg("Be precise."), last],
        system="Use short answers.",
        max_tokens=12,
    )


@pytest.mark.parametrize("image", [False, True], ids=["tool-text", "tool-image"])
def test_anthropic_tool_result_normalization(client, image):
    name, args = (
        ("render_chart", {"kind": "bar"})
        if image
        else ("get_weather", {"location": "SF"})
    )
    text = dict(type="text", text="Rendered chart.")
    content = (
        [
            text,
            dict(
                type="image",
                source=dict(type="base64", media_type="image/png", data="aW1n"),
            ),
        ]
        if image
        else "72F"
    )
    inputs = [
        _msg([dict(type="tool_use", id="toolu_1", name=name, input=args)], "assistant"),
        _msg([dict(type="tool_result", tool_use_id="toolu_1", content=content)]),
    ]
    expected = [
        _msg(
            "",
            "assistant",
            tool_calls=[
                dict(
                    id="toolu_1",
                    type="function",
                    function=dict(name=name, arguments=json.dumps(args)),
                )
            ],
        ),
        _msg(
            [text, dict(type="image")] if image else "72F",
            "tool",
            tool_call_id="toolu_1",
            name=None,
        ),
    ]
    _, fake = _normalized(client, "messages", inputs, expected)
    if image:
        assert fake.generate.call_args.kwargs["image"] == ["data:image/png;base64,aW1n"]
    else:
        assert (
            apply_chat_template(None, fake.config, expected, return_messages=True)[0][
                "content"
            ]
            == ""
        )


def _assert_chat_and_responses_messages(client, messages, expected, **extra):
    with _endpoint(model_type="qwen3_5") as fake:
        for api, field in [("chat", "messages"), ("responses", "input")]:
            fake.template.reset_mock()
            response = _post(client, api, **{field: messages}, **extra)
            assert response.status_code == 200, response.text
            fake.template.assert_called_once()
            assert fake.template.call_args.args[2] == expected, api
            assert fake.template.call_args.kwargs["tools"] == extra.get("tools"), api


@pytest.mark.parametrize("omitted_reasoning", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize(
    "content", ["Inspecting.", [{"type": "output_text", "text": "Inspecting."}]]
)
def test_chat_and_responses_preserve_same_tool_history(
    client, omitted_reasoning, content
):
    expected = [
        {"role": "user", "content": "Where is the entry point?"},
        {
            "role": "assistant",
            "content": "Inspecting.",
            "reasoning_content": "Check the entry point first.",
            "reasoning": "Check the entry point first.",
            "tool_calls": [
                {
                    "id": "call_saved",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "/src/app.py"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_saved",
            "name": "read_file",
            "content": "It initializes SQLite.",
        },
        {"role": "user", "content": "Summarize what you learned."},
    ]
    messages = copy.deepcopy(expected)
    assistant = messages[1]
    assistant["content"] = content
    del assistant[omitted_reasoning]
    function = assistant["tool_calls"][0]["function"]
    function["arguments"] = json.dumps(function["arguments"])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    _assert_chat_and_responses_messages(
        client,
        messages,
        expected,
        tools=tools,
        tool_choice="auto",
    )


@pytest.mark.parametrize(
    "roles", [("system", "system"), ("system", "developer"), ("developer", "system")]
)
def test_chat_and_responses_merge_instruction_messages_identically(client, roles):
    messages = [
        {"role": roles[0], "content": "Be concise."},
        {"role": roles[1], "content": "Preserve exact paths."},
        {"role": "user", "content": "Say hello."},
    ]
    _assert_chat_and_responses_messages(
        client,
        messages,
        [
            {"role": "system", "content": "Be concise.\n\nPreserve exact paths."},
            messages[2],
        ],
    )


def test_responses_endpoint_places_function_output_image_after_tool_result(client):
    image_url = "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    # Exercise real templating here: placement of the image in the prompt is the contract.
    with _endpoint() as fake:
        fake.template.side_effect = apply_chat_template
        response = _post(
            client,
            "/responses",
            input=_function_result(
                [_input_image(image_url, detail="high")], name="view_image"
            ),
        )
    assert response.status_code == 200
    prompt = fake.generate.call_args.kwargs["prompt"]
    assert prompt.index("Tool:") < prompt.index("<image>") and image_url not in prompt
    assert fake.generate.call_args.kwargs["image"] == [image_url]


def test_responses_endpoint_rejects_image_file_id(client):
    response = _post(
        client,
        "responses",
        input=_function_result([dict(type="input_image", file_id="file-image")]),
    )
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "input_image.file_id is not supported by this server. Provide image_url instead."
    )


def test_responses_input_tokens_endpoint_forwards_adapter_path(client):
    generator = NS(
        _cpu_preprocess=MagicMock(return_value={"input_ids": mx.array([[1, 2, 3]])})
    )
    with _endpoint(generator=generator) as fake:
        response = _post(client, "/responses/input_tokens", adapter_path="adapter-a")
    assert response.status_code == 200 and response.json() == {"input_tokens": 3}
    assert fake.cache.call_args.args == ("demo", "adapter-a")


def test_responses_previous_response_id_replays_stored_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    with _endpoint() as fake:
        fake.generate.side_effect = [
            _result("First answer", prompt_tokens=3, generation_tokens=2),
            _result("Second answer", prompt_tokens=7, generation_tokens=2),
        ]
        first = _post(client, "responses", input="First")
        assert first.status_code == 200
        previous = first.json()["id"]
        second = _post(
            client, "responses", input="Second", previous_response_id=previous
        )
    assert second.status_code == 200
    assert fake.template.call_args_list[1].args[2] == [
        _msg("First"),
        _msg("First answer", "assistant"),
        _msg("Second"),
    ]
    assert client.get(f"/v1/responses/{previous}").status_code == 200
    items = client.get(f"/v1/responses/{previous}/input_items")
    assert (
        items.status_code == 200
        and items.json()["data"][0]["content"][0]["text"] == "First"
    )


@pytest.mark.parametrize(
    "kind,stream", [("shell", False), ("shell", True), ("function", True)]
)
def test_responses_native_tool_calls(client, kind, stream):
    server.response_store.clear()
    server.response_store_order.clear()
    name, args = (
        ("shell", {"command": "pwd"})
        if kind == "shell"
        else ("get_weather", {"location": "SF"})
    )
    result = _result(
        "<tool_call>" + json.dumps(dict(name=name, arguments=args)) + "</tool_call>",
        finish_reason="stop",
    )
    tool = (
        {"type": "shell"}
        if kind == "shell"
        else dict(
            type="function",
            name=name,
            parameters=dict(type="object", properties={"location": {"type": "string"}}),
        )
    )
    with _endpoint(result=result, chunks=[result], parser=_JSON_TOOLS):
        response = _post(
            client, "responses", input="run tool", tools=[tool], stream=stream
        )
    assert response.status_code == 200
    if not stream:
        item = response.json()["output"][0]
        assert item["type"] == "shell_call" and item["action"] == dict(
            type="exec", command="pwd"
        )
    elif kind == "shell":
        assert (
            '"type": "shell_call"' in response.text
            and '"command": "pwd"' in response.text
        )
        assert "<tool_call>" not in response.text
    else:
        done = next(
            data
            for event, data in _sse_events(response.text)
            if event == "response.function_call_arguments.done"
        )
        assert done["item_id"].startswith("fc_") and done["name"] == name
        assert json.loads(done["arguments"]) == args


@pytest.mark.parametrize(
    "api,family",
    [
        ("responses", "cohere2_moe"),
        ("chat", "cohere2_moe"),
        ("messages", "gemma4"),
        ("messages", "custom"),
    ],
)
def test_endpoint_thinking_markers(client, api, family):
    tokens, options, expected = _THINKING_CASES[family]
    preopened = family == "cohere2_moe"
    payload = dict(options, max_tokens=16)
    if api == "responses":
        payload["reasoning"] = {"effort": "high", "summary": "auto"}
    elif not preopened:
        payload["enable_thinking"] = True
    with _endpoint(
        model_type=family,
        template="prompt<|START_THINKING|>" if preopened else "prompt",
        chunks=[
            _result(tokens[0].text, generation_tokens=1),
            _result("".join(t.text for t in tokens[1:]), finish_reason="stop"),
        ],
        generator=None if api == "responses" else _streaming(tokens, 8),
    ) as fake:
        response = _post(client, api, stream=True, **payload)
    assert _thinking_text(response, api) == expected
    if api == "responses":
        _assert_fields(
            fake.template.call_args.kwargs,
            enable_thinking=True,
            reasoning=True,
            reasoning_effort="high",
        )
    elif api == "chat":
        assert _joined(_deltas(response), "reasoning") == expected[0]
    assert all(
        marker not in response.text
        for marker in (
            "<|channel>",
            "<channel|>",
            "<|END_THINKING|>",
            "<|START_TEXT|>",
            "<|END_TEXT|>",
            *options.values(),
        )
    )


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_stream_endpoints_do_not_clear_mlx_cache_on_close(client, api):
    with _endpoint(generator=_streaming([_token("ok", finish_reason="stop")])):
        with (
            patch.object(openai.mx, "clear_cache") as clear,
            patch.object(openai.gc, "collect") as collect,
        ):
            response = _post(
                client,
                api,
                stream=True,
                **({"max_tokens": 4} if api == "chat" else {"max_output_tokens": 4}),
            )
    assert response.status_code == 200
    clear.assert_not_called()
    collect.assert_not_called()


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_v1_stream_endpoints_reject_over_context_before_sse(
    client, monkeypatch, api, stream
):
    generator = _streaming([])
    error = server.PromptTooLongError(
        "Request needs 9 context tokens (5 prompt + 4 max generation), but MAX_KV_SIZE is 8."
    )
    (
        generator.validate_context_budget if stream else generator.generate
    ).side_effect = error
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    with _endpoint(generator=generator):
        response = _post(
            client,
            api,
            stream=stream,
            **({"max_tokens": 4} if api == "chat" else {"max_output_tokens": 4}),
        )
    assert (
        response.status_code == 400 and "MAX_KV_SIZE is 8" in response.json()["detail"]
    )
    if stream:
        generator.generate.assert_not_called()


@pytest.mark.parametrize("encoding", ["base64", "data-uri", "path"])
def test_chat_completions_decodes_input_audio_base64(client, encoding):
    raw = b"RIFF$\x00\x00\x00WAVEfmt "
    data = base64.b64encode(raw).decode("ascii")
    if encoding == "data-uri":
        data = "data:audio/wav;base64," + data
    elif encoding == "path":
        data = "/tmp/audio.wav"
    with _endpoint() as fake:
        response = _post(
            client,
            "/chat/completions",
            messages=[
                _msg(
                    [
                        dict(type="text", text="Describe the audio."),
                        dict(
                            type="input_audio",
                            input_audio=dict(data=data, format="wav"),
                        ),
                    ]
                )
            ],
        )
    assert response.status_code == 200
    audio = fake.generate.call_args.kwargs["audio"]
    assert audio == [data] if encoding == "path" else audio[0].getvalue() == raw


def test_generation_metrics_record_speculative_stats():
    metrics = generation.GenerationMetrics()
    expected = dict(draft_kind="dflash", draft_rounds=3, draft_n_accepted=4, draft_n=9)
    metrics.record_chunk(NS(generation_tokens=1, emitted_at=10.0))
    metrics.record_chunk(NS(generation_tokens=6, emitted_at=10.5, **expected))
    _assert_fields(vars(metrics), **expected)


def test_chat_completions_streaming_emits_timings_on_finish(client):
    tokens = [
        _token(text, i, finish, prompt_tps=20.0, cached_tokens=2)
        for i, (text, finish) in enumerate([("hi", None), ("!", "stop")])
    ]
    response = _stream_response(
        client, tokens, prompt_tokens=10, stream_options={"include_usage": True}
    )
    choices, usage, final = _chat_events(response, "stop")
    assert usage["choices"] == [] and usage["timings"]["cache_n"] == 2
    assert usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    tokens = [c for c in choices if c["choices"][0]["delta"].get("content") is not None]
    assert tokens[0]["timings"]["predicted_per_second"] is None
    assert tokens[1]["timings"]["predicted_per_second"] > 0
    assert final["timings"]["predicted_per_second"] > 0
    assert (
        usage["timings"]["predicted_per_second"]
        == final["timings"]["predicted_per_second"]
    )


@pytest.mark.parametrize("api", ["chat", "messages"])
def test_response_template_tool_calls(client, api):
    config = NS(model_type="muse_glimmer")
    tool = _tool(api=api)
    if api == "messages":
        config.thinking_start_token = "to=self<|message|>"
        config.thinking_end_token = "<|eom|>"
        tool["input_schema"].update(
            properties={"city": {"type": "string"}}, required=["city"]
        )
    payload = (
        dict(stream=True, stream_options={"include_usage": True})
        if api == "chat"
        else dict(thinking=dict(type="enabled", budget_tokens=4), max_tokens=8)
    )
    token = _token(_MUSE_CALL, finish_reason="stop", prompt_tps=20, cached_tokens=2)
    with _endpoint(
        config=config,
        processor=NS(config=config, tokenizer=_MuseResponseTemplateTokenizer()),
        result=_result(_MUSE_CALL, prompt_tokens=7, generation_tokens=6),
        generator=_streaming([token], 10) if api == "chat" else None,
        parser=load_tool_module("atem"),
    ):
        response = _post(client, api, tools=[tool], **payload)
    assert response.status_code == 200
    if api == "chat":
        choices, usage, tool_event = _chat_events(response, "tool_calls")
        deltas = [c["choices"][0]["delta"] for c in choices]
        call = tool_event["choices"][0]["delta"]["tool_calls"][0]["function"]
        assert tool_event.get("usage") is None
        assert _joined(deltas, "reasoning_content") == "I need the weather tool."
        assert _joined(deltas, "content") == ""
        assert usage["choices"] == []
        assert usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
        arguments = json.loads(call["arguments"])
    else:
        body = response.json()
        assert body["stop_reason"] == "tool_use"
        assert body["content"][0] == dict(
            type="thinking", thinking="I need the weather tool.", signature=""
        )
        call = body["content"][1]
        assert call["type"] == "tool_use"
        arguments = call["input"]
    assert call["name"] == "get_weather" and arguments == {"city": "Warsaw"}
    assert "to=self" not in response.text and "<atem:" not in response.text


def test_chat_completions_endpoint_falls_back_from_video_to_images(client):
    from mlx_vlm.generate import video

    frames = [object(), object()]
    with (
        _endpoint(model_type="mage_vl") as fake,
        patch.object(
            video, "sample_video_frames", return_value=(frames, 2.0)
        ) as sample,
    ):
        response = _post(
            client,
            "/chat/completions",
            messages=[
                _msg(
                    [
                        dict(type="video_url", video_url={"url": "clip.mp4"}),
                        dict(type="text", text="Describe this video."),
                    ]
                )
            ],
        )
    assert response.status_code == 200
    _assert_fields(fake.template.call_args.kwargs, num_images=2, video=None)
    _assert_fields(fake.generate.call_args.kwargs, image=frames, video=[])
    sample.assert_called_once_with(["clip.mp4"], 2.0, None)


def test_anthropic_messages_streaming_emits_tool_use_events(client):
    token = _token(
        '<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call> After the call.',
        finish_reason="stop",
    )
    response = _stream_response(
        client,
        [token],
        "messages",
        endpoint=dict(parser=_JSON_TOOLS),
        tools=[_tool(api="messages")],
    )
    assert response.status_code == 200
    for fragment in (
        '"type": "tool_use"',
        '"name": "get_weather"',
        '"type": "input_json_delta"',
        '"partial_json": "{\\"location\\": \\"SF\\"}"',
        '"text": "After the call."',
        '"stop_reason": "tool_use"',
    ):
        assert fragment in response.text


ANTHROPIC_TOOLS = [_tool(name, "messages") for name in ("get_time", "get_weather")]


def test_anthropic_count_tokens_applies_tool_choice(client):
    with (
        _endpoint() as fake,
        patch.object(anthropic, "prepare_inputs", return_value={}),
        patch.object(anthropic, "_count_prompt_tokens", return_value=7),
    ):
        response = _post(
            client,
            "/v1/messages/count_tokens",
            messages=[_msg("Weather in Paris?")],
            tools=ANTHROPIC_TOOLS,
            tool_choice=dict(type="tool", name="get_time"),
        )
    assert response.status_code == 200 and response.json() == {"input_tokens": 7}
    assert [t["function"]["name"] for t in fake.template.call_args.kwargs["tools"]] == [
        "get_time"
    ]


def test_cache_endpoints_report_disabled_stats_and_reset(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    response = client.get("/v1/cache/stats")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    response = client.post("/v1/cache/reset")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    manager = NS(
        stats_snapshot=MagicMock(return_value={"hits": 2, "pool_used": 1}),
        clear=MagicMock(),
    )
    monkeypatch.setattr(server.runtime, "apc_manager", manager)

    response = client.get("/v1/cache/stats")
    assert response.status_code == 200
    assert response.json() == {"hits": 2, "pool_used": 1, "enabled": True}

    response = client.post("/v1/cache/reset")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "status": "cleared"}
    manager.clear.assert_called_once_with()


class _Detokenizer:
    last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = str(token)

    def finalize(self):
        pass


class _Batch:
    unprocessed_prompts = []
    has_pending_prompts = False

    def __init__(self, uids, steps, **kwargs):
        self.uids, self.steps, self.kwargs = uids, steps, kwargs
        self.active, self.inserted, self.sizes = {}, [], []
        if kwargs.get("draft_model") is not None:
            self.apc = NS(prepare_prefill=MagicMock())

    def insert(self, *args, **kwargs):
        uid = next(self.uids)
        self.active[uid] = 0
        self.inserted.append(uid)
        return (uid,)

    def remove(self, uid):
        return self.active.pop(uid, None) is not None

    def next(self, **kwargs):
        self.sizes.append(len(self.active))
        responses = []
        for uid, step in sorted(self.active.items()):
            done = step + 1 == self.steps
            responses.append(
                NS(
                    uid=uid,
                    token=uid * 10 + step,
                    token_logprob=0.0,
                    finish_reason="length" if done else None,
                )
            )
            if done:
                del self.active[uid]
            else:
                self.active[uid] += 1
        return [], responses


class _IdleBatch(_Batch):
    closed = False

    @property
    def has_work(self):
        return bool(self.active)

    def close(self):
        self.closed = True


def _worker_setup(
    monkeypatch, *, steps=1, draft_kind=None, idle=False, initialize=True
):
    instances, uids = [], count(1)

    def make_batch(*args, **kwargs):
        batch = (_IdleBatch if idle else _Batch)(uids, steps, **kwargs)
        instances.append(batch)
        return batch

    monkeypatch.setattr(generation, "BatchGenerator", make_batch)
    monkeypatch.setattr(
        generation, "make_streaming_detokenizer", lambda _: _Detokenizer()
    )
    gen = _generator()

    def fake_initialize():
        gen.model = NS(language_model=object())
        gen.processor, gen.config, gen.tokenizer = (NS(), NS(), NS())
        gen.draft_model, gen.draft_kind = (object() if draft_kind else None), draft_kind

    if initialize:
        gen._initialize_model = fake_initialize
    gen._gpu_embed = lambda raw, images=None, apc_semantic_hash=None: (
        mx.array([[raw["request_id"]]], dtype=mx.int32),
        {},
    )
    return gen, instances


def _enqueue(gen, request_id=1, **kwargs):
    queue = Queue()
    gen.requests.put(
        generation.QueuedGenerationRequest(
            rqueue=queue,
            raw_inputs={"request_id": request_id},
            prompt_tokens=1,
            args=Args(**kwargs),
        )
    )
    return queue


@contextmanager
def _running(gen):
    worker = Thread(target=gen._run, daemon=True)
    worker.start()
    try:
        yield worker
    finally:
        gen._stop = True
        gen.requests.put(None)
        worker.join(timeout=2)
        assert not worker.is_alive()


def _drain(queue):
    context = queue.get(timeout=1)
    assert isinstance(context, server.GenerationContext)
    tokens = []
    while (item := queue.get(timeout=1)) is not None:
        tokens.append(item)
    return context, tokens


def _ready_generator(**kwargs):
    defaults = dict(
        wait_until_ready=lambda: None,
        _cpu_preprocess=lambda prompt, images, audio: {"input_ids": [1, 2, 3]},
        _cancel=lambda uid: None,
    )
    return _generator(**(defaults | kwargs))


def _capture_requests(gen, *, prompt_tokens=1, uid=None):
    queued = []

    def put(request):
        queued.append(request)
        request.rqueue.put(
            server.GenerationContext(
                uid=uid or len(queued), prompt_tokens=prompt_tokens
            )
        )

    gen.requests = NS(put=put)
    return queued


def _step_tokens(tokenizer, responses, *, progress=(), trim_space=True):
    gen, queue = _generator(), Queue()
    processor = NS(
        detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=trim_space)
    )
    active = {
        1: dict(
            rqueue=queue,
            streamer=_ServerTokenStreamer(
                tokenizer, server.make_streaming_detokenizer(processor)
            ),
            prompt_tps=None,
            cached_tokens=0,
        )
    }
    for token, finish in responses:
        row = NS(uid=1, token=token, token_logprob=0.0, finish_reason=finish)
        gen._step(NS(next=lambda **kw: (progress, [row])), active)
    return list(queue.queue)


class TestResponseGenerator:
    """Tests for the ResponseGenerator continuous batching engine."""

    def test_context_limit_precedes_image_hashing(self, monkeypatch):
        gen = _ready_generator(apc_manager=object(), apc_mode="block")
        gen._preprocess_request = lambda *a: dict(
            input_ids=mx.array([[1, 2, 3, 4, 5]]), pixel_values=mx.zeros((1, 3, 2, 2))
        )
        image_hash = MagicMock(wraps=apc.hash_image_payload)
        monkeypatch.setattr(apc, "hash_image_payload", image_hash)
        monkeypatch.setenv("MAX_KV_SIZE", "8")
        with pytest.raises(server.PromptTooLongError, match="MAX_KV_SIZE is 8"):
            gen.generate("prompt", args=Args(max_tokens=4))
        assert gen.requests.empty()
        image_hash.assert_not_called()

    def test_tokenizer_and_budget_criteria_share_lock(self):
        gen = _ready_generator(_tokenizer_lock=Lock())
        lock, active, maximum = Lock(), 0, 0

        def work():
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1

        def preprocess(*a, **kw):
            work()
            return {"input_ids": mx.array([[99]])}

        def criteria(*a):
            work()
            return object()

        gen._preprocess_request, gen._make_thinking_budget_criteria = (
            preprocess,
            criteria,
        )
        queued = _capture_requests(gen)

        def generate(_):
            _, tokens = gen.generate(
                "prompt", args=Args(max_tokens=1, thinking_budget=512)
            )
            tokens.close()

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(generate, range(4)))
        assert maximum == 1 and len(queued) == 4
        assert all(request.thinking_budget_criteria is not None for request in queued)

    def test_mutable_images_have_distinct_semantic_hashes(self):
        gen = _ready_generator(
            apc_manager=object(),
            apc_mode="block",
            model=NS(language_model=NS()),
            processor=NS(),
        )
        pixels = [mx.full((1, 3, 2, 2), value, dtype=mx.float32) for value in (0, 1)]
        gen._preprocess_request = MagicMock(
            side_effect=[
                dict(input_ids=mx.array([[1, 2]]), pixel_values=value)
                for value in pixels
            ]
        )
        queued = _capture_requests(gen, prompt_tokens=2)
        for _ in pixels:
            _, tokens = gen.generate(
                "prompt", images=["mutable-image.png"], args=Args(max_tokens=1)
            )
            tokens.close()
        assert queued[0].images == queued[1].images
        assert queued[0].apc_semantic_hash != queued[1].apc_semantic_hash
        for request, value in zip(queued, pixels):
            assert request.apc_semantic_hash == apc.semantic_extra_hash(
                image_hash=hash_image_payload(pixel_values=value),
                model=gen.model.language_model,
                processor=gen.processor,
            )

    def test_runtime_context_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_KV_SIZE", "8")
        _reset_runtime(
            monkeypatch,
            model_cache={"config": NS(text_config=NS(max_position_embeddings=16))},
        )
        _assert_fields(
            server._server_runtime_snapshot(),
            loaded_context_size=16,
            configured_context_limit=8,
            effective_context_limit=8,
        )

    @pytest.mark.parametrize("value,expected", [("bad", 600.0), ("0", None)])
    def test_queue_timeout_settings(self, monkeypatch, value, expected):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", value)
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        assert server.get_token_queue_timeout() == expected

    @pytest.mark.parametrize("debug", [False, True])
    def test_decode_log_detail_and_frequency(self, monkeypatch, caplog, debug):
        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "2")
        caplog.set_level(
            logging.DEBUG if debug else logging.INFO, logger="mlx_vlm.server"
        )
        info = dict(
            request_id="req-1",
            queued_at=time.perf_counter() - 0.1,
            generated_tokens=0,
            decode_started_at=None,
        )
        for n in range(1, 4 if debug else 3):
            Generator._log_decode_progress(
                1, info, token=n, text=str(n), finish_reason="stop" if n == 3 else None
            )
        messages = [r.getMessage() for r in caplog.records]
        if debug:
            assert any(
                "Decode progress: request=req-1 generated_tokens=1" in m
                and "token_number=1 token_id=1 text='1'" in m
                for m in messages
            )
            assert not any("Token streamed:" in m for m in messages)
            assert any("Decode started: request=req-1" in m for m in messages)
            assert any(
                "Decode completed: request=req-1 generated_tokens=3" in m
                for m in messages
            )
        else:
            progress = [m for m in messages if m.startswith("Decode progress:")]
            assert len(progress) == 1 and "generated_tokens=2" in progress[0]
            assert all(
                field not in progress[0]
                for field in ("token_number=", "token_id=", "text=")
            )

    def test_prefill_progress_logging(self, caplog):
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        gen = Generator.__new__(Generator)
        prompt_batch = NS(
            _processed_prompt_columns=2,
            _inputs_embeds=mx.zeros((1, 4, 8)),
            uids=[1],
            _suffix_lens=[6],
            _cached_tokens_per_row=[0],
            _left_padding_per_row=[0],
            _right_pad_per_row=None,
        )
        active = {1: {"request_id": "req-1", "prefill_processed": -1}}

        gen._log_prefill_progress(NS(_prompt_batch=prompt_batch), active)

        assert "Prefill progress: request=req-1 tokens=2/6 (33.3%)" in caplog.text

    @pytest.mark.parametrize(
        "delayed", [False, True], ids=["timeout-cancels", "delayed-token"]
    )
    def test_token_queue_timeout(self, monkeypatch, delayed):
        cancelled = []
        gen = _ready_generator(_cancel=cancelled.append)
        queued = _capture_requests(gen, uid="req-1")
        monkeypatch.setattr(
            server.runtime.config, "token_queue_timeout", 0.5 if delayed else 0.01
        )
        _, tokens = gen.generate("hello")
        token = NS(text="hi")

        def deliver():
            queued[0].rqueue.put(token)
            queued[0].rqueue.put(None)

        if delayed:
            timer = Timer(0.15, deliver)
            timer.start()
            try:
                assert next(tokens) is token
                with pytest.raises(StopIteration):
                    next(tokens)
            finally:
                timer.join(timeout=1)
        else:
            with pytest.raises(RuntimeError, match="Timed out waiting for 0.01s"):
                next(tokens)
        assert cancelled == ([] if delayed else ["req-1"])

    def test_close_cancels_blocked_iterator(self):
        cancelled = []
        result = []

        class BlockingQueue(Queue):
            def __init__(self):
                super().__init__()
                self.waiting = Event()

            def get(self, *args, **kwargs):
                self.waiting.set()
                return super().get(*args, **kwargs)

        rqueue = BlockingQueue()
        token_iter = generation._TokenIterator(rqueue, "req-1", cancelled.append, None)

        def consume():
            try:
                result.append(next(token_iter))
            except Exception as exc:
                result.append(exc)

        thread = Thread(target=consume)
        thread.start()
        assert rqueue.waiting.wait(timeout=1.0)

        token_iter.close()

        assert cancelled == ["req-1"]

        rqueue.put(None)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert isinstance(result[0], StopIteration)

    def test_step_streams_spm_subword_tokens_immediately(self):
        tokenizer = NS(
            vocab={"▁hello": 0, "world": 1, "!": 2},
            decode=lambda tokens: "".join(
                {0: " hello", 1: "world", 2: "!"}[t] for t in tokens
            ).lstrip(),
        )
        items = _step_tokens(
            tokenizer,
            [(0, None), (1, None), (2, None), (99, "stop")],
            progress=[NS(uid=1, prompt_tps=184.431, cached_tokens=7)],
        )
        assert [t.text for t in items[:-1]] == ["hello", "world", "!", ""]
        assert items[-1] is None
        _assert_fields(
            vars(items[0]), prompt_tps=pytest.approx(184.431), cached_tokens=7
        )

    def test_finalize_flushes_incomplete_utf8(self):
        tokenizer = NS(
            vocab={"<0xF0>": 0, "<0x9F>": 1},
            decode=lambda tokens: bytes({0: 0xF0, 1: 0x9F}[t] for t in tokens).decode(
                "utf-8", errors="replace"
            ),
        )
        processor = NS(detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False))
        streamer = _ServerTokenStreamer(
            tokenizer, server.make_streaming_detokenizer(processor)
        )
        assert streamer.advance(0, None) == streamer.advance(1, None) == ""
        assert streamer.finalize() == "\ufffd"

    def test_run_batches_eight_streaming_requests(self, monkeypatch):
        gen, batches = _worker_setup(monkeypatch, steps=2)
        queues = [_enqueue(gen, i, max_tokens=2) for i in range(8)]
        with _running(gen):
            results = [_drain(queue) for queue in queues]
        assert batches[0].inserted == list(range(1, 9)) and batches[0].sizes[:2] == [
            8,
            8,
        ]
        assert len({ctx.uid for ctx, _ in results}) == 8
        for ctx, tokens in results:
            assert ctx.prompt_tokens == 1
            assert [(t.text, t.finish_reason) for t in tokens] == [
                (str(ctx.uid * 10), None),
                (str(ctx.uid * 10 + 1), "length"),
            ]

    @pytest.mark.parametrize("draft_kind", ["dflash", "eagle3", "mtp"])
    def test_speculative_batch_options_and_apc(self, monkeypatch, draft_kind):
        gen, batches = _worker_setup(monkeypatch, draft_kind=draft_kind)
        monkeypatch.setattr(generation, "_get_draft_block_size_from_env", lambda: 6)
        manager = gen.apc_manager = NS(close=MagicMock())
        gen.prefill_step_size = 3072
        queues = [_enqueue(gen, i, max_tokens=1, temperature=0) for i in range(2)]
        with _running(gen):
            for queue in queues:
                _, tokens = _drain(queue)
                assert len(tokens) == 1 and tokens[0].finish_reason == "length"
        batch = batches[0]
        _assert_fields(
            batch.kwargs,
            draft_model=gen.draft_model,
            draft_kind=draft_kind,
            draft_block_size=6,
            greedy_sampling=True,
            compute_logprobs=False,
            prefill_step_size=3072,
            apc_manager=manager,
        )
        assert batch.apc.prepare_prefill.call_count == 2 and batch.sizes == [2]
        batch.apc.prepare_prefill.assert_called_with(1, prefill_step_size=3072)
        manager.close.assert_called_once_with()

    def test_new_sampler_recreates_idle_batch(self, monkeypatch):
        gen, batches = _worker_setup(monkeypatch, idle=True)
        gen._make_sampler = lambda args: f"sampler-{args.temperature}"
        with _running(gen):
            for i, temperature in enumerate([0.0, 0.6]):
                _, tokens = _drain(
                    _enqueue(gen, i, max_tokens=1, temperature=temperature)
                )
                assert len(tokens) == 1 and tokens[0].finish_reason == "length"
        assert [b.kwargs["sampler"] for b in batches] == ["sampler-0.0", "sampler-0.6"]
        assert batches[0].closed

    def test_generate_arguments_to_generate_kwargs(self):
        args = Args()
        _assert_fields(
            vars(args),
            max_tokens=server.DEFAULT_MAX_TOKENS,
            temperature=server.DEFAULT_TEMPERATURE,
            enable_thinking=False,
            logit_bias=None,
        )
        options = dict(
            max_tokens=50,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={3: -0.5},
            enable_thinking=False,
            thinking_budget=100,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[lambda t, logits: logits],
        )
        args = Args(**options, tenant_id="tenant-a")
        _assert_fields(args.to_generate_kwargs(), **options, apc_tenant="tenant-a")
        assert _generator()._make_sampler(args).top_k == options["top_k"]

    @pytest.mark.parametrize(
        "ids,wrapped",
        [([1, 10, 3], True), ([1, 10, 3, 20], False), ([1, 2, 3], True)],
        ids=["open", "closed", "self-opening"],
    )
    def test_structured_processors_wait_for_thinking(self, monkeypatch, ids, wrapped):
        repetition, structured = (lambda t, l: l), (lambda t, l: l)
        monkeypatch.setattr(
            generation,
            "make_logits_processors",
            lambda *a: [repetition] if wrapped else [],
        )
        gen = _generator(
            tokenizer=NS(
                encode=lambda text, **kw: {"<think>": [10], "</think>": [20]}[text]
            )
        )
        args = Args(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured],
        )
        processors = gen._make_logits_processors(args, mx.array([ids]))
        if wrapped:
            assert processors[0] is repetition
            assert isinstance(processors[1], generation.ThinkingAwareLogitsProcessor)
            assert processors[1].processor is structured
        else:
            assert processors == [structured]

    @pytest.mark.parametrize(
        "options,env,expected",
        [
            (
                dict(max_tokens=256, enable_thinking=True),
                "0",
                dict(max_tokens=256, enable_thinking=True),
            ),
            (
                dict(reasoning_effort="low"),
                "0",
                dict(enable_thinking=True, reasoning=True, reasoning_effort="low"),
            ),
            ({}, "1", dict(enable_thinking=True)),
            ({}, "0", dict(enable_thinking=False)),
        ],
        ids=["explicit", "effort", "default-on", "default-off"],
    )
    def test_build_generation_arguments(self, monkeypatch, options, env, expected):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", env)
        req = _chat_request(**options)
        if not options:
            assert "enable_thinking" not in req.model_fields_set
        _assert_fields(vars(server._build_gen_args(req)), **expected)
        if "max_tokens" in options:
            # Older callers pass plain objects without Pydantic field tracking.
            legacy = NS(**req.model_dump())
            legacy.repetition_context_size = None
            _assert_fields(vars(server._build_gen_args(legacy)), **expected)

    def test_server_cli_sets_thinking_defaults(self, monkeypatch):
        flags = [
            ("model", "PRELOAD_MODEL", "demo"),
            ("image-model", "PRELOAD_IMAGE_MODEL", "image-demo"),
            ("tts-model", "PRELOAD_TTS_MODEL", "tts-demo"),
            ("stt-model", "PRELOAD_STT_MODEL", "stt-demo"),
            ("reranker-model", "PRELOAD_RERANKER_MODEL", "reranker-demo"),
            ("thinking-budget", "THINKING_BUDGET", "128"),
            ("thinking-start-token", "THINKING_START_TOKEN", "<|START_THINKING|>"),
            ("thinking-eos-token", "THINKING_END_TOKEN", "<|END_THINKING|>"),
            ("api-key", "SERVER_API_KEY", "admin-token"),
        ]
        expected = {"MLX_VLM_" + env: value for _, env, value in flags}
        expected["MLX_VLM_ENABLE_THINKING"] = "1"
        argv = [
            "mlx_vlm.server",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--enable-thinking",
        ]
        argv += [arg for flag, _, value in flags for arg in ("--" + flag, value)]
        monkeypatch.setattr(sys, "argv", argv)
        with patch.dict(os.environ), patch.object(cli.uvicorn, "run") as run:
            for key in [
                *expected,
                "MLX_VLM_PRELOAD_ADAPTER",
                "MLX_VLM_VISION_CACHE_SIZE",
                "MLX_VLM_MAX_TOKENS",
                "PREFILL_STEP_SIZE",
                "KV_GROUP_SIZE",
                "KV_QUANT_SCHEME",
                "QUANTIZED_KV_START",
            ]:
                os.environ.pop(key, None)
            cli.main()
            _assert_fields(os.environ, **expected)
            assert run.call_args.kwargs["host"] == "127.0.0.1"

    def test_lifespan_continues_when_optional_preload_fails(self, monkeypatch):
        kinds = dict(
            MODEL="text_generation",
            TTS_MODEL="audio_tts",
            STT_MODEL="audio_stt",
            EMBEDDING_MODEL="embedding",
            RERANKER_MODEL="reranker",
        )
        for key, kind in kinds.items():
            monkeypatch.setenv("MLX_VLM_PRELOAD_" + key, kind)
        calls = []

        def load(model_path, adapter_path=None, *, model_kind="auto"):
            calls.append(model_kind)
            if model_kind == "audio_stt":
                raise server.HTTPException(
                    status_code=500, detail="Failed to load audio model: boom"
                )
            return NS(), None, NS(model_type=model_kind)

        monkeypatch.setattr(server._app_module, "get_cached_model", load)
        monkeypatch.setattr(server.runtime, "audio_queue", None)
        monkeypatch.setattr(server.runtime, "preload_failures", {})

        async def run():
            async with server._app_module.lifespan(server.app):
                pass

        asyncio.run(run())
        assert calls == list(kinds.values())
        failure = server.runtime.preload_failures["audio_stt"]
        assert (
            failure["model"] == "audio_stt"
            and "Failed to load audio model" in failure["error"]
        )
        assert "audio_tts" not in server.runtime.preload_failures

    @pytest.mark.parametrize("image", [False, True])
    def test_gpu_embed_hashes_pixel_values_without_image_ref(self, image):
        embedding = NS(
            to_dict=lambda: dict(
                inputs_embeds=mx.zeros((1, 2, 4)), position_ids=None, rope_deltas=None
            )
        )
        gen = NS(
            model=NS(get_input_embeddings=lambda *a, **kw: embedding), vision_cache=None
        )
        pixels = mx.array([[[[1.0, 2.0]]]])
        semantic_hash = (
            apc.semantic_extra_hash(image_hash=hash_image_payload(pixel_values=pixels))
            if image
            else None
        )
        raw = dict(input_ids=mx.array([[1, 2]]), attention_mask=mx.array([[1, 1]]))
        if image:
            raw["pixel_values"] = pixels
        _, kwargs = Generator._gpu_embed(
            gen, raw, images=None, apc_semantic_hash=semantic_hash
        )
        assert "position_ids" not in kwargs and "rope_deltas" not in kwargs
        assert (
            kwargs["_apc_semantic_hash"] == semantic_hash
            if image
            else "_apc_semantic_hash" not in kwargs
        )

    @pytest.mark.parametrize(
        "format,expected",
        [
            (
                dict(
                    type="json_schema",
                    name="animal",
                    schema=dict(
                        type="object",
                        properties={"animal": {"type": "string"}},
                        required=["animal"],
                    ),
                ),
                {"required": ["animal"]},
            ),
            ({"type": "json_object"}, {"type": "object"}),
            ({"type": "object"}, {"type": "object"}),
        ],
    )
    def test_response_format_schema(self, format, expected):
        schema = server._extract_response_format_schema(
            NS(response_format=None, text={"format": format})
        )
        _assert_fields(schema, **expected)

    def test_structured_processor_factory(self):
        req = NS(
            response_format=dict(
                type="json_schema",
                json_schema=dict(name="animal", schema={"type": "object"}),
            ),
            text=None,
        )
        proc = NS(tokenizer=object())

        with patch.object(
            server, "build_json_schema_logits_processor", return_value="processor"
        ) as mock_build:
            processors = server._build_structured_logits_processors(req, proc)

        assert processors == ["processor"]
        assert mock_build.call_args.args[1] == {"type": "object"}


@pytest.mark.parametrize(
    "text,preopened,expected,tags",
    [
        ("<think>Thinking.</think>Answer.", False, ("Thinking.", "Answer."), 2),
        ("got it<channel|>42", False, ("got it", "42"), 0),
        ("thought\ngot it<channel|>42", False, ("got it", "42"), 0),
        ("Unterminated reasoning", True, ("Unterminated reasoning", ""), 0),
    ],
)
def test_thinking_text(text, preopened, expected, tags):
    assert server._split_thinking(text, starts_in_thinking=preopened) == expected
    assert server._count_thinking_tag_tokens(text) == tags


def _feed_thinking(state, chunks, last=False):
    return [
        state.feed(text, last=last and i == len(chunks) - 1)
        for i, text in enumerate(chunks)
    ]


def _thoughts(deltas):
    return tuple(
        _joined([vars(delta) for delta in deltas], key)
        for key in ("reasoning", "content")
    )


@pytest.mark.parametrize(
    "chunks,field,expected",
    [
        (["hello <", ""], "content", ["hello ", "<"]),
        (["<think>", "cut off </thi"], "reasoning", [None, "cut off </thi"]),
    ],
)
def test_incomplete_thinking_markers(chunks, field, expected):
    deltas = _feed_thinking(server.ThinkingStreamState(), chunks, last=True)
    assert [getattr(delta, field) for delta in deltas] == expected


@pytest.mark.parametrize(
    "family,enabled",
    [("gemma4", False), ("gemma4", True)],
)
def test_thinking_stream_markers(family, enabled):
    tokens, options, expected = _THINKING_CASES[family]
    state = server.ThinkingStreamState(enable_thinking=enabled, **options)
    assert _thoughts(_feed_thinking(state, [t.text for t in tokens])) == expected


def test_response_template_thinking_stream():
    state = server.make_response_stream_state(
        NS(tokenizer=_MuseResponseTemplateTokenizer()),
        thinking_start_token="unused-start",
        thinking_end_token="unused-end",
    )
    deltas = _feed_thinking(
        state,
        [
            "to=self<|mes",
            "sage|>Muse reasoning.<|eom|><|start|>assistant ",
            "to=user<|message|>Muse answer.",
        ],
        last=True,
    )
    assert _thoughts(deltas) == ("Muse reasoning.", "Muse answer.")
    assert any(delta.thinking_closed for delta in deltas)


def test_kv_bits_independent_of_model_path(monkeypatch):
    monkeypatch.setenv("KV_BITS", "3.5")
    monkeypatch.setenv("MAX_KV_SIZE", "0")
    assert generation.get_quantized_kv_bits() == 3.5
    assert generation.get_max_kv_size("mlx-community/gemma-4-31B-it-QAT-mxfp4") is None


class TestRuntimeConfig:
    def test_apply_changes_validates_and_coerces(self):
        cfg = RuntimeConfig.from_env()
        applied, rejected = cfg.apply_changes(
            {
                "kv_bits": "8",  # str -> float coercion
                "apc_enabled": "true",
                "vision_cache_size": "50",
                "not_a_knob": 1,
            }
        )
        assert applied == {"kv_bits": 8.0, "apc_enabled": True, "vision_cache_size": 50}
        assert rejected == [{"name": "not_a_knob", "reason": "unknown knob"}]
        assert cfg.kv_bits == 8.0
        assert cfg.apc_enabled is True
        assert cfg.vision_cache_size == 50

    def test_reload_kinds_excludes_live_knobs(self):
        cfg = RuntimeConfig.from_env()
        before = cfg.fingerprint()
        applied, _ = cfg.apply_changes(
            {"max_kv_size": 4096, "token_queue_timeout": 30.0}
        )
        assert applied == {"max_kv_size": 4096, "token_queue_timeout": 30.0}
        assert cfg.reload_kinds(applied) == set()
        assert cfg.fingerprint() == before

    @pytest.mark.parametrize(
        "payload",
        [[1, 2, 3], {"op": "bogus", "values": {}}, {"op": "replace", "values": "x"}],
    )
    def test_settings_patch_rejects_invalid_payload(self, client, monkeypatch, payload):
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        assert client.patch("/v1/settings", json=payload).status_code == 400

    def test_max_kv_size_is_live_context_limit(self, monkeypatch):
        import mlx_vlm.server.generation as generation

        monkeypatch.setattr(server.runtime.config, "max_kv_size", 4096)
        assert generation.get_configured_context_limit() == 4096

        monkeypatch.setattr(server.runtime.config, "max_kv_size", None)
        monkeypatch.delenv("MAX_KV_SIZE", raising=False)
        assert generation.get_configured_context_limit() is None


def test_runtime_config_enum_knobs_reject_invalid():
    cfg = RuntimeConfig.from_env()
    applied, rejected = cfg.apply_changes({"kv_quant_scheme": "bogus"})
    assert applied == {}
    assert rejected[0]["name"] == "kv_quant_scheme"
    assert "bogus" in rejected[0]["reason"]
    assert cfg.kv_quant_scheme == "uniform"

    applied, rejected = cfg.apply_changes({"kv_quant_scheme": "turboquant"})
    assert applied == {"kv_quant_scheme": "turboquant"}
    assert rejected == []


class TestReranking:
    @pytest.mark.parametrize("preloaded", [False, True])
    def test_endpoint(self, client, monkeypatch, preloaded):
        monkeypatch.delenv("MLX_VLM_PRELOAD_RERANKER_MODEL", raising=False)
        body = dict(query="query", documents=["first", "second", "third"])
        if not preloaded:
            missing = client.post("/v1/rerank", json=body)
            assert missing.status_code == 400
            assert "No reranker model specified" in missing.json()["detail"]
        if preloaded:
            monkeypatch.setenv("MLX_VLM_PRELOAD_RERANKER_MODEL", "reranker")
        else:
            body.update(model="reranker", top_n=2, return_documents=True)
        with (
            _endpoint() as fake,
            patch.object(
                reranking, "score_documents", return_value=([0.2, 0.9, 0.5], 12)
            ),
        ):
            response = client.post("/v1/rerank", json=body)
        assert response.status_code == 200
        if preloaded:
            assert response.json()["model"] == "reranker"
        else:
            assert response.json() == dict(
                model="reranker",
                results=[
                    dict(index=i, relevance_score=s, document=d)
                    for i, s, d in [(1, 0.9, "second"), (2, 0.5, "third")]
                ],
                usage=dict(prompt_tokens=12, total_tokens=12),
            )
        fake.cache.assert_called_once_with("reranker", model_kind="reranker")

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("  text  ", dict(text="text")),
            ({"text": " text "}, dict(text="text")),
            ({"image_url": {"url": " image.png "}}, dict(image="image.png")),
            ({"video": " video.mp4 "}, dict(video="video.mp4")),
        ],
    )
    def test_normalization(self, value, expected):
        assert reranking.normalize_item(value, "query") == reranking.RerankItem(
            **expected
        )

    @pytest.mark.parametrize("value", ["", "   ", {}, {"text": " "}, {"image": {}}])
    def test_empty_items(self, value):
        with pytest.raises(ValueError):
            reranking.normalize_item(value, "query")

    @pytest.mark.parametrize(
        "family,image,instruction,error",
        [
            ("qwen3", True, "instruction", "do not support image or video"),
            ("modernbert", True, None, "do not support image or video"),
            (
                "modernbert",
                False,
                "rank legal documents",
                "do not support custom instructions",
            ),
        ],
    )
    def test_unsupported_inputs(self, family, image, instruction, error):
        query = reranking.RerankItem(
            **({"image": "query.png"} if image else {"text": "query"})
        )
        with pytest.raises(ValueError, match=error):
            reranking.score_documents(
                NS(),
                NS(),
                NS(model_type=family),
                query,
                [reranking.RerankItem(text="document")],
                instruction,
            )

    def test_multimodal_order(self):
        messages = reranking._vl_messages(
            reranking.RerankItem(text="query", image="query.png"),
            reranking.RerankItem(text="document", video="document.mp4"),
            "rank candidates",
        )
        assert messages[1]["content"] == [
            dict(type="text", text="<Instruct>: rank candidates"),
            dict(type="text", text="<Query>:"),
            dict(type="image"),
            dict(type="text", text="query"),
            dict(type="text", text="\n<Document>:"),
            dict(type="video"),
            dict(type="text", text="document"),
        ]

    def test_batch_order(self, monkeypatch):
        batches = []

        def score(model, processor, query, documents, instruction):
            batches.append([d.text for d in documents])
            return [float(d.text) for d in documents], len(documents)

        monkeypatch.setenv("MLX_VLM_RERANK_BATCH_SIZE", "2")
        monkeypatch.setattr(reranking, "_score_text_batch", score)
        result = reranking.score_documents(
            NS(),
            NS(),
            NS(model_type="qwen3"),
            reranking.RerankItem(text="query"),
            [reranking.RerankItem(text=str(i)) for i in range(5)],
            "instruction",
        )
        assert result == ([0.0, 1.0, 2.0, 3.0, 4.0], 5)
        assert batches == [["0", "1"], ["2", "3"], ["4"]]

    def test_sequence_classifier(self):
        inputs = dict(
            input_ids=[[1, 2, 3, 0], [1, 4, 5, 6]],
            attention_mask=[[1, 1, 1, 0], [1, 1, 1, 1]],
            token_type_ids=[[0, 0, 1, 0], [0, 0, 1, 1]],
        )
        tokenizer = MagicMock(
            spec=["model_max_length"],
            model_max_length=6,
            return_value={k: np.array(v) for k, v in inputs.items()},
        )
        model = MagicMock(return_value=NS(logits=mx.array([[-2.0], [2.0]])))
        scores, tokens = reranking.score_documents(
            model,
            tokenizer,
            NS(model_type="bert", max_position_embeddings=4),
            reranking.RerankItem(text="query"),
            [reranking.RerankItem(text=t) for t in ("first", "second")],
            None,
        )
        assert scores == pytest.approx([1 / (1 + math.exp(2)), 1 / (1 + math.exp(-2))])
        assert tokens == 7 and set(model.call_args.kwargs) == set(inputs)
        tokenizer.assert_called_once_with(
            ["query", "query"],
            ["first", "second"],
            padding=True,
            truncation=True,
            max_length=4,
            return_tensors="np",
        )

    def test_attention_and_pooling(self):
        padding = mx.array([[0, 1, 1], [1, 1, 0]])
        mask = reranking._attention_mask(padding)
        assert mask.shape == (2, 1, 3, 3)
        assert mask[:, 0].tolist() == [
            [[False, False, False], [False, True, False], [False, True, True]],
            [[True, False, False], [True, True, False], [False, False, False]],
        ]
        assert reranking._attention_mask(mx.ones((2, 3))) == "causal"
        model = NS(language_model=NS(lm_head=lambda hidden: hidden))
        tokenizer = NS(
            unk_token_id=None, convert_tokens_to_ids=lambda t: {"no": 0, "yes": 1}[t]
        )
        hidden = mx.array(
            [
                [[9.0, -9.0], [2.0, 4.0], [1.0, 5.0]],
                [[4.0, 1.0], [8.0, 2.0], [-9.0, 9.0]],
            ]
        )
        assert reranking._binary_scores(
            model, hidden, padding, tokenizer
        ) == pytest.approx([1 / (1 + math.exp(-4)), 1 / (1 + math.exp(6))])

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3],
            {"input_ids": [1, 2, 3]},
            NS(input_ids=[1, 2, 3]),
            NS(input_ids=[[1, 2, 3]]),
            mx.array([1, 2, 3]),
        ],
    )
    def test_input_ids(self, value):
        assert reranking._input_ids(value) == [1, 2, 3]

    def test_packaged_template(self, tmp_path, monkeypatch):
        (tmp_path / "chat_template.jinja").write_text("template", encoding="utf-8")
        processor = NS(chat_template=None, tokenizer=NS(chat_template=None))
        monkeypatch.setattr(reranking, "get_model_path", lambda path: tmp_path)
        reranking.ensure_chat_template(processor, "reranker")
        assert (
            processor.chat_template == processor.tokenizer.chat_template == "template"
        )

    @pytest.mark.parametrize("family", ["qwen3", "bert", "deberta_v2"])
    def test_loader_and_isolated_cache(self, monkeypatch, family):
        registry = server.ModelCacheRegistry()
        text_cache = dict(
            cache_key=("language", None, "text_generation"),
            model_kind="text_generation",
        )
        registry.set("text_generation", text_cache)
        monkeypatch.setattr(server.runtime, "model_cache", registry)
        model, processor = NS(config=NS(model_type=family)), object()
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, processor)
        )
        template = MagicMock()
        monkeypatch.setattr(
            server._app_module, "ensure_reranker_chat_template", template
        )
        if family == "deberta_v2":
            with pytest.raises(
                server.HTTPException, match="Unsupported reranker model type"
            ) as exc:
                server.get_cached_model("reranker", None, model_kind="reranker")
            assert exc.value.status_code == 400
        else:
            assert server.get_cached_model("reranker", None, model_kind="reranker") == (
                model,
                processor,
                model.config,
            )
            assert registry.for_kind("text_generation") is text_cache
            assert registry.for_kind("reranker")["cache_key"] == (
                "reranker",
                None,
                "reranker",
                server.runtime.config.fingerprint(kinds={"reranker"}),
            )
            if family == "bert":
                template.assert_not_called()


@dataclass
class _FakeAlignedResult:
    text: str
    sentences: list


@pytest.mark.parametrize(
    "item,expected",
    [
        ("just text", {"text": "just text"}),
        (
            _FakeAlignedResult(
                "Hello world. Bye.",
                [
                    dict(
                        text="Hello world.",
                        start=0.0,
                        end=0.9,
                        tokens=[
                            dict(id=1, text="Hello", start=0.0, duration=0.4, end=0.4)
                        ],
                    ),
                    dict(text="Bye.", start=1.0, end=1.3),
                ],
            ),
            dict(
                text="Hello world. Bye.",
                segments=[
                    dict(id=0, text="Hello world.", start=0.0, end=0.9),
                    dict(id=1, text="Bye.", start=1.0, end=1.3),
                ],
            ),
        ),
    ],
    ids=["plain-text", "aligned-sentences"],
)
def test_audio_result_serialization(item, expected):
    chunks = [
        json.dumps(
            server_audio._sanitize_for_json(server_audio._stt_item_to_dict(part))
        )
        + "\n"
        for part in server_audio._iter_stt_items(item)
    ]
    result = server_audio._transcription_result_from_chunks(chunks)
    _assert_fields(result, **expected)


@pytest.mark.parametrize(
    "results,expected",
    [
        (
            [
                GenerationResult(text="Hello", token=5, diffusion_canvas_index=1),
                GenerationResult(text=" world", token=6, diffusion_canvas_index=1),
                GenerationResult(
                    diffusion_block_complete=True, diffusion_canvas_index=1
                ),
                GenerationResult(text="!", token=7, diffusion_canvas_index=2),
                GenerationResult(
                    diffusion_block_complete=True, diffusion_canvas_index=2
                ),
                GenerationResult(text="", finish_reason="stop", token=7),
            ],
            [("Hello world", None), ("!", None), ("", "stop")],
        ),
        (
            [
                GenerationResult(is_draft=True, draft_text="[Mask]"),
                GenerationResult(
                    diffusion_block_complete=True, diffusion_canvas_index=1
                ),
                GenerationResult(text="Hi", token=4, diffusion_canvas_index=2),
                GenerationResult(text=".", finish_reason="stop", token=5),
            ],
            [("Hi.", "stop")],
        ),
    ],
    ids=["groups_by_block", "skips_drafts_and_empty_blocks"],
)
def test_diffusion_block_chunks(results, expected):
    chunks = generation._diffusion_block_chunks(iter(results))
    assert [(chunk.text, chunk.finish_reason) for chunk in chunks] == expected


@pytest.fixture
def settings_client(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("APC_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("MLX_VLM_CACHE_HOME", str(tmp_path / "cache"))
    _reset_runtime(monkeypatch, config=RuntimeConfig.from_env())
    monkeypatch.setattr(
        server._app_module, "is_image_generation_model", lambda _: False
    )

    # Exercise the real cache factory / APC manager while avoiding model downloads.
    generators = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.model = NS(make_cache=lambda: [KVCache()])
            self.processor = NS()
            self.config = NS(model_type="test")
            self.apc_manager = kwargs["apc_manager"]
            self.stopped = False
            generators.append(self)

        def wait_until_ready(self):
            return self.model, self.processor, self.config

        def stop_and_join(self):
            self.stopped = True
            if self.apc_manager is not None:
                self.apc_manager.close()

    monkeypatch.setattr(server._app_module, "ResponseGenerator", FakeGenerator)
    client = TestClient(server.app)
    yield client, generators
    for generator in generators:
        if not generator.stopped:
            generator.stop_and_join()
    client.close()


def test_apc_patch_reaches_next_model_request(settings_client, tmp_path):
    client, generators = settings_client
    cfg = server.runtime.config
    assert cfg.apc_enabled is False
    server.get_cached_model("demo")
    assert server.runtime.apc_manager is None
    settings = {
        "kv_quant_scheme": "turboquant",
        "apc_enabled": True,
        "apc_disk_enabled": True,
        "apc_disk_path": str(tmp_path / "live"),
        "apc_block_size": 32,
        "apc_num_blocks": 8,
        "apc_disk_max_gb": 0,
        "apc_memory_max_gb": 1,
        "apc_memory_reserve_gb": 0.25,
        "apc_disk_queue_max_gb": 0,
        "apc_disk_shard_max_blocks": 64,
        "apc_checkpoint_entries": 3,
        "apc_checkpoint_interval_tokens": 128,
        "apc_checkpoint_guard_tokens": 2,
    }
    response = client.patch("/v1/settings", json=settings)
    assert response.status_code == 200
    assert response.json()["applied"] == settings
    assert response.json()["rejected"] == []
    assert response.json()["reload_kinds"] == ["text_generation"]
    assert cfg.kv_quant_scheme == "turboquant" and cfg.apc_enabled is True
    assert len(generators) == 1  # Configuration changes apply at the next load.

    body = client.get("/v1/settings").json()
    assert settings.keys() <= {knob["name"] for knob in body["schema"]}
    assert {name: body["current"][name] for name in settings} == settings
    server.get_cached_model("demo")
    assert len(generators) == 2 and generators[0].stopped
    manager = server.runtime.apc_manager
    assert manager is generators[-1].apc_manager
    assert (manager.block_size, manager.num_blocks) == (32, 8)
    _apc_budget(manager, 1 << 30, 1 << 28, None, 0)
    assert manager._exact_cache_max == 3
    assert manager.checkpoint_interval_tokens == 128
    assert manager.exact_cache_guard_tokens == 2
    assert manager.disk.dir.parent == tmp_path / "live"
    assert manager.disk._shard_max_blocks == 64
    assert client.get("/v1/cache/stats").json()["memory_max_bytes"] == 1 << 30

    response = client.patch("/v1/settings", json=settings)
    assert response.json()["reload_kinds"] == []  # A no-op retains the model/cache.
    server.get_cached_model("demo")
    assert len(generators) == 2

    client.patch("/v1/settings", json={"apc_memory_max_gb": 0})
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.memory_max_bytes == 0
    assert all(not thread.is_alive() for thread in manager.disk._workers)

    body = client.patch(
        "/v1/settings", json={"op": "replace", "values": {"kv_quant_scheme": "uniform"}}
    ).json()
    assert body["op"] == "replace"
    assert cfg.kv_quant_scheme == "uniform" and cfg.apc_enabled is False
    server.get_cached_model("demo")
    assert server.runtime.apc_manager is None
    assert client.get("/v1/cache/stats").json() == {"enabled": False}


def _apc_budget(manager, memory, reserve, disk, queue):
    assert (manager.memory_max_bytes, manager.memory_reserve_bytes) == (memory, reserve)
    assert (manager.disk.max_bytes, manager.disk.queue_max_bytes) == (disk, queue)


def test_apc_zero_null_and_replace_override_environment(
    settings_client, monkeypatch, tmp_path
):
    client, _ = settings_client
    monkeypatch.setattr(apc, "_metal_working_set_bytes", lambda: 40 << 30)
    monkeypatch.setenv("APC_ENABLED", "1")
    monkeypatch.setenv("APC_DISK_ENABLED", "0")
    monkeypatch.setenv("APC_DISK_PATH", str(tmp_path / "environment"))
    budgets = "memory_max_gb memory_reserve_gb disk_max_gb disk_queue_max_gb".split()
    for name in budgets:
        monkeypatch.setenv("APC_" + name.upper(), "2")
    monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
    before_env = dict(os.environ)
    zero_values = {"apc_" + name: 0 for name in budgets}
    body = client.patch(
        "/v1/settings", json={**zero_values, "apc_disk_enabled": True}
    ).json()
    assert all(body["current"][name] == 0 for name in zero_values)
    server.get_cached_model("demo")
    manager = server.runtime.apc_manager
    _apc_budget(manager, 0, 0, None, 0)

    client.patch(
        "/v1/settings",
        json={**{name: None for name in zero_values}, "apc_disk_path": None},
    )
    server.get_cached_model("demo")
    manager = server.runtime.apc_manager
    _apc_budget(manager, 4 << 30, 4 << 30, 20 << 30, 1 << 30)
    assert manager.disk.dir.parent == apc.default_disk_path()

    body = client.patch("/v1/settings", json={"op": "replace", "values": {}}).json()
    assert body["reload_kinds"] == ["text_generation"]
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.memory_max_bytes == 2 << 30
    assert server.runtime.apc_manager.disk is None
    assert dict(os.environ) == before_env


@pytest.mark.parametrize(
    "name,value",
    [
        ("apc_block_size", 0),
        ("apc_num_blocks", -1),
        ("apc_checkpoint_entries", -1),
        ("apc_checkpoint_entries", 1.5),
        ("apc_checkpoint_guard_tokens", 0),
        ("apc_checkpoint_interval_tokens", -1),
        ("apc_disk_shard_max_blocks", 0),
        ("apc_memory_max_gb", -1),
        ("apc_memory_max_gb", "NaN"),
        ("apc_memory_reserve_gb", "inf"),
        ("apc_disk_queue_max_gb", 1e308),
        ("apc_disk_max_gb", -1),
        ("apc_num_blocks", True),
        ("apc_disk_path", []),
        ("apc_disk_enabled", None),
    ],
)
def test_invalid_apc_values_are_rejected_without_reload(settings_client, name, value):
    client, _ = settings_client
    client.patch("/v1/settings", json={"apc_enabled": True})
    before = client.get("/v1/settings").json()
    response = client.patch("/v1/settings", json={name: value})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == {} and body["rejected"][0]["name"] == name
    assert body["reload_kinds"] == []
    assert body["current"] == before["current"]
    assert body["fingerprint"] == before["fingerprint"]


@pytest.mark.parametrize("fail", [False, True])
def test_generation_worker_closes_disk_after_final_store(tmp_path, fail):
    disk = apc.DiskBlockStore(tmp_path)
    manager = apc.APCManager(num_blocks=1, disk=disk)
    generator = server.ResponseGenerator.__new__(server.ResponseGenerator)
    generator.apc_manager = manager

    def finish_request():
        cache = KVCache()
        cache.keys = cache.values = mx.ones((1, 1, 32, 4))
        cache.offset = 32
        manager.store_exact_cache(list(range(32)), [cache])
        if fail:
            raise RuntimeError("worker failed")

    generator._run_impl = MagicMock(side_effect=finish_request)
    errors = []

    def run_worker():
        try:
            generator._run()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run_worker, daemon=True)
    try:
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        if fail:
            assert len(errors) == 1 and str(errors[0]) == "worker failed"
        else:
            assert errors == []
        assert disk.num_exact_indexed == 1
        assert disk.pending_bytes == 0
        assert all(not thread.is_alive() for thread in disk._workers)
    finally:
        if any(thread.is_alive() for thread in disk._workers):
            manager.close()


@pytest.mark.parametrize(
    "mixed_text", [False, True], ids=["image-after-call", "text-with-image"]
)
def test_function_output_preserves_visual_input(mixed_text):
    image_url = (
        "https://example.com/result.png"
        if mixed_text
        else "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    )
    call_id = "call_analyze_image" if mixed_text else "call_view_image"
    output = (
        [
            {"type": "input_text", "text": "Rendered result"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        if mixed_text
        else [_input_image(image_url, detail="high")]
    )
    items = _function_result(
        output, name=None if mixed_text else "view_image", call_id=call_id
    )
    messages, images = _response_items_to_chat(items)
    assert images == [image_url]
    expected = [
        _msg(
            ("Rendered result\n" if mixed_text else "")
            + "[Image output attached in the next message]",
            role="tool",
            tool_call_id=call_id,
        ),
        _msg([{"type": "image"}], role="user"),
    ]
    assert (messages if mixed_text else messages[-2:]) == expected


def test_message_image_stays_on_its_original_user_turn():
    image_url = "https://example.com/first-turn.png"
    items = [
        _msg(
            [
                {"type": "input_text", "text": "First turn"},
                _input_image(image_url),
            ],
            type="message",
        ),
        _msg(
            [{"type": "output_text", "text": "I see it."}],
            role="assistant",
            type="message",
        ),
        _input_message("Second turn"),
    ]

    messages, images = _response_items_to_chat(items)
    assert images == [image_url]
    assert messages == [
        _msg([dict(type="text", text="First turn"), dict(type="image")]),
        _msg("I see it.", "assistant"),
        _msg("Second turn"),
    ]


def test_message_metadata_survives_image_extraction_without_mutating_input():
    image_url = "https://example.com/result.png"
    items = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Inspecting."}],
            "reasoning_content": "Use the saved path.",
            "reasoning": "Outdated alias.",
            "tool_calls": [
                {
                    "id": "call_saved",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/src/app.py"}',
                    },
                }
            ],
        },
        {
            "type": "message",
            "role": "tool",
            "tool_call_id": "call_saved",
            "name": "read_file",
            "content": [
                {"type": "input_text", "text": "File preview"},
                {"type": "input_image", "image_url": image_url},
            ],
        },
    ]
    original = copy.deepcopy(items)

    messages, images = _response_items_to_chat(items)

    assert images == [image_url]
    assert messages[0]["reasoning_content"] == "Use the saved path."
    assert messages[0]["reasoning"] == "Use the saved path."
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "/src/app.py"
    }
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call_saved",
        "name": "read_file",
        "content": "File preview",
    }
    assert messages[2] == {"role": "user", "content": [{"type": "image"}]}
    assert items == original


def test_unknown_function_output_blocks_remain_text():
    unknown = {"type": "custom_output", "value": {"answer": 42}}
    messages, images = _response_items_to_chat(
        _function_result([unknown], call_id="call_custom")
    )

    assert images == []
    assert json.loads(messages[0]["content"]) == [unknown]


@pytest.mark.parametrize(
    "chunks,start_marker,end_marker,expected,inside",
    [
        (["text<tool_call>"], "<tool_call>", "</tool_call>", "text<tool_call>", True),
        (
            ["Before ", "<tool_call>", '{"name": "a"}', " trailing"],
            "<tool_call>",
            "",
            "Before",
            True,
        ),
        (["A literal <tool"], "<tool_call>", "</tool_call>", "A literal <tool", False),
        (
            [*MINICPM_MULTICALL, ""],
            "<function",
            "</function>",
            "BeforeBetweenAfter",
            False,
        ),
        (
            ["<tool_call>a</tool_call>", "\n", "<tool_call>b</tool_call>", "\n"],
            "<tool_call>",
            "</tool_call>",
            "",
            False,
        ),
        (
            list("<tool_call>a</tool_call>\n<tool_call>b</tool_call>\n"),
            "<tool_call>",
            "</tool_call>",
            "",
            False,
        ),
        (
            ["<tool_call>a</tool_call>", "\n", "Done", "."],
            "<tool_call>",
            "</tool_call>",
            "Done.",
            False,
        ),
        (
            list("A<tool_call>x</tool_call> \n<tool_call>y</tool_call>B"),
            "<tool_call>",
            "</tool_call>",
            "A \nB",
            False,
        ),
        (
            list("A<tool_call>x</tool"),
            "<tool_call>",
            "</tool_call>",
            "A<tool_call>x</tool",
            True,
        ),
        (
            list("  <tool_call>x</tool_call>B"),
            "<tool_call>",
            "</tool_call>",
            "B",
            False,
        ),
        (
            ["<tool_call>x</tool_call> B", "  "],
            "<tool_call>",
            "</tool_call>",
            "B",
            False,
        ),
        (
            ["Before ", "[TOOL_CALLS]foo[ARGS]{}", "\nAfter"],
            "[TOOL_CALLS]",
            "",
            "Before After",
            False,
        ),
    ],
    ids=[
        "start-marker",
        "missing-end-marker",
        "unfinished-start-marker",
        "minicpm-character-chunks",
        "whitespace-between-calls",
        "whitespace-between-calls-character-chunks",
        "text-after-call",
        "whitespace-between-text",
        "unfinished-call",
        "leading-whitespace",
        "trailing-whitespace",
        "no-end-marker-ends-at-newline",
    ],
)
def test_tool_stream_finalization(chunks, start_marker, end_marker, expected, inside):
    state = ToolCallStreamState(start_marker, end_marker)
    visible = [
        state.feed(chunk, last=i == len(chunks) - 1) for i, chunk in enumerate(chunks)
    ]
    assert "".join(delta for delta in visible if delta) == expected
    assert state.in_tool_call is inside


@pytest.mark.parametrize(
    "parser,text",
    [
        ("qwen3_coder", "<tool_call>a</tool_call>\n<tool_call>b</tool_call>\n"),
        ("qwen3_coder", "Hi <tool_call>a</tool_call>\n<tool_call>b</tool_call>\n bye"),
        ("qwen3_coder", "<tool_call>a</tool_call> \nDone."),
        ("qwen3_coder", "A<tool_call>x</tool_call>\nB<tool_call>unfinished"),
        ("qwen3_coder", "No calls\n\n"),
        ("mistral", 'Before [TOOL_CALLS]foo[ARGS]{"a": 1}\nAfter'),
        ("mistral", "[TOOL_CALLS]foo[ARGS]{}\n[TOOL_CALLS]bar[ARGS]{}"),
    ],
)
def test_tool_stream_matches_non_streamed_content(parser, text):
    # Streamed content equals the stripped content process_tool_calls leaves,
    # up to the separator it substitutes for each call, whatever the chunking.
    module = load_tool_module(parser)
    expected = " ".join(process_tool_calls(text, module, None).remaining_text.split())
    for chunks in ([text], list(text)):
        state = ToolCallStreamState(module.tool_call_start, module.tool_call_end)
        streamed = "".join(
            state.feed(chunk, last=i == len(chunks) - 1) or ""
            for i, chunk in enumerate(chunks)
        )
        assert " ".join(streamed.split()) == expected
        assert streamed == streamed.strip()


_WEATHER_CALL = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'


def test_chat_fallback_stream_parses_tool_calls(client):
    # Without a response generator the stream_generate fallback streamed the
    # raw tool-call markup as content and never emitted tool_calls.
    result = _result(f"Checking.{_WEATHER_CALL}", finish_reason="stop")
    with _endpoint(chunks=[result], parser=_JSON_TOOLS):
        response = _post(client, stream=True, tools=[_tool()])
    deltas = _deltas(response)
    assert _joined(deltas, "content") == "Checking."
    calls = [call for delta in deltas for call in delta.get("tool_calls") or []]
    assert [call["function"]["name"] for call in calls] == ["get_weather"]
    reasons = [
        choice["finish_reason"]
        for chunk in _data(response)
        for choice in chunk.get("choices") or []
        if choice.get("finish_reason")
    ]
    assert reasons == ["tool_calls"]


@pytest.mark.parametrize("api", ["chat", "responses", "messages"])
def test_stream_without_finish_token_flushes_held_text(client, api):
    # The iterator stops without a finish reason: the unfinished call is not a
    # call, so its text is content, as in the non-streamed response.
    tool = _tool(api="messages") if api == "messages" else _tool()
    if api == "responses":
        tool = dict(type="function", name="get_weather", parameters={"type": "object"})
    response = _stream_response(
        client,
        [_token("A <tool_call>unfinished")],
        api,
        endpoint=dict(parser=_JSON_TOOLS),
        tools=[tool],
    )
    deltas = _deltas(response, api)
    if api == "chat":
        text = _joined(deltas, "content")
    elif api == "messages":
        text = _joined(deltas, "text")
    else:
        text = _joined(
            [d for d in deltas if d.get("type") == "response.output_text.delta"],
            "delta",
        )
    assert text == "A <tool_call>unfinished"


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_tool_call_content_keeps_angle_bracket_text(client, api):
    result = _result(f"Use <b>bold</b>.<|im_end|> {_WEATHER_CALL}")
    tool = (
        dict(type="function", name="get_weather", parameters={"type": "object"})
        if api == "responses"
        else _tool()
    )
    with _endpoint(result=result, parser=_JSON_TOOLS):
        response = _post(client, api, tools=[tool])
    assert response.status_code == 200, response.text
    body = response.json()
    if api == "chat":
        message = body["choices"][0]["message"]
        assert message["content"] == "Use <b>bold</b>."
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    else:
        texts = [
            part["text"]
            for item in body["output"]
            if item.get("type") == "message"
            for part in item["content"]
        ]
        assert (
            "Use <b>bold</b>." in texts or body.get("output_text") == "Use <b>bold</b>."
        )


# HTTP audio endpoints


@pytest.fixture
def audio_client(reset_audio_runtime):
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
def reset_audio_runtime(monkeypatch):
    if server.runtime.audio_queue is not None:
        server.runtime.audio_queue.stop_and_join()
    for kind in ("MODEL", "ADAPTER", "IMAGE_MODEL", "TTS_MODEL", "STT_MODEL"):
        monkeypatch.delenv("MLX_VLM_PRELOAD_" + kind, raising=False)
    _reset_runtime(monkeypatch, audio_queue=None, metrics=server.ServerMetricsStore())
    yield
    if server.runtime.audio_queue is not None:
        server.runtime.audio_queue.stop_and_join()
        server.runtime.audio_queue = None


def _fake_audio_write(target, audio, sample_rate, format="wav"):
    payload = f"{format}:{sample_rate}:{np.array(audio).shape[0]}".encode()
    if hasattr(target, "write"):
        target.write(payload)
    else:
        Path(target).write_bytes(payload)


class _AudioModel:
    sample_rate = 16000

    def __init__(self, result=None):
        self.result, self.calls = result, []

    def generate(self, value, **kwargs):
        key = "text" if self.result is None else "path"
        self.calls.append({key: value, **kwargs})
        if self.result is not None:
            assert Path(value).exists()
            return self.result
        return iter(
            [
                NS(
                    audio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    sample_rate=self.sample_rate,
                )
            ]
        )


def _audio_backend(monkeypatch, result=None):
    model = _AudioModel(result)
    loader = Mock(return_value=(model, None, NS(model_type="audio")))
    monkeypatch.setattr(server, "get_cached_model", loader)
    monkeypatch.setattr(server_audio, "audio_write", _fake_audio_write)
    return model, loader


def _upload(client, endpoint="transcriptions", **options):
    return client.post(
        "/v1/audio/" + endpoint,
        files={"file": ("test.wav", b"audio-bytes", "audio/wav")},
        data={"model": "fake-stt", **options},
    )


def test_audio_speech_returns_audio_bytes(audio_client, monkeypatch):
    fake_model, loader = _audio_backend(monkeypatch)

    response = audio_client.post(
        "/v1/audio/speech",
        json={
            "model": "fake-tts",
            "input": "Hello world",
            "voice": "alloy",
            "speed": 1.25,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].lower() == "audio/wav"
    assert (
        response.headers["content-disposition"].lower()
        == "attachment; filename=speech.wav"
    )
    assert response.content == b"wav:16000:3"
    _assert_fields(fake_model.calls[0], text="Hello world", voice="alloy", speed=1.25)
    loader.assert_called_once_with("fake-tts", model_kind="audio_tts")

    metrics = audio_client.get("/metrics").json()
    assert metrics["latest"]["endpoint"] == "/v1/audio/speech"
    assert metrics["latest"]["backend"] == "audio_queue"


def test_audio_speech_stream_bad_model_returns_error_before_headers(
    audio_client, monkeypatch
):
    def raise_not_found(model, **kwargs):
        raise HTTPException(status_code=404, detail=f"missing {model}")

    monkeypatch.setattr(server, "get_cached_model", raise_not_found)

    response = audio_client.post(
        "/v1/audio/speech",
        json={"model": "missing-model", "input": "hi", "stream": True},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "missing missing-model"


@pytest.mark.parametrize(
    "endpoint, options, text, forwarded",
    [
        (
            "transcriptions",
            {"prompt": "prior context", "language": "en"},
            "This is a test transcription.",
            {"context": "prior context", "language": "en"},
        ),
        ("transcriptions", {"response_format": "text"}, "Plain text transcript.", {}),
        ("translations", {}, "Translated transcript.", {"task": "translate"}),
    ],
    ids=["transcription-json", "transcription-text", "translation"],
)
def test_audio_transcription_request(
    audio_client, monkeypatch, endpoint, options, text, forwarded
):
    fake, loader = _audio_backend(monkeypatch, {"text": text})
    monkeypatch.setattr(
        server_audio,
        "audio_read",
        lambda buffer, always_2d=False: (np.zeros(160, dtype=np.float32), 16000),
    )
    response = _upload(audio_client, endpoint, **options)
    assert response.status_code == 200
    if options.get("response_format") == "text":
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == text
    else:
        assert response.json() == {"text": text}
    assert {key: fake.calls[0][key] for key in forwarded} == forwarded
    loader.assert_called_once_with("fake-stt", model_kind="audio_stt")


@pytest.mark.usefixtures("reset_audio_runtime")
def test_audio_tts_and_stt_caches_are_independent(monkeypatch):
    fake_tts = NS(model_type="fake_tts")
    fake_stt = NS(model_type="fake_stt")

    def fake_load_audio_model(model_path):
        return {"fake-tts": fake_tts, "fake-stt": fake_stt}[model_path]

    monkeypatch.setattr(server, "load_audio_model", fake_load_audio_model)

    tts_model, _, _ = server.get_cached_model("fake-tts", model_kind="audio_tts")
    stt_model, _, _ = server.get_cached_model("fake-stt", model_kind="audio_stt")

    assert tts_model is fake_tts
    assert stt_model is fake_stt
    assert server.runtime.model_cache.for_kind("tts")["model"] is fake_tts
    assert server.runtime.model_cache.for_kind("stt")["model"] is fake_stt

    cached_tts_model, _, _ = server.get_cached_model("fake-tts", model_kind="audio_tts")

    assert cached_tts_model is fake_tts


def test_audio_transcriptions_undecodable_upload_returns_400(audio_client, monkeypatch):
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: pytest.fail("model must not load for a bad upload"),
    )

    response = audio_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("broken.m4a", b"not-audio", "audio/mp4")},
        data={"model": "fake-stt"},
    )

    assert response.status_code == 400
    assert "Invalid audio file" in response.json()["detail"]


# Realtime voice sessions


class _FakeStreamingSession:
    input_sample_rate = 16000
    output_sample_rate = 22050
    frame_samples = 1280

    def __init__(self):
        self.closed = False
        self.push_threads = []

    def push_audio(self, samples, sample_rate):
        import threading

        self.push_threads.append(threading.current_thread().name)
        assert sample_rate == 16000
        assert samples.shape == (1280,)
        return [
            NS(
                kind="assistant_text_delta",
                frame_index=0,
                token_id=42,
                delta="hello",
                text="hello",
            ),
            NS(kind="function_delta", frame_index=0, token_id=43, delta="{", text="{"),
            NS(
                kind="audio",
                frame_index=0,
                samples=mx.zeros((1764,)),
                sample_rate=22050,
                audio_codes=mx.zeros((31,), dtype=mx.int32),
            ),
        ]

    def flush(self, pad_partial=True):
        self.closed = True
        return [NS(kind="done", frame_index=1)]

    def cancel(self):
        self.closed = True
        return [NS(kind="cancelled", frame_index=0)]


class _FakeLoadedModel:
    def __init__(self):
        self.session = None
        self.configs = []

    def create_streaming_session(self, **kwargs):
        self.configs.append(kwargs)
        self.session = _FakeStreamingSession()
        return self.session


def test_realtime_loader_uses_generic_load_and_model_session(monkeypatch):
    calls = []
    created = object()
    processor = object()

    class Model:
        def create_session(self, value):
            calls.append(("create_session", value))
            return created

    def fake_load(model_name, **kwargs):
        calls.append(("load", model_name, kwargs))
        return Model(), processor

    monkeypatch.setattr("mlx_vlm.utils.load", fake_load)

    assert realtime.load_realtime_voicechat("mlx-community/model") is created
    assert calls == [
        (
            "load",
            "mlx-community/model",
            {"lazy": True, "strict": True, "trust_remote_code": False},
        ),
        ("create_session", processor),
    ]


@pytest.fixture
def realtime_client(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    if server.runtime.realtime_engine is not None:
        server.runtime.realtime_engine.stop_and_join()
    loaded = _FakeLoadedModel()
    engine = realtime.RealtimeVoiceChatEngine(loader=lambda _: loaded)
    monkeypatch.setattr(server.runtime, "realtime_engine", engine)
    with TestClient(server.app) as client:
        yield client, loaded, engine
    if server.runtime.realtime_engine is not None:
        server.runtime.realtime_engine.stop_and_join()
        server.runtime.realtime_engine = None


@contextmanager
def _voice_session(client, **settings):
    with client.websocket_connect("/v1/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            dict(
                type="session.update", session=dict(model="fake-voicechat", **settings)
            )
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["state"] == "ready"
        yield websocket


def _send_pcm(websocket, rate=16000):
    pcm = np.zeros(1280, dtype="<i2").tobytes()
    websocket.send_json(
        dict(
            type="input_audio_buffer.append",
            audio=base64.b64encode(pcm).decode(),
            sample_rate=rate,
        )
    )


def test_realtime_websocket_streams_json_events(realtime_client):
    client, loaded, _ = realtime_client
    with _voice_session(client, system_prompt="Be brief.", seed=7) as websocket:
        _send_pcm(websocket)
        text = websocket.receive_json()
        function = websocket.receive_json()
        audio = websocket.receive_json()
        assert text["type"] == "response.text.delta"
        assert text["delta"] == "hello"
        assert function["type"] == "response.function.delta"
        assert function["delta"] == "{"
        assert audio["type"] == "response.audio.delta"
        assert len(base64.b64decode(audio["delta"])) == 1764 * 2

        websocket.send_json({"type": "input_audio_buffer.commit"})
        assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
        assert websocket.receive_json()["type"] == "response.done"

    assert loaded.configs[0]["system_prompt"] == "Be brief."
    assert loaded.configs[0]["seed"] == 7
    assert loaded.configs[0]["max_streaming_seconds"] is None
    assert loaded.session.push_threads == ["mlx-vlm-realtime"]


@pytest.mark.parametrize(("value", "expected"), [(None, None), (30, 30.0)])
def test_realtime_websocket_accepts_explicit_session_limit(
    realtime_client, value, expected
):
    client, loaded, _ = realtime_client
    with _voice_session(client, max_streaming_seconds=value) as websocket:
        websocket.send_json({"type": "response.cancel"})
        assert websocket.receive_json()["type"] == "response.cancelled"

    assert loaded.configs[0]["max_streaming_seconds"] == expected


def test_realtime_websocket_rejects_second_active_session(realtime_client):
    client, _, _ = realtime_client
    with client.websocket_connect("/v1/realtime") as first:
        assert first.receive_json()["type"] == "session.created"
        with client.websocket_connect("/v1/realtime") as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "server_busy"


def test_realtime_websocket_requires_native_pcm_rate(realtime_client):
    client, _, _ = realtime_client
    with _voice_session(client) as websocket:
        _send_pcm(websocket, rate=24000)
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "inference_error"
