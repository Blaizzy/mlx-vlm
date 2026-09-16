import asyncio
import base64
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from transformers.utils.chat_parsing import ResponseParser, parse_response

import mlx_vlm.reranker_loader as reranker_loader
import mlx_vlm.server as server
import mlx_vlm.server.anthropic as server_anthropic
import mlx_vlm.server.cli as server_cli
import mlx_vlm.server.generation as server_generation
import mlx_vlm.server.openai as server_openai
import mlx_vlm.server.reranking as server_reranking
import mlx_vlm.speculative.utils as speculative_utils
from mlx_vlm import apc as apc_module
from mlx_vlm.apc import hash_image_payload
from mlx_vlm.generate import GenerationResult
from mlx_vlm.generate.image import ImageGenerationResult
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server.runtime_config import RuntimeConfig
from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer, _ServerTokenStreamer
from mlx_vlm.tools.parsers import minicpm5

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


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _gemma_thinking_channel_chunks():
    return [
        server.StreamingToken(text="", token=100, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=45518, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=107, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=101, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=236832, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text="<|channel>thought\n<channel|>7",
            token=808,
            logprobs=0.0,
            finish_reason=None,
        ),
        server.StreamingToken(
            text=" *", token=236743, logprobs=0.0, finish_reason=None
        ),
        server.StreamingToken(text="", token=236828, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text=" 8", token=578, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text=" =", token=236743, logprobs=0.0, finish_reason=None
        ),
        server.StreamingToken(text="", token=236810, logprobs=0.0, finish_reason=None),
        server.StreamingToken(text="", token=236825, logprobs=0.0, finish_reason=None),
        server.StreamingToken(
            text=" 56", token=106, logprobs=0.0, finish_reason="stop"
        ),
    ]


@pytest.mark.parametrize(
    "input_value",
    [
        "",
        " \n\t ",
        [],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": [{"type": "input_text", "text": " "}]}],
    ],
)
def test_responses_endpoint_rejects_empty_effective_input(client, input_value):
    with patch.object(server_openai, "get_cached_model") as mock_get_cached_model:
        response = client.post(
            "/v1/responses", json={"model": "demo", "input": input_value}
        )

    assert response.status_code == 400
    assert "non-empty message content" in response.json()["detail"]
    mock_get_cached_model.assert_not_called()


def test_chat_request_schema_requires_model():
    assert "model" in server.ChatRequest.model_json_schema()["required"]


def test_chat_request_schema_declares_tool_choice_fields():
    properties = server.ChatRequest.model_json_schema()["properties"]

    assert "tools" in properties
    assert "tool_choice" in properties


def test_chat_request_schema_allows_one_or_two_resize_shape_values():
    resize_shape = server.ChatRequest.model_json_schema()["properties"]["resize_shape"]
    lengths = {
        (item["minItems"], item["maxItems"])
        for item in resize_shape["anyOf"]
        if item.get("type") == "array"
    }

    assert lengths == {(1, 1), (2, 2)}


def test_chat_completions_tool_choice_none_disables_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="<tool_call>\n<function=")
    )
    config = SimpleNamespace(model_type="qwen3_5")
    result = GenerationResult(
        text="No tool call.", prompt_tokens=5, generation_tokens=3
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

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
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Use the tool."}],
                "tools": tools,
                "tool_choice": "none",
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["tool_calls"] is None
    assert mock_template.call_args.kwargs["tools"] is None
    assert mock_template.call_args.kwargs["tool_choice"] == "none"


def test_chat_completions_tool_parser_override(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    # A template with no tool markers: inference alone selects no parser.
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="a plain template, no tool markers")
    )
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
        prompt_tokens=5,
        generation_tokens=3,
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

    def post(extra):
        with (
            patch.object(
                server, "get_cached_model", return_value=(model, processor, config)
            ),
            patch.object(server, "apply_chat_template", return_value="prompt"),
            patch.object(server, "generate", return_value=result),
        ):
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": tools,
                    **extra,
                },
            )

    # Without an override the markerless template routes to no parser: no calls.
    base = post({})
    assert base.status_code == 200
    assert base.json()["choices"][0]["message"]["tool_calls"] is None

    # The override forces json_tools, which parses the emitted call.
    overridden = post({"tool_parser": "json_tools"})
    assert overridden.status_code == 200
    calls = overridden.json()["choices"][0]["message"]["tool_calls"]
    assert calls and calls[0]["function"]["name"] == "get_weather"

    # An unknown parser name is rejected at request validation.
    assert post({"tool_parser": "bogus"}).status_code == 422


def test_chat_completions_forced_tool_choice_filters_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=5, generation_tokens=2)
    tools = [
        {
            "type": "function",
            "function": {"name": "get_time", "parameters": {"type": "object"}},
        },
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        },
    ]
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}

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
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Say hello."},
                ],
                "tools": tools,
                "tool_choice": tool_choice,
            },
        )

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert messages[0]["content"].startswith("Be concise.")
    assert "must call the 'get_weather' function" in messages[0]["content"]
    assert "must call the 'get_weather' function" in messages[-1]["content"]
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == ["get_weather"]
    assert mock_template.call_args.kwargs["tool_choice"] == tool_choice


@pytest.mark.parametrize(
    ("tools", "tool_choice", "detail"),
    [
        ([], "required", "requires at least one tool"),
        (
            [{"type": "function", "function": {"name": "get_weather"}}],
            {"type": "function", "function": {"name": "missing"}},
            "unknown function 'missing'",
        ),
        ([], "sometimes", "Invalid tool_choice"),
    ],
)
def test_chat_completions_rejects_invalid_tool_choice(
    client, tools, tool_choice, detail
):
    with patch.object(server, "get_cached_model") as mock_get_cached_model:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": tools,
                "tool_choice": tool_choice,
            },
        )

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    mock_get_cached_model.assert_not_called()


def test_speculative_server_dispatches_mtp_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("mtp")
        is speculative_utils._mtp_rounds_batch
    )


@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_positioned_target_sampler_honors_top_k(top_p):
    sampler = server_generation._PositionedTargetSampler(
        temperature=1.0, top_p=top_p, top_k=2, seed=42
    )
    logits = mx.array([[0.0, 1.0, 2.0, 3.0]], dtype=mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    repeated = mx.repeat(logprobs, 32, axis=0)

    tokens = sampler.sample_target(
        repeated, row_ids=[0] * 32, positions=list(range(32))
    )
    mx.eval(tokens)

    assert set(tokens.tolist()) <= {2, 3}


def test_server_passes_top_k_to_positioned_sampler():
    generator = server.ResponseGenerator.__new__(server.ResponseGenerator)
    args = server_generation.GenerationArguments(max_tokens=1, temperature=1.0, top_k=7)

    sampler = generator._make_sampler(args)

    assert sampler.top_k == 7


def test_speculative_server_dispatches_eagle3_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("eagle3")
        is speculative_utils._eagle3_rounds_batch
    )


def test_speculative_server_keeps_dflash_default_batch_loop():
    assert (
        speculative_utils.get_speculative_rounds_batch("dflash")
        is speculative_utils._dflash_rounds_batch
    )


def test_speculative_server_rejects_unknown_draft_kind():
    with pytest.raises(ValueError):
        speculative_utils.get_speculative_rounds_batch("nope")


def test_speculative_server_hidden_state_picks_last_layer_for_mtp():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    assert speculative_utils.speculative_hidden_state("mtp", out) is h[-1]


def test_speculative_server_reads_batch_coalesce_env(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", raising=False)
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.005)

    monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "2.5")
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.0025)

    monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "bad")
    assert server.get_speculative_batch_coalesce_s() == pytest.approx(0.005)


def test_get_cached_model_omitted_adapter_inherits_loaded_adapter(monkeypatch):
    class FakeResponseGenerator:
        def __init__(self, model_path, adapter_path=None, **kwargs):
            self.model_path = model_path
            self.adapter_path = adapter_path
            self.model = SimpleNamespace()
            self.processor = SimpleNamespace()
            self.config = SimpleNamespace(model_type="qwen2_vl")

        def wait_until_ready(self):
            return self.model, self.processor, self.config

        def stop_and_join(self):
            pass

    monkeypatch.setattr(server._app_module, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
    monkeypatch.setattr(server.runtime, "model_cache", {})
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    server.get_cached_model("demo-model", "adapter-a")
    server.get_cached_model("demo-model")

    cache_key = server.runtime.model_cache["cache_key"]
    assert cache_key[:3] == ("demo-model", "adapter-a", "text_generation")
    assert cache_key[3] == server.runtime.config.fingerprint(kinds={"text_generation"})
    assert server.runtime.model_cache["adapter_path"] == "adapter-a"


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
        {
            "model_path": "old-model",
            "adapter_path": None,
            "response_generator": response_generator,
            "apc_manager": manager,
        },
    )
    monkeypatch.setattr(server.runtime, "model_cache", registry)
    monkeypatch.setattr(server.runtime, "response_generator", response_generator)
    monkeypatch.setattr(server.runtime, "apc_manager", manager)
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

    monkeypatch.setattr(server_generation, "load", reject_model)
    monkeypatch.setattr(server._app_module._apc, "from_env", lambda *_, **__: None)
    monkeypatch.setattr(server.runtime, "model_cache", {})
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "google-bert/bert-base-multilingual-cased",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Failed to load model: Model type bert not supported."
    )
    assert client.get("/health").status_code == 200


def _unstarted_response_generator():
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen.model_path = "demo"
    gen.adapter_path = None
    gen.model = None
    gen.processor = None
    gen.config = None
    gen.stop_tokens = set()
    gen.vision_cache = None
    gen.draft_model = None
    gen.draft_kind = None
    gen.draft_model_path = None
    gen.draft_kind_override = None
    gen.kv_bits = None
    gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
    gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
    gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
    gen.top_logprobs_k = 0
    gen.apc_manager = None
    gen.apc_mode = None
    gen.tokenizer = None
    gen.requests = Queue()
    gen._stop = False
    gen._ready = Event()
    gen._load_error = None
    gen._cancelled = set()
    gen._cancel_lock = Lock()
    return gen


def test_server_caches_apc_mode_when_model_initializes(monkeypatch):
    config = SimpleNamespace(eos_token_id=[])
    language_model = SimpleNamespace()
    model = SimpleNamespace(language_model=language_model)
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    gen = _unstarted_response_generator()
    gen.apc_manager = object()

    monkeypatch.delenv("MLX_VLM_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("MLX_VLM_DRAFT_KIND", raising=False)
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, config),
    )
    apc_mode = MagicMock(return_value="exact")
    monkeypatch.setattr(apc_module, "model_apc_mode", apc_mode)

    gen._initialize_model()

    assert gen.apc_mode == "exact"
    apc_mode.assert_called_once_with(language_model)


def test_server_serves_ar_requests_after_drafter_mismatch(monkeypatch):
    class FakeDetokenizer:
        def __init__(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = str(token)

        def finalize(self):
            pass

    class FakeBatchGenerator:
        def __init__(self, *args, **kwargs):
            self.unprocessed_prompts = []
            self.has_pending_prompts = False

        def insert(self, *args, **kwargs):
            return (1,)

        def next(self, **kwargs):
            return [], [
                SimpleNamespace(
                    uid=1, token=7, token_logprob=0.0, finish_reason="length"
                )
            ]

    target_config = SimpleNamespace(
        model_type="gemma4_text", hidden_size=5376, eos_token_id=[]
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=target_config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    drafter = SimpleNamespace(
        config=SimpleNamespace(model_type="gemma4_assistant", backbone_hidden_size=1536)
    )
    gen = _unstarted_response_generator()

    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "assistant")
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(
        server_generation,
        "make_streaming_detokenizer",
        lambda _processor: FakeDetokenizer(),
    )
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, target_config),
    )
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.load_drafter",
        lambda *_args, **_kwargs: (drafter, "mtp"),
    )
    gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
        mx.array([[raw_inputs["token"]]], dtype=mx.int32),
        {},
    )

    rqueue = Queue()
    gen.requests.put(
        server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"token": 1},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=1),
        )
    )
    worker = Thread(target=gen._run, daemon=True)
    worker.start()
    try:
        ctx = rqueue.get(timeout=1)
        token = rqueue.get(timeout=1)
        done = rqueue.get(timeout=1)
    finally:
        gen._stop = True
        gen.requests.put(None)
        worker.join(timeout=2)

    assert isinstance(ctx, server.GenerationContext)
    assert token.text == "7"
    assert token.finish_reason == "length"
    assert done is None
    assert gen.draft_model is None
    assert gen.draft_kind is None


def test_ar_thread_exception_reaches_pending_client_queue(monkeypatch):
    class FakeBatchGenerator:
        def __init__(self, *_args, **_kwargs):
            self.has_work = False

        def close(self):
            pass

    gen = _unstarted_response_generator()

    def initialize_model():
        gen.model = SimpleNamespace(language_model=object())
        gen.processor = SimpleNamespace()
        gen.config = SimpleNamespace()
        gen.tokenizer = SimpleNamespace()

    error = RuntimeError("vision embedding failed")
    gen._initialize_model = initialize_model
    gen._gpu_embed = MagicMock(side_effect=error)
    monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)

    rqueue = Queue()
    gen.requests.put(
        server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        )
    )

    worker = Thread(target=gen._run, daemon=True)
    worker.start()
    try:
        assert rqueue.get(timeout=1) is error
        assert rqueue.get(timeout=1) is None
        assert worker.is_alive()
    finally:
        gen._stop = True
        gen.requests.put(None)
        worker.join(timeout=2)


def test_models_endpoint_lists_single_file_safetensors_models(client, monkeypatch):
    monkeypatch.setenv("MLX_VLM_MODEL_DISCOVERY", "hf-cache")

    def repo(repo_id, file_names):
        return SimpleNamespace(
            repo_id=repo_id,
            repo_type="model",
            last_modified=123.0,
            refs={
                "main": SimpleNamespace(
                    files=[
                        SimpleNamespace(file_path=SimpleNamespace(name=file_name))
                        for file_name in file_names
                    ]
                )
            },
        )

    monkeypatch.setattr(
        server,
        "scan_cache_dir",
        lambda: SimpleNamespace(
            repos=[
                repo(
                    "local/single-file-model",
                    ["config.json", "model.safetensors", "tokenizer_config.json"],
                ),
                repo(
                    "local/sharded-model",
                    [
                        "config.json",
                        "model.safetensors.index.json",
                        "tokenizer_config.json",
                    ],
                ),
                repo("missing/weights", ["config.json", "tokenizer_config.json"]),
            ]
        ),
    )

    response = client.get("/v1/models")

    assert response.status_code == 200
    ids = {model["id"] for model in response.json()["data"]}
    assert "local/single-file-model" in ids
    assert "local/sharded-model" in ids
    assert "missing/weights" not in ids


def test_models_endpoint_includes_loaded_local_model_without_hf_cache(
    client, monkeypatch
):
    monkeypatch.delenv("MLX_VLM_MODEL_DISCOVERY", raising=False)
    monkeypatch.setattr(
        server,
        "scan_cache_dir",
        MagicMock(side_effect=server.CacheNotFound("missing cache", "/missing")),
    )
    monkeypatch.setitem(server.runtime.model_cache, "model_path", "/models/local-qwen")

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "id": "/models/local-qwen",
            "object": "model",
            "created": response.json()["data"][0]["created"],
        }
    ]


def test_response_generator_diffusion_forwards_generation_options(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace()
    gen.processor = SimpleNamespace()
    gen.config = SimpleNamespace(eos_token_id=3)
    gen.tokenizer = SimpleNamespace(all_special_ids=[0])
    gen.prefill_step_size = 3072
    apc_manager = SimpleNamespace()
    gen.apc_manager = apc_manager
    gen.apc_mode = "exact"
    captured = {}

    def fake_stream_diffusion_generate_from_kwargs(
        model,
        processor,
        tokenizer,
        input_ids,
        pixel_values,
        attention_mask,
        skip_special_token_ids,
        kwargs,
        *,
        skip_special_tokens=False,
        on_result=None,
    ):
        captured.update(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            skip_special_token_ids=skip_special_token_ids,
            kwargs=dict(kwargs),
            skip_special_tokens=skip_special_tokens,
        )
        on_result(
            GenerationResult(
                text="ok",
                token=7,
                prompt_tokens=2,
                generation_tokens=1,
                total_tokens=3,
                prompt_tps=10.0,
                generation_tps=5.0,
                cached_tokens=1,
                finish_reason="length",
            )
        )
        if False:
            yield None

    monkeypatch.setattr(
        server_generation,
        "stream_diffusion_generate_from_kwargs",
        fake_stream_diffusion_generate_from_kwargs,
    )
    args = server.GenerationArguments(
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
    rqueue = Queue()

    gen._generate_diffusion(
        uid=1,
        rqueue=rqueue,
        raw_inputs={
            "input_ids": mx.array([[11, 12]], dtype=mx.int32),
            "pixel_values": "pixels",
            "attention_mask": "mask",
            "mm_token_type_ids": "types",
        },
        args=args,
        cancelled=set(),
        apc_semantic_hash=73,
    )

    chunk = rqueue.get(timeout=1)
    assert chunk.text == "ok"
    assert chunk.finish_reason == "length"
    assert chunk.generation_tps == 5.0
    assert chunk.cached_tokens == 1
    assert captured["input_ids"].tolist() == [[11, 12]]
    assert captured["pixel_values"] == "pixels"
    assert captured["attention_mask"] == "mask"
    assert captured["skip_special_token_ids"] == {0}
    assert captured["kwargs"]["_apc_manager"] is apc_manager
    assert captured["kwargs"]["_apc_semantic_hash"] == 73
    assert captured["skip_special_tokens"] is True
    assert captured["kwargs"] == {
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "mm_token_type_ids": "types",
        "prefill_step_size": 3072,
        "seed": 123,
        "max_denoising_steps": 7,
        "block_length": 16,
        "num_to_transfer": 3,
        "max_transfer_per_step": 2,
        "editing_threshold": 0.8,
        "max_post_steps": 5,
        "stability_steps": 1,
        "diffusion_full_canvas": True,
        "diffusion_min_canvas_length": 4,
        "diffusion_max_canvas_length": 8,
        "diffusion_sampler": "entropy-bound",
        "threshold": 0.7,
        "min_threshold": 0.4,
        "_apc_manager": apc_manager,
        "_apc_semantic_hash": 73,
    }


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
def test_management_endpoints_require_configured_api_key(
    client, monkeypatch, method, path
):
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


def test_images_generations_forwards_prompt_expansion_model(client, monkeypatch):
    calls = []

    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (
            SimpleNamespace(),
            None,
            SimpleNamespace(model_type="ideogram4"),
        ),
    )

    def fake_generate_image(model, request, **kwargs):
        calls.append(request)
        result = _fake_image_result(seed=request.seed)
        result.metadata["revised_prompt"] = '{"compositional_deconstruction":{}}'
        return result

    monkeypatch.setattr(server_openai, "generate_image", fake_generate_image)

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "ideogram-ai/ideogram-4-fp8",
            "prompt": "A red cube.",
            "seed": 10,
            "size": "256x256",
            "steps": 1,
            "auto_json_caption": True,
            "prompt_expansion_model": "tiny-text-model",
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    assert calls[0].extra == {
        "auto_json_caption": True,
        "prompt_expansion_model": "tiny-text-model",
    }
    assert (
        response.json()["data"][0]["revised_prompt"]
        == '{"compositional_deconstruction":{}}'
    )


def test_images_generations_writes_paths(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (
            SimpleNamespace(),
            None,
            SimpleNamespace(model_type="bonsai"),
        ),
    )

    def fake_generate_image(model, request, **kwargs):
        return _fake_image_result(seed=request.seed, output_path=kwargs["output_path"])

    monkeypatch.setattr(server_openai, "generate_image", fake_generate_image)

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "bonsai-ternary",
            "prompt": "bonsai",
            "n": 2,
            "seed": 20,
            "size": "256x256",
            "steps": 1,
            "response_format": "path",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    paths = [Path(item["path"]) for item in payload["data"]]
    assert [path.name for path in paths] == ["image-20.png", "image-21.png"]
    assert all(path.exists() for path in paths)
    assert all(item["b64_json"] is None for item in payload["data"])


def test_images_edits_returns_b64_json(client, monkeypatch):
    calls = []
    cache_calls = []

    def fake_get_cached_model(model, **kwargs):
        cache_calls.append((model, kwargs))
        return SimpleNamespace(), None, SimpleNamespace(model_type="flux2")

    monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)

    def fake_edit_image(model, request, **kwargs):
        calls.append((request, kwargs))
        return _fake_image_result(seed=request.seed)

    monkeypatch.setattr(server_openai, "edit_image", fake_edit_image)

    response = client.post(
        "/v1/images/edits",
        json={
            "model": "black-forest-labs/FLUX.2-klein-9b-kv",
            "prompt": "add sunglasses",
            "image": ["reference.png"],
            "n": 2,
            "seed": 30,
            "size": "256x256",
            "steps": 1,
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["size"] == "16x16"
    assert [item["seed"] for item in payload["data"]] == [30, 31]
    assert all(item["b64_json"] for item in payload["data"])
    assert [call[0].seed for call in calls] == [30, 31]
    assert calls[0][0].image_paths == ("reference.png",)
    assert cache_calls == [
        ("black-forest-labs/FLUX.2-klein-9b-kv", {"model_kind": "image_edit"})
    ]


def test_images_edits_writes_paths(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        server,
        "get_cached_model",
        lambda model, **kwargs: (
            SimpleNamespace(),
            None,
            SimpleNamespace(model_type="flux2"),
        ),
    )

    def fake_edit_image(model, request, **kwargs):
        return _fake_image_result(seed=request.seed, output_path=kwargs["output_path"])

    monkeypatch.setattr(server_openai, "edit_image", fake_edit_image)

    response = client.post(
        "/v1/images/edits",
        json={
            "model": "black-forest-labs/FLUX.2-klein-9b-kv",
            "prompt": "add sunglasses",
            "image": "reference.png",
            "n": 2,
            "seed": 40,
            "size": "256x256",
            "steps": 1,
            "response_format": "path",
            "output_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    paths = [Path(item["path"]) for item in payload["data"]]
    assert [path.name for path in paths] == ["edit-40.png", "edit-41.png"]
    assert all(path.exists() for path in paths)
    assert all(item["b64_json"] is None for item in payload["data"])


def test_responses_endpoint_forwards_new_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done", prompt_tokens=8, generation_tokens=4, total_tokens=12
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
                "thinking_end_token": "</think>",
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["enable_thinking"] is False
    assert mock_template.call_args.kwargs["thinking_budget"] == 24
    assert mock_template.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_template.call_args.kwargs["thinking_end_token"] == "</think>"
    assert mock_generate.call_args.kwargs["max_tokens"] == 12
    assert mock_generate.call_args.kwargs["top_k"] == 40
    assert mock_generate.call_args.kwargs["min_p"] == 0.08
    assert mock_generate.call_args.kwargs["repetition_penalty"] == 1.15
    assert mock_generate.call_args.kwargs["logit_bias"] == {12: -1.5}
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_generate.call_args.kwargs["thinking_budget"] == 24
    assert mock_generate.call_args.kwargs["thinking_start_token"] == "<think>"
    assert mock_generate.call_args.kwargs["thinking_end_token"] == "</think>"


def test_responses_endpoint_merges_developer_message_with_instructions(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done", prompt_tokens=8, generation_tokens=4, total_tokens=12
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
                "instructions": "Top-level instructions.",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {"type": "input_text", "text": "Developer instructions."}
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello"}],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "system",
            "content": "Top-level instructions.\n\nDeveloper instructions.",
        },
        {"role": "user", "content": "Hello"},
    ]


def test_responses_endpoint_places_function_output_image_after_tool_result(
    client, monkeypatch
):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    image_url = "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done", prompt_tokens=8, generation_tokens=4, total_tokens=12
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/responses",
            json={
                "model": "demo",
                "input": [
                    {
                        "type": "function_call",
                        "name": "view_image",
                        "arguments": "{}",
                        "call_id": "call_view_image",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_view_image",
                        "output": [
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": "high",
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    prompt = mock_generate.call_args.kwargs["prompt"]
    assert prompt.index("Tool:") < prompt.index("<image>")
    assert image_url not in prompt
    assert mock_generate.call_args.kwargs["image"] == [image_url]


def test_responses_endpoint_rejects_image_file_id(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "demo",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_view_image",
                    "output": [{"type": "input_image", "file_id": "file-image"}],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "input_image.file_id is not supported by this server. "
        "Provide image_url instead."
    )


def test_responses_input_tokens_endpoint_forwards_adapter_path(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    get_cached_model = MagicMock(return_value=(model, processor, config))
    response_generator = SimpleNamespace(
        _cpu_preprocess=MagicMock(
            return_value={"input_ids": mx.array([[1, 2, 3]], dtype=mx.int32)}
        )
    )

    monkeypatch.setattr(server.runtime, "response_generator", response_generator)
    monkeypatch.setattr(server, "get_cached_model", get_cached_model)
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))

    response = client.post(
        "/responses/input_tokens",
        json={"model": "demo", "input": "Hello", "adapter_path": "adapter-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 3}
    assert get_cached_model.call_args.args == ("demo", "adapter-a")


def test_responses_previous_response_id_replays_stored_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    first = GenerationResult(text="First answer", prompt_tokens=3, generation_tokens=2)
    second = GenerationResult(
        text="Second answer", prompt_tokens=7, generation_tokens=2
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", side_effect=[first, second]),
    ):
        first_response = client.post(
            "/v1/responses", json={"model": "demo", "input": "First"}
        )
        assert first_response.status_code == 200
        previous_response_id = first_response.json()["id"]

        second_response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "previous_response_id": previous_response_id,
                "input": "Second",
            },
        )

    assert second_response.status_code == 200
    replayed_messages = mock_template.call_args_list[1].args[2]
    assert replayed_messages == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second"},
    ]
    retrieved = client.get(f"/v1/responses/{previous_response_id}")
    assert retrieved.status_code == 200
    input_items = client.get(f"/v1/responses/{previous_response_id}/input_items")
    assert input_items.status_code == 200
    assert input_items.json()["data"][0]["content"][0]["text"] == "First"


def test_responses_endpoint_returns_native_shell_call_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name":"shell","arguments":{"command":"pwd"}}</tool_call>',
        prompt_tokens=8,
        generation_tokens=4,
    )
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/responses",
            json={"model": "demo", "input": "run pwd", "tools": [{"type": "shell"}]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["type"] == "shell_call"
    assert payload["output"][0]["action"] == {"type": "exec", "command": "pwd"}


def _sse_events(body):
    events = []
    for block in body.split("\n\n"):
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event_type and data:
            events.append((event_type, data))
    return events


def test_responses_streaming_emits_native_tool_call_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        GenerationResult(
            text='<tool_call>{"name":"shell","arguments":{"command":"pwd"}}</tool_call>',
            prompt_tokens=8,
            generation_tokens=4,
            prompt_tps=0.0,
            generation_tps=0.0,
            peak_memory=0.0,
        )
    ]
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "run pwd",
                "stream": True,
                "tools": [{"type": "shell"}],
            },
        )

    assert response.status_code == 200
    body = response.text
    assert '"type": "shell_call"' in body
    assert '"command": "pwd"' in body
    assert "<tool_call>" not in body


def test_responses_streaming_uses_prompt_opened_thinking_without_flag(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="cohere2_moe")
    chunks = [
        GenerationResult(text="North reasoning.", prompt_tokens=8, generation_tokens=1),
        GenerationResult(
            text="<|END_THINKING|><|START_TEXT|>North answer.<|END_TEXT|>",
            prompt_tokens=8,
            generation_tokens=4,
            finish_reason="stop",
        ),
    ]
    template_kwargs = {}

    def fake_apply_chat_template(*args, **kwargs):
        template_kwargs.update(kwargs)
        return "prompt<|START_THINKING|>"

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", side_effect=fake_apply_chat_template
        ),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "CohereLabs/North-Mini-Code-1.0-w4a16",
                "input": "hello",
                "reasoning": {"effort": "high", "summary": "auto"},
                "stream": True,
            },
        )

    assert response.status_code == 200
    events = _sse_events(response.text)
    reasoning = "".join(
        data["delta"]
        for event_type, data in events
        if event_type == "response.reasoning_text.delta"
    )
    content = "".join(
        data["delta"]
        for event_type, data in events
        if event_type == "response.output_text.delta"
    )

    assert reasoning == "North reasoning."
    assert content == "North answer."
    assert template_kwargs["enable_thinking"] is True
    assert template_kwargs["reasoning"] is True
    assert template_kwargs["reasoning_effort"] == "high"
    assert "<|END_THINKING|>" not in response.text
    assert "<|START_TEXT|>" not in response.text
    assert "<|END_TEXT|>" not in response.text


def test_responses_streaming_emits_function_call_arguments_done(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        GenerationResult(
            text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
            prompt_tokens=8,
            generation_tokens=4,
            finish_reason="stop",
        )
    ]
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "weather?",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    events = _sse_events(response.text)
    done = next(
        data
        for event_type, data in events
        if event_type == "response.function_call_arguments.done"
    )
    assert done["item_id"].startswith("fc_")
    assert done["name"] == "get_weather"
    assert done["arguments"] == '{"location": "SF"}'


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        ),
        (
            "/v1/responses",
            {"model": "demo", "input": "Hello", "max_output_tokens": 4, "stream": True},
        ),
    ],
)
def test_stream_endpoints_do_not_clear_mlx_cache_on_close(
    client, monkeypatch, path, payload
):
    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text="ok", token=1, logprobs=0.0, finish_reason="stop"
                    )
                ]
            )

    calls = {"clear_cache": 0, "collect": 0}
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())
    monkeypatch.setattr(
        server, "get_cached_model", MagicMock(return_value=(model, processor, config))
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))
    monkeypatch.setattr(
        server_openai.mx,
        "clear_cache",
        lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
    )
    monkeypatch.setattr(
        server_openai.gc,
        "collect",
        lambda: calls.__setitem__("collect", calls["collect"] + 1),
    )

    response = client.post(path, json=payload)

    assert response.status_code == 200
    assert calls == {"clear_cache": 0, "collect": 0}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        ),
        (
            "/v1/responses",
            {"model": "demo", "input": "Hello", "max_output_tokens": 4, "stream": True},
        ),
    ],
)
def test_v1_stream_endpoints_reject_over_context_before_sse(
    client, monkeypatch, path, payload
):
    class OverBudgetResponseGenerator:
        generate_called = False

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            raise server.PromptTooLongError(
                "Request needs 9 context tokens "
                "(5 prompt + 4 max generation), but MAX_KV_SIZE is 8."
            )

        def generate(self, *args, **kwargs):
            self.generate_called = True
            raise AssertionError("streaming should not start")

    response_generator = OverBudgetResponseGenerator()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(server.runtime, "response_generator", response_generator)
    monkeypatch.setattr(
        server, "get_cached_model", MagicMock(return_value=(model, processor, config))
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))

    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert "MAX_KV_SIZE is 8" in response.json()["detail"]
    assert response_generator.generate_called is False


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            },
        ),
        ("/v1/responses", {"model": "demo", "input": "Hello", "max_output_tokens": 4}),
    ],
)
def test_v1_non_stream_endpoints_reject_over_context(
    client, monkeypatch, path, payload
):
    class OverBudgetResponseGenerator:
        def generate(self, *args, **kwargs):
            raise server.PromptTooLongError(
                "Request needs 9 context tokens "
                "(5 prompt + 4 max generation), but MAX_KV_SIZE is 8."
            )

    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(
        server.runtime, "response_generator", OverBudgetResponseGenerator()
    )
    monkeypatch.setattr(
        server, "get_cached_model", MagicMock(return_value=(model, processor, config))
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))

    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert "MAX_KV_SIZE is 8" in response.json()["detail"]


def test_chat_completions_endpoint_forwards_explicit_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
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


def test_chat_completions_streaming_uses_prompt_opened_thinking_without_flag(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="cohere2_moe")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="North reasoning.",
                        token=1,
                        logprobs=0.0,
                        finish_reason=None,
                    ),
                    server.StreamingToken(
                        text="<|END_THINK", token=2, logprobs=0.0, finish_reason=None
                    ),
                    server.StreamingToken(
                        text="ING|><|START_TEXT|>North answer.<|END_TEXT|>",
                        token=3,
                        logprobs=0.0,
                        finish_reason="stop",
                    ),
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt<|START_THINKING|>"
        ),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "CohereLabs/North-Mini-Code-1.0-w4a16",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [
        chunk["choices"][0]["delta"]
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("delta")
    ]

    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == (
        "North reasoning."
    )
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == (
        "North reasoning."
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == "North answer."
    assert "<|END_THINKING|>" not in response.text
    assert "<|START_TEXT|>" not in response.text
    assert "<|END_TEXT|>" not in response.text


@pytest.mark.parametrize(
    "audio_data_factory",
    [
        lambda raw: base64.b64encode(raw).decode("ascii"),
        lambda raw: f"data:audio/wav;base64,{base64.b64encode(raw).decode('ascii')}",
    ],
)
def test_chat_completions_decodes_input_audio_base64(client, audio_data_factory):
    raw_audio = b"RIFF$\x00\x00\x00WAVEfmt "
    captured = {}

    def fake_generate(prompt, images=None, audio=None, **kwargs):
        captured["audio"] = audio
        return GenerationResult(
            text="audio ok",
            prompt_tokens=8,
            generation_tokens=4,
            total_tokens=12,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        )

    with (
        patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", side_effect=fake_generate),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the audio."},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data_factory(raw_audio),
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert captured["audio"][0].getvalue() == raw_audio


def test_chat_completions_preserves_input_audio_references(client):
    audio_path = "/tmp/audio.wav"
    captured = {}

    def fake_generate(prompt, images=None, audio=None, **kwargs):
        captured["audio"] = audio
        return GenerationResult(
            text="audio ok",
            prompt_tokens=8,
            generation_tokens=4,
            total_tokens=12,
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=0.1,
        )

    with (
        patch.object(
            server,
            "get_cached_model",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(model_type="qwen2_vl"),
            ),
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", side_effect=fake_generate),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the audio."},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_path, "format": "wav"},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert captured["audio"] == [audio_path]


def test_generation_metrics_record_speculative_stats():
    metrics = server_generation.GenerationMetrics()

    metrics.record_chunk(SimpleNamespace(generation_tokens=1, emitted_at=10.0))
    metrics.record_chunk(
        SimpleNamespace(
            generation_tokens=6,
            emitted_at=10.5,
            draft_kind="dflash",
            draft_rounds=3,
            draft_n_accepted=4,
            draft_n=9,
        )
    )

    assert metrics.draft_kind == "dflash"
    assert metrics.draft_rounds == 3
    assert metrics.draft_n_accepted == 4
    assert metrics.draft_n == 9


def test_speculative_lifetime_counters_survive_reset():
    from mlx_vlm.speculative.common import (
        _record_speculative_round,
        speculative_stats_since,
        speculative_stats_snapshot,
    )

    drafter = SimpleNamespace(accept_lens=[], draft_lens=[])

    assert speculative_stats_since(drafter, speculative_stats_snapshot(drafter)) == (
        None,
        None,
        None,
    )

    snapshot = speculative_stats_snapshot(drafter)
    _record_speculative_round(drafter, 3, 7)
    _record_speculative_round(drafter, 2.5, 7)
    drafter.accept_lens = []
    drafter.draft_lens = []
    _record_speculative_round(drafter, 1.5, 7)

    rounds, accepted, drafted = speculative_stats_since(drafter, snapshot)
    assert (rounds, accepted, drafted) == (3, 7, 21)

    later_snapshot = speculative_stats_snapshot(drafter)
    _record_speculative_round(drafter, 2, 7)
    rounds, accepted, drafted = speculative_stats_since(drafter, later_snapshot)
    assert (rounds, accepted, drafted) == (1, 2, 7)


def test_chat_completions_streaming_emits_timings_on_finish(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=10), iter(
                [
                    server.StreamingToken(
                        text="hi",
                        token=1,
                        logprobs=0.0,
                        finish_reason=None,
                        prompt_tps=20.0,
                        cached_tokens=2,
                    ),
                    server.StreamingToken(
                        text="!",
                        token=2,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                    ),
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    timed_chunk = next(chunk for chunk in chunks if chunk.get("usage") is not None)
    assert timed_chunk["choices"] == []
    assert timed_chunk["timings"]["cache_n"] == 2
    assert timed_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    token_chunks = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["delta"].get("content") is not None
    ]
    assert token_chunks[0]["timings"]["predicted_per_second"] is None
    assert token_chunks[1]["timings"]["predicted_per_second"] > 0
    terminal_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] == "stop"
    )
    assert terminal_chunk["timings"]["predicted_per_second"] > 0
    assert (
        timed_chunk["timings"]["predicted_per_second"]
        == terminal_chunk["timings"]["predicted_per_second"]
    )


def test_chat_completions_streaming_response_template_tool_calls(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer())
    config = SimpleNamespace(model_type="muse_glimmer")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=10), iter(
                [
                    server.StreamingToken(
                        text=(
                            "to=self<|message|>I need the weather tool.<|eom|>"
                            "<|start|>assistant to=get_weather<|message|>"
                            '<atem:function_calls><atem:invoke name="get_weather">'
                            '<atem:parameter name="city">Warsaw</atem:parameter>'
                            "</atem:invoke></atem:function_calls>"
                        ),
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                    )
                ]
            )

    from mlx_vlm.tools.parsers import atem as tool_module

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert response.status_code == 200
    chunks = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_chunk = next(
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] == "tool_calls"
    )
    usage_chunk = next(chunk for chunk in chunks if chunk.get("usage") is not None)
    reasoning = "".join(
        chunk["choices"][0]["delta"].get("reasoning_content") or ""
        for chunk in chunks
        if chunk["choices"]
    )
    content = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in chunks
        if chunk["choices"]
    )
    tool_call = tool_chunk["choices"][0]["delta"]["tool_calls"][0]

    assert tool_chunk.get("usage") is None
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Warsaw"}
    assert reasoning == "I need the weather tool."
    assert content == ""
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_chat_completions_endpoint_falls_back_from_video_to_images(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="mage_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
        total_tokens=12,
        prompt_tps=10.0,
        generation_tps=5.0,
        peak_memory=0.1,
    )
    frames = [object(), object()]
    from mlx_vlm.generate import video as video_module

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result) as mock_generate,
        patch.object(
            video_module, "sample_video_frames", return_value=(frames, 2.0)
        ) as mock_sample,
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                            {"type": "text", "text": "Describe this video."},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["num_images"] == 2
    assert mock_template.call_args.kwargs["video"] is None
    assert mock_generate.call_args.kwargs["image"] == frames
    assert mock_generate.call_args.kwargs["video"] == []
    mock_sample.assert_called_once_with(["clip.mp4"], 2.0, None)


def test_chat_completions_endpoint_preserves_assistant_reasoning_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
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
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": "Hello",
                        "reasoning_content": "Prior thought",
                    },
                    {"role": "user", "content": "Continue"},
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2][1] == {
        "role": "assistant",
        "content": "Hello",
        "reasoning_content": "Prior thought",
        "reasoning": "Prior thought",
    }


def test_anthropic_messages_endpoint_maps_text_and_images(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=8,
        generation_tokens=4,
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
        patch.object(server, "generate", return_value=result) as mock_generate,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "system": "You are concise.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe it."},
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/image.png",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 12,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "done"}]
    assert payload["stop_reason"] == "end_turn"
    assert payload["usage"] == {
        "input_tokens": 8,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 4,
    }
    assert mock_template.call_args.args[2] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Describe it."},
    ]
    assert mock_generate.call_args.kwargs["image"] == ["https://example.com/image.png"]
    assert mock_generate.call_args.kwargs["max_tokens"] == 12


def test_anthropic_messages_endpoint_accepts_system_role_in_messages(
    client, monkeypatch
):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=4, generation_tokens=2)

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
            "/v1/messages",
            json={
                "model": "demo",
                "system": "Use short answers.",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "Be precise."}],
                    },
                    {"role": "user", "content": "Introduce the project."},
                ],
                "max_tokens": 12,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {"role": "system", "content": "Use short answers."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Be precise."},
        {"role": "user", "content": "Introduce the project."},
    ]


def test_anthropic_messages_endpoint_converts_tool_result_inputs(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=5,
        generation_tokens=2,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
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
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"location": "SF"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "72F",
                            }
                        ],
                    },
                ],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "SF"}, ensure_ascii=False),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "72F", "name": None},
    ]
    normalized = apply_chat_template(
        None, config, mock_template.call_args.args[2], return_messages=True
    )
    assert normalized[0]["content"] == ""


def test_anthropic_messages_endpoint_preserves_tool_result_images(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=5,
        generation_tokens=2,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
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
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "render_chart",
                                "input": {"kind": "bar"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "Rendered chart."},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "aW1n",
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "render_chart",
                        "arguments": json.dumps({"kind": "bar"}, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": [{"type": "text", "text": "Rendered chart."}, {"type": "image"}],
            "name": None,
        },
    ]
    assert mock_generate.call_args.kwargs["image"] == ["data:image/png;base64,aW1n"]


def test_anthropic_nonstreaming_preserves_thinking_with_tool_use(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    config = SimpleNamespace(
        model_type="muse_glimmer",
        thinking_start_token="to=self<|message|>",
        thinking_end_token="<|eom|>",
    )
    processor = SimpleNamespace(
        config=config, tokenizer=_MuseResponseTemplateTokenizer()
    )
    result = GenerationResult(
        text=(
            "to=self<|message|>I need the weather tool.<|eom|>"
            "<|start|>assistant to=get_weather<|message|>"
            '<atem:function_calls><atem:invoke name="get_weather">'
            '<atem:parameter name="city">Warsaw</atem:parameter>'
            "</atem:invoke></atem:function_calls>"
        ),
        prompt_tokens=7,
        generation_tokens=6,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
    )
    from mlx_vlm.tools.parsers import atem as tool_module

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "thinking": {"type": "enabled", "budget_tokens": 4},
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0] == {
        "type": "thinking",
        "thinking": "I need the weather tool.",
        "signature": "",
    }
    assert payload["content"][1]["type"] == "tool_use"
    assert payload["content"][1]["name"] == "get_weather"
    assert payload["content"][1]["input"] == {"city": "Warsaw"}
    assert "to=self" not in response.text
    assert "<atem:" not in response.text


def test_anthropic_messages_streaming_splits_gemma_thinking_channel_content(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="gemma4")

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                _gemma_thinking_channel_chunks()
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "What's 7*8?"}],
                "max_tokens": 16,
                "stream": True,
                "enable_thinking": True,
            },
        )

    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    deltas = [
        event["delta"] for event in events if event.get("type") == "content_block_delta"
    ]

    assert "".join(delta.get("text") or "" for delta in deltas) == "7 * 8 = 56"
    assert "".join(delta.get("thinking") or "" for delta in deltas) == ""
    assert "<|channel>" not in response.text
    assert "<channel|>" not in response.text


def test_anthropic_messages_streaming_uses_custom_thinking_markers(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="custom")

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text="<analysis>Custom reasoning.</analysis>Custom answer.",
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                    )
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 16,
                "stream": True,
                "enable_thinking": True,
                "thinking_start_token": "<analysis>",
                "thinking_end_token": "</analysis>",
            },
        )

    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    deltas = [
        event["delta"] for event in events if event.get("type") == "content_block_delta"
    ]

    assert "".join(delta.get("thinking") or "" for delta in deltas) == (
        "Custom reasoning."
    )
    assert "".join(delta.get("text") or "" for delta in deltas) == "Custom answer."


def test_anthropic_messages_streaming_emits_tool_use_events(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
    )

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text=(
                            '<tool_call>{"name":"get_weather","arguments":'
                            '{"location":"SF"}}</tool_call> After the call.'
                        ),
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                    )
                ]
            )

    monkeypatch.setattr(server.runtime, "response_generator", FakeResponseGenerator())

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object"},
                    }
                ],
                "max_tokens": 4,
                "stream": True,
            },
        )

    assert response.status_code == 200
    body = response.text
    assert '"type": "tool_use"' in body
    assert '"name": "get_weather"' in body
    assert '"type": "input_json_delta"' in body
    assert '"partial_json": "{\\"location\\": \\"SF\\"}"' in body
    assert '"text": " After the call."' in body
    assert '"stop_reason": "tool_use"' in body


ANTHROPIC_TOOLS = [
    {
        "name": "get_time",
        "description": "Get the current time",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_weather",
        "description": "Get the current weather",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _anthropic_tool_choice_request(client, tool_choice, tools=ANTHROPIC_TOOLS):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=5, generation_tokens=2)

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
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": 32,
            },
        )
    return response, mock_template


def test_anthropic_messages_tool_choice_none_disables_tools(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, mock_template = _anthropic_tool_choice_request(client, {"type": "none"})

    assert response.status_code == 200
    assert mock_template.call_args.kwargs["tools"] is None
    assert mock_template.call_args.kwargs["tool_choice"] == "none"


def test_anthropic_messages_any_tool_choice_adds_instruction(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, mock_template = _anthropic_tool_choice_request(client, {"type": "any"})

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert "must call one or more" in messages[-1]["content"]
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == [
        "get_time",
        "get_weather",
    ]
    assert mock_template.call_args.kwargs["tool_choice"] == "required"


@pytest.mark.parametrize(
    ("tool_choice", "tools", "message"),
    [
        (
            {"type": "tool", "name": "missing"},
            ANTHROPIC_TOOLS,
            "unknown function 'missing'",
        ),
        ({"type": "any"}, [], "requires at least one tool"),
    ],
)
def test_anthropic_messages_rejects_unsatisfiable_tool_choice(
    client, monkeypatch, tool_choice, tools, message
):
    monkeypatch.setattr(server.runtime, "response_generator", None)

    response, _ = _anthropic_tool_choice_request(client, tool_choice, tools=tools)

    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert message in payload["error"]["message"]


def test_anthropic_count_tokens_applies_tool_choice(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server_anthropic, "prepare_inputs", return_value={}),
        patch.object(server_anthropic, "_count_prompt_tokens", return_value=7),
    ):
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": ANTHROPIC_TOOLS,
                "tool_choice": {"type": "tool", "name": "get_time"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 7}
    selected_tools = mock_template.call_args.kwargs["tools"]
    assert [tool["function"]["name"] for tool in selected_tools] == ["get_time"]


def test_cache_endpoints_report_disabled_stats_and_reset(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "apc_manager", None)

    response = client.get("/v1/cache/stats")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    response = client.post("/v1/cache/reset")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    manager = SimpleNamespace(
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


# ── Continuous batching / ResponseGenerator tests ─────────────────────


class TestResponseGenerator:
    """Tests for the ResponseGenerator continuous batching engine."""

    def _bare_generator(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.draft_model = None
        gen.wait_until_ready = lambda: None
        gen._cpu_preprocess = lambda prompt, images, audio: {"input_ids": [1, 2, 3]}
        return gen

    def test_generate_rejects_requests_over_configured_context_limit(self, monkeypatch):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen.apc_manager = object()
        gen.apc_mode = "block"
        gen._preprocess_request = lambda prompt, images, audio, videos: {
            "input_ids": mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
            "pixel_values": mx.zeros((1, 3, 2, 2), dtype=mx.float32),
        }
        gen.requests = Queue()
        image_hash = MagicMock(wraps=apc_module.hash_image_payload)
        monkeypatch.setattr(apc_module, "hash_image_payload", image_hash)

        monkeypatch.setenv("MAX_KV_SIZE", "8")

        with pytest.raises(server.PromptTooLongError, match="MAX_KV_SIZE is 8"):
            gen.generate("prompt", args=server.GenerationArguments(max_tokens=4))

        assert gen.requests.empty()
        image_hash.assert_not_called()

    def test_generate_serializes_budget_criteria_with_tokenizer_preprocessing(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen._tokenizer_lock = Lock()
        gen._cancel = lambda uid: None

        state_lock = Lock()
        active = 0
        max_active = 0
        queued = []
        next_uid = 0

        def tokenizer_work():
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with state_lock:
                active -= 1

        def preprocess(prompt, images=None, audio=None, videos=None):
            del prompt, images, audio, videos
            tokenizer_work()
            return {"input_ids": mx.array([[99]], dtype=mx.int32)}

        def make_criteria(args, input_ids):
            del args, input_ids
            tokenizer_work()
            return object()

        class Requests:
            def put(self, request):
                nonlocal next_uid
                next_uid += 1
                queued.append(request)
                request.rqueue.put(
                    server.GenerationContext(uid=next_uid, prompt_tokens=1)
                )

        gen._preprocess_request = preprocess
        gen._make_thinking_budget_criteria = make_criteria
        gen.requests = Requests()

        def generate_one(_):
            _, token_iter = gen.generate(
                "prompt",
                args=server.GenerationArguments(max_tokens=1, thinking_budget=512),
            )
            token_iter.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(generate_one, range(4)))

        assert max_active == 1
        assert len(queued) == 4
        assert all(request.thinking_budget_criteria is not None for request in queued)

    def test_generate_precomputes_semantic_hash_from_processed_image_content(self):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.wait_until_ready = lambda: None
        gen.draft_model = None
        gen.apc_manager = object()
        gen.apc_mode = "block"
        gen.model = SimpleNamespace(language_model=SimpleNamespace())
        gen.processor = SimpleNamespace()
        gen._cancel = lambda uid: None

        pixel_values = iter(
            [
                mx.zeros((1, 3, 2, 2), dtype=mx.float32),
                mx.ones((1, 3, 2, 2), dtype=mx.float32),
            ]
        )
        queued = []

        def preprocess(prompt, images=None, audio=None, videos=None):
            del prompt, images, audio, videos
            return {
                "input_ids": mx.array([[1, 2]], dtype=mx.int32),
                "pixel_values": next(pixel_values),
            }

        class Requests:
            def put(self, request):
                queued.append(request)
                request.rqueue.put(
                    server.GenerationContext(uid=len(queued), prompt_tokens=2)
                )

        gen._preprocess_request = preprocess
        gen.requests = Requests()

        for _ in range(2):
            _, token_iter = gen.generate(
                "prompt",
                images=["mutable-image.png"],
                args=server.GenerationArguments(max_tokens=1),
            )
            token_iter.close()

        assert queued[0].images == queued[1].images
        assert queued[0].apc_semantic_hash != queued[1].apc_semantic_hash
        assert queued[0].apc_semantic_hash == apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(
                pixel_values=mx.zeros((1, 3, 2, 2), dtype=mx.float32)
            ),
            model=gen.model.language_model,
            processor=gen.processor,
        )
        assert queued[1].apc_semantic_hash == apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(
                pixel_values=mx.ones((1, 3, 2, 2), dtype=mx.float32)
            ),
            model=gen.model.language_model,
            processor=gen.processor,
        )

    def test_server_runtime_snapshot_reports_effective_context_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_KV_SIZE", "8")
        monkeypatch.setattr(
            server.runtime,
            "model_cache",
            {
                "config": SimpleNamespace(
                    text_config=SimpleNamespace(max_position_embeddings=16)
                )
            },
        )
        monkeypatch.setattr(server.runtime, "response_generator", None)
        monkeypatch.setattr(server.runtime, "apc_manager", None)

        runtime = server._server_runtime_snapshot()

        assert runtime["loaded_context_size"] == 16
        assert runtime["configured_context_limit"] == 8
        assert runtime["effective_context_limit"] == 8

    def test_generate_arguments_defaults(self):
        args = server.GenerationArguments()
        assert args.max_tokens == server.DEFAULT_MAX_TOKENS
        assert args.temperature == server.DEFAULT_TEMPERATURE
        assert args.enable_thinking is False
        assert args.logit_bias is None

    def test_token_queue_timeout_invalid_values_fall_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "bad")
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_can_disable_timeout(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0")
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())

        assert server.get_token_queue_timeout() is None

    def test_debug_decode_logging_adds_token_details(self, monkeypatch, caplog):
        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "2")
        caplog.set_level(logging.DEBUG, logger="mlx_vlm.server")
        info = {
            "request_id": "req-1",
            "queued_at": time.perf_counter() - 0.1,
            "generated_tokens": 0,
            "decode_started_at": None,
        }

        for token_number in range(1, 4):
            server.ResponseGenerator._log_decode_progress(
                1,
                info,
                token=token_number,
                text=str(token_number),
                finish_reason="stop" if token_number == 3 else None,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Decode progress: request=req-1 generated_tokens=1" in m
            and "token_number=1 token_id=1 text='1'" in m
            for m in messages
        )
        assert not any("Token streamed:" in m for m in messages)
        assert any("Decode started: request=req-1" in m for m in messages)
        assert any(
            "Decode completed: request=req-1 generated_tokens=3" in m for m in messages
        )

    def test_info_decode_logging_uses_interval_without_token_details(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "2")
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        info = {
            "request_id": "req-1",
            "queued_at": time.perf_counter(),
            "generated_tokens": 0,
            "decode_started_at": None,
        }

        for token_number in range(1, 3):
            server.ResponseGenerator._log_decode_progress(
                1, info, token=token_number, text=str(token_number), finish_reason=None
            )

        progress = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Decode progress:")
        ]
        assert len(progress) == 1
        assert "generated_tokens=2" in progress[0]
        assert "token_number=" not in progress[0]
        assert "token_id=" not in progress[0]
        assert "text=" not in progress[0]

    def test_chunked_prefill_logging_reports_partial_progress(self, caplog):
        caplog.set_level(logging.INFO, logger="mlx_vlm.server")
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        prompt_batch = SimpleNamespace(
            _processed_prompt_columns=2,
            _inputs_embeds=mx.zeros((1, 4, 8)),
            uids=[1],
            _suffix_lens=[6],
            _cached_tokens_per_row=[0],
            _left_padding_per_row=[0],
            _right_pad_per_row=None,
        )
        active = {1: {"request_id": "req-1", "prefill_processed": -1}}

        gen._log_prefill_progress(SimpleNamespace(_prompt_batch=prompt_batch), active)

        assert "Prefill progress: request=req-1 tokens=2/6 (33.3%)" in caplog.text

    def test_token_iterator_reports_timeout_and_cancels_request(self, monkeypatch):
        gen = self._bare_generator()
        cancelled = []

        class Requests:
            def put(self, item):
                rqueue = item.rqueue
                rqueue.put(SimpleNamespace(uid="req-1"))

        gen.requests = Requests()
        gen._cancel = cancelled.append
        monkeypatch.setattr(server.runtime.config, "token_queue_timeout", 0.01)

        _, token_iter = gen.generate("hello")

        with pytest.raises(RuntimeError, match="Timed out waiting for 0.01s"):
            next(token_iter)

        assert cancelled == ["req-1"]

    def test_token_iterator_close_cancels_while_next_blocks(self):
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
        token_iter = server_generation._TokenIterator(
            rqueue, "req-1", cancelled.append, None
        )

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

    def test_token_iterator_waits_past_timeout_for_delayed_token(self, monkeypatch):
        import threading

        gen = self._bare_generator()
        cancelled = []
        token = SimpleNamespace(text="hi")
        timeout_s = 0.05
        delay_s = timeout_s * 3

        class Requests:
            def put(self, item):
                rqueue: Queue = item.rqueue
                rqueue.put(SimpleNamespace(uid="req-1"))

                def deliver():
                    rqueue.put(token)
                    rqueue.put(None)

                threading.Timer(delay_s, deliver).start()

        gen.requests = Requests()
        gen._cancel = cancelled.append
        monkeypatch.setattr(
            server.runtime.config, "token_queue_timeout", timeout_s * 10
        )

        _, token_iter = gen.generate("hello")

        start = time.monotonic()
        assert next(token_iter) is token
        assert time.monotonic() - start >= delay_s * 0.5
        with pytest.raises(StopIteration):
            next(token_iter)
        assert cancelled == []

    def test_step_streams_spm_subword_tokens_immediately(self):
        class SentencePieceTokenizer:
            vocab = {"▁hello": 0, "world": 1, "!": 2}

            def decode(self, tokens):
                parts = []
                for token in tokens:
                    parts.append({0: " hello", 1: "world", 2: "!"}[token])
                return "".join(parts).lstrip()

        class SingleResponseBatch:
            def __init__(self, response):
                self.response = response

            def next(self, **kwargs):
                return [], [self.response]

        tokenizer = SentencePieceTokenizer()
        processor = SimpleNamespace(detokenizer=SPMStreamingDetokenizer(tokenizer))
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "streamer": _ServerTokenStreamer(
                    tokenizer, server.make_streaming_detokenizer(processor)
                ),
            }
        }

        for token in [0, 1, 2]:
            gen._step(
                SingleResponseBatch(
                    SimpleNamespace(
                        uid=1, token=token, token_logprob=0.0, finish_reason=None
                    )
                ),
                active,
            )
        gen._step(
            SingleResponseBatch(
                SimpleNamespace(
                    uid=1, token=99, token_logprob=0.0, finish_reason="stop"
                )
            ),
            active,
        )

        segments = []
        while not rqueue.empty():
            item = rqueue.get()
            if item is not None:
                segments.append(item.text)

        assert segments == ["hello", "world", "!", ""]

    def test_server_token_streamer_flushes_incomplete_utf8_on_finalize(self):
        class ByteFallbackTokenizer:
            vocab = {"<0xF0>": 0, "<0x9F>": 1}

            def decode(self, tokens):
                byte_values = {0: 0xF0, 1: 0x9F}
                return bytes(byte_values[token] for token in tokens).decode(
                    "utf-8", errors="replace"
                )

        tokenizer = ByteFallbackTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        streamer = _ServerTokenStreamer(
            tokenizer, server.make_streaming_detokenizer(processor)
        )

        assert streamer.advance(0, None) == ""
        assert streamer.advance(1, None) == ""
        assert streamer.finalize() == "\ufffd"

    def test_run_batches_eight_streaming_requests(self, monkeypatch):
        batch_state = {}

        class FakeDetokenizer:
            def __init__(self):
                self.last_segment = ""

            def reset(self):
                self.last_segment = ""

            def add_token(self, token):
                self.last_segment = str(token)

            def finalize(self):
                pass

        class FakeBatchGenerator:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self._next_uid = 1
                self._active = {}
                self.inserted_uids = []
                self.next_active_sizes = []
                batch_state["instance"] = self

            def insert(self, *args, **kwargs):
                del args, kwargs
                uid = self._next_uid
                self._next_uid += 1
                self._active[uid] = 0
                self.inserted_uids.append(uid)
                return (uid,)

            def remove(self, uid):
                return self._active.pop(uid, None) is not None

            @property
            def unprocessed_prompts(self):
                return []

            @property
            def has_pending_prompts(self):
                return False

            def next(self, **kwargs):
                del kwargs
                self.next_active_sizes.append(len(self._active))
                responses = []
                finished = []
                for uid in sorted(self._active):
                    step = self._active[uid]
                    token = uid * 10 + step
                    finish_reason = None if step == 0 else "length"
                    responses.append(
                        SimpleNamespace(
                            uid=uid,
                            token=token,
                            token_logprob=0.0,
                            finish_reason=finish_reason,
                        )
                    )
                    if finish_reason is None:
                        self._active[uid] = step + 1
                    else:
                        finished.append(uid)
                for uid in finished:
                    del self._active[uid]
                return [], responses

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            server_generation, "make_streaming_detokenizer", lambda _: FakeDetokenizer()
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.model_path = "demo"
        gen.adapter_path = None
        gen.model = None
        gen.processor = None
        gen.config = None
        gen.stop_tokens = set()
        gen.vision_cache = None
        gen.draft_model = None
        gen.draft_kind = None
        gen.kv_bits = None
        gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
        gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
        gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
        gen.top_logprobs_k = 0
        gen.apc_manager = None
        gen.tokenizer = SimpleNamespace()
        gen.requests = Queue()
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None
        gen._cancelled = set()
        gen._cancel_lock = Lock()

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = None
            gen.draft_kind = None
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
            mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
            {},
        )

        request_queues = []
        for request_id in range(8):
            rqueue = Queue()
            request_queues.append(rqueue)
            gen.requests.put(
                server_generation.QueuedGenerationRequest(
                    rqueue=rqueue,
                    raw_inputs={"request_id": request_id},
                    prompt_tokens=1,
                    args=server.GenerationArguments(max_tokens=2),
                )
            )

        worker = Thread(target=gen._run, daemon=True)
        worker.start()

        streamed_by_uid = {}
        try:
            for rqueue in request_queues:
                ctx = rqueue.get(timeout=1)
                assert isinstance(ctx, server.GenerationContext)
                assert ctx.prompt_tokens == 1

                items = []
                while True:
                    item = rqueue.get(timeout=1)
                    if item is None:
                        break
                    items.append((item.text, item.finish_reason))
                streamed_by_uid[ctx.uid] = items
        finally:
            gen._stop = True
            gen.requests.put(None)
            worker.join(timeout=2)

        batch_gen = batch_state["instance"]
        assert batch_gen.inserted_uids == list(range(1, 9))
        assert batch_gen.next_active_sizes[:2] == [8, 8]
        assert len(streamed_by_uid) == 8
        for uid, items in streamed_by_uid.items():
            assert items == [(str(uid * 10), None), (str(uid * 10 + 1), "length")]

    @pytest.mark.parametrize("draft_kind", ["dflash", "eagle3", "mtp"])
    def test_run_routes_speculative_decode_through_batch_generator(
        self, monkeypatch, draft_kind
    ):
        batch_state = {}
        draft_model = object()

        class FakeDetokenizer:
            def __init__(self):
                self.last_segment = ""

            def reset(self):
                self.last_segment = ""

            def add_token(self, token):
                self.last_segment = str(token)

            def finalize(self):
                pass

        class FakeBatchGenerator:
            def __init__(self, *args, **kwargs):
                del args
                batch_state["kwargs"] = kwargs
                self._next_uid = 1
                self._active = {}
                self.next_active_sizes = []
                self.apc = SimpleNamespace(prepare_prefill=MagicMock())
                batch_state["instance"] = self

            def insert(self, *args, **kwargs):
                del args, kwargs
                uid = self._next_uid
                self._next_uid += 1
                self._active[uid] = True
                return (uid,)

            def remove(self, uid):
                return self._active.pop(uid, None) is not None

            @property
            def unprocessed_prompts(self):
                return []

            @property
            def has_pending_prompts(self):
                return False

            def next(self, **kwargs):
                del kwargs
                self.next_active_sizes.append(len(self._active))
                responses = [
                    SimpleNamespace(
                        uid=uid,
                        token=uid + 100,
                        token_logprob=0.0,
                        finish_reason="length",
                    )
                    for uid in sorted(self._active)
                ]
                self._active.clear()
                return [], responses

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            server_generation, "_get_draft_block_size_from_env", lambda: 6
        )
        monkeypatch.setattr(
            server_generation, "make_streaming_detokenizer", lambda _: FakeDetokenizer()
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.model_path = "demo"
        gen.adapter_path = None
        gen.model = None
        gen.processor = None
        gen.config = None
        gen.stop_tokens = set()
        gen.vision_cache = None
        gen.draft_model = None
        gen.draft_kind = None
        gen.kv_bits = None
        gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
        gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
        gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
        gen.top_logprobs_k = 0
        apc_manager = SimpleNamespace(close=MagicMock())
        gen.apc_manager = apc_manager
        gen.prefill_step_size = 3072
        gen.tokenizer = SimpleNamespace()
        gen.requests = Queue()
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None
        gen._cancelled = set()
        gen._cancel_lock = Lock()

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = draft_model
            gen.draft_kind = draft_kind
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
            mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
            {},
        )

        request_queues = []
        for request_id in range(2):
            rqueue = Queue()
            request_queues.append(rqueue)
            gen.requests.put(
                server_generation.QueuedGenerationRequest(
                    rqueue=rqueue,
                    raw_inputs={"request_id": request_id},
                    prompt_tokens=1,
                    args=server.GenerationArguments(max_tokens=1, temperature=0),
                )
            )

        worker = Thread(target=gen._run, daemon=True)
        worker.start()

        try:
            for rqueue in request_queues:
                ctx = rqueue.get(timeout=1)
                assert isinstance(ctx, server.GenerationContext)
                item = rqueue.get(timeout=1)
                assert item.finish_reason == "length"
                assert rqueue.get(timeout=1) is None
        finally:
            gen._stop = True
            gen.requests.put(None)
            worker.join(timeout=2)

        kwargs = batch_state["kwargs"]
        assert kwargs["draft_model"] is draft_model
        assert kwargs["draft_kind"] == draft_kind
        assert kwargs["draft_block_size"] == 6
        assert kwargs["greedy_sampling"] is True
        assert kwargs["compute_logprobs"] is False
        assert kwargs["prefill_step_size"] == 3072
        assert kwargs["apc_manager"] is apc_manager
        coordinator = batch_state["instance"].apc
        assert coordinator.prepare_prefill.call_count == 2
        coordinator.prepare_prefill.assert_called_with(1, prefill_step_size=3072)
        apc_manager.close.assert_called_once_with()
        assert batch_state["instance"].next_active_sizes == [2]

    def test_idle_batch_generator_is_recreated_for_new_sampler(self, monkeypatch):
        created = []
        next_uid = [1]

        class FakeDetokenizer:
            def __init__(self):
                self.last_segment = ""

            def reset(self):
                self.last_segment = ""

            def add_token(self, token):
                self.last_segment = str(token)

            def finalize(self):
                pass

        class FakeBatchGenerator:
            def __init__(self, *args, **kwargs):
                del args
                self.sampler = kwargs.get("sampler")
                self.closed = False
                self._active = {}
                created.append(self)

            def insert(self, *args, **kwargs):
                del args, kwargs
                uid = next_uid[0]
                next_uid[0] += 1
                self._active[uid] = True
                return (uid,)

            @property
            def has_work(self):
                return bool(self._active)

            @property
            def unprocessed_prompts(self):
                return []

            @property
            def has_pending_prompts(self):
                return False

            def next(self, **kwargs):
                del kwargs
                responses = [
                    SimpleNamespace(
                        uid=uid, token=uid, token_logprob=0.0, finish_reason="length"
                    )
                    for uid in list(self._active)
                ]
                self._active.clear()
                return [], responses

            def remove(self, uid):
                return self._active.pop(uid, None) is not None

            def close(self):
                self.closed = True

        monkeypatch.setattr(server_generation, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(
            server_generation, "make_streaming_detokenizer", lambda _: FakeDetokenizer()
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.model_path = "demo"
        gen.adapter_path = None
        gen.model = None
        gen.processor = None
        gen.config = None
        gen.stop_tokens = set()
        gen.vision_cache = None
        gen.draft_model = None
        gen.draft_kind = None
        gen.kv_bits = None
        gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
        gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
        gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
        gen.top_logprobs_k = 0
        gen.apc_manager = None
        gen.tokenizer = SimpleNamespace()
        gen.requests = Queue()
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None
        gen._cancelled = set()
        gen._cancel_lock = Lock()
        gen._make_sampler = lambda args: f"sampler-{args.temperature}"

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = None
            gen.draft_kind = None
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._gpu_embed = lambda raw_inputs, images=None, apc_semantic_hash=None: (
            mx.array([[raw_inputs["request_id"]]], dtype=mx.int32),
            {},
        )

        worker = Thread(target=gen._run, daemon=True)
        worker.start()

        def run_request(request_id, temperature):
            rqueue = Queue()
            gen.requests.put(
                server_generation.QueuedGenerationRequest(
                    rqueue=rqueue,
                    raw_inputs={"request_id": request_id},
                    prompt_tokens=1,
                    args=server.GenerationArguments(
                        max_tokens=1, temperature=temperature
                    ),
                )
            )
            ctx = rqueue.get(timeout=1)
            assert isinstance(ctx, server.GenerationContext)
            item = rqueue.get(timeout=1)
            assert item.finish_reason == "length"
            assert rqueue.get(timeout=1) is None

        try:
            run_request(1, 0.0)
            run_request(2, 0.6)
        finally:
            gen._stop = True
            gen.requests.put(None)
            worker.join(timeout=2)

        assert [bg.sampler for bg in created] == ["sampler-0.0", "sampler-0.6"]
        assert created[0].closed is True

    def test_step_attaches_prompt_metrics_from_prompt_progress(self):
        class SimpleTokenizer:
            vocab = {"hi": 0}

            def decode(self, tokens):
                return "hi" if tokens else ""

        class PromptProgressBatch:
            def next(self, **kwargs):
                return (
                    [SimpleNamespace(uid=1, prompt_tps=184.431, cached_tokens=7)],
                    [
                        SimpleNamespace(
                            uid=1, token=0, token_logprob=0.0, finish_reason="stop"
                        )
                    ],
                )

        tokenizer = SimpleTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "streamer": _ServerTokenStreamer(
                    tokenizer, server.make_streaming_detokenizer(processor)
                ),
                "prompt_tps": None,
                "cached_tokens": 0,
            }
        }

        gen._step(PromptProgressBatch(), active)

        item = rqueue.get()
        assert item.prompt_tps == pytest.approx(184.431)
        assert item.cached_tokens == 7
        assert rqueue.get() is None

    def test_generate_arguments_to_generate_kwargs(self):
        processor = lambda tokens, logits: logits
        args = server.GenerationArguments(
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
            logits_processors=[processor],
            tenant_id="tenant-a",
        )
        kw = args.to_generate_kwargs()
        assert kw["max_tokens"] == 50
        assert kw["top_k"] == 40
        assert kw["min_p"] == 0.05
        assert kw["repetition_penalty"] == 1.15
        assert kw["repetition_context_size"] == 512
        assert kw["presence_penalty"] == 0.2
        assert kw["presence_context_size"] == 256
        assert kw["frequency_penalty"] == 0.3
        assert kw["frequency_context_size"] == 128
        assert kw["logit_bias"] == {3: -0.5}
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 100
        assert kw["thinking_start_token"] == "<think>"
        assert kw["thinking_end_token"] == "</think>"
        assert kw["logits_processors"] == [processor]
        assert kw["apc_tenant"] == "tenant-a"

    def test_server_generation_delays_structured_processors_for_thinking_prompt(
        self, monkeypatch
    ):
        class SimpleTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"<think>": [10], "</think>": [20]}[text]

        repetition_processor = lambda tokens, logits: logits
        structured_processor = lambda tokens, logits: logits

        monkeypatch.setattr(
            server_generation,
            "make_logits_processors",
            lambda *_args: [repetition_processor],
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.tokenizer = SimpleTokenizer()
        args = server.GenerationArguments(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured_processor],
        )

        processors = gen._make_logits_processors(
            args, mx.array([[1, 10, 3]], dtype=mx.int32)
        )

        assert processors[0] is repetition_processor
        assert isinstance(processors[1], server_generation.ThinkingAwareLogitsProcessor)
        assert processors[1].processor is structured_processor

    def test_server_generation_keeps_structured_processors_active_without_open_thinking(
        self, monkeypatch
    ):
        class SimpleTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"<think>": [10], "</think>": [20]}[text]

        structured_processor = lambda tokens, logits: logits
        monkeypatch.setattr(
            server_generation, "make_logits_processors", lambda *_args: []
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.tokenizer = SimpleTokenizer()
        args = server.GenerationArguments(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured_processor],
        )

        processors = gen._make_logits_processors(
            args, mx.array([[1, 10, 3, 20]], dtype=mx.int32)
        )

        assert processors == [structured_processor]

    def test_server_generation_delays_structured_processors_for_self_opening_model(
        self, monkeypatch
    ):
        """Regression test for issue #1911."""

        class SimpleTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"<think>": [10], "</think>": [20]}[text]

        repetition_processor = lambda tokens, logits: logits
        structured_processor = lambda tokens, logits: logits

        monkeypatch.setattr(
            server_generation,
            "make_logits_processors",
            lambda *_args: [repetition_processor],
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.tokenizer = SimpleTokenizer()
        args = server.GenerationArguments(
            enable_thinking=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            logits_processors=[structured_processor],
        )

        processors = gen._make_logits_processors(
            args, mx.array([[1, 2, 3]], dtype=mx.int32)
        )

        assert processors[0] is repetition_processor
        assert isinstance(processors[1], server_generation.ThinkingAwareLogitsProcessor)
        assert processors[1].processor is structured_processor

    def test_build_gen_args_from_chat_request(self):
        req = SimpleNamespace(
            max_tokens=256,
            max_output_tokens=None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=None,
            repetition_context_size=None,
            presence_penalty=None,
            presence_context_size=None,
            frequency_penalty=None,
            frequency_context_size=None,
            logit_bias=None,
            enable_thinking=True,
            thinking_budget=None,
            thinking_start_token=None,
            thinking_end_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True

    def test_build_gen_args_maps_chat_reasoning_effort(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            reasoning_effort="low",
        )

        args = server._build_gen_args(req)

        assert args.enable_thinking is True
        assert args.reasoning is True
        assert args.reasoning_effort == "low"

    def test_build_gen_args_uses_server_thinking_default_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "1")
        req = server.ChatRequest(
            model="demo", messages=[server.ChatMessage(role="user", content="hi")]
        )

        assert "enable_thinking" not in req.model_fields_set
        assert server._build_gen_args(req).enable_thinking is True

        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        req = server.ChatRequest(
            model="demo", messages=[server.ChatMessage(role="user", content="hi")]
        )

        assert server._build_gen_args(req).enable_thinking is False

    def test_server_cli_sets_thinking_defaults(self, monkeypatch):
        for env_var in (
            "MLX_VLM_ENABLE_THINKING",
            "MLX_VLM_PRELOAD_MODEL",
            "MLX_VLM_PRELOAD_ADAPTER",
            "MLX_VLM_PRELOAD_IMAGE_MODEL",
            "MLX_VLM_PRELOAD_TTS_MODEL",
            "MLX_VLM_PRELOAD_STT_MODEL",
            "MLX_VLM_PRELOAD_RERANKER_MODEL",
            "MLX_VLM_MODEL_DISCOVERY",
            "MLX_VLM_VISION_CACHE_SIZE",
            "MLX_VLM_MAX_TOKENS",
            "MLX_VLM_THINKING_BUDGET",
            "MLX_VLM_THINKING_START_TOKEN",
            "MLX_VLM_THINKING_END_TOKEN",
            "MLX_VLM_SERVER_API_KEY",
            "PREFILL_STEP_SIZE",
            "KV_GROUP_SIZE",
            "KV_QUANT_SCHEME",
            "QUANTIZED_KV_START",
        ):
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "mlx_vlm.server",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--model",
                "demo",
                "--image-model",
                "image-demo",
                "--tts-model",
                "tts-demo",
                "--stt-model",
                "stt-demo",
                "--reranker-model",
                "reranker-demo",
                "--model-discovery",
                "served",
                "--enable-thinking",
                "--thinking-budget",
                "128",
                "--thinking-start-token",
                "<|START_THINKING|>",
                "--thinking-eos-token",
                "<|END_THINKING|>",
                "--api-key",
                "admin-token",
            ],
        )
        run_calls = []
        monkeypatch.setattr(
            server_cli.uvicorn,
            "run",
            lambda *args, **kwargs: run_calls.append((args, kwargs)),
        )

        try:
            server_cli.main()

            assert os.environ["MLX_VLM_ENABLE_THINKING"] == "1"
            assert os.environ["MLX_VLM_THINKING_BUDGET"] == "128"
            assert os.environ["MLX_VLM_THINKING_START_TOKEN"] == "<|START_THINKING|>"
            assert os.environ["MLX_VLM_THINKING_END_TOKEN"] == "<|END_THINKING|>"
            assert os.environ["MLX_VLM_PRELOAD_MODEL"] == "demo"
            assert os.environ["MLX_VLM_PRELOAD_IMAGE_MODEL"] == "image-demo"
            assert os.environ["MLX_VLM_PRELOAD_TTS_MODEL"] == "tts-demo"
            assert os.environ["MLX_VLM_PRELOAD_STT_MODEL"] == "stt-demo"
            assert os.environ["MLX_VLM_PRELOAD_RERANKER_MODEL"] == "reranker-demo"
            assert os.environ["MLX_VLM_MODEL_DISCOVERY"] == "served"
            assert os.environ["MLX_VLM_SERVER_API_KEY"] == "admin-token"
            assert run_calls[0][1]["host"] == "127.0.0.1"
        finally:
            for env_var in (
                "MLX_VLM_ENABLE_THINKING",
                "MLX_VLM_PRELOAD_MODEL",
                "MLX_VLM_PRELOAD_ADAPTER",
                "MLX_VLM_PRELOAD_IMAGE_MODEL",
                "MLX_VLM_PRELOAD_TTS_MODEL",
                "MLX_VLM_PRELOAD_STT_MODEL",
                "MLX_VLM_PRELOAD_RERANKER_MODEL",
                "MLX_VLM_MODEL_DISCOVERY",
                "MLX_VLM_VISION_CACHE_SIZE",
                "MLX_VLM_MAX_TOKENS",
                "MLX_VLM_THINKING_BUDGET",
                "MLX_VLM_THINKING_START_TOKEN",
                "MLX_VLM_THINKING_END_TOKEN",
                "MLX_VLM_SERVER_API_KEY",
            ):
                os.environ.pop(env_var, None)

    def test_lifespan_continues_when_optional_preload_fails(self, monkeypatch):
        preload_env = {
            "MLX_VLM_PRELOAD_MODEL": "language-demo",
            "MLX_VLM_PRELOAD_TTS_MODEL": "tts-demo",
            "MLX_VLM_PRELOAD_STT_MODEL": "stt-demo",
            "MLX_VLM_PRELOAD_EMBEDDING_MODEL": "embed-demo",
            "MLX_VLM_PRELOAD_RERANKER_MODEL": "reranker-demo",
        }
        for key, value in preload_env.items():
            monkeypatch.setenv(key, value)
        calls = []

        def fake_get_cached_model(model_path, adapter_path=None, *, model_kind="auto"):
            calls.append(model_kind)
            if model_kind == "audio_stt":
                raise server.HTTPException(
                    status_code=500, detail="Failed to load audio model: boom"
                )
            return SimpleNamespace(), None, SimpleNamespace(model_type=model_kind)

        monkeypatch.setattr(
            server._app_module, "get_cached_model", fake_get_cached_model
        )
        monkeypatch.setattr(server.runtime, "audio_queue", None)
        server.runtime.preload_failures.clear()

        async def run_lifespan():
            async with server._app_module.lifespan(server.app):
                pass

        asyncio.run(run_lifespan())

        assert calls == [
            "text_generation",
            "audio_tts",
            "audio_stt",
            "embedding",
            "reranker",
        ]
        failure = server.runtime.preload_failures["audio_stt"]
        assert failure["model"] == "stt-demo"
        assert "Failed to load audio model" in failure["error"]
        assert "audio_tts" not in server.runtime.preload_failures
        server.runtime.preload_failures.clear()

    def test_gpu_embed_hashes_pixel_values_without_image_ref(self):
        class Embed:
            def to_dict(self):
                return {"inputs_embeds": mx.zeros((1, 2, 4))}

        class Model:
            def get_input_embeddings(
                self, input_ids, pixel_values, mask=None, **kwargs
            ):
                return Embed()

        response_generator = SimpleNamespace(model=Model(), vision_cache=None)
        pixel_values = mx.array([[[[1.0, 2.0]]]])
        semantic_hash = apc_module.semantic_extra_hash(
            image_hash=hash_image_payload(pixel_values=pixel_values)
        )

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=None,
            apc_semantic_hash=semantic_hash,
        )

        assert gen_kwargs["_apc_semantic_hash"] == semantic_hash

    def test_gpu_embed_drops_none_embedding_fields(self):
        class Embed:
            def to_dict(self):
                return {
                    "inputs_embeds": mx.zeros((1, 2, 4)),
                    "position_ids": None,
                    "rope_deltas": None,
                }

        class Model:
            def get_input_embeddings(
                self, input_ids, pixel_values, mask=None, **kwargs
            ):
                return Embed()

        response_generator = SimpleNamespace(model=Model(), vision_cache=None)

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {"input_ids": mx.array([[1, 2]]), "attention_mask": mx.array([[1, 1]])},
            images=None,
        )

        assert "position_ids" not in gen_kwargs
        assert "rope_deltas" not in gen_kwargs
        assert "_apc_semantic_hash" not in gen_kwargs

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

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_extract_responses_text_format_json_object_aliases(self, format_type):
        req = SimpleNamespace(
            response_format=None, text={"format": {"type": format_type}}
        )

        assert server._extract_response_format_schema(req) == {"type": "object"}

    def test_build_structured_logits_processors_uses_tokenizer(self):
        req = SimpleNamespace(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "animal", "schema": {"type": "object"}},
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

    def test_think_tags(self):
        text = "<think>Thinking.</think>Answer."
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Thinking."
        assert content == "Answer."

    @pytest.mark.parametrize("prefix", ["", "thought\n"])
    def test_channel_close_only(self, prefix):
        assert server._split_thinking(f"{prefix}got it<channel|>42") == ("got it", "42")

    def test_unterminated_thinking_without_markers_is_reasoning(self):
        text = "The user is asking me to say OK. This is a simple request"
        reasoning, content = server._split_thinking(text, starts_in_thinking=True)
        assert reasoning == text
        assert content == ""


class TestThinkingStreamState:
    """Tests for streaming thinking tag parsing."""

    def test_last_chunk_releases_text_held_for_an_unfinished_marker(self):
        state = server.ThinkingStreamState()

        assert state.feed("hello <").content == "hello "
        assert state.feed("", last=True).content == "<"

    def test_last_chunk_releases_reasoning_held_for_an_unfinished_marker(self):
        state = server.ThinkingStreamState()

        state.feed("<think>")
        assert state.feed("cut off </thi", last=True).reasoning == "cut off </thi"

    @pytest.mark.parametrize("enable_thinking", [False, True])
    def test_gemma_channel_markers_and_content_in_same_delta(self, enable_thinking):
        state = server.ThinkingStreamState(enable_thinking=enable_thinking)
        reasoning = []
        content = []

        for token in _gemma_thinking_channel_chunks():
            delta = state.feed(token.text)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)

        assert "".join(reasoning) == ""
        assert "".join(content) == "7 * 8 = 56"

    def test_response_template_markers_split_across_chunks(self):
        state = server.make_response_stream_state(
            SimpleNamespace(tokenizer=_MuseResponseTemplateTokenizer()),
            thinking_start_token="unused-start",
            thinking_end_token="unused-end",
        )
        reasoning = []
        content = []

        chunks = (
            "to=self<|mes",
            "sage|>Muse reasoning.<|eom|><|start|>assistant ",
            "to=user<|message|>Muse answer.",
        )
        thinking_closed = False
        for index, chunk in enumerate(chunks):
            delta = state.feed(chunk, last=index == len(chunks) - 1)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
            thinking_closed = thinking_closed or delta.thinking_closed

        assert "".join(reasoning) == "Muse reasoning."
        assert "".join(content) == "Muse answer."
        assert thinking_closed is True

    def test_cohere_text_markers_are_suppressed_across_chunks(self):
        state = server.ThinkingStreamState(enable_thinking=True)
        reasoning = []
        content = []

        for chunk in [
            "Custom reasoning.",
            "<|END_THINKING|><|START_",
            "TEXT|>Custom answer.<|END_",
            "TEXT|>",
        ]:
            delta = state.feed(chunk)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)

        assert "".join(reasoning) == "Custom reasoning."
        assert "".join(content) == "Custom answer."


class TestToolCallStreamState:
    """Tests for tool-call markup suppression in streaming."""

    def test_suppresses_on_start_marker(self):
        state = server.ToolCallStreamState("<tool_call>", "</tool_call>")
        assert state.feed("text<tool_call>") == "text"
        assert state.in_tool_call is True


class TestProcessToolCalls:
    """Tests for tool call parsing from model output."""

    minicpm5_call = (
        '<function name="write_file"><param name="content">'
        "<![CDATA[  <html>\nA & B\n</html>  ]]></param>"
        '<param name="version">123</param><param name="count">3</param>'
        '<param name="enabled">True</param></function>'
    )

    def test_minicpm5_cdata_and_argument_types(self):
        tools = [
            {
                "function": {
                    "name": "write_file",
                    "parameters": {"properties": {"version": {"type": "string"}}},
                }
            }
        ]
        result = minicpm5.parse_tool_call(self.minicpm5_call, tools)

        assert result == {
            "name": "write_file",
            "arguments": {
                "content": "  <html>\nA & B\n</html>  ",
                "version": "123",
                "count": 3,
                "enabled": True,
            },
        }

    def test_minicpm5_multiple_calls_and_streamed_markup(self):
        text = f'Before{self.minicpm5_call}Between<function name="get_time"></function>After'
        result = server.process_tool_calls(text, minicpm5, tools=None)

        assert result.remaining_text == "Before Between After"
        assert [call["function"]["name"] for call in result.calls] == [
            "write_file",
            "get_time",
        ]
        assert json.loads(result.calls[1]["function"]["arguments"]) == {}

        state = server.ToolCallStreamState(
            minicpm5.tool_call_start, minicpm5.tool_call_end
        )
        visible = "".join(state.feed(char) or "" for char in text)
        visible += state.feed("", last=True) or ""
        assert visible == "BeforeBetweenAfter"

    @pytest.mark.parametrize(
        "text",
        [
            '<function name="lookup"><param name="value">unfinished',
            '<function name=""></function>',
            '<function name="lookup"><param>3</param></function>',
        ],
    )
    def test_minicpm5_rejects_malformed_calls(self, text):
        with pytest.raises(ValueError):
            minicpm5.parse_tool_call(text)


class TestCountThinkingTagTokens:
    """Tests for thinking tag token counting."""

    def test_think_tags(self):
        assert server._count_thinking_tag_tokens("<think>text</think>answer") == 2


class TestQuantizedKVBits:
    @pytest.mark.parametrize(
        "model_path",
        [
            "mlx-community/gemma-4-31B-it-qat-mxfp4",
            "mlx-community/gemma-4-31B-it-QAT-mxfp4",
            "/models/qat-experiments/llama-3",
            "some-org/qatar-news-llm",
        ],
    )
    def test_kv_bits_not_suppressed_by_model_path(self, monkeypatch, model_path):
        # KV cache quantization is independent of how the weights were trained,
        # so nothing in the model path may suppress it (#1333).
        monkeypatch.setenv("KV_BITS", "3.5")
        monkeypatch.setenv("MAX_KV_SIZE", "0")
        assert server_generation.get_quantized_kv_bits() == 3.5
        assert server_generation.get_max_kv_size(model_path) is None


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

    def test_settings_patch_requires_json_object(self, client, monkeypatch):
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        r = client.patch("/v1/settings", json=[1, 2, 3])
        assert r.status_code == 400


class TestRuntimeConfigAdditions:
    def test_max_kv_size_is_live_context_limit(self, monkeypatch):
        import mlx_vlm.server.generation as server_generation

        monkeypatch.setattr(server.runtime.config, "max_kv_size", 4096)
        assert server_generation.get_configured_context_limit() == 4096

        monkeypatch.setattr(server.runtime.config, "max_kv_size", None)
        monkeypatch.delenv("MAX_KV_SIZE", raising=False)
        assert server_generation.get_configured_context_limit() is None

    def test_settings_patch_replace_semantics(self, client, monkeypatch):
        monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
        cfg = server.runtime.config
        assert cfg.apc_enabled is False

        client.patch(
            "/v1/settings", json={"kv_quant_scheme": "turboquant", "apc_enabled": True}
        )
        assert cfg.kv_quant_scheme == "turboquant"
        assert cfg.apc_enabled is True

        r = client.patch(
            "/v1/settings",
            json={"op": "replace", "values": {"kv_quant_scheme": "uniform"}},
        )
        body = r.json()
        assert body["op"] == "replace"
        assert cfg.kv_quant_scheme == "uniform"
        assert cfg.apc_enabled is False

        r = client.patch("/v1/settings", json={"op": "bogus", "values": {}})
        assert r.status_code == 400

        r = client.patch("/v1/settings", json={"op": "replace", "values": "x"})
        assert r.status_code == 400


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
    def test_requires_model(self, client, monkeypatch):
        monkeypatch.delenv("MLX_VLM_PRELOAD_RERANKER_MODEL", raising=False)

        response = client.post("/v1/rerank", json={"query": "q", "documents": ["d"]})

        assert response.status_code == 400
        assert "No reranker model specified" in response.json()["detail"]

    def test_sorts_limits_and_returns_documents(self, client, monkeypatch):
        cache_calls = []

        def fake_get_cached_model(model, *, model_kind):
            cache_calls.append((model, model_kind))
            return object(), object(), SimpleNamespace(model_type="qwen3")

        monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)
        monkeypatch.setattr(
            server_reranking, "score_documents", lambda *args: ([0.2, 0.9, 0.5], 12)
        )

        response = client.post(
            "/v1/rerank",
            json={
                "model": "reranker",
                "query": "query",
                "documents": ["first", "second", "third"],
                "top_n": 2,
                "return_documents": True,
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "model": "reranker",
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": "second"},
                {"index": 2, "relevance_score": 0.5, "document": "third"},
            ],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }
        assert cache_calls == [("reranker", "reranker")]

    def test_uses_preloaded_model(self, client, monkeypatch):
        monkeypatch.setenv("MLX_VLM_PRELOAD_RERANKER_MODEL", "preloaded")
        seen = []

        def fake_get_cached_model(model, *, model_kind):
            seen.append((model, model_kind))
            return object(), object(), SimpleNamespace(model_type="qwen3")

        monkeypatch.setattr(server, "get_cached_model", fake_get_cached_model)
        monkeypatch.setattr(
            server_reranking, "score_documents", lambda *args: ([0.7], 4)
        )

        response = client.post(
            "/v1/rerank", json={"query": "query", "documents": ["document"]}
        )

        assert response.status_code == 200
        assert response.json()["model"] == "preloaded"
        assert seen == [("preloaded", "reranker")]

    @pytest.mark.parametrize(
        "value,label,expected",
        [
            ("  text  ", "query", server_reranking.RerankItem(text="text")),
            ({"text": " text "}, "query", server_reranking.RerankItem(text="text")),
            (
                {"image_url": {"url": " image.png "}},
                "documents[0]",
                server_reranking.RerankItem(image="image.png"),
            ),
            (
                {"video": " video.mp4 "},
                "documents[0]",
                server_reranking.RerankItem(video="video.mp4"),
            ),
        ],
    )
    def test_normalizes_items(self, value, label, expected):
        assert server_reranking.normalize_item(value, label) == expected

    @pytest.mark.parametrize("value", ["", "   ", {}, {"text": " "}, {"image": {}}])
    def test_rejects_empty_items(self, value):
        with pytest.raises(ValueError):
            server_reranking.normalize_item(value, "query")

    def test_text_model_rejects_media(self):
        with pytest.raises(ValueError, match="do not support image or video"):
            server_reranking.score_documents(
                object(),
                object(),
                SimpleNamespace(model_type="qwen3"),
                server_reranking.RerankItem(image="image.png"),
                [server_reranking.RerankItem(text="document")],
                "instruction",
            )

    def test_vl_messages_preserve_content_order(self):
        messages = server_reranking._vl_messages(
            server_reranking.RerankItem(text="query", image="query.png"),
            server_reranking.RerankItem(text="document", video="document.mp4"),
            "rank candidates",
        )

        assert messages[1]["content"] == [
            {"type": "text", "text": "<Instruct>: rank candidates"},
            {"type": "text", "text": "<Query>:"},
            {"type": "image"},
            {"type": "text", "text": "query"},
            {"type": "text", "text": "\n<Document>:"},
            {"type": "video"},
            {"type": "text", "text": "document"},
        ]

    def test_batches_without_reordering(self, monkeypatch):
        batches = []

        def fake_score_batch(model, processor, query, documents, instruction):
            del model, processor, query, instruction
            batches.append([document.text for document in documents])
            return [float(document.text) for document in documents], len(documents)

        monkeypatch.setenv("MLX_VLM_RERANK_BATCH_SIZE", "2")
        monkeypatch.setattr(server_reranking, "_score_text_batch", fake_score_batch)
        documents = [server_reranking.RerankItem(text=str(index)) for index in range(5)]

        scores, tokens = server_reranking.score_documents(
            object(),
            object(),
            SimpleNamespace(model_type="qwen3"),
            server_reranking.RerankItem(text="query"),
            documents,
            "instruction",
        )

        assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert tokens == 5
        assert batches == [["0", "1"], ["2", "3"], ["4"]]

    def test_sequence_classifier_scores_tokenized_pairs(self):
        calls = []

        class Tokenizer:
            model_max_length = 6

            def __call__(self, queries, documents, **kwargs):
                calls.append((queries, documents, kwargs))
                return {
                    "input_ids": np.array([[1, 2, 3, 0], [1, 4, 5, 6]]),
                    "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
                    "token_type_ids": np.array([[0, 0, 1, 0], [0, 0, 1, 1]]),
                }

        class Model:
            def __call__(self, **inputs):
                assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
                return SimpleNamespace(logits=mx.array([[-2.0], [2.0]]))

        scores, tokens = server_reranking.score_documents(
            Model(),
            Tokenizer(),
            SimpleNamespace(model_type="bert", max_position_embeddings=4),
            server_reranking.RerankItem(text="query"),
            [
                server_reranking.RerankItem(text="first"),
                server_reranking.RerankItem(text="second"),
            ],
            None,
        )

        assert scores == pytest.approx([1 / (1 + math.exp(2)), 1 / (1 + math.exp(-2))])
        assert tokens == 7
        assert calls == [
            (
                ["query", "query"],
                ["first", "second"],
                {
                    "padding": True,
                    "truncation": True,
                    "max_length": 4,
                    "return_tensors": "np",
                },
            )
        ]

    @pytest.mark.parametrize(
        "query,documents,instruction,error",
        [
            (
                server_reranking.RerankItem(image="query.png"),
                [server_reranking.RerankItem(text="document")],
                None,
                "do not support image or video",
            ),
            (
                server_reranking.RerankItem(text="query"),
                [server_reranking.RerankItem(text="document")],
                "rank legal documents",
                "do not support custom instructions",
            ),
        ],
    )
    def test_sequence_classifier_rejects_unsupported_inputs(
        self, query, documents, instruction, error
    ):
        with pytest.raises(ValueError, match=error):
            server_reranking.score_documents(
                object(),
                object(),
                SimpleNamespace(model_type="modernbert"),
                query,
                documents,
                instruction,
            )

    def test_attention_mask_combines_padding_and_causality(self):
        mask = server_reranking._attention_mask(mx.array([[0, 1, 1], [1, 1, 0]]))

        assert mask.shape == (2, 1, 3, 3)
        assert mask[0, 0].tolist() == [
            [False, False, False],
            [False, True, False],
            [False, True, True],
        ]
        assert mask[1, 0].tolist() == [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ]

    def test_attention_mask_uses_native_causal_path_without_padding(self):
        assert server_reranking._attention_mask(mx.ones((2, 3))) == "causal"

    def test_binary_scores_pool_last_non_padding_token(self):
        model = SimpleNamespace(
            language_model=SimpleNamespace(lm_head=lambda hidden_states: hidden_states)
        )
        tokenizer = SimpleNamespace(
            unk_token_id=None,
            convert_tokens_to_ids=lambda token: {"no": 0, "yes": 1}[token],
        )
        hidden_states = mx.array(
            [
                [[9.0, -9.0], [2.0, 4.0], [1.0, 5.0]],
                [[4.0, 1.0], [8.0, 2.0], [-9.0, 9.0]],
            ]
        )

        scores = server_reranking._binary_scores(
            model, hidden_states, mx.array([[0, 1, 1], [1, 1, 0]]), tokenizer
        )

        assert scores == pytest.approx([1 / (1 + math.exp(-4)), 1 / (1 + math.exp(6))])

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3],
            {"input_ids": [1, 2, 3]},
            SimpleNamespace(input_ids=[1, 2, 3]),
            SimpleNamespace(input_ids=[[1, 2, 3]]),
            mx.array([1, 2, 3]),
        ],
    )
    def test_input_ids_accept_tokenizer_return_types(self, value):
        assert server_reranking._input_ids(value) == [1, 2, 3]

    def test_ensure_chat_template_loads_packaged_template(self, tmp_path, monkeypatch):
        (tmp_path / "chat_template.jinja").write_text("template", encoding="utf-8")
        processor = SimpleNamespace(
            chat_template=None, tokenizer=SimpleNamespace(chat_template=None)
        )
        monkeypatch.setattr(server_reranking, "get_model_path", lambda path: tmp_path)

        server_reranking.ensure_chat_template(processor, "reranker")

        assert processor.chat_template == "template"
        assert processor.tokenizer.chat_template == "template"

    def test_model_uses_isolated_cache(self, monkeypatch):
        registry = server.ModelCacheRegistry()
        text_cache = {
            "cache_key": ("language", None, "text_generation"),
            "model_kind": "text_generation",
        }
        registry.set("text_generation", text_cache)
        monkeypatch.setattr(server.runtime, "model_cache", registry)
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
        processor = object()
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, processor)
        )
        monkeypatch.setattr(
            server._app_module, "ensure_reranker_chat_template", lambda *args: None
        )

        loaded = server.get_cached_model("reranker", None, model_kind="reranker")

        assert loaded == (model, processor, model.config)
        assert registry.for_kind("text_generation") is text_cache
        assert registry.for_kind("reranker")["cache_key"] == (
            "reranker",
            None,
            "reranker",
            server.runtime.config.fingerprint(kinds={"reranker"}),
        )

    def test_loader_rejects_unsupported_family(self, monkeypatch):
        monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
        model = SimpleNamespace(config=SimpleNamespace(model_type="deberta_v2"))
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, object())
        )

        with pytest.raises(
            server.HTTPException, match="Unsupported reranker model type"
        ) as exc:
            server.get_cached_model("reranker", None, model_kind="reranker")

        assert exc.value.status_code == 400

    def test_loader_skips_chat_template_for_sequence_classifier(self, monkeypatch):
        monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
        model = SimpleNamespace(config=SimpleNamespace(model_type="bert"))
        processor = object()
        monkeypatch.setattr(
            reranker_loader, "load_reranker", lambda path: (model, processor)
        )
        monkeypatch.setattr(
            server._app_module,
            "ensure_reranker_chat_template",
            lambda *args: pytest.fail("sequence classifiers do not use chat templates"),
        )

        loaded = server.get_cached_model("reranker", None, model_kind="reranker")

        assert loaded == (model, processor, model.config)


@dataclass
class _FakeAlignedToken:
    id: int
    text: str
    start: float
    duration: float
    end: float = 0.0

    def __post_init__(self):
        self.end = self.start + self.duration


@dataclass
class _FakeAlignedSentence:
    text: str
    tokens: list
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self):
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end


@dataclass
class _FakeAlignedResult:
    text: str
    sentences: list


def _fake_parakeet_result():
    first = _FakeAlignedSentence(
        "Hello world.",
        [
            _FakeAlignedToken(1, "Hello", 0.0, 0.4),
            _FakeAlignedToken(2, " world.", 0.4, 0.5),
        ],
    )
    second = _FakeAlignedSentence("Bye.", [_FakeAlignedToken(3, "Bye.", 1.0, 0.3)])
    return _FakeAlignedResult("Hello world. Bye.", [first, second])


class TestSTTSegmentSerialization:
    """Serialization of STT results into OpenAI-style transcription payloads.

    Regression coverage for NeMo-alignment models (Parakeet/Canary) whose
    ``AlignedResult`` exposes ``sentences`` rather than ``segments`` (issue 2183).
    """

    def test_pipeline_preserves_nemo_segments(self):
        from mlx_vlm.server.audio import (
            _iter_stt_items,
            _sanitize_for_json,
            _stt_item_to_dict,
            _transcription_result_from_chunks,
        )

        chunks = [
            json.dumps(_sanitize_for_json(_stt_item_to_dict(item))) + "\n"
            for item in _iter_stt_items(_fake_parakeet_result())
        ]
        result = _transcription_result_from_chunks(chunks)

        assert result["text"].startswith("Hello world.")
        assert len(result.get("segments") or []) == 2

    def test_plain_text_item_unchanged(self):
        from mlx_vlm.server.audio import _stt_item_to_dict

        assert _stt_item_to_dict("just text") == {"text": "just text"}


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
    chunks = server_generation._diffusion_block_chunks(iter(results))
    assert [(chunk.text, chunk.finish_reason) for chunk in chunks] == expected
