import asyncio
import base64
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

import mlx_vlm.server as server
import mlx_vlm.server.cli as server_cli
import mlx_vlm.server.generation as server_generation
import mlx_vlm.server.openai as server_openai
import mlx_vlm.speculative.utils as speculative_utils
from mlx_vlm.apc import hash_image_payload
from mlx_vlm.generate import GenerationResult
from mlx_vlm.generate.image import ImageGenerationResult
from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer, _ServerTokenStreamer


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


def test_chat_completions_endpoint_requires_model(client):
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 422
    detail = response.json().get("detail", [])
    assert any(err.get("loc") == ["body", "model"] for err in detail)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": " \n\t "}],
        [{"role": "user", "content": [{"type": "text", "text": " "}]}],
    ],
)
def test_chat_completions_endpoint_rejects_empty_effective_input(client, messages):
    with patch.object(server_openai, "get_cached_model") as mock_get_cached_model:
        response = client.post(
            "/chat/completions",
            json={"model": "demo", "messages": messages},
        )

    assert response.status_code == 400
    assert "non-empty message content" in response.json()["detail"]
    mock_get_cached_model.assert_not_called()


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
            "/v1/responses",
            json={"model": "demo", "input": input_value},
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


def test_chat_completions_required_tool_choice_adds_instruction(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(text="done", prompt_tokens=5, generation_tokens=2)
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
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": tools,
                "tool_choice": "required",
            },
        )

    assert response.status_code == 200
    messages = mock_template.call_args.args[2]
    assert messages[0]["role"] == "user"
    assert "must call one or more" in messages[0]["content"]
    assert mock_template.call_args.kwargs["tools"] == tools
    assert mock_template.call_args.kwargs["tool_choice"] == "required"


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
            [
                {
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            ],
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


def test_speculative_server_samples_first_bonus_like_decode_step():
    seen = {}
    logits = mx.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 1.0, 0.0]],
        ],
        dtype=mx.float32,
    )

    def sampler(logprobs):
        seen["shape"] = logprobs.shape
        seen["values"] = logprobs
        return mx.argmax(logprobs, axis=-1)

    tokens = server_generation._sample_last_token(logits, sampler)
    expected_logprobs = logits[:, -1, :] - mx.logsumexp(
        logits[:, -1, :], axis=-1, keepdims=True
    )
    mx.eval(tokens, seen["values"], expected_logprobs)

    assert seen["shape"] == (2, 3)
    assert tokens.tolist() == [2, 0]
    assert bool(mx.allclose(seen["values"], expected_logprobs).item())


def test_speculative_server_samples_first_bonus_with_positioned_sampler():
    seen = {}
    logits = mx.array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 1.0, 0.0]],
        ],
        dtype=mx.float32,
    )

    class Sampler:
        def __call__(self, logprobs):
            raise AssertionError("positioned sampler was not used")

        def sample_target(self, logprobs, *, row_ids, positions):
            seen["shape"] = logprobs.shape
            seen["row_ids"] = list(row_ids)
            seen["positions"] = list(positions)
            return mx.argmax(logprobs, axis=-1)

    tokens = server_generation._sample_last_token(
        logits,
        Sampler(),
        row_ids=[0, 0],
        positions=[0, 0],
    )
    mx.eval(tokens)

    assert seen == {
        "shape": (2, 3),
        "row_ids": [0, 0],
        "positions": [0, 0],
    }
    assert tokens.tolist() == [2, 0]


def test_positioned_target_sampler_is_batch_grouping_invariant():
    sampler = server_generation._PositionedTargetSampler(
        temperature=0.7, top_p=1.0, seed=42
    )
    logits = mx.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=mx.float32,
    )
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    batched = sampler.sample_target(
        logprobs,
        row_ids=[0, 0],
        positions=[5, 5],
    )
    single_0 = sampler.sample_target(
        logprobs[0:1],
        row_ids=[0],
        positions=[5],
    )
    single_1 = sampler.sample_target(
        logprobs[1:2],
        row_ids=[0],
        positions=[5],
    )
    mx.eval(batched, single_0, single_1)

    assert batched.tolist() == [single_0.item(), single_1.item()]


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


def test_speculative_server_prefill_kwargs_are_drafter_specific():
    drafter = SimpleNamespace(config=SimpleNamespace(target_layer_ids=[1, 2, 3]))

    assert speculative_utils.speculative_prefill_kwargs("mtp", drafter) == {
        "return_hidden": True,
        "return_shared_kv": True,
    }
    assert speculative_utils.speculative_prefill_kwargs("dflash", drafter) == {
        "capture_layer_ids": [1, 2, 3],
    }


def test_speculative_server_hidden_state_picks_last_layer_for_mtp():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    assert speculative_utils.speculative_hidden_state("mtp", out) is h[-1]


def test_speculative_server_hidden_state_concatenates_for_dflash():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    result = speculative_utils.speculative_hidden_state("dflash", out)
    assert result.shape == (1, 1, 8)


def test_speculative_prompt_cache_uses_unbatched_cache_for_single_mtp(monkeypatch):
    lm = object()
    unbatched_cache = object()
    batched_cache = object()

    monkeypatch.setattr(
        speculative_utils.cache, "make_prompt_cache", lambda target: unbatched_cache
    )

    result = speculative_utils.make_speculative_prompt_cache(
        lm,
        draft_kind="mtp",
        batch_size=1,
        left_padding=[0],
        make_cache=lambda *args, **kwargs: batched_cache,
    )

    assert result is unbatched_cache


def test_speculative_prompt_cache_uses_batched_cache_for_batch_or_dflash(monkeypatch):
    lm = object()
    batched_cache = object()

    monkeypatch.setattr(
        speculative_utils.cache, "make_prompt_cache", lambda target: pytest.fail()
    )

    assert (
        speculative_utils.make_speculative_prompt_cache(
            lm,
            draft_kind="mtp",
            batch_size=2,
            left_padding=[0, 1],
            make_cache=lambda *args, **kwargs: batched_cache,
        )
        is batched_cache
    )
    assert (
        speculative_utils.make_speculative_prompt_cache(
            lm,
            draft_kind="dflash",
            batch_size=1,
            left_padding=[0],
            make_cache=lambda *args, **kwargs: batched_cache,
        )
        is batched_cache
    )


def test_speculative_server_reads_draft_block_size_env(monkeypatch):
    monkeypatch.delenv("MLX_VLM_DRAFT_BLOCK_SIZE", raising=False)
    assert server._get_draft_block_size_from_env() is None

    monkeypatch.setenv("MLX_VLM_DRAFT_BLOCK_SIZE", "3")
    assert server._get_draft_block_size_from_env() == 3


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

    assert server.runtime.model_cache["cache_key"] == (
        "demo-model",
        "adapter-a",
        "text_generation",
    )
    assert server.runtime.model_cache["adapter_path"] == "adapter-a"


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
    gen.kv_bits = None
    gen.kv_group_size = server.DEFAULT_KV_GROUP_SIZE
    gen.kv_quant_scheme = server.DEFAULT_KV_QUANT_SCHEME
    gen.quantized_kv_start = server.DEFAULT_QUANTIZED_KV_START
    gen.top_logprobs_k = 0
    gen.apc_manager = None
    gen.tokenizer = None
    gen.requests = Queue()
    gen._stop = False
    gen._ready = Event()
    gen._load_error = None
    gen._cancelled = set()
    gen._cancel_lock = Lock()
    return gen


def test_server_demotes_incompatible_mtp_drafter_to_ar(monkeypatch):
    target_config = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=5376,
        eos_token_id=[],
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=target_config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    drafter = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma4_assistant",
            backbone_hidden_size=1536,
        )
    )
    gen = _unstarted_response_generator()

    monkeypatch.setenv("MLX_VLM_DRAFT_MODEL", "assistant")
    monkeypatch.setenv("MLX_VLM_DRAFT_KIND", "mtp")
    monkeypatch.setattr(
        server_generation,
        "load_model_resources",
        lambda *_args, **_kwargs: (model, processor, target_config),
    )
    monkeypatch.setattr(
        "mlx_vlm.speculative.drafters.load_drafter",
        lambda *_args, **_kwargs: (drafter, "mtp"),
    )

    gen._initialize_model()

    assert gen.model is model
    assert gen.processor is processor
    assert gen.draft_model is None
    assert gen.draft_kind is None


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
                    uid=1,
                    token=7,
                    token_logprob=0.0,
                    finish_reason="length",
                )
            ]

    target_config = SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=5376,
        eos_token_id=[],
    )
    model = SimpleNamespace(language_model=SimpleNamespace(config=target_config))
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    drafter = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma4_assistant",
            backbone_hidden_size=1536,
        )
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
    gen._gpu_embed = lambda raw_inputs, images=None: (
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


def test_speculative_thread_exception_reaches_client_queue(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = "dflash"
    gen.stop_tokens = set()

    rqueue = Queue()
    pending = [
        server_generation.QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        )
    ]
    calls = {"count": 0}

    def collect_pending_requests(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return pending, False
        return [], True

    error = RuntimeError("speculative prefill failed")
    gen._collect_pending_requests = collect_pending_requests
    gen._gpu_embed = MagicMock(side_effect=error)
    monkeypatch.setattr(
        "mlx_vlm.speculative.utils.speculative_prefill_kwargs",
        lambda *_args, **_kwargs: {},
    )

    gen._run_speculative()

    assert rqueue.get(timeout=1) is error
    assert rqueue.get(timeout=1) is None


def test_speculative_thread_exception_skips_broken_queues(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = "dflash"
    gen.stop_tokens = set()

    class BrokenQueue:
        def put(self, item):
            raise RuntimeError("client went away")

    good_queue = Queue()
    pending = [
        server_generation.QueuedGenerationRequest(
            rqueue=BrokenQueue(),
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        ),
        server_generation.QueuedGenerationRequest(
            rqueue=good_queue,
            raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
            prompt_tokens=1,
            args=server.GenerationArguments(max_tokens=2),
        ),
    ]
    calls = {"count": 0}

    def collect_pending_requests(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return pending, False
        return [], True

    error = RuntimeError("speculative prefill failed")
    gen._collect_pending_requests = collect_pending_requests
    gen._gpu_embed = MagicMock(side_effect=error)

    gen._run_speculative()

    assert good_queue.get(timeout=1) is error
    assert good_queue.get(timeout=1) is None


def test_speculative_thread_exception_clears_runtime_cache(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace(language_model=SimpleNamespace())
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = "dflash"
    gen.stop_tokens = set()
    rqueue = Queue()

    calls = {"clear_cache": 0, "collect": 0}
    collect_calls = {"count": 0}

    def collect_pending_requests(**_kwargs):
        collect_calls["count"] += 1
        if collect_calls["count"] > 1:
            return [], True
        return [
            server_generation.QueuedGenerationRequest(
                rqueue=rqueue,
                raw_inputs={"input_ids": mx.array([[1]], dtype=mx.int32)},
                prompt_tokens=1,
                args=server.GenerationArguments(max_tokens=2),
            )
        ], False

    gen._collect_pending_requests = collect_pending_requests
    gen._gpu_embed = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        server_generation.mx,
        "clear_cache",
        lambda: calls.__setitem__("clear_cache", calls["clear_cache"] + 1),
    )
    monkeypatch.setattr(
        server_generation.gc,
        "collect",
        lambda: calls.__setitem__("collect", calls["collect"] + 1),
    )

    gen._run_speculative()

    assert calls == {"clear_cache": 1, "collect": 1}


def test_models_endpoint_lists_single_file_safetensors_models(client, monkeypatch):
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


def test_models_endpoint_deduplicates_loaded_model_from_hf_cache(client, monkeypatch):
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
                    "local/sharded-model",
                    [
                        "config.json",
                        "model.safetensors.index.json",
                        "tokenizer_config.json",
                    ],
                ),
            ]
        ),
    )
    monkeypatch.setitem(server.runtime.model_cache, "model_path", "local/sharded-model")

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]].count(
        "local/sharded-model"
    ) == 1


def test_response_generator_diffusion_forwards_generation_options(monkeypatch):
    gen = _unstarted_response_generator()
    gen.model = SimpleNamespace()
    gen.processor = SimpleNamespace()
    gen.config = SimpleNamespace(eos_token_id=3)
    gen.tokenizer = SimpleNamespace(all_special_ids=[0])
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
    monkeypatch.setattr(server_generation, "get_prefill_step_size", lambda: 2048)
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
    )

    chunk = rqueue.get(timeout=1)
    assert chunk.text == "ok"
    assert chunk.finish_reason == "length"
    assert chunk.generation_tps == 5.0
    assert captured["input_ids"].tolist() == [[11, 12]]
    assert captured["pixel_values"] == "pixels"
    assert captured["attention_mask"] == "mask"
    assert captured["skip_special_token_ids"] == {0}
    assert captured["skip_special_tokens"] is True
    assert captured["kwargs"] == {
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "mm_token_type_ids": "types",
        "prefill_step_size": 2048,
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
def test_management_endpoints_allow_requests_without_configured_api_key(
    client, monkeypatch, method, path
):
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)

    response = getattr(client, method)(path)

    assert response.status_code == 200


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
        path,
        headers={"Authorization": "Bearer wrong-token"},
    )
    valid = getattr(client, method)(
        path,
        headers={"Authorization": "Bearer secret-token"},
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


def test_images_generations_returns_b64_json(client, monkeypatch):
    calls = []
    cache_calls = []

    def fake_get_cached_model(model, **kwargs):
        cache_calls.append((model, kwargs))
        return SimpleNamespace(), None, SimpleNamespace(model_type="bonsai")

    monkeypatch.setattr(
        server,
        "get_cached_model",
        fake_get_cached_model,
    )

    def fake_generate_image(model, request, **kwargs):
        calls.append(request)
        return _fake_image_result(seed=request.seed)

    monkeypatch.setattr(server_openai, "generate_image", fake_generate_image)

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "bonsai-ternary",
            "prompt": "bonsai",
            "n": 2,
            "seed": 10,
            "size": "256x256",
            "steps": 1,
            "response_format": "b64_json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["size"] == "256x256"
    assert [item["seed"] for item in payload["data"]] == [10, 11]
    assert all(item["b64_json"] for item in payload["data"])
    assert [call.seed for call in calls] == [10, 11]
    assert cache_calls == [("bonsai-ternary", {"model_kind": "image_generation"})]


def test_image_generation_lock_uses_image_cache_kind(monkeypatch):
    text_lock = object()
    image_lock = object()
    registry = server.ModelCacheRegistry()
    registry.set("text_generation", {"generation_lock": text_lock})
    registry.set("image_generation", {"generation_lock": image_lock})
    monkeypatch.setattr(server.runtime, "model_cache", registry)

    assert server_openai._runtime_cache_get("generation_lock") is text_lock
    assert (
        server_openai._runtime_cache_get("generation_lock", kind="image_generation")
        is image_lock
    )


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


class _RecordingSpeculativeLM:
    def __init__(self, draft_kind):
        self.calls = []
        self.draft_kind = draft_kind
        self._position_ids = "stale"
        self._rope_deltas = "stale"

    def __call__(self, inputs, cache=None, **kwargs):
        self.calls.append({"inputs": inputs, "cache": cache, **kwargs})
        batch_size, seq_len = inputs.shape
        logits = mx.broadcast_to(
            mx.array([[[0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=mx.float32),
            (batch_size, seq_len, 5),
        )
        hidden = mx.ones((batch_size, seq_len, 2), dtype=mx.float32)
        if self.draft_kind == "mtp":
            return SimpleNamespace(
                logits=logits,
                hidden_states=[hidden],
                shared_kv_states={"full_attention": ("k", "v")},
            )
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden, hidden],
            shared_kv_states=None,
        )


def _run_speculative_prefill_once(monkeypatch, *, draft_kind, request_specs):
    lm = _RecordingSpeculativeLM(draft_kind)
    gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
    gen.model = SimpleNamespace(language_model=lm)
    gen.processor = SimpleNamespace()
    gen.draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2]), accept_lens=[]
    )
    gen.draft_kind = draft_kind
    gen.stop_tokens = {99}
    gen.requests = Queue()
    gen._stop = False
    gen._make_sampler = lambda args: None
    gen.tokenizer = SimpleNamespace(
        decode=lambda tokens: "".join(str(tok) for tok in tokens)
    )

    specs_iter = iter(request_specs)

    def fake_gpu_embed(raw_inputs, images=None):
        del raw_inputs, images
        spec = next(specs_iter)
        return spec["input_ids"], spec["gen_kwargs"]

    gen._gpu_embed = fake_gpu_embed

    monkeypatch.setattr(server_generation, "_make_cache", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        server_generation, "_get_draft_block_size_from_env", lambda: None
    )
    monkeypatch.setattr(
        server_generation, "get_speculative_batch_coalesce_s", lambda: 0.0
    )

    class _FakeDetokenizer:
        def __init__(self):
            self.last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = str(token)

        def finalize(self):
            pass

    monkeypatch.setattr(
        server_generation,
        "make_streaming_detokenizer",
        lambda processor: _FakeDetokenizer(),
    )

    def fake_rounds(*args, **kwargs):
        del args
        gen.round_kwargs = kwargs
        gen._stop = True
        yield ([4] * int(kwargs["first_bonus"].shape[0]), None)

    monkeypatch.setattr(server_generation, "run_speculative_server_rounds", fake_rounds)

    args = server.GenerationArguments(max_tokens=2, temperature=0)
    for spec in request_specs:
        gen.requests.put(
            server_generation.QueuedGenerationRequest(
                rqueue=Queue(),
                raw_inputs={"input_ids": spec["input_ids"]},
                prompt_tokens=int(spec["input_ids"].shape[1]),
                args=args,
            )
        )

    gen._run_speculative()
    call = lm.calls[0]
    call["round_kwargs"] = gen.round_kwargs
    return call


def test_speculative_server_threads_greedy_flag_to_mtp_loop(monkeypatch):
    call = _run_speculative_prefill_once(
        monkeypatch,
        draft_kind="mtp",
        request_specs=[
            {
                "input_ids": mx.array([[11, 12, 13]], dtype=mx.int32),
                "gen_kwargs": {"inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32)},
            },
            {
                "input_ids": mx.array([[21, 22, 23]], dtype=mx.int32),
                "gen_kwargs": {"inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32)},
            },
        ],
    )

    assert call["round_kwargs"]["greedy_sampling"] is True


def test_speculative_server_prefill_threads_gemma4_per_layer_inputs(monkeypatch):
    call = _run_speculative_prefill_once(
        monkeypatch,
        draft_kind="mtp",
        request_specs=[
            {
                "input_ids": mx.array([[11, 12, 13]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32),
                    "per_layer_inputs": mx.array(
                        [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]], dtype=mx.float32
                    ),
                },
            },
            {
                "input_ids": mx.array([[21, 22]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.full((1, 2, 4), 7.0, dtype=mx.float32),
                    "per_layer_inputs": mx.array(
                        [[[4.0, 4.5], [5.0, 5.5]]], dtype=mx.float32
                    ),
                },
            },
        ],
    )

    assert call["return_hidden"] is True
    assert call["return_shared_kv"] is True
    assert call["per_layer_inputs"].shape == (2, 3, 2)
    assert call["per_layer_inputs"].tolist()[1][0] == [0.0, 0.0]
    assert call["inputs_embeds"].shape == (2, 3, 4)


def test_speculative_server_prefill_threads_qwen_dflash_prompt_kwargs(monkeypatch):
    call = _run_speculative_prefill_once(
        monkeypatch,
        draft_kind="dflash",
        request_specs=[
            {
                "input_ids": mx.array([[31, 32, 33]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.ones((1, 3, 4), dtype=mx.float32),
                    "image_grid_thw": mx.array([[1, 2, 3]], dtype=mx.int32),
                    "_apc_image_hash": 123,
                    "_apc_tenant": "tenant-a",
                },
            },
            {
                "input_ids": mx.array([[41, 42]], dtype=mx.int32),
                "gen_kwargs": {
                    "inputs_embeds": mx.full((1, 2, 4), 9.0, dtype=mx.float32),
                    "image_grid_thw": mx.array([[4, 5, 6]], dtype=mx.int32),
                },
            },
        ],
    )

    assert call["capture_layer_ids"] == [1, 2]
    assert call["image_grid_thw"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert call["inputs_embeds"].shape == (2, 3, 4)
    assert call["inputs_embeds"].tolist()[1][0] == [0.0, 0.0, 0.0, 0.0]
    assert "_apc_image_hash" not in call
    assert "_apc_tenant" not in call


def test_read_tenant_id_prefers_apc_header(monkeypatch):
    monkeypatch.setenv("APC_DEFAULT_TENANT", "env-tenant")
    request = SimpleNamespace(
        headers={"x-apc-tenant": "header-tenant", "x-tenant-id": "legacy-tenant"}
    )

    assert server._read_tenant_id(request) == "header-tenant"


def test_read_tenant_id_uses_legacy_header_before_env(monkeypatch):
    monkeypatch.setenv("APC_DEFAULT_TENANT", "env-tenant")
    request = SimpleNamespace(headers={"x-tenant-id": "legacy-tenant"})

    assert server._read_tenant_id(request) == "legacy-tenant"


def test_read_tenant_id_uses_env_default(monkeypatch):
    monkeypatch.setenv("APC_DEFAULT_TENANT", "env-tenant")

    assert server._read_tenant_id(SimpleNamespace(headers={})) == "env-tenant"
    assert server._read_tenant_id(None) == "env-tenant"


def test_read_tenant_id_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("APC_DEFAULT_TENANT", raising=False)

    assert server._read_tenant_id(SimpleNamespace(headers={})) == "default"


def test_responses_endpoint_forwards_new_sampling_args(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
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


@pytest.mark.parametrize(
    ("include_adapter", "adapter_path", "expected_adapter"),
    [
        (False, None, server._INHERIT_ADAPTER),
        (True, "adapter-a", "adapter-a"),
        (True, None, None),
    ],
)
def test_responses_endpoint_forwards_adapter_path_or_inherits(
    client, monkeypatch, include_adapter, adapter_path, expected_adapter
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=1,
        generation_tokens=1,
        total_tokens=2,
    )
    get_cached_model = MagicMock(return_value=(model, processor, config))
    payload = {"model": "demo", "input": "Hello"}
    if include_adapter:
        payload["adapter_path"] = adapter_path

    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server, "get_cached_model", get_cached_model)
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))
    monkeypatch.setattr(server, "generate", MagicMock(return_value=result))

    response = client.post("/responses", json=payload)

    assert response.status_code == 200
    assert get_cached_model.call_args.args == ("demo", expected_adapter)


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


def test_responses_endpoint_returns_function_call_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
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
        patch.object(
            server, "apply_chat_template", return_value="prompt"
        ) as mock_template,
        patch.object(server, "generate", return_value=result),
        patch.object(server, "_infer_tool_parser_from_processor", return_value="demo"),
        patch.object(server, "load_tool_module", return_value=tool_module),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "demo",
                "input": "weather?",
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
    payload = response.json()
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["name"] == "get_weather"
    assert payload["output"][0]["arguments"] == '{"location": "SF"}'
    assert (
        mock_template.call_args.kwargs["tools"][0]["function"]["name"] == "get_weather"
    )


def test_responses_endpoint_returns_reasoning_items(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="<think>Check briefly.</think>\n\nDone.",
        prompt_tokens=8,
        generation_tokens=4,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/responses", json={"model": "demo", "input": "hello"}
        )

    assert response.status_code == 200
    payload = response.json()
    assert [item["type"] for item in payload["output"]] == ["reasoning", "message"]
    assert payload["output"][0]["summary"][0]["text"] == "Check briefly."
    assert payload["output"][1]["content"][0]["text"] == "Done."
    assert payload["output_text"] == "Done."


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
            json={
                "model": "demo",
                "input": "run pwd",
                "tools": [{"type": "shell"}],
            },
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


def test_responses_streaming_emits_reasoning_events(client):
    server.response_store.clear()
    server.response_store_order.clear()
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    chunks = [
        GenerationResult(text="<think>Check", prompt_tokens=8, generation_tokens=1),
        GenerationResult(
            text=" briefly.</think>\n\nDone.",
            prompt_tokens=8,
            generation_tokens=4,
            finish_reason="stop",
        ),
    ]

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "stream_generate", return_value=iter(chunks)),
        patch.object(server.runtime, "response_generator", None),
    ):
        response = client.post(
            "/v1/responses",
            json={"model": "demo", "input": "hello", "stream": True},
        )

    assert response.status_code == 200
    events = _sse_events(response.text)
    reasoning_events = [
        data
        for event_type, data in events
        if event_type == "response.reasoning_text.delta"
    ]
    text_delta_events = [
        data
        for event_type, data in events
        if event_type == "response.output_text.delta"
    ]
    done_event = next(
        data for event_type, data in events if event_type == "response.output_text.done"
    )
    completed = next(
        data["response"]
        for event_type, data in events
        if event_type == "response.completed"
    )

    assert "".join(event["delta"] for event in reasoning_events) == "Check briefly."
    assert "".join(event["delta"] for event in text_delta_events) == "Done."
    assert reasoning_events[0]["timings"]["predicted_per_second"] is None
    assert reasoning_events[1]["timings"]["predicted_per_second"] > 0
    assert (
        text_delta_events[0]["timings"]["predicted_per_second"]
        == reasoning_events[1]["timings"]["predicted_per_second"]
    )
    assert done_event["timings"]["predicted_per_second"] > 0
    assert [item["type"] for item in completed["output"]] == ["reasoning", "message"]
    assert completed["output"][0]["summary"][0]["text"] == "Check briefly."
    assert completed["output_text"] == "Done."


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
            {
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 4,
                "stream": True,
            },
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
                        text="ok",
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
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
            {
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 4,
                "stream": True,
            },
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
        (
            "/v1/responses",
            {
                "model": "demo",
                "input": "Hello",
                "max_output_tokens": 4,
            },
        ),
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


def test_chat_completions_streaming_forwards_explicit_sampling_args(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    captured = {}

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            captured["prompt"] = prompt
            captured["images"] = images
            captured["audio"] = audio
            captured["args"] = args
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="done", token=1, logprobs=0.0, finish_reason="stop"
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
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 12,
                "top_k": 40,
                "min_p": 0.08,
                "repetition_penalty": 1.15,
                "logit_bias": {"12": -1.5},
            },
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert captured["args"].max_tokens == 12
    assert captured["args"].top_k == 40
    assert captured["args"].min_p == 0.08
    assert captured["args"].repetition_penalty == 1.15
    assert captured["args"].logit_bias == {12: -1.5}


def test_chat_completions_streaming_splits_gemma_thinking_channel_content(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="gemma4")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
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
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "What's 7*8?"}],
                "stream": True,
                "enable_thinking": True,
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

    assert "".join(delta.get("content") or "" for delta in deltas) == "7 * 8 = 56"
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == ""
    assert "<|channel>" not in response.text
    assert "<channel|>" not in response.text


def test_chat_completions_streaming_uses_custom_thinking_markers(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="custom")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
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
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "enable_thinking": True,
                "thinking_start_token": "<analysis>",
                "thinking_end_token": "</analysis>",
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
        "Custom reasoning."
    )
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == (
        "Custom reasoning."
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == ("Custom answer.")


def test_chat_completions_streaming_keeps_plain_output_as_content_when_thinking_enabled(
    client, monkeypatch
):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="lfm2_vl")

    class FakeResponseGenerator:
        tokenizer = SimpleNamespace(decode=lambda tokens: "")

        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=8), iter(
                [
                    server.StreamingToken(
                        text="Hello", token=1, logprobs=0.0, finish_reason=None
                    ),
                    server.StreamingToken(
                        text="!", token=2, logprobs=0.0, finish_reason="stop"
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
                "model": "liquidai/LFM2.5-VL-1.6B",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "enable_thinking": True,
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

    assert "".join(delta.get("content") or "" for delta in deltas) == "Hello!"
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == ""
    assert "".join(delta.get("reasoning") or "" for delta in deltas) == ""


def test_chat_completions_response_uses_reasoning_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="custom")
    result = GenerationResult(
        text="<analysis>Custom reasoning.</analysis>Custom answer.",
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
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "enable_thinking": True,
                "thinking_start_token": "<analysis>",
                "thinking_end_token": "</analysis>",
            },
        )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "Custom reasoning."
    assert message["reasoning"] == "Custom reasoning."
    assert message["content"] == "Custom answer."


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


def test_generation_timings_from_metrics():
    metrics = SimpleNamespace(
        cached_tokens=2,
        prompt_tps=20.0,
        generation_tps=8.0,
        token_times=[],
        peak_memory=0.5,
    )
    timings = server.GenerationTimings.from_metrics(metrics, 10, 4)

    assert (timings.prompt_n, timings.cache_n, timings.predicted_n) == (8, 2, 4)
    assert timings.prompt_ms == pytest.approx(500.0)
    assert timings.prompt_per_token_ms == pytest.approx(62.5)
    assert timings.prompt_per_second == pytest.approx(16.0)
    assert timings.predicted_ms == pytest.approx(500.0)
    assert timings.predicted_per_token_ms == pytest.approx(125.0)
    assert timings.predicted_per_second == pytest.approx(8.0)
    assert timings.peak_memory == pytest.approx(0.5)

    metrics = SimpleNamespace(
        cached_tokens=9,
        prompt_tps=None,
        generation_tps=None,
        token_times=[],
        peak_memory=0.0,
    )
    timings = server.GenerationTimings.from_metrics(metrics, 4, 1)
    assert timings.prompt_n == 0
    assert timings.prompt_ms == 0.0
    assert timings.prompt_per_token_ms == 0.0
    assert timings.predicted_ms == 0.0
    assert timings.predicted_per_token_ms == 0.0


def test_generation_metrics_reports_chunk_and_aggregate_rates():
    metrics = server_generation.GenerationMetrics()

    first_rate = metrics.record_chunk(
        SimpleNamespace(generation_tokens=1, emitted_at=10.0)
    )
    second_rate = metrics.record_chunk(
        SimpleNamespace(generation_tokens=4, emitted_at=10.25)
    )

    assert first_rate is None
    assert second_rate == pytest.approx(12.0)
    assert metrics.rate == pytest.approx(12.0)


def test_chat_completions_returns_timings(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=10,
        generation_tokens=4,
        prompt_tps=20.0,
        generation_tps=8.0,
        peak_memory=0.1,
        cached_tokens=2,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/chat/completions",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 12,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert (body["timings"]["cache_n"], body["timings"]["prompt_n"]) == (2, 8)
    assert body["timings"]["predicted_per_second"] == 8.0


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


def test_chat_completions_streaming_tool_calls_emit_usage_chunk(client, monkeypatch):
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
                        text=(
                            '<tool_call>{"name":"get_weather",'
                            '"arguments":{"location":"SF"}}</tool_call>'
                        ),
                        token=1,
                        logprobs=0.0,
                        finish_reason="stop",
                        prompt_tps=20.0,
                        cached_tokens=2,
                    )
                ]
            )

    tool_module = SimpleNamespace(
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        parse_tool_call=lambda call, tools: json.loads(call),
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

    assert tool_chunk.get("usage") is None
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_chat_completions_endpoint_flattens_text_content_parts(client):
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
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First text block."},
                            {"type": "text", "text": "Second text block."},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert mock_template.call_args.args[2] == [
        {
            "role": "user",
            "content": "First text block. Second text block.",
        }
    ]


def test_chat_completions_endpoint_forwards_video_content(client):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="gemma4")
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
        patch.object(server, "generate", return_value=result) as mock_generate,
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
    assert mock_template.call_args.kwargs["video"] == ["clip.mp4"]
    assert mock_template.call_args.args[2] == [
        {"role": "user", "content": "Describe this video."}
    ]
    assert mock_generate.call_args.kwargs["video"] == ["clip.mp4"]


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
        {"role": "system", "content": "Use short answers.\nBe precise."},
        {"role": "user", "content": "Hello"},
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


def test_anthropic_messages_usage_reports_cached_tokens(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text="done",
        prompt_tokens=10,
        generation_tokens=4,
        cached_tokens=6,
        prompt_tps=20.0,
        generation_tps=8.0,
        peak_memory=0.1,
    )

    with (
        patch.object(
            server, "get_cached_model", return_value=(model, processor, config)
        ),
        patch.object(server, "apply_chat_template", return_value="prompt"),
        patch.object(server, "generate", return_value=result),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "input_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 6,
        "output_tokens": 4,
    }


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
            "content": [
                {"type": "text", "text": "Rendered chart."},
                {"type": "image"},
            ],
            "name": None,
        },
    ]
    assert mock_generate.call_args.kwargs["image"] == ["data:image/png;base64,aW1n"]


def test_anthropic_messages_endpoint_returns_tool_use_blocks(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "response_generator", None)
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")
    result = GenerationResult(
        text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
        prompt_tokens=7,
        generation_tokens=6,
        prompt_tps=0.0,
        generation_tps=0.0,
        peak_memory=0.0,
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
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
                "max_tokens": 8,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["type"] == "tool_use"
    assert payload["content"][0]["name"] == "get_weather"
    assert payload["content"][0]["input"] == {"location": "SF"}


def test_anthropic_messages_streaming_uses_anthropic_events(client, monkeypatch):
    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace(model_type="qwen2_vl")

    class FakeResponseGenerator:
        def validate_context_budget(self, prompt, images=None, audio=None, args=None):
            return None

        def generate(self, prompt, images=None, audio=None, args=None):
            return server.GenerationContext(uid=1, prompt_tokens=3), iter(
                [
                    server.StreamingToken(
                        text="Hel",
                        token=1,
                        logprobs=0.0,
                        finish_reason=None,
                        cached_tokens=2,
                    ),
                    server.StreamingToken(
                        text="lo",
                        token=2,
                        logprobs=0.0,
                        finish_reason="stop",
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
            "/v1/messages",
            json={
                "model": "demo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        )

    assert response.status_code == 200
    body = response.text
    assert "event: message_start" in body
    assert "event: content_block_start" in body
    assert "event: content_block_delta" in body
    assert '"text": "Hel"' in body
    assert "event: message_delta" in body
    assert '"stop_reason": "end_turn"' in body
    assert '"cache_read_input_tokens": 2' in body
    assert '"input_tokens": 1' in body
    assert "event: message_stop" in body


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
                        text='<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>',
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
    assert '"stop_reason": "tool_use"' in body


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


def test_metrics_endpoint_reports_empty_state(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "model_cache", {})

    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["latest"] is None
    assert payload["recent"] == []
    assert payload["summary"]["requests_started"] == 0
    assert payload["summary"]["requests_completed"] == 0
    assert payload["summary"]["requests_failed"] == 0
    assert payload["server"]["loaded_model"] is None
    assert payload["server"]["apc"] == {"enabled": False}


def test_metrics_store_logs_request_lifecycle(caplog):
    caplog.set_level(logging.INFO, logger="mlx_vlm.server")
    metrics = server.ServerMetricsStore()
    metrics.begin_request(endpoint="/chat/completions", model="demo", stream=True)
    metrics.record_success(
        {
            "endpoint": "/chat/completions",
            "model": "demo",
            "stream": True,
            "backend": "continuous_batching",
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "generated_tokens": 4,
            "request_elapsed_s": 0.5,
            "decode_elapsed_s": 0.1,
            "prefill_tok_s": 100.0,
            "decode_tok_s": 40.0,
            "finish_reason": "stop",
        }
    )

    assert "Request started: endpoint=/chat/completions model=demo" in caplog.text
    assert "Request completed: endpoint=/chat/completions model=demo" in caplog.text
    assert "prefill=100.0 tok/s decode=40.0 tok/s" in caplog.text


def test_metrics_endpoint_records_chat_completion_metrics(client, monkeypatch):
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    monkeypatch.setattr(server.runtime, "response_generator", None)

    config = SimpleNamespace(
        text_config=SimpleNamespace(max_position_embeddings=4096),
    )
    processor = SimpleNamespace()
    model = SimpleNamespace()
    monkeypatch.setattr(
        server.runtime,
        "model_cache",
        {
            "model_path": "demo-model",
            "adapter_path": None,
            "config": config,
            "processor": processor,
        },
    )
    monkeypatch.setattr(
        server,
        "get_cached_model",
        MagicMock(return_value=(model, processor, config)),
    )
    monkeypatch.setattr(server, "apply_chat_template", MagicMock(return_value="prompt"))
    monkeypatch.setattr(
        server,
        "generate",
        MagicMock(
            return_value=GenerationResult(
                text="Hello there",
                prompt_tokens=12,
                generation_tokens=5,
                prompt_tps=120.0,
                generation_tps=50.0,
                peak_memory=1.25,
            )
        ),
    )

    response = client.post(
        "/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()

    latest = payload["latest"]
    assert latest["endpoint"] == "/chat/completions"
    assert latest["model"] == "demo-model"
    assert latest["stream"] is False
    assert latest["backend"] == "generate"
    assert latest["prompt_tokens"] == 12
    assert latest["completion_tokens"] == 5
    assert latest["generated_tokens"] == 5
    assert latest["prefill_tok_s"] == 120.0
    assert latest["decode_tok_s"] == 50.0
    assert latest["peak_memory_gb"] == 1.25
    assert latest["image_count"] == 0
    assert latest["audio_count"] == 0
    assert latest["apc_enabled"] is False

    assert len(payload["recent"]) == 1
    assert payload["summary"]["requests_started"] == 1
    assert payload["summary"]["requests_completed"] == 1
    assert payload["summary"]["requests_failed"] == 0
    assert payload["summary"]["prompt_tokens_total"] == 12
    assert payload["summary"]["completion_tokens_total"] == 5
    assert payload["summary"]["generated_tokens_total"] == 5
    assert payload["server"]["loaded_model"] == "demo-model"
    assert payload["server"]["loaded_context_size"] == 4096


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
        gen._cpu_preprocess = lambda prompt, images, audio: {
            "input_ids": mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        }
        gen.requests = Queue()

        monkeypatch.setenv("MAX_KV_SIZE", "8")

        with pytest.raises(server.PromptTooLongError, match="MAX_KV_SIZE is 8"):
            gen.generate("prompt", args=server.GenerationArguments(max_tokens=4))

        assert gen.requests.empty()

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
                args=server.GenerationArguments(
                    max_tokens=1,
                    thinking_budget=512,
                ),
            )
            token_iter.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(generate_one, range(4)))

        assert max_active == 1
        assert len(queued) == 4
        assert all(request.thinking_budget_criteria is not None for request in queued)

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

    def test_token_queue_timeout_defaults_to_long_prefill_window(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", raising=False)

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_accepts_namespaced_env(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "42.5")

        assert server.get_token_queue_timeout() == 42.5

    def test_token_queue_timeout_invalid_values_fall_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "bad")

        assert server.get_token_queue_timeout() == 600.0

    def test_token_queue_timeout_can_disable_timeout(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0")

        assert server.get_token_queue_timeout() is None

    def test_log_progress_interval_is_configurable(self, monkeypatch):
        monkeypatch.delenv("MLX_VLM_LOG_PROGRESS_INTERVAL", raising=False)
        assert server.get_log_progress_interval() == 10

        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "7")
        assert server.get_log_progress_interval() == 7

        monkeypatch.setenv("MLX_VLM_LOG_PROGRESS_INTERVAL", "-1")
        assert server.get_log_progress_interval() == 0

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
                1,
                info,
                token=token_number,
                text=str(token_number),
                finish_reason=None,
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

    def test_decode_logging_uses_one_rate_field(self, monkeypatch, caplog):
        times = iter([10.0, 10.25])
        monkeypatch.setattr(server_generation.time, "perf_counter", lambda: next(times))
        caplog.set_level(logging.DEBUG, logger="mlx_vlm.server")
        info = {
            "request_id": "req-1",
            "queued_at": 9.0,
            "generated_tokens": 0,
            "decode_started_at": None,
            "last_token_at": None,
        }

        for token_number in range(1, 3):
            server.ResponseGenerator._log_decode_progress(
                1,
                info,
                token=token_number,
                text=str(token_number),
                finish_reason="stop" if token_number == 2 else None,
            )

        progress = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Decode progress:")
        ]
        assert "rate=n/a" in progress[0]
        assert "rate=4.0 tok/s" in progress[1]
        assert not any("token_rate=" in message for message in progress)
        completed = next(
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("Decode completed:")
        )
        assert "rate=4.0 tok/s" in completed
        assert "token_rate=" not in completed

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
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0.01")

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
            rqueue,
            "req-1",
            cancelled.append,
            None,
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
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", str(timeout_s * 10))

        _, token_iter = gen.generate("hello")

        start = time.monotonic()
        assert next(token_iter) is token
        assert time.monotonic() - start >= delay_s * 0.5
        with pytest.raises(StopIteration):
            next(token_iter)
        assert cancelled == []

    def test_collect_pending_requests_coalesces_after_first_item(self, monkeypatch):
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.requests = Queue()
        gen._stop = False
        first = object()
        second = object()
        gen.requests.put(first)

        def fake_sleep(seconds):
            assert seconds == pytest.approx(0.005)
            gen.requests.put(second)

        monkeypatch.setattr(server.time, "sleep", fake_sleep)

        pending, should_stop = gen._collect_pending_requests(
            active=False, coalesce_s=0.005
        )

        assert pending == [first, second]
        assert should_stop is False

    def test_step_streams_spm_subword_tokens_immediately(self):
        class SentencePieceTokenizer:
            vocab = {
                "▁hello": 0,
                "world": 1,
                "!": 2,
            }

            def decode(self, tokens):
                parts = []
                for token in tokens:
                    parts.append(
                        {
                            0: " hello",
                            1: "world",
                            2: "!",
                        }[token]
                    )
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
                    tokenizer,
                    server.make_streaming_detokenizer(processor),
                ),
            }
        }

        for token in [0, 1, 2]:
            gen._step(
                SingleResponseBatch(
                    SimpleNamespace(
                        uid=1,
                        token=token,
                        token_logprob=0.0,
                        finish_reason=None,
                    )
                ),
                active,
            )
        gen._step(
            SingleResponseBatch(
                SimpleNamespace(
                    uid=1,
                    token=99,
                    token_logprob=0.0,
                    finish_reason="stop",
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
            vocab = {
                "<0xF0>": 0,
                "<0x9F>": 1,
            }

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
            tokenizer,
            server.make_streaming_detokenizer(processor),
        )

        assert streamer.advance(0, None) == ""
        assert streamer.advance(1, None) == ""
        assert streamer.finalize() == "\ufffd"

    def test_step_streams_multiple_utf8_emojis_with_text_between_them(self):
        class MixedEmojiTokenizer:
            vocab = {
                "hi": 0,
                "<0xF0>": 1,
                "<0x9F>": 2,
                "<0x98>": 3,
                "<0x80>": 4,
                "▁mid": 5,
                "<0x82>": 6,
                "▁wow": 7,
                "<0x8E>": 8,
                "▁done": 9,
            }

            def decode(self, tokens):
                text = ""
                byte_buffer = bytearray()
                byte_values = {
                    1: 0xF0,
                    2: 0x9F,
                    3: 0x98,
                    4: 0x80,
                    6: 0x82,
                    8: 0x8E,
                }
                regular = {0: "hi", 5: "▁mid", 7: "▁wow", 9: "▁done"}

                def flush_bytes():
                    nonlocal text, byte_buffer
                    if byte_buffer:
                        text += byte_buffer.decode("utf-8", errors="replace")
                        byte_buffer = bytearray()

                for token in tokens:
                    if token in byte_values:
                        byte_buffer.append(byte_values[token])
                    else:
                        flush_bytes()
                        text += regular[token].replace("▁", " ")
                flush_bytes()
                return text

        class SingleResponseBatch:
            def __init__(self, response):
                self.response = response

            def next(self, **kwargs):
                return [], [self.response]

        tokenizer = MixedEmojiTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "streamer": _ServerTokenStreamer(
                    tokenizer,
                    server.make_streaming_detokenizer(processor),
                ),
            }
        }

        for token in [0, 1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 1, 2, 3, 8, 9, 1, 2, 3, 4]:
            gen._step(
                SingleResponseBatch(
                    SimpleNamespace(
                        uid=1,
                        token=token,
                        token_logprob=0.0,
                        finish_reason=None,
                    )
                ),
                active,
            )
        gen._step(
            SingleResponseBatch(
                SimpleNamespace(
                    uid=1,
                    token=99,
                    token_logprob=0.0,
                    finish_reason="stop",
                )
            ),
            active,
        )

        segments = []
        while not rqueue.empty():
            item = rqueue.get()
            if item is not None:
                segments.append(item.text)

        streamed_text = "".join(segments)
        assert segments == [
            "hi",
            "",
            "",
            "",
            "😀",
            " mid",
            "",
            "",
            "",
            "😂",
            " wow",
            "",
            "",
            "",
            "😎",
            " done",
            "",
            "",
            "",
            "😀",
            "",
        ]
        assert streamed_text == "hi😀 mid😂 wow😎 done😀"
        assert "\ufffd" not in streamed_text

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
            server_generation,
            "make_streaming_detokenizer",
            lambda _: FakeDetokenizer(),
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
        gen._gpu_embed = lambda raw_inputs, images=None: (
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
            assert items == [
                (str(uid * 10), None),
                (str(uid * 10 + 1), "length"),
            ]

    def test_run_routes_mtp_through_batch_generator(self, monkeypatch):
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
            server_generation,
            "_get_draft_block_size_from_env",
            lambda: 6,
        )
        monkeypatch.setattr(
            server_generation,
            "make_streaming_detokenizer",
            lambda _: FakeDetokenizer(),
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
            gen.draft_model = draft_model
            gen.draft_kind = "mtp"
            gen.tokenizer = SimpleNamespace()

        gen._initialize_model = fake_initialize_model
        gen._run_speculative = lambda: pytest.fail("MTP should use BatchGenerator")
        gen._gpu_embed = lambda raw_inputs, images=None: (
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
        assert kwargs["draft_kind"] == "mtp"
        assert kwargs["draft_block_size"] == 6
        assert kwargs["greedy_sampling"] is True
        assert kwargs["compute_logprobs"] is False
        assert batch_state["instance"].next_active_sizes == [2]

    def test_run_coalesces_idle_mtp_batch_generator(self, monkeypatch):
        monkeypatch.setenv("MLX_VLM_SPEC_BATCH_COALESCE_MS", "37")
        calls = []
        draft_model = object()

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        gen.draft_model = None
        gen.draft_kind = None
        gen._stop = False
        gen._ready = Event()
        gen._load_error = None

        def fake_initialize_model():
            gen.model = SimpleNamespace(language_model=object())
            gen.processor = SimpleNamespace()
            gen.config = SimpleNamespace()
            gen.stop_tokens = set()
            gen.draft_model = draft_model
            gen.draft_kind = "mtp"
            gen.tokenizer = SimpleNamespace()

        def fake_collect_pending_requests(*, active, idle_timeout=0.1, coalesce_s=0.0):
            del idle_timeout
            calls.append((active, coalesce_s))
            return [], True

        gen._initialize_model = fake_initialize_model
        gen._run_speculative = lambda: pytest.fail("MTP should use BatchGenerator")
        gen._collect_pending_requests = fake_collect_pending_requests

        gen._run()

        assert calls == [(False, 0.037)]

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
                        uid=uid,
                        token=uid,
                        token_logprob=0.0,
                        finish_reason="length",
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
            server_generation,
            "make_streaming_detokenizer",
            lambda _: FakeDetokenizer(),
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
        gen._gpu_embed = lambda raw_inputs, images=None: (
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
                            uid=1,
                            token=0,
                            token_logprob=0.0,
                            finish_reason="stop",
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
                    tokenizer,
                    server.make_streaming_detokenizer(processor),
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

    def test_generate_arguments_to_template_kwargs(self):
        args = server.GenerationArguments(
            enable_thinking=False,
            thinking_budget=50,
            thinking_end_token="</think>",
        )
        kw = args.to_template_kwargs()
        assert kw["enable_thinking"] is False
        assert kw["thinking_budget"] == 50
        assert kw["thinking_end_token"] == "</think>"

    def test_generate_arguments_omits_none_optionals(self):
        args = server.GenerationArguments()
        kw = args.to_generate_kwargs()
        assert "repetition_penalty" not in kw
        assert (
            kw["repetition_context_size"]
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert "presence_penalty" not in kw
        assert (
            kw["presence_context_size"]
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert "frequency_penalty" not in kw
        assert (
            kw["frequency_context_size"]
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert "logit_bias" not in kw
        assert "thinking_budget" not in kw

    def test_server_generation_builds_repetition_logits_processors(self, monkeypatch):
        custom_processor = lambda tokens, logits: logits
        calls = []

        def fake_make_logits_processors(
            logit_bias,
            repetition_penalty,
            repetition_context_size,
            presence_penalty,
            presence_context_size,
            frequency_penalty,
            frequency_context_size,
        ):
            calls.append(
                (
                    logit_bias,
                    repetition_penalty,
                    repetition_context_size,
                    presence_penalty,
                    presence_context_size,
                    frequency_penalty,
                    frequency_context_size,
                )
            )
            return ["repetition-processor"]

        monkeypatch.setattr(
            server_generation, "make_logits_processors", fake_make_logits_processors
        )

        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        args = server.GenerationArguments(
            repetition_penalty=1.2,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={5: -0.5},
            logits_processors=[custom_processor],
        )

        processors = gen._make_logits_processors(args)

        assert calls == [({5: -0.5}, 1.2, 512, 0.2, 256, 0.3, 128)]
        assert processors == ["repetition-processor", custom_processor]

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
            args,
            mx.array([[1, 10, 3]], dtype=mx.int32),
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
            server_generation,
            "make_logits_processors",
            lambda *_args: [],
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
            args,
            mx.array([[1, 10, 3, 20]], dtype=mx.int32),
        )

        assert processors == [structured_processor]

    def test_build_gen_args_from_openai_request(self):
        req = SimpleNamespace(
            max_output_tokens=128,
            temperature=0.5,
            top_p=0.9,
            top_k=32,
            min_p=0.1,
            repetition_penalty=1.2,
            repetition_context_size=512,
            presence_penalty=0.2,
            presence_context_size=256,
            frequency_penalty=0.3,
            frequency_context_size=128,
            logit_bias={"5": -1.0},
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
            thinking_end_token=None,
        )
        args = server._build_gen_args(req, tenant_id="tenant-a")
        assert args.max_tokens == 128
        assert args.top_k == 32
        assert args.repetition_context_size == 512
        assert args.presence_penalty == 0.2
        assert args.presence_context_size == 256
        assert args.frequency_penalty == 0.3
        assert args.frequency_context_size == 128
        assert args.logit_bias == {5: -1.0}  # string keys converted to int
        assert args.to_generate_kwargs()["apc_tenant"] == "tenant-a"

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

    def test_build_gen_args_preserves_diffusion_options(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
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

        args = server._build_gen_args(req)

        expected = {
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
        }
        assert args.diffusion_kwargs() == expected
        for key, value in expected.items():
            assert args.to_generate_kwargs()[key] == value

    def test_build_gen_args_uses_model_generation_config_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            server.runtime.model_cache,
            "config",
            SimpleNamespace(temperature=1.0, top_p=0.95, top_k=64),
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        args = server._build_gen_args(req)

        assert args.temperature == 1.0
        assert args.top_p == 0.95
        assert args.top_k == 64

    def test_build_gen_args_request_sampling_overrides_model_generation_config(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            server.runtime.model_cache,
            "config",
            SimpleNamespace(temperature=1.0, top_p=0.95, top_k=64),
        )
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )

        args = server._build_gen_args(req)

        assert args.temperature == 0.0
        assert args.top_p == 1.0
        assert args.top_k == 0

    def test_build_gen_args_defaults_penalty_context_sizes_when_omitted(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
        )

        args = server._build_gen_args(req)

        assert (
            args.repetition_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.presence_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.frequency_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )

    def test_build_gen_args_defaults_penalty_context_sizes_when_null(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            repetition_penalty=1.1,
            repetition_context_size=None,
            presence_penalty=0.2,
            presence_context_size=None,
            frequency_penalty=0.3,
            frequency_context_size=None,
        )

        args = server._build_gen_args(req)

        assert (
            args.repetition_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.presence_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )
        assert (
            args.frequency_context_size
            == server_generation.DEFAULT_REPETITION_CONTEXT_SIZE
        )

    def test_build_gen_args_preserves_explicit_penalty_context_sizes(self):
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            repetition_context_size=64,
            presence_context_size=32,
            frequency_context_size=16,
        )

        args = server._build_gen_args(req)

        assert args.repetition_context_size == 64
        assert args.presence_context_size == 32
        assert args.frequency_context_size == 16

    def test_build_gen_args_uses_server_thinking_default_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "1")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        assert "enable_thinking" not in req.model_fields_set
        assert server._build_gen_args(req).enable_thinking is True

        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        assert server._build_gen_args(req).enable_thinking is False

    def test_build_gen_args_uses_server_thinking_token_defaults_when_omitted(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_THINKING_BUDGET", "256")
        monkeypatch.setenv("MLX_VLM_THINKING_START_TOKEN", "<analysis>")
        monkeypatch.setenv("MLX_VLM_THINKING_END_TOKEN", "</analysis>")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
        )

        assert "thinking_budget" not in req.model_fields_set
        assert "thinking_start_token" not in req.model_fields_set
        assert "thinking_end_token" not in req.model_fields_set
        args = server._build_gen_args(req)

        assert args.thinking_budget == 256
        assert args.thinking_start_token == "<analysis>"
        assert args.thinking_end_token == "</analysis>"

    def test_build_gen_args_request_thinking_overrides_server_default(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "1")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            enable_thinking=False,
        )

        assert server._build_gen_args(req).enable_thinking is False

        monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            enable_thinking=True,
        )

        assert server._build_gen_args(req).enable_thinking is True

    def test_build_gen_args_request_thinking_tokens_override_server_defaults(
        self, monkeypatch
    ):
        monkeypatch.setenv("MLX_VLM_THINKING_BUDGET", "256")
        monkeypatch.setenv("MLX_VLM_THINKING_START_TOKEN", "<analysis>")
        monkeypatch.setenv("MLX_VLM_THINKING_END_TOKEN", "</analysis>")
        req = server.ChatRequest(
            model="demo",
            messages=[server.ChatMessage(role="user", content="hi")],
            thinking_budget=32,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
        )

        args = server._build_gen_args(req)

        assert args.thinking_budget == 32
        assert args.thinking_start_token == "<think>"
        assert args.thinking_end_token == "</think>"

    def test_server_cli_sets_thinking_defaults(self, monkeypatch):
        for env_var in (
            "MLX_VLM_ENABLE_THINKING",
            "MLX_VLM_PRELOAD_MODEL",
            "MLX_VLM_PRELOAD_ADAPTER",
            "MLX_VLM_PRELOAD_IMAGE_MODEL",
            "MLX_VLM_PRELOAD_TTS_MODEL",
            "MLX_VLM_PRELOAD_STT_MODEL",
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
                "MLX_VLM_VISION_CACHE_SIZE",
                "MLX_VLM_MAX_TOKENS",
                "MLX_VLM_THINKING_BUDGET",
                "MLX_VLM_THINKING_START_TOKEN",
                "MLX_VLM_THINKING_END_TOKEN",
                "MLX_VLM_SERVER_API_KEY",
            ):
                os.environ.pop(env_var, None)

    def test_lifespan_preloads_configured_model_kinds(self, monkeypatch):
        preload_env = {
            "MLX_VLM_PRELOAD_MODEL": "language-demo",
            "MLX_VLM_PRELOAD_ADAPTER": "adapter-demo",
            "MLX_VLM_PRELOAD_IMAGE_MODEL": "image-demo",
            "MLX_VLM_PRELOAD_TTS_MODEL": "tts-demo",
            "MLX_VLM_PRELOAD_STT_MODEL": "stt-demo",
        }
        for key, value in preload_env.items():
            monkeypatch.setenv(key, value)
        calls = []

        def fake_get_cached_model(model_path, adapter_path=None, *, model_kind="auto"):
            calls.append((model_path, adapter_path, model_kind))
            return SimpleNamespace(), None, SimpleNamespace(model_type=model_kind)

        monkeypatch.setattr(
            server._app_module, "get_cached_model", fake_get_cached_model
        )
        monkeypatch.setattr(server.runtime, "audio_queue", None)

        async def run_lifespan():
            async with server._app_module.lifespan(server.app):
                pass

        asyncio.run(run_lifespan())

        assert calls == [
            ("language-demo", "adapter-demo", "text_generation"),
            ("image-demo", None, "image_generation"),
            ("tts-demo", None, "audio_tts"),
            ("stt-demo", None, "audio_stt"),
        ]
        for key in preload_env:
            assert key not in os.environ

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

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=None,
        )

        assert gen_kwargs["_apc_image_hash"] == hash_image_payload(
            pixel_values=pixel_values
        )

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
            {
                "input_ids": mx.array([[1, 2]]),
                "attention_mask": mx.array([[1, 1]]),
            },
            images=None,
        )

        assert "position_ids" not in gen_kwargs
        assert "rope_deltas" not in gen_kwargs

    def test_gpu_embed_prefers_image_ref_for_apc_hash(self):
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
        images = ["image-a.png"]

        _, gen_kwargs = server.ResponseGenerator._gpu_embed(
            response_generator,
            {
                "input_ids": mx.array([[1, 2]]),
                "pixel_values": pixel_values,
                "attention_mask": mx.array([[1, 1]]),
            },
            images=images,
        )

        assert gen_kwargs["_apc_image_hash"] == hash_image_payload(image_ref=images)
        assert gen_kwargs["_apc_image_hash"] != hash_image_payload(
            pixel_values=pixel_values
        )

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

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_extract_chat_response_format_json_object_aliases(self, format_type):
        req = SimpleNamespace(
            response_format={"type": format_type},
            text=None,
        )

        assert server._extract_response_format_schema(req) == {"type": "object"}

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_extract_responses_text_format_json_object_aliases(self, format_type):
        req = SimpleNamespace(
            response_format=None,
            text={"format": {"type": format_type}},
        )

        assert server._extract_response_format_schema(req) == {"type": "object"}

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

    @pytest.mark.parametrize("format_type", ["json_object", "object"])
    def test_build_structured_logits_processors_for_json_object_aliases(
        self, format_type
    ):
        req = SimpleNamespace(
            response_format={"type": format_type},
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

    def test_custom_thinking_markers(self):
        text = "<analysis>Custom reasoning.</analysis>Custom answer."
        reasoning, content = server._split_thinking(text, "<analysis>", "</analysis>")
        assert reasoning == "Custom reasoning."
        assert content == "Custom answer."

    def test_cohere_thinking_markers_strip_text_markers(self):
        text = (
            "<|START_THINKING|>Custom reasoning.<|END_THINKING|>"
            "<|START_TEXT|>Custom answer.<|END_TEXT|>"
        )
        reasoning, content = server._split_thinking(text)
        assert reasoning == "Custom reasoning."
        assert content == "Custom answer."


class TestThinkingStreamState:
    """Tests for streaming thinking tag parsing."""

    def test_prompt_must_end_with_open_thinking_marker_to_start_in_thinking(self):
        assert server.prompt_has_open_thinking("prompt", enable_thinking=True) is False
        assert (
            server.prompt_has_open_thinking("prompt<think>\n", enable_thinking=True)
            is True
        )
        assert (
            server.prompt_has_open_thinking(
                "User: Say <think> literally\nAssistant:", enable_thinking=True
            )
            is False
        )
        assert (
            server.prompt_has_open_thinking(
                "prompt<analysis>",
                enable_thinking=True,
                thinking_start_token="<analysis>",
                thinking_end_token="</analysis>",
            )
            is True
        )

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

    def test_think_close_can_emit_reasoning_tail_and_content(self):
        state = server.ThinkingStreamState(enable_thinking=True)

        first = state.feed("thinking")
        second = state.feed(" tail</think>\n\nAnswer")

        assert first.reasoning == "thinking"
        assert first.content is None
        assert first.thinking_closed is False
        assert second.reasoning == " tail"
        assert second.content == "Answer"
        assert second.thinking_closed is True

    def test_custom_markers_split_same_delta_content(self):
        state = server.ThinkingStreamState(
            enable_thinking=False,
            thinking_start_token="<analysis>",
            thinking_end_token="</analysis>",
        )

        first = state.feed("<ana")
        second = state.feed("lysis>Custom reasoning.</analysis>Custom answer.")

        assert first.reasoning is None
        assert first.content is None
        assert second.reasoning == "Custom reasoning."
        assert second.content == "Custom answer."
        assert second.thinking_closed is True

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
        assert msg.reasoning_content == "thought"

    def test_reasoning_content_field(self):
        msg = server.ChatMessage(
            role="assistant", content="answer", reasoning_content="thought"
        )
        assert msg.reasoning_content == "thought"
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

    def test_literal_less_than_is_not_suppressed(self):
        in_tc, content = server.suppress_tool_call_content(
            "if n <", False, "<tool_call>", "<"
        )
        assert in_tc is False
        assert content == "<"


class TestProcessToolCalls:
    """Tests for tool call parsing from model output."""

    def test_no_tool_calls(self):
        # Minimal tool module mock
        module = SimpleNamespace(tool_call_start="<tc>", tool_call_end="</tc>")
        result = server.process_tool_calls("Just text.", module, [])
        assert result["calls"] == []
        assert result["remaining_text"] == "Just text."

    def test_parser_can_return_multiple_tool_calls(self):
        module = SimpleNamespace(
            tool_call_start="<tc>",
            tool_call_end="</tc>",
            parse_tool_call=lambda call, tools: [
                {"name": "grep", "arguments": {"pattern": "foo"}},
                {"name": "read", "arguments": {"path": "file.py"}},
            ],
        )

        result = server.process_tool_calls("Before <tc>[]</tc> after", module, [])

        assert result["remaining_text"] == "Before   after"
        assert [call["function"]["name"] for call in result["calls"]] == [
            "grep",
            "read",
        ]
        assert json.loads(result["calls"][0]["function"]["arguments"]) == {
            "pattern": "foo"
        }
        assert json.loads(result["calls"][1]["function"]["arguments"]) == {
            "path": "file.py"
        }


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
