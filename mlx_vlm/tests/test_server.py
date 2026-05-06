import time
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
from mlx_vlm.apc import hash_image_payload
from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer


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


def test_speculative_server_dispatches_mtp_batch_loop():
    assert server._get_speculative_rounds_batch("mtp") is server._mtp_rounds_batch


def test_speculative_server_keeps_dflash_default_batch_loop():
    assert server._get_speculative_rounds_batch("dflash") is server._dflash_rounds_batch


def test_speculative_server_rejects_unknown_draft_kind():
    with pytest.raises(ValueError):
        server._get_speculative_rounds_batch("nope")


def test_speculative_server_prefill_kwargs_are_drafter_specific():
    drafter = SimpleNamespace(config=SimpleNamespace(target_layer_ids=[1, 2, 3]))

    assert server._speculative_prefill_kwargs("mtp", drafter) == {
        "return_hidden": True,
        "return_shared_kv": True,
    }
    assert server._speculative_prefill_kwargs("dflash", drafter) == {
        "capture_layer_ids": [1, 2, 3],
    }


def test_speculative_server_hidden_state_picks_last_layer_for_mtp():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    assert server._speculative_hidden_state("mtp", out) is h[-1]


def test_speculative_server_hidden_state_concatenates_for_dflash():
    h = [mx.zeros((1, 1, 4)), mx.ones((1, 1, 4))]
    out = SimpleNamespace(hidden_states=h)

    result = server._speculative_hidden_state("dflash", out)
    assert result.shape == (1, 1, 8)


def test_speculative_server_reads_draft_block_size_env(monkeypatch):
    monkeypatch.delenv("MLX_VLM_DRAFT_BLOCK_SIZE", raising=False)
    assert server._get_draft_block_size_from_env() is None

    monkeypatch.setenv("MLX_VLM_DRAFT_BLOCK_SIZE", "3")
    assert server._get_draft_block_size_from_env() == 3


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

    monkeypatch.setattr(server, "_make_cache", lambda *args, **kwargs: [])
    monkeypatch.setattr(server, "_get_draft_block_size_from_env", lambda: None)

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
        server, "make_streaming_detokenizer", lambda processor: _FakeDetokenizer()
    )

    def fake_rounds(*args, **kwargs):
        del args
        gen._stop = True
        yield ([4] * int(kwargs["first_bonus"].shape[0]), None)

    monkeypatch.setattr(
        server, "_get_speculative_rounds_batch", lambda kind: fake_rounds
    )

    args = server.GenerationArguments(max_tokens=2, temperature=0)
    for spec in request_specs:
        gen.requests.put(
            (
                Queue(),
                {"input_ids": spec["input_ids"]},
                int(spec["input_ids"].shape[1]),
                args,
                None,
            )
        )

    gen._run_speculative()
    return lm.calls[0]


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


def test_chat_completions_endpoint_flattens_text_content_parts(client):
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


def test_cache_endpoints_report_disabled_stats_and_reset(client, monkeypatch):
    monkeypatch.setattr(server, "apc_manager", None)

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
    monkeypatch.setattr(server, "apc_manager", manager)

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

    def test_token_iterator_reports_timeout_and_cancels_request(self, monkeypatch):
        gen = self._bare_generator()
        cancelled = []

        class Requests:
            def put(self, item):
                rqueue = item[0]
                rqueue.put(SimpleNamespace(uid="req-1"))

        gen.requests = Requests()
        gen._cancel = cancelled.append
        monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "0.01")

        _, token_iter = gen.generate("hello")

        with pytest.raises(RuntimeError, match="Timed out waiting for 0.01s"):
            next(token_iter)

        assert cancelled == ["req-1"]

    def test_token_iterator_waits_past_timeout_for_delayed_token(self, monkeypatch):
        import threading

        gen = self._bare_generator()
        cancelled = []
        token = SimpleNamespace(text="hi")
        timeout_s = 0.05
        delay_s = timeout_s * 3

        class Requests:
            def put(self, item):
                rqueue: Queue = item[0]
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

    def test_step_uses_streaming_detokenizer_for_utf8_byte_tokens(self):
        class ByteFallbackTokenizer:
            vocab = {
                "hi": 0,
                "<0xF0>": 1,
                "<0x9F>": 2,
                "<0x98>": 3,
                "<0x80>": 4,
            }

            def decode(self, tokens):
                text = ""
                byte_buffer = bytearray()
                byte_values = {1: 0xF0, 2: 0x9F, 3: 0x98, 4: 0x80}

                def flush_bytes():
                    nonlocal text, byte_buffer
                    if byte_buffer:
                        text += byte_buffer.decode("utf-8", errors="replace")
                        byte_buffer = bytearray()

                for token in tokens:
                    if token == 0:
                        flush_bytes()
                        text += "hi"
                    else:
                        byte_buffer.append(byte_values[token])
                flush_bytes()
                return text

        class SingleResponseBatch:
            def __init__(self, response):
                self.response = response

            def next(self, **kwargs):
                return [], [self.response]

        tokenizer = ByteFallbackTokenizer()
        processor = SimpleNamespace(
            detokenizer=SPMStreamingDetokenizer(tokenizer, trim_space=False)
        )
        gen = server.ResponseGenerator.__new__(server.ResponseGenerator)
        rqueue = Queue()
        active = {
            1: {
                "rqueue": rqueue,
                "detokenizer": server.make_streaming_detokenizer(processor),
            }
        }

        for token in [0, 1, 2, 3, 4]:
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

        streamed_text = ""
        while not rqueue.empty():
            item = rqueue.get()
            if item is not None:
                streamed_text += item.text

        assert streamed_text == "hi😀"
        assert "\ufffd" not in streamed_text

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
            tenant_id="tenant-a",
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
        assert kw["apc_tenant"] == "tenant-a"

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
        args = server._build_gen_args(req, tenant_id="tenant-a")
        assert args.max_tokens == 128
        assert args.top_k == 32
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
            logit_bias=None,
            enable_thinking=True,
            thinking_budget=None,
            thinking_start_token=None,
        )
        args = server._build_gen_args(req)
        assert args.max_tokens == 256
        assert args.enable_thinking is True

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
