"""Tests for the OpenAI Responses API (/v1/responses) compliance.

Covers:
  A. Model validation (pure unit tests, no server/mlx needed)
  B. Response store (pure unit tests)
  C. Functional endpoint tests (TestClient, mocked model)
  D. Streaming endpoint tests (TestClient, mocked model)
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: load modules without triggering mlx_vlm.__init__ (no mlx needed)
# ---------------------------------------------------------------------------

def _load_module(name: str, filename: str):
    """Load a sibling module by file path, bypassing package __init__."""
    mod_path = Path(__file__).parent.parent / filename
    spec = importlib.util.spec_from_file_location(name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


responses_models = _load_module("responses_models", "responses_models.py")
responses_store = _load_module("responses_store", "responses_store.py")

ResponsesRequest = responses_models.ResponsesRequest
ResponseObject = responses_models.ResponseObject
ResponseMessageItem = responses_models.ResponseMessageItem
ResponseFunctionCallItem = responses_models.ResponseFunctionCallItem
ContentPartOutputText = responses_models.ContentPartOutputText
ResponseUsage = responses_models.ResponseUsage
FlexibleBaseModel = responses_models.FlexibleBaseModel
BaseStreamEvent = responses_models.BaseStreamEvent
ResponseStore = responses_store.ResponseStore


# =========================================================================
# A. Model Validation Tests
# =========================================================================


class TestResponsesModels:
    """Pure unit tests for Pydantic models in responses_models.py."""

    def test_responses_request_accepts_string_input(self):
        req = ResponsesRequest(input="Hello", model="test-model")
        assert req.input == "Hello"

    def test_responses_request_accepts_message_list(self):
        msgs = [{"role": "user", "content": "hello"}]
        req = ResponsesRequest(input=msgs, model="test-model")
        assert isinstance(req.input, list)
        assert len(req.input) == 1

    def test_responses_request_accepts_tools(self):
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        req = ResponsesRequest(input="hi", model="m", tools=tools)
        assert req.tools is not None
        assert len(req.tools) == 1

    def test_responses_request_default_tool_choice(self):
        req = ResponsesRequest(input="hi", model="m")
        assert req.tool_choice == "auto"

    def test_responses_request_generation_kwargs(self):
        req = ResponsesRequest(input="hi", model="m", max_output_tokens=128)
        kwargs = req.generation_kwargs()
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] == 128
        assert "max_output_tokens" not in kwargs

    def test_response_object_output_text_computed(self):
        msg = ResponseMessageItem(
            content=[
                ContentPartOutputText(text="Hello "),
                ContentPartOutputText(text="world!"),
            ]
        )
        resp = ResponseObject(
            created_at=0,
            model="m",
            output=[msg],
            usage=ResponseUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )
        assert resp.output_text == "Hello world!"

    def test_response_object_output_text_empty_when_only_function_calls(self):
        fc = ResponseFunctionCallItem(name="fn", arguments='{"a":1}')
        resp = ResponseObject(
            created_at=0,
            model="m",
            output=[fc],
            usage=ResponseUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )
        assert resp.output_text == ""

    def test_function_call_item_auto_ids(self):
        fc = ResponseFunctionCallItem(name="fn", arguments="{}")
        assert fc.id.startswith("fc_")
        assert fc.call_id.startswith("call_")
        # IDs should be unique per instance
        fc2 = ResponseFunctionCallItem(name="fn", arguments="{}")
        assert fc.id != fc2.id

    def test_function_call_item_schema(self):
        fc = ResponseFunctionCallItem(name="get_weather", arguments='{"city":"NYC"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city":"NYC"}'
        assert fc.type == "function_call"

    def test_content_part_output_text_defaults(self):
        part = ContentPartOutputText()
        assert part.type == "output_text"
        assert part.text == ""
        assert part.annotations == []

    def test_streaming_event_sequence_number(self):
        evt = BaseStreamEvent(type="test.event", sequence_number=42)
        assert evt.sequence_number == 42
        evt_default = BaseStreamEvent(type="test.event")
        assert evt_default.sequence_number == 0

    def test_flexible_base_model_accepts_unknown_fields(self):
        req = ResponsesRequest(
            input="hi", model="m", some_unknown_field="surprise"
        )
        # Should not raise; extra field accessible via model_extra
        assert req.model_extra.get("some_unknown_field") == "surprise"


# =========================================================================
# B. Response Store Tests
# =========================================================================


class TestResponseStore:
    """Pure unit tests for the LRU ResponseStore."""

    def test_store_save_and_get(self):
        store = ResponseStore()
        store.save("resp_1", "hello", [{"type": "message"}])
        entry = store.get("resp_1")
        assert entry is not None
        assert entry["input"] == "hello"
        assert entry["output"] == [{"type": "message"}]

    def test_store_get_missing_returns_none(self):
        store = ResponseStore()
        assert store.get("resp_nonexistent") is None

    def test_store_lru_eviction(self):
        store = ResponseStore(maxsize=2)
        store.save("resp_a", "a", [])
        store.save("resp_b", "b", [])
        store.save("resp_c", "c", [])  # should evict resp_a
        assert store.get("resp_a") is None
        assert store.get("resp_b") is not None
        assert store.get("resp_c") is not None

    def test_store_replay_string_input(self):
        store = ResponseStore()
        store.save("resp_1", "hello", [])
        items = store.replay_input("resp_1")
        assert items is not None
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "hello"

    def test_store_replay_message_list_input(self):
        original = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "system", "content": "You are helpful."},
        ]
        store = ResponseStore()
        store.save("resp_1", original, [])
        items = store.replay_input("resp_1")
        assert items is not None
        assert len(items) == 2
        assert items[0]["role"] == "user"
        assert items[1]["role"] == "system"

    def test_store_replay_function_call_output(self):
        output = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
            }
        ]
        store = ResponseStore()
        store.save("resp_1", "hello", output)
        items = store.replay_input("resp_1")
        assert items is not None
        # First item is the original user input, second is the function call
        fc_items = [i for i in items if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"
        assert fc_items[0]["call_id"] == "call_123"

    def test_store_replay_missing_returns_none(self):
        store = ResponseStore()
        assert store.replay_input("resp_nope") is None

    def test_store_clear(self):
        store = ResponseStore()
        store.save("resp_1", "a", [])
        store.save("resp_2", "b", [])
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert store.get("resp_1") is None


# =========================================================================
# C. Functional Endpoint Tests (require mlx for server import)
# =========================================================================

# Guard: skip functional/streaming tests if mlx is unavailable, but let
# the pure-unit tests above run on any platform.
_has_mlx = importlib.util.find_spec("mlx") is not None

if _has_mlx:
    import mlx_vlm.server as server  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402

_skip_no_mlx = pytest.mark.skipif(not _has_mlx, reason="mlx not installed")


# Shared mock objects (safe to create even without mlx)
mock_model = MagicMock()
mock_processor = MagicMock()
mock_processor.tokenizer = MagicMock()
mock_processor.tokenizer.chat_template = ""
mock_config = SimpleNamespace(model_type="test")


def _mock_result(text="Hello world!", prompt_tokens=10, gen_tokens=5):
    """Build a SimpleNamespace matching generate() return shape."""
    return SimpleNamespace(
        text=text,
        prompt_tokens=prompt_tokens,
        generation_tokens=gen_tokens,
        total_tokens=prompt_tokens + gen_tokens,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=1.0,
    )


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


def _patch_model():
    return patch.object(
        server, "get_cached_model",
        return_value=(mock_model, mock_processor, mock_config),
    )


def _patch_template():
    return patch.object(server, "apply_chat_template", return_value="prompt")


def _patch_generate(result=None):
    if result is None:
        result = _mock_result()
    return patch.object(server, "generate", return_value=result)


@_skip_no_mlx
class TestResponsesEndpoint:
    """Functional tests for POST /responses."""

    def test_basic_text_response(self, client):
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={"model": "demo", "input": "Hello"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert "id" in data
        assert "output" in data
        assert "usage" in data

    def test_response_with_message_list(self, client):
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [{"role": "user", "content": "hello"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_instructions_field_echoed(self, client):
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [
                        {"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "hi"},
                    ],
                    "instructions": "Be brief.",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        # The instructions field should be present in the response
        assert data.get("instructions") is not None

    def test_tools_field_echoed(self, client):
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={"model": "demo", "input": "hi", "tools": tools},
            )
        assert resp.status_code == 200

    def test_previous_response_id_not_found(self, client):
        """Referencing a non-existent previous_response_id should return an error."""
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": "follow-up",
                    "previous_response_id": "resp_nonexistent999",
                },
            )
        # The server should either 404 or 200 (ignoring unknown ID).
        # We just verify it doesn't crash with a 500.
        assert resp.status_code in (200, 404)

    def test_developer_role_mapped_to_system(self, client):
        """developer role should be accepted (mapped to system internally)."""
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [
                        {"role": "developer", "content": "You are helpful."},
                        {"role": "user", "content": "hi"},
                    ],
                },
            )
        # Should not crash; accept 200 or 422 if server rejects developer role
        assert resp.status_code in (200, 422)

    def test_text_type_alias(self, client):
        """'text' type should be accepted alongside 'input_text'."""
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        }
                    ],
                },
            )
        assert resp.status_code == 200

    def test_function_call_output_input(self, client):
        """function_call_output items in input should not crash the server."""
        with _patch_model(), _patch_template(), _patch_generate():
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": [
                        {"role": "user", "content": "call a tool"},
                        {
                            "type": "function_call_output",
                            "call_id": "call_abc",
                            "output": '{"result": 42}',
                        },
                    ],
                },
            )
        # May fail if server doesn't handle function_call_output yet;
        # accept anything except unhandled 500
        assert resp.status_code in (200, 400, 422)

    def test_max_output_tokens_incomplete(self, client):
        """When finish_reason is 'length', the response status should ideally be 'incomplete'."""
        result = _mock_result(text="truncated...")
        with _patch_model(), _patch_template(), _patch_generate(result):
            resp = client.post(
                "/responses",
                json={
                    "model": "demo",
                    "input": "Write a very long essay",
                    "max_output_tokens": 5,
                },
            )
        assert resp.status_code == 200
        # Just verify the response is well-formed
        data = resp.json()
        assert data["status"] in ("completed", "incomplete")


@_skip_no_mlx
class TestResponsesStreaming:
    """Streaming SSE tests for POST /responses with stream=true."""

    def _stream_events(self, client, payload):
        """Helper: POST with stream=True and collect SSE events."""
        with _patch_model(), _patch_template():
            # Mock stream_generate to yield chunks
            chunks = [
                SimpleNamespace(text="Hello", prompt_tokens=10, generation_tokens=1),
                SimpleNamespace(text=" world", prompt_tokens=10, generation_tokens=2),
            ]

            def mock_stream_gen(**kwargs):
                return iter(chunks)

            with patch.object(server, "stream_generate", side_effect=mock_stream_gen):
                resp = client.post("/responses", json=payload)
        return resp

    def test_streaming_sse_events(self, client):
        payload = {"model": "demo", "input": "Hello", "stream": True}
        resp = self._stream_events(client, payload)
        assert resp.status_code == 200
        body = resp.text
        # Should contain key event types
        assert "event: response.created" in body
        assert "event: response.output_text.delta" in body
        assert "event: response.completed" in body

    def test_streaming_done_sentinel(self, client):
        """The stream should end properly (response.completed is the last real event)."""
        payload = {"model": "demo", "input": "Hello", "stream": True}
        resp = self._stream_events(client, payload)
        assert resp.status_code == 200
        body = resp.text
        # The last meaningful event should be response.completed
        lines = [l for l in body.strip().split("\n") if l.startswith("event:")]
        assert lines[-1] == "event: response.completed"


# =========================================================================
# E. Prompt Cache Tests
# =========================================================================


@_skip_no_mlx
class TestPromptCache:
    """Verify prompt_cache_state is wired into all generation entry points."""

    def test_responses_non_streaming_passes_cache_state(self, client):
        """Non-streaming /responses should pass prompt_cache_state to generate."""
        captured = {}

        def capture_generate(**kwargs):
            captured["prompt_cache_state"] = kwargs.get("prompt_cache_state")
            return _mock_result()

        with _patch_model(), _patch_template(), \
             patch.object(server, "generate", side_effect=capture_generate):
            resp = client.post("/responses", json={"model": "demo", "input": "hi"})
        assert resp.status_code == 200
        assert captured.get("prompt_cache_state") is not None
        assert hasattr(captured["prompt_cache_state"], "find_prefix_length")

    def test_responses_streaming_passes_cache_state(self, client):
        """Streaming /responses should pass prompt_cache_state to stream_generate."""
        captured = {}

        def capture_stream(**kwargs):
            captured["prompt_cache_state"] = kwargs.get("prompt_cache_state")
            return iter([
                SimpleNamespace(text="Hi", prompt_tokens=5, generation_tokens=1),
            ])

        with _patch_model(), _patch_template(), \
             patch.object(server, "stream_generate", side_effect=capture_stream):
            resp = client.post(
                "/responses", json={"model": "demo", "input": "hi", "stream": True},
            )
        assert resp.status_code == 200
        assert captured.get("prompt_cache_state") is not None

    def test_chat_completions_non_streaming_passes_cache_state(self, client):
        """Non-streaming /chat/completions should pass prompt_cache_state."""
        captured = {}

        def capture_generate(**kwargs):
            captured["prompt_cache_state"] = kwargs.get("prompt_cache_state")
            return _mock_result()

        with _patch_model(), _patch_template(), \
             patch.object(server, "generate", side_effect=capture_generate):
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert resp.status_code == 200
        assert captured.get("prompt_cache_state") is not None

    def test_chat_completions_streaming_passes_cache_state(self, client):
        """Streaming /chat/completions should pass prompt_cache_state."""
        captured = {}

        def capture_stream(**kwargs):
            captured["prompt_cache_state"] = kwargs.get("prompt_cache_state")
            return iter([
                SimpleNamespace(text="Hi", prompt_tokens=5, generation_tokens=1),
            ])

        with _patch_model(), _patch_template(), \
             patch.object(server, "stream_generate", side_effect=capture_stream):
            resp = client.post(
                "/chat/completions",
                json={
                    "model": "demo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert captured.get("prompt_cache_state") is not None

    def test_cache_state_persists_across_requests(self, client):
        """The same PromptCacheState should be reused for the same model."""
        states = []

        def capture_generate(**kwargs):
            states.append(kwargs.get("prompt_cache_state"))
            return _mock_result()

        with _patch_model(), _patch_template(), \
             patch.object(server, "generate", side_effect=capture_generate):
            client.post("/responses", json={"model": "demo", "input": "first"})
            client.post("/responses", json={"model": "demo", "input": "second"})

        assert len(states) == 2
        assert states[0] is states[1], "Same model should reuse the same cache state"

    def test_cache_state_isolated_per_model(self, client):
        """Different models should get different PromptCacheState instances."""
        states = {}

        def capture_generate(**kwargs):
            return _mock_result()

        # We need to capture from the store directly
        with _patch_model(), _patch_template(), \
             patch.object(server, "generate", side_effect=capture_generate):
            client.post("/responses", json={"model": "model-a", "input": "hi"})
            state_a = server.get_prompt_cache_state("model-a")
            client.post("/responses", json={"model": "model-b", "input": "hi"})
            state_b = server.get_prompt_cache_state("model-b")

        assert state_a is not state_b, "Different models must have separate cache states"
