"""Compaction protocol, persistence, tool boundaries, and failure atomicity."""

import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mlx_vlm.server as server
from mlx_vlm.server import compaction, openai
from mlx_vlm.tests.test_server import (
    _data,
    _endpoint,
    _post,
    _result,
    _streaming,
    _token,
)


def message(text, role="user"):
    return {"type": "message", "role": role, "content": text}


def history():
    return [
        message("Follow the user's constraints.", "system"),
        message("Project ORCHID. The chosen port is 7319. Never edit secrets.env."),
        {
            "type": "function_call",
            "name": "read_file",
            "call_id": "c1",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "old log entry\n" * 1500,
        },
        message("Read the log. Next: verify configuration.", "assistant"),
        message("What is the port?"),
    ]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_VLM_COMPACTION_KEY_FILE", str(tmp_path / "key"))
    monkeypatch.setattr(server.runtime.config, "max_kv_size", None)
    server.response_store.clear()
    server.response_store_order.clear()


@pytest.fixture
def mocked():
    with (
        _endpoint(
            config=NS(model_type="qwen2_vl", max_position_embeddings=32768),
            result=_result(
                "Goal: ORCHID. Port: 7319. Constraint: never edit secrets.env.",
                cached_tokens=32,
            ),
        ) as fake,
        patch.object(openai, "prepare_inputs") as prepare,
    ):
        fake.template.side_effect = (
            lambda processor, config, messages, **kw: json.dumps(messages)
        )
        prepare.side_effect = lambda processor, prompts, **kw: {
            "input_ids": np.zeros((1, max(1, len(prompts) // 4)), dtype=np.int32)
        }
        yield fake, TestClient(server.app)


@pytest.mark.parametrize("path", ["/responses/compact", "/v1/responses/compact"])
def test_compact_round_trip_and_token_count(mocked, path):
    fake, client = mocked
    original = history()
    result = _post(client, path, input=original, keep_tokens=0, max_output_tokens=256)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["object"] == "response.compaction"
    assert data["usage"]["input_tokens_details"]["cached_tokens"] == 32
    item = data["output"][0]
    assert item["type"] == "compaction"
    assert "ORCHID" not in item["encrypted_content"]
    restored = compaction.resolve([item], model="demo", tenant=None)
    assert restored[0] == original[0] and restored[-1] == original[-1]
    assert restored[1]["role"] == "assistant"
    assert "7319" in restored[1]["content"][0]["text"]
    assert "old log entry" not in json.dumps(restored)
    # No registry or model cache is needed to decode; key reload represents restart.
    assert compaction.resolve([item], model="demo", tenant=None) == restored
    for inputs in ([item], original + [item], restored):
        response = _post(client, "/responses/input_tokens", input=inputs)
        assert response.status_code == 200
        if inputs == [item]:
            expected = response.json()
        assert response.json() == expected
    continuation = _post(
        client,
        "responses",
        input=[item, message("Continue")],
        instructions="New current instructions",
    )
    assert continuation.status_code == 200
    rendered = json.loads(fake.generate.call_args.kwargs["prompt"])
    assert "New current instructions" in rendered[0]["content"]
    assert rendered[-1]["content"] == "Continue"


@pytest.mark.parametrize("change", ["tenant", "model", "tamper", "foreign"])
def test_capsules_reject_invalid_scope_and_payload(change):
    item = compaction.seal([message("private")], model="demo", tenant="one")
    model, tenant = "demo", "one"
    if change == "tenant":
        tenant = "two"
    elif change == "model":
        model = "other"
    elif change == "tamper":
        value = item["encrypted_content"]
        at = len(compaction.CAPSULE_PREFIX) + 50
        item["encrypted_content"] = (
            value[:at] + ("A" if value[at] != "A" else "B") + value[at + 1 :]
        )
    else:
        item["encrypted_content"] = "openai-issued-opaque-state"
    with pytest.raises(HTTPException) as caught:
        compaction.resolve([item], model=model, tenant=tenant)
    assert caught.value.status_code == 400


def test_latest_capsule_replaces_previous_context():
    first = compaction.seal([message("old")], model="demo", tenant=None)
    second = compaction.seal([message("new")], model="demo", tenant=None)
    assert compaction.resolve(
        [first, message("between"), second, message("last")], model="demo", tenant=None
    ) == [message("new"), message("last")]


def test_resent_codex_instructions_do_not_accumulate_in_capsules():
    prefix = message("Current permission constraints.", "developer")
    other = message("A separate requirement.", "developer")
    original = [prefix, other, message("handoff", "assistant")]
    for _ in range(5):
        capsule = compaction.seal(original, model="demo", tenant=None)
        original = compaction.resolve(
            [capsule, {**prefix, "id": "new-client-id"}, message("Continue")],
            model="demo",
            tenant=None,
        )
        assert sum(x.get("content") == prefix["content"] for x in original) == 1
        assert other in original


def test_pending_parallel_tools_cannot_be_split():
    items = [
        message("start"),
        {"type": "function_call", "call_id": "a"},
        {"type": "function_call", "call_id": "b"},
        {"type": "function_call_output", "call_id": "a"},
        message("steering during tools"),
        {"type": "function_call_output", "call_id": "b"},
        message("next"),
    ]
    assert compaction.safe_boundaries(items) == [0, 6]


@pytest.mark.parametrize("failure", ["empty", "length", "too_large"])
def test_summary_failures_preserve_original_context(mocked, failure):
    fake, client = mocked
    original = history()
    saved = copy.deepcopy(original)
    result = _result("" if failure == "empty" else "huge " * 10000)
    if failure == "length":
        result = _result("partial summary", generation_tokens=256)
    fake.generate.return_value = result
    response = _post(
        client,
        "/responses/compact",
        input=original,
        max_output_tokens=256,
        keep_tokens=0,
    )
    assert response.status_code in (400, 502), response.text
    assert original == saved and not server.response_store


def test_short_history_is_a_noop(mocked):
    fake, client = mocked
    items = [message("hello")]
    response = _post(client, "/responses/compact", input=items)
    assert response.status_code == 200 and response.json()["output"] == items
    fake.generate.assert_not_called()


def test_summary_budget_overrides_legacy_max_tokens(mocked):
    fake, client = mocked
    response = _post(
        client,
        "/responses/compact",
        input=history(),
        keep_tokens=0,
        max_output_tokens=256,
        max_tokens=4096,
    )
    assert response.status_code == 200, response.text
    assert fake.generate.call_args.kwargs["max_tokens"] == 256


def test_auto_below_threshold_only_generates_the_answer(mocked):
    fake, client = mocked
    response = _post(
        client,
        "responses",
        input="hello",
        max_output_tokens=64,
        context_management=[{"type": "compaction", "compact_threshold": 1000}],
    )
    assert response.status_code == 200, response.text
    assert all(item["type"] != "compaction" for item in response.json()["output"])
    assert fake.generate.call_count == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("short", [False, True])
def test_codex_compaction_trigger_returns_only_one_capsule(mocked, stream, short):
    fake, client = mocked
    items = [message("hello")] if short else history()
    response = _post(
        client,
        "responses",
        input=items + [{"type": "compaction_trigger"}],
        stream=stream,
    )
    assert response.status_code == 200, response.text
    if stream:
        events = _data(response)
        data = next(x["response"] for x in events if x["type"] == "response.completed")
        assert not any(x["type"] == "response.output_text.delta" for x in events)
    else:
        data = response.json()
    assert [x["type"] for x in data["output"]] == ["compaction"]
    assert data["output_text"] == ""
    assert fake.generate.call_count == (0 if short else 1)
    followup = _post(client, "responses", input=data["output"] + [message("Continue")])
    assert followup.status_code == 200
    assert "compaction_trigger" not in fake.generate.call_args.kwargs["prompt"]


def test_codex_trigger_must_be_terminal(mocked):
    _, client = mocked
    response = _post(
        client, "responses", input=[{"type": "compaction_trigger"}, message("later")]
    )
    assert response.status_code == 400


def test_fixed_instructions_are_excluded_from_reduction_target(mocked):
    _, client = mocked
    inputs = history()
    options = dict(input=inputs, instructions="Static instruction. " * 3000)
    before = _post(client, "/responses/input_tokens", **options).json()["input_tokens"]
    response = _post(client, "/responses/compact", keep_tokens=0, **options)
    assert response.status_code == 200, response.text
    after = _post(
        client,
        "/responses/input_tokens",
        input=response.json()["output"],
        instructions=options["instructions"],
    ).json()["input_tokens"]
    assert before * 0.6 < after < before


def test_responses_tool_arguments_are_normalized_without_mutating_input():
    from mlx_vlm.server.responses_state import _response_items_to_chat

    items = [
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c1",
            "arguments": '{"cmd":"cat log.txt"}',
        }
    ]
    original = copy.deepcopy(items)
    messages, _ = _response_items_to_chat(items)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "cmd": "cat log.txt"
    }
    assert messages[0]["content"] == ""
    assert items == original


@pytest.mark.parametrize("threshold", [1, 100000])
def test_impossible_output_budget_rejected_before_generation(mocked, threshold):
    fake, client = mocked
    response = _post(
        client,
        "responses",
        input=history(),
        max_output_tokens=32768,
        context_management=[{"type": "compaction", "compact_threshold": threshold}],
    )
    assert response.status_code == 400
    fake.generate.assert_not_called()


@pytest.mark.parametrize(
    "item",
    [
        {"type": "item_reference", "id": "missing"},
        {"type": "message", "role": "user", "content": [{"type": "input_audio"}]},
    ],
)
def test_unsupported_content_is_not_silently_summarized(mocked, item):
    fake, client = mocked
    response = _post(client, "/responses/compact", input=[item] + history())
    assert response.status_code == 400
    fake.generate.assert_not_called()


def test_tail_with_image_and_pending_tool_call_is_retained():
    instructions = message("Keep the original constraints.", "developer")
    tail = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect this"},
                {"type": "input_image", "image_url": "data:image/png;base64,example"},
            ],
        },
        {
            "type": "function_call",
            "call_id": "pending",
            "name": "read_file",
            "arguments": "{}",
        },
        message("A correction while the tool is running"),
    ]
    original = [
        instructions,
        message("old " * 1000),
        message("done", "assistant"),
        *tail,
    ]

    async def count(items):
        return len(json.dumps(items))

    async def summarize(items):
        assert items == original[:3]
        return "Prior work completed.", None

    result = asyncio.run(
        compaction.compact(
            original,
            count=count,
            summarize=summarize,
            keep_tokens=0,
            target_tokens=2000,
        )
    )
    assert result.changed and result.items[0] == instructions
    assert result.items[-len(tail) :] == tail
    assert (
        compaction.resolve(
            [compaction.seal(result.items, model="demo", tenant=None)],
            model="demo",
            tenant=None,
        )
        == result.items
    )


@pytest.mark.parametrize(
    "config",
    [
        NS(text_config=NS(max_position_embeddings=8192)),
        NS(text_config={"max_position_embeddings": 8192}),
        {"text_config": {"max_position_embeddings": 8192}},
    ],
)
def test_context_limit_respects_model_and_server(config, monkeypatch):
    assert openai._compaction_context_limit(config) == 8192
    monkeypatch.setattr(server.runtime.config, "max_kv_size", 4096)
    assert openai._compaction_context_limit(config) == 4096


@pytest.mark.parametrize("stream", [False, True])
def test_auto_compaction_replay_and_stream_indices(mocked, stream):
    fake, client = mocked
    response = _post(
        client,
        "responses",
        input=history(),
        max_output_tokens=64,
        context_management=[{"type": "compaction", "compact_threshold": 1000}],
        stream=stream,
    )
    assert response.status_code == 200, response.text
    if stream:
        events = _data(response)
        final = next(x["response"] for x in events if x["type"] == "response.completed")
        compact_events = [
            x
            for x in events
            if x["type"].startswith("response.output_item.")
            and x["item"]["type"] == "compaction"
        ]
        assert [x["type"] for x in compact_events] == [
            "response.output_item.added",
            "response.output_item.done",
        ]
        assert all(x["output_index"] == 0 for x in compact_events)
        text_events = [
            x for x in events if "output_index" in x and x not in compact_events
        ]
        assert all(x["output_index"] >= 1 for x in text_events)
    else:
        final = response.json()
    assert final["output"][0]["type"] == "compaction"
    resolved = compaction.resolve(
        history() + final["output"], model="demo", tenant=None
    )
    assert "old log entry" not in json.dumps(resolved)
    continued = _post(
        client, "responses", input="next", previous_response_id=final["id"]
    )
    assert continued.status_code == 200
    assert "old log entry" not in fake.generate.call_args.kwargs["prompt"]


def test_stored_compaction_survives_parent_eviction(mocked):
    _, client = mocked
    first = _post(client, "responses", input="original").json()
    capsule = compaction.seal([message("compacted")], model="demo", tenant=None)
    second = _post(
        client, "responses", input=[capsule], previous_response_id=first["id"]
    ).json()
    server.response_store.pop(first["id"])
    response = _post(
        client, "responses", input="next", previous_response_id=second["id"]
    )
    assert response.status_code == 200


@pytest.mark.parametrize("threshold", [0, -1, "bad"])
def test_invalid_threshold_rejected(mocked, threshold):
    _, client = mocked
    response = _post(
        client,
        "responses",
        context_management=[{"type": "compaction", "compact_threshold": threshold}],
    )
    assert response.status_code == 422


def test_summary_uses_generation_worker_and_closes_iterator(mocked, monkeypatch):
    _, client = mocked
    generator = _streaming([_token("Goal: ORCHID. Port: 7319.", finish_reason="stop")])
    generator._cpu_preprocess = lambda prompt, images, audio: {
        "input_ids": np.zeros((1, max(1, len(prompt) // 4)), dtype=np.int32)
    }
    monkeypatch.setattr(server.runtime, "response_generator", generator)
    response = _post(client, "/responses/compact", input=history(), keep_tokens=0)
    assert response.status_code == 200, response.text
    assert generator.generate.call_count == 1
