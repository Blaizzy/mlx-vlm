"""Speculative-compat guards: real 400s, preflighted before SSE streams start.

Under a drafter the server's speculative round-loops sample with the sampler
from ``_make_sampler`` and never apply ``_make_logits_processors`` — so
``logit_bias``, the repetition/presence/frequency penalties, compositions of
the exclusive-priority branch fields (``top_n_sigma``/``p_less``/
``typical_p``), and ``top_k``/``min_p`` composed with a branch field are
silently ignored. (Plain ``top_k``/``min_p`` are #1647's lane and stay
unflagged.) ``generate()`` already rejected ``logits_processors`` and
``thinking_budget``, but as ``ValueError``:

- non-stream requests reported a 500 instead of a 400;
- stream=true requests ran ``generate()`` inside an already-started SSE
  generator, so the rejection surfaced as error data under HTTP 200.

The guards now live in the module-level ``preflight_speculative_args``
(HTTPException(400)); the /chat/completions, /responses, and /v1/messages
stream branches call it BEFORE any StreamingResponse is constructed, and
``generate()`` keeps it for defense in depth.
``MLX_VLM_SPEC_IGNORE_UNSUPPORTED_SAMPLING=1`` downgrades the sampling-field
guard to a warning for deployments that prefer the old silent behavior.

Model-free: unit tests use fakes; endpoint tests patch the request path up to
the preflight (same approach as test_server_anthropic_stream_preflight).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mlx_vlm.server as server
import mlx_vlm.server.anthropic as server_anthropic
import mlx_vlm.server.openai as server_openai
from mlx_vlm.server.generation import (
    GenerationArguments,
    ResponseGenerator,
    _unsupported_speculative_sampling_fields,
    preflight_speculative_args,
)


def _fields(**kwargs):
    return _unsupported_speculative_sampling_fields(GenerationArguments(**kwargs))


def _preflight(draft=True, **kwargs):
    """Run the guard; return the HTTPException or None."""
    try:
        preflight_speculative_args(
            object() if draft else None, GenerationArguments(**kwargs)
        )
    except HTTPException as e:
        return e
    return None


# ---------------------------------------------------------------------------
# Which sampling fields the speculative path would silently ignore
# ---------------------------------------------------------------------------


def test_supported_sampler_shapes_not_flagged():
    assert _fields(temperature=0.7, top_p=0.95) == []
    assert _fields(temperature=0.7, top_n_sigma=2.0) == []
    assert _fields(temperature=0.7, p_less=True) == []
    assert _fields(temperature=0.7, typical_p=0.8) == []


def test_branch_composition_flagged():
    assert _fields(temperature=0.7, top_n_sigma=2.0, typical_p=0.8) == ["typical_p"]
    assert _fields(temperature=0.7, top_n_sigma=1.0, p_less=True, typical_p=0.5) == [
        "p_less",
        "typical_p",
    ]


def test_branch_plus_top_k_min_p_flagged():
    # The top_n_sigma/p_less/typical_p branch samplers are built without
    # top_k/min_p, so composing drops them.
    assert _fields(temperature=0.7, top_n_sigma=2.0, top_k=64) == ["top_k"]
    assert _fields(temperature=0.7, typical_p=0.8, min_p=0.05) == ["min_p"]
    assert _fields(temperature=0.7, p_less=True, top_k=64, min_p=0.05) == [
        "top_k",
        "min_p",
    ]


def test_plain_top_k_min_p_not_flagged():
    # Plain top_k/min_p are silently dropped on main today too, but
    # implementing them row-level is #1647's lane — the guard must not
    # reject requests that PR makes correct.
    assert _fields(temperature=0.7, top_k=64) == []
    assert _fields(temperature=0.7, min_p=0.05) == []


def test_logits_processor_fields_flagged():
    assert _fields(repetition_penalty=1.3) == ["repetition_penalty"]
    assert _fields(repetition_penalty=1.0) == []
    assert _fields(logit_bias={5: 2.0}) == ["logit_bias"]
    assert _fields(presence_penalty=0.5, frequency_penalty=0.5) == [
        "presence_penalty",
        "frequency_penalty",
    ]


def test_greedy_ignores_every_sampler_knob():
    assert (
        _fields(
            temperature=0,
            top_n_sigma=2.0,
            p_less=True,
            typical_p=0.8,
            top_k=64,
            min_p=0.05,
        )
        == []
    )


# ---------------------------------------------------------------------------
# The extracted preflight raises 400; escape hatch covers sampling fields only
# ---------------------------------------------------------------------------


def test_preflight_rejects_offending_args():
    e = _preflight(logits_processors=[lambda tokens, logits: logits])
    assert e is not None and e.status_code == 400
    assert "response_format" in str(e.detail)
    e = _preflight(thinking_budget=512)
    assert e is not None and e.status_code == 400
    assert "thinking_budget" in str(e.detail)
    e = _preflight(repetition_penalty=1.3)
    assert e is not None and e.status_code == 400
    assert "repetition_penalty" in str(e.detail)
    assert "MLX_VLM_SPEC_IGNORE_UNSUPPORTED_SAMPLING" in str(e.detail)
    e = _preflight(temperature=0.7, top_n_sigma=2.0, top_k=64)
    assert e is not None and e.status_code == 400 and "top_k" in str(e.detail)


def test_preflight_accepts_supported_args():
    assert _preflight(temperature=0.7, top_p=0.95) is None
    assert _preflight(temperature=0.7, typical_p=0.8) is None
    assert _preflight(temperature=0.7, top_k=64, min_p=0.05) is None
    assert _preflight(temperature=0) is None


def test_preflight_noop_without_drafter():
    assert (
        _preflight(
            draft=False,
            repetition_penalty=1.3,
            thinking_budget=512,
            logits_processors=[lambda tokens, logits: logits],
        )
        is None
    )


def test_preflight_escape_hatch_covers_sampling_fields_only(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SPEC_IGNORE_UNSUPPORTED_SAMPLING", "1")
    assert _preflight(repetition_penalty=1.3) is None
    assert _preflight(temperature=0.7, top_n_sigma=2.0, top_k=64) is None
    # The response_format/thinking_budget guards have no escape hatch.
    assert _preflight(logits_processors=[lambda tokens, logits: logits]) is not None
    assert _preflight(thinking_budget=512) is not None
    monkeypatch.delenv("MLX_VLM_SPEC_IGNORE_UNSUPPORTED_SAMPLING")
    assert _preflight(repetition_penalty=1.3) is not None


def test_generate_keeps_guard_defense_in_depth():
    """generate() must reject on its own — with a 400, not a ValueError —
    so the non-stream path is covered even if an endpoint forgets to
    preflight."""

    class _Sentinel(Exception):
        pass

    def _boom(prompt, images=None, audio=None, videos=None):
        raise _Sentinel

    fake = SimpleNamespace(
        draft_model=object(),
        wait_until_ready=lambda: None,
        _preprocess_request=_boom,
    )
    with pytest.raises(HTTPException) as excinfo:
        ResponseGenerator.generate(
            fake, "hi", args=GenerationArguments(repetition_penalty=1.3)
        )
    assert excinfo.value.status_code == 400
    # Clean args reach preprocessing (the sentinel), not a guard rejection.
    with pytest.raises(_Sentinel):
        ResponseGenerator.generate(fake, "hi", args=GenerationArguments())


# ---------------------------------------------------------------------------
# Router wiring: the stream preflight records the failure and raises before
# any SSE byte can be produced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,endpoint",
    [(server_openai, "/chat/completions"), (server_anthropic, "/v1/messages")],
)
def test_stream_preflight_helper_with_fake_runtime(monkeypatch, module, endpoint):
    failures = []
    monkeypatch.setattr(
        server.runtime, "response_generator", SimpleNamespace(draft_model=object())
    )
    monkeypatch.setattr(
        server.runtime,
        "metrics",
        SimpleNamespace(record_failure=lambda **kwargs: failures.append(kwargs)),
    )
    with pytest.raises(HTTPException) as excinfo:
        module._preflight_speculative_stream_args(
            endpoint=endpoint,
            model="demo",
            args=GenerationArguments(repetition_penalty=1.3),
        )
    assert excinfo.value.status_code == 400
    assert failures and failures[0]["stream"] is True
    assert failures[0]["endpoint"] == endpoint

    # Supported args pass through silently.
    module._preflight_speculative_stream_args(
        endpoint=endpoint, model="demo", args=GenerationArguments(temperature=0.7)
    )
    # No drafter -> no-op even with offending args.
    monkeypatch.setattr(
        server.runtime, "response_generator", SimpleNamespace(draft_model=None)
    )
    module._preflight_speculative_stream_args(
        endpoint=endpoint,
        model="demo",
        args=GenerationArguments(repetition_penalty=1.3),
    )
    # No response generator at all -> no-op.
    monkeypatch.setattr(server.runtime, "response_generator", None)
    module._preflight_speculative_stream_args(
        endpoint=endpoint,
        model="demo",
        args=GenerationArguments(repetition_penalty=1.3),
    )
    assert len(failures) == 1


def test_endpoints_preflight_before_stream_generator_starts():
    """Structural audit: in every streaming endpoint the speculative
    preflight call must appear before the stream generator is even defined,
    i.e. before any SSE byte can be produced."""
    for endpoint in (
        server_openai.chat_completions_endpoint,
        server_openai.responses_endpoint,
        server_anthropic.anthropic_messages_endpoint,
    ):
        src = inspect.getsource(endpoint)
        assert "_preflight_speculative_stream_args" in src, endpoint.__name__
        assert src.index("_preflight_speculative_stream_args") < src.index(
            "async def stream_generator"
        ), f"{endpoint.__name__}: preflight after generator definition"


# ---------------------------------------------------------------------------
# End-to-end through the Anthropic router: 400 with the proper error
# envelope, not error data under an HTTP 200 SSE stream
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _patched_request_path(gen_args):
    """Patch every step between the endpoint entry and the preflights so the
    request reaches them without loading a model."""

    async def _budget_ok(**_kwargs):
        return None

    model = SimpleNamespace()
    processor = SimpleNamespace()
    config = SimpleNamespace()
    return (
        patch.object(
            server_anthropic,
            "get_cached_model",
            return_value=(model, processor, config),
        ),
        patch.object(
            server_anthropic,
            "_anthropic_messages_to_internal",
            return_value=(["msg"], [], None, None),
        ),
        patch.object(
            server_anthropic, "_infer_tool_parser_from_processor", return_value=None
        ),
        patch.object(server_anthropic, "_build_gen_args", return_value=gen_args),
        patch.object(server_anthropic, "apply_chat_template", return_value="prompt"),
        patch.object(server_anthropic, "_preflight_stream_context_budget", _budget_ok),
    )


_STREAM_BODY = {
    "model": "demo",
    "max_tokens": 16,
    "stream": True,
    "messages": [{"role": "user", "content": "Hello"}],
}


def test_stream_with_drafter_returns_400_not_sse_200(client, monkeypatch):
    monkeypatch.setattr(
        server.runtime, "response_generator", SimpleNamespace(draft_model=object())
    )
    patches = _patched_request_path(GenerationArguments(repetition_penalty=1.3))
    for p in patches:
        p.start()
    try:
        response = client.post("/v1/messages", json=_STREAM_BODY)
    finally:
        for p in patches:
            p.stop()

    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "repetition_penalty" in payload["error"]["message"]
    assert not payload["error"]["message"].startswith("400:")


def test_stream_with_drafter_and_supported_args_starts_stream(client, monkeypatch):
    monkeypatch.setattr(
        server.runtime, "response_generator", SimpleNamespace(draft_model=object())
    )
    patches = _patched_request_path(GenerationArguments(temperature=0.7, top_p=0.95))
    for p in patches:
        p.start()
    try:
        with client.stream("POST", "/v1/messages", json=_STREAM_BODY) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
    finally:
        for p in patches:
            p.stop()
