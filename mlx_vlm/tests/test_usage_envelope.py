"""Tests for the streaming/non-streaming usage envelope fixes.

Covers two related fixes:

  * Streaming /chat/completions chunks were being constructed with a
    plain ``dict`` for ``usage``. Pydantic coerced it into ``UsageStats``
    with ``prompt_tps``, ``peak_memory``, and ``prompt_tokens_details``
    silently filled from defaults — clients always saw 0. Pass a full
    ``UsageStats`` so chunks report real rates.
  * ``cached_tokens`` is plumbed from APC's per-row ``prefix_len``
    through ``PromptProgress`` → ``StreamingToken`` →
    ``UsageStats.prompt_tokens_details.cached_tokens``, mirroring
    OpenAI's ``input_tokens_details.cached_tokens``.

``generation_tps`` is intentionally untouched — its source is not yet
wired in continuous batching (#1113 is the open PR for that).

These tests run without a real model — they exercise the data-class
plumbing only.
"""

from __future__ import annotations

from mlx_vlm.generate import PromptProgress
from mlx_vlm.server import (
    GenerationContext,
    InputTokensDetails,
    OpenAIUsage,
    PromptTokensDetails,
    StreamingToken,
    UsageStats,
)


def test_prompt_progress_carries_cached_tokens():
    """``PromptProcessingBatch.prompt_progress()`` builds these from
    ``apc_meta[i]['prefix_len']`` so the server can attribute APC hits
    per request.
    """
    p = PromptProgress(
        uid=7,
        prompt_tokens=128,
        prompt_tps=42.0,
        prompt_time=3.0,
        cached_tokens=96,
    )
    assert p.cached_tokens == 96
    # Default still 0 so old call sites continue to work.
    assert PromptProgress(uid=1, prompt_tokens=10).cached_tokens == 0


def test_streaming_token_carries_prompt_tps_and_cached_tokens():
    tok = StreamingToken(
        text="x",
        token=42,
        logprobs=0.0,
        finish_reason=None,
        prompt_tps=120.0,
        cached_tokens=128,
    )
    assert tok.cached_tokens == 128
    assert tok.prompt_tps == 120.0


def test_streaming_token_defaults_preserve_old_callers():
    tok = StreamingToken(
        text="x", token=1, logprobs=0.0, finish_reason=None,
    )
    assert tok.cached_tokens == 0
    assert tok.prompt_tps is None


def test_generation_context_unchanged():
    """``GenerationContext`` deliberately stays minimal — cached_tokens
    only becomes known after admission, so it rides on
    ``StreamingToken``/``UsageStats`` instead.
    """
    ctx = GenerationContext(uid=1, prompt_tokens=100)
    assert ctx.uid == 1
    assert ctx.prompt_tokens == 100


def test_usage_stats_serialises_full_envelope():
    """The streaming chat fix means a chunk's ``usage`` field now
    serialises every UsageStats field, including ``cached_tokens``.
    """
    usage = UsageStats(
        prompt_tokens=200,
        completion_tokens=50,
        total_tokens=250,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=128),
        prompt_tps=120.0,
        generation_tps=33.5,
        peak_memory=4.2,
    )
    payload = usage.model_dump()
    assert payload["prompt_tokens_details"]["cached_tokens"] == 128
    assert payload["prompt_tps"] == 120.0
    assert payload["generation_tps"] == 33.5
    assert payload["peak_memory"] == 4.2


def test_usage_stats_defaults_when_no_cache_hit():
    """Without an APC hit, ``cached_tokens`` defaults to 0 so the
    field is always present in the envelope.
    """
    usage = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    payload = usage.model_dump()
    assert payload["prompt_tokens_details"]["cached_tokens"] == 0
    assert payload["prompt_tps"] == 0.0
    assert payload["generation_tps"] == 0.0


def test_openai_usage_input_tokens_details():
    """OpenAI's /responses spec puts cached_tokens under
    ``input_tokens_details.cached_tokens``. We mirror that field name.
    """
    usage = OpenAIUsage(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        input_tokens_details=InputTokensDetails(cached_tokens=80),
    )
    assert usage.input_tokens_details.cached_tokens == 80
    payload = usage.model_dump()
    assert payload["input_tokens_details"]["cached_tokens"] == 80


def test_openai_usage_default_input_tokens_details():
    usage = OpenAIUsage(input_tokens=100, output_tokens=10, total_tokens=110)
    assert usage.input_tokens_details.cached_tokens == 0
