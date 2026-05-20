import argparse
import asyncio
import gc
import json
import logging
import os
import re
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

logger = logging.getLogger("mlx_vlm.server")

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import scan_cache_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypeAlias, TypedDict

from . import apc as _apc
from .generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    BatchGenerator,
    _make_cache,
    _merge_prefill_prompt_kwargs,
    generate,
    normalize_resize_shape,
    stream_generate,
)
from .prompt_utils import apply_chat_template, extract_text_from_content
from .sample_utils import top_p_sampling
from .speculative.utils import (
    make_speculative_prompt_cache,
    run_speculative_server_rounds,
    speculative_hidden_state,
    speculative_prefill_kwargs,
)
from .structured import build_json_schema_logits_processor
from .tokenizer_utils import _ServerTokenStreamer, make_streaming_detokenizer
from .tool_parsers import _infer_tool_parser_from_processor, load_tool_module
from .utils import load, prepare_inputs
from .version import __version__
from .vision_cache import VisionFeatureCache

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080
DEFAULT_TOKEN_QUEUE_TIMEOUT = 600.0
DEFAULT_SPECULATIVE_BATCH_COALESCE_MS = 5.0
DEFAULT_ENABLE_THINKING = False
METRICS_HISTORY_LIMIT = 100
METRICS_RECENT_LIMIT = 32


class PromptTooLongError(ValueError):
    """Raised when a request exceeds the configured server context budget."""


def _get_draft_block_size_from_env():
    draft_block_size_str = os.environ.get("MLX_VLM_DRAFT_BLOCK_SIZE")
    return int(draft_block_size_str) if draft_block_size_str else None


def get_prefill_step_size():
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))


def get_server_max_tokens():
    return int(os.environ.get("MLX_VLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def get_token_queue_timeout():
    raw_timeout = os.environ.get("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "")
    if raw_timeout == "":
        return DEFAULT_TOKEN_QUEUE_TIMEOUT
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_TOKEN_QUEUE_TIMEOUT=%r; falling back to %ss.",
            raw_timeout,
            DEFAULT_TOKEN_QUEUE_TIMEOUT,
        )
        return DEFAULT_TOKEN_QUEUE_TIMEOUT
    if timeout <= 0:
        return None
    return timeout


def get_speculative_batch_coalesce_s():
    raw = os.environ.get(
        "MLX_VLM_SPEC_BATCH_COALESCE_MS", str(DEFAULT_SPECULATIVE_BATCH_COALESCE_MS)
    )
    try:
        return max(0.0, float(raw)) / 1000.0
    except ValueError:
        return DEFAULT_SPECULATIVE_BATCH_COALESCE_MS / 1000.0


def get_server_enable_thinking():
    raw = os.environ.get("MLX_VLM_ENABLE_THINKING")
    if raw is None:
        return DEFAULT_ENABLE_THINKING
    return raw.lower() in ("1", "true", "yes", "on")


def get_quantized_kv_bits(model: str):
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        print(f"Model {model} is quantization aware, KV cache will not be quantized.")
        return None
    return kv_bits


def get_kv_group_size():
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme():
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str):
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits(model) is not None:
        print(f"Model {model} uses QuantizedKVCache, can't set max KV size.")
        return None
    return max_kv_tokens


def get_configured_context_limit():
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    return max_kv_tokens or None


def _count_prompt_tokens(raw_inputs: dict) -> int:
    input_ids = raw_inputs["input_ids"]
    return input_ids.size if hasattr(input_ids, "size") else len(input_ids)


def _check_configured_context_budget(prompt_tokens: int, max_tokens: int):
    context_limit = get_configured_context_limit()
    requested_tokens = prompt_tokens + max(0, int(max_tokens or 0))
    if context_limit is not None and requested_tokens > context_limit:
        raise PromptTooLongError(
            "Request needs "
            f"{requested_tokens} context tokens "
            f"({prompt_tokens} prompt + {max_tokens} max generation), "
            f"but MAX_KV_SIZE is {context_limit}."
        )


def get_quantized_kv_start():
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


def get_top_logprobs_k():
    """Max per-token top_logprobs honored by the server (0 = disabled).

    Set via TOP_LOGPROBS_K env var. OpenAI caps this at 20. When 0, requests
    with top_logprobs>0 still succeed but the top_logprobs list stays empty.
    """
    k = int(os.environ.get("TOP_LOGPROBS_K", 0))
    return max(0, min(k, 20))


def _token_window_rate(token_times: List[float], window: int) -> Optional[float]:
    if len(token_times) < 2:
        return None
    subset = token_times[-window:]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _token_window_rate_first(token_times: List[float], window: int) -> Optional[float]:
    if len(token_times) < 2:
        return None
    subset = token_times[:window]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


class ServerMetricsStore:
    """Rolling request metrics and lifetime counters for the server."""

    def __init__(self, history_limit: int = METRICS_HISTORY_LIMIT):
        self.history_limit = history_limit
        self.reset()

    def reset(self):
        self.started_at = time.time()
        self._lock = Lock()
        self._latest: Optional[dict] = None
        self._recent = deque(maxlen=self.history_limit)
        self._requests_started = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._streaming_requests = 0
        self._in_flight = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._generated_tokens_total = 0
        self._request_time_total_s = 0.0
        self._decode_time_total_s = 0.0
        self._last_request_at: Optional[float] = None
        self._last_error: Optional[dict] = None

    def begin_request(self, *, endpoint: str, model: str, stream: bool):
        del endpoint, model
        with self._lock:
            self._requests_started += 1
            self._in_flight += 1
            if stream:
                self._streaming_requests += 1

    def record_success(self, envelope: dict):
        payload = dict(envelope)
        with self._lock:
            self._requests_completed += 1
            self._in_flight = max(0, self._in_flight - 1)
            self._latest = payload
            self._recent.append(payload)
            self._last_request_at = payload.get("timestamp_unix")
            self._prompt_tokens_total += int(payload.get("prompt_tokens") or 0)
            self._completion_tokens_total += int(payload.get("completion_tokens") or 0)
            self._generated_tokens_total += int(payload.get("generated_tokens") or 0)
            self._request_time_total_s += float(payload.get("request_elapsed_s") or 0.0)
            self._decode_time_total_s += float(payload.get("decode_elapsed_s") or 0.0)

    def record_failure(self, *, endpoint: str, model: str, stream: bool, error: str):
        with self._lock:
            self._requests_failed += 1
            self._in_flight = max(0, self._in_flight - 1)
            self._last_request_at = time.time()
            self._last_error = {
                "timestamp_unix": self._last_request_at,
                "endpoint": endpoint,
                "model": model,
                "stream": bool(stream),
                "error": error,
            }

    def snapshot(self) -> dict:
        with self._lock:
            latest = dict(self._latest) if self._latest is not None else None
            recent = [dict(item) for item in list(self._recent)[-METRICS_RECENT_LIMIT:]]
            requests_completed = self._requests_completed
            avg_request_time = (
                self._request_time_total_s / requests_completed
                if requests_completed > 0
                else 0.0
            )
            avg_request_tok_s = (
                self._completion_tokens_total / self._request_time_total_s
                if self._request_time_total_s > 0
                else 0.0
            )
            avg_decode_tok_s = (
                self._generated_tokens_total / self._decode_time_total_s
                if self._decode_time_total_s > 0
                else 0.0
            )
            last_error = (
                dict(self._last_error) if self._last_error is not None else None
            )
            last_request_at = self._last_request_at
            return {
                "latest": latest,
                "recent": recent,
                "summary": {
                    "uptime_s": max(0.0, time.time() - self.started_at),
                    "requests_started": self._requests_started,
                    "requests_completed": requests_completed,
                    "requests_failed": self._requests_failed,
                    "streaming_requests": self._streaming_requests,
                    "in_flight": self._in_flight,
                    "prompt_tokens_total": self._prompt_tokens_total,
                    "completion_tokens_total": self._completion_tokens_total,
                    "generated_tokens_total": self._generated_tokens_total,
                    "avg_request_time_s": avg_request_time,
                    "avg_request_tok_s": avg_request_tok_s,
                    "avg_decode_tok_s": avg_decode_tok_s,
                    "last_request_at": last_request_at,
                    "last_error": last_error,
                },
            }


def _prompt_eval_time_from_tps(
    prompt_tokens: int, prompt_tps: Optional[float]
) -> Optional[float]:
    if prompt_tokens <= 0 or prompt_tps is None or prompt_tps <= 0:
        return None
    return prompt_tokens / prompt_tps


def _decode_elapsed_from_metrics(
    token_times: List[float],
    generated_tokens: int,
    generation_tps: Optional[float],
) -> Optional[float]:
    if len(token_times) >= 2:
        elapsed = token_times[-1] - token_times[0]
        if elapsed > 0:
            return elapsed
    if generated_tokens > 0 and generation_tps is not None and generation_tps > 0:
        return generated_tokens / generation_tps
    return None


def _build_metrics_envelope(
    *,
    endpoint: str,
    model: str,
    stream: bool,
    backend: str,
    prompt_tokens: int,
    completion_tokens: int,
    generated_tokens: int,
    request_elapsed_s: float,
    request_started_s: float,
    token_times: Optional[List[float]] = None,
    prompt_tps: Optional[float] = None,
    generation_tps: Optional[float] = None,
    peak_memory_gb: Optional[float] = None,
    finish_reason: Optional[str] = None,
    image_count: int = 0,
    audio_count: int = 0,
    structured_output: bool = False,
    thinking_enabled: bool = False,
    tool_parser: Optional[str] = None,
    tool_calls: bool = False,
) -> dict:
    token_times = token_times or []
    ttft_s = max(0.0, token_times[0] - request_started_s) if token_times else None
    decode_elapsed_s = _decode_elapsed_from_metrics(
        token_times, generated_tokens, generation_tps
    )
    decode_tok_s = None
    if generation_tps is not None and generation_tps > 0:
        decode_tok_s = generation_tps
    elif decode_elapsed_s is not None and decode_elapsed_s > 0 and generated_tokens > 0:
        decode_tok_s = generated_tokens / decode_elapsed_s
    prompt_eval_time_s = _prompt_eval_time_from_tps(prompt_tokens, prompt_tps)
    request_tok_s = (
        completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0
    )
    return {
        "timestamp_unix": time.time(),
        "endpoint": endpoint,
        "model": model,
        "stream": bool(stream),
        "backend": backend,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "generated_tokens": int(generated_tokens),
        "reasoning_tokens": max(0, int(generated_tokens) - int(completion_tokens)),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "prompt_eval_time_s": prompt_eval_time_s,
        "prefill_tok_s": prompt_tps,
        "ttft_s": ttft_s,
        "decode_elapsed_s": decode_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "request_tok_s": request_tok_s,
        "decode_tok_s": decode_tok_s,
        "sliding_decode_tok_s_first_32": _token_window_rate_first(token_times, 32),
        "sliding_decode_tok_s_first_64": _token_window_rate_first(token_times, 64),
        "sliding_decode_tok_s_last_32": _token_window_rate(token_times, 32),
        "sliding_decode_tok_s_last_64": _token_window_rate(token_times, 64),
        "peak_memory_gb": peak_memory_gb,
        "finish_reason": finish_reason,
        "image_count": int(image_count),
        "audio_count": int(audio_count),
        "structured_output": bool(structured_output),
        "thinking_enabled": bool(thinking_enabled),
        "tool_parser": tool_parser,
        "tool_calls": bool(tool_calls),
        "apc_enabled": apc_manager is not None,
    }


def _server_runtime_snapshot() -> dict:
    config = model_cache.get("config")
    text_config = getattr(config, "text_config", None)
    native_context_size = getattr(text_config, "max_position_embeddings", None)
    configured_context_limit = get_configured_context_limit()
    effective_context_limit = (
        min(native_context_size, configured_context_limit)
        if native_context_size is not None and configured_context_limit is not None
        else configured_context_limit or native_context_size
    )
    queue_depth = 0
    if response_generator is not None and hasattr(response_generator, "requests"):
        try:
            queue_depth = response_generator.requests.qsize()
        except Exception:
            queue_depth = 0
    return {
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
        "loaded_context_size": native_context_size,
        "configured_context_limit": configured_context_limit,
        "effective_context_limit": effective_context_limit,
        "loaded_tool_parser": (
            _infer_tool_parser_from_processor(model_cache.get("processor"))
            if model_cache.get("processor")
            else None
        ),
        "continuous_batching_enabled": response_generator is not None,
        "request_queue_depth": queue_depth,
        "apc": (
            {"enabled": False}
            if apc_manager is None
            else {"enabled": True, **apc_manager.stats_snapshot()}
        ),
    }


# =============================================================================
# ResponseGenerator - Concurrent Request Handling with Threaded Batching
# =============================================================================


@dataclass
class GenerationArguments:
    """Arguments for a generation request."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None
    # Per-tenant salt for APC. When set, it's mixed into ``extra_hash`` so
    # cached blocks from one tenant can't be reused (or detected via timing)
    # by another. None = no salt = single-tenant behaviour.
    tenant_id: Optional[str] = None

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs dict for generate()/stream_generate()."""
        kw = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "enable_thinking": self.enable_thinking,
        }
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        if self.logit_bias is not None:
            kw["logit_bias"] = self.logit_bias
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.logits_processors is not None:
            kw["logits_processors"] = self.logits_processors
        if self.tenant_id is not None:
            kw["apc_tenant"] = self.tenant_id
        return kw

    def to_template_kwargs(self) -> dict:
        """Convert to kwargs for apply_chat_template()."""
        kw = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        return kw


@dataclass
class GenerationContext:
    """Context returned when a request is queued."""

    uid: int
    prompt_tokens: int


@dataclass
class StreamingToken:
    """A single token response during streaming generation."""

    text: str
    token: int
    logprobs: float
    finish_reason: Optional[str]
    peak_memory: float = 0.0
    prompt_tps: Optional[float] = None
    top_logprobs: Optional[List[Tuple[int, float]]] = None


class ResponseGenerator:
    """
    Continuous batching for concurrent requests via a single GPU thread.

    A dedicated thread owns all GPU work (BatchGenerator). FastAPI async
    handlers submit requests to a queue and read tokens back from
    per-request queues. Multiple requests are batched together for
    higher throughput — same pattern as mlx-lm's server.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        vision_cache=None,
        kv_bits=None,
        kv_group_size=DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start=DEFAULT_QUANTIZED_KV_START,
        top_logprobs_k=0,
        apc_manager: Optional["_apc.APCManager"] = None,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.config = None
        self.stop_tokens = set()
        self.vision_cache = vision_cache
        self.draft_model = None
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.top_logprobs_k = top_logprobs_k
        self.apc_manager = apc_manager
        self.tokenizer = None
        self.requests: Queue = Queue()
        self._stop = False
        self._ready = Event()
        self._load_error: Optional[Exception] = None
        self._cancelled: set = set()
        self._cancel_lock = Lock()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_join(self):
        self._stop = True
        self.requests.put(None)
        self._thread.join(timeout=5.0)

    def wait_until_ready(self, timeout: Optional[float] = None):
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for generation thread to load model.")
        if self._load_error is not None:
            raise self._load_error
        return self.model, self.processor, self.config

    def _cancel(self, uid):
        with self._cancel_lock:
            self._cancelled.add(uid)

    def _drain_cancellations(self) -> set:
        with self._cancel_lock:
            pending, self._cancelled = self._cancelled, set()
            return pending

    def _initialize_model(self):
        model, processor, config = load_model_resources(
            self.model_path, self.adapter_path
        )

        stop_tokens = set()
        if hasattr(config, "eos_token_id"):
            if isinstance(config.eos_token_id, list):
                stop_tokens.update(config.eos_token_id)
            elif config.eos_token_id is not None:
                stop_tokens.add(config.eos_token_id)

        draft_model = None
        draft_kind = os.environ.get("MLX_VLM_DRAFT_KIND")
        draft_model_path = os.environ.get("MLX_VLM_DRAFT_MODEL")
        if draft_model_path:
            from .speculative.drafters import load_drafter

            print(
                f"Loading speculative drafter ({draft_kind or 'auto'}): "
                f"{draft_model_path}"
            )
            draft_model, resolved_kind = load_drafter(draft_model_path, kind=draft_kind)
            if draft_kind is None:
                print(f"  → auto-detected --draft-kind={resolved_kind!r}.")
            elif resolved_kind != draft_kind:
                print(
                    f"  → drafter requires --draft-kind={resolved_kind!r}; "
                    f"using {resolved_kind!r} instead of {draft_kind!r}."
                )
            draft_kind = resolved_kind
            print("Drafter ready — speculative decoding enabled.")

        self.model = model
        self.processor = processor
        self.config = config
        self.stop_tokens = stop_tokens
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    def generate(
        self,
        prompt: str,
        images: Optional[List] = None,
        audio: Optional[List] = None,
        args: Optional[GenerationArguments] = None,
    ) -> Tuple[GenerationContext, Iterator[StreamingToken]]:
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        if self.draft_model is not None and args.logits_processors is not None:
            raise ValueError(
                "Structured response_format is not supported with speculative decoding."
            )
        rqueue: Queue = Queue()

        # CPU preprocessing (tokenize, load images) on caller thread.
        # GPU work (vision encoder) deferred to GPU thread.
        raw_inputs = self._cpu_preprocess(prompt, images, audio)
        prompt_tokens = _count_prompt_tokens(raw_inputs)
        _check_configured_context_budget(prompt_tokens, args.max_tokens)

        self.requests.put((rqueue, raw_inputs, prompt_tokens, args, images))

        # Block until the GPU thread sends back the context
        ctx = rqueue.get()
        if isinstance(ctx, Exception):
            raise ctx

        uid = ctx.uid

        def token_iterator():
            # Mark ended before yielding the final token so a consumer that
            # closes immediately after seeing finish_reason isn't treated
            # as a client abort.
            ended = False
            queue_timeout = get_token_queue_timeout()
            try:
                while True:
                    try:
                        item = rqueue.get(timeout=queue_timeout)
                    except QueueEmpty as exc:
                        timeout_label = (
                            "without a timeout"
                            if queue_timeout is None
                            else f"for {queue_timeout:g}s"
                        )
                        raise RuntimeError(
                            "Timed out waiting "
                            f"{timeout_label} for the next generated token. "
                            "Increase MLX_VLM_TOKEN_QUEUE_TIMEOUT for long "
                            "prefills, or reduce the prompt size."
                        ) from exc
                    if item is None:
                        ended = True
                        break
                    if isinstance(item, Exception):
                        ended = True
                        raise item
                    if getattr(item, "finish_reason", None):
                        ended = True
                    yield item
                    if ended:
                        break
            finally:
                if not ended:
                    self._cancel(uid)

        return ctx, token_iterator()

    def _cpu_preprocess(self, prompt, images=None, audio=None) -> dict:
        """CPU-only: tokenize text, load/resize images. Thread-safe."""
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if self.model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)
        return prepare_inputs(
            self.processor,
            images=images,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )

    # -- internals --

    def _make_sampler(self, args: GenerationArguments) -> Optional[Callable]:
        if args.temperature == 0:
            return None

        def sampler(logprobs: mx.array) -> mx.array:
            if args.top_p > 0 and args.top_p < 1.0:
                return top_p_sampling(logprobs, args.top_p, args.temperature)
            else:
                return mx.random.categorical(logprobs * (1 / args.temperature))

        return sampler

    def _gpu_embed(self, raw_inputs: dict, images=None) -> Tuple[mx.array, dict]:
        """GPU-only: run vision encoder if needed. Must run on GPU thread."""
        input_ids = raw_inputs.get("input_ids")
        pixel_values = raw_inputs.get("pixel_values")
        mask = raw_inputs.get("attention_mask")
        data_kwargs = {
            k: v
            for k, v in raw_inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        # Pass vision cache for image feature caching
        if (
            pixel_values is not None
            and self.vision_cache is not None
            and images is not None
        ):
            data_kwargs["vision_cache"] = self.vision_cache
            data_kwargs["_image_key"] = images

        # Always call get_input_embeddings — BatchGenerator requires inputs_embeds
        embed = self.model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **data_kwargs
        )
        # Remove cache kwargs before passing to BatchGenerator
        data_kwargs.pop("vision_cache", None)
        data_kwargs.pop("_image_key", None)
        gen_kwargs = {**data_kwargs, **embed.to_dict()}
        if images is not None:
            gen_kwargs["_apc_image_hash"] = _apc.hash_image_payload(image_ref=images)
        elif pixel_values is not None:
            gen_kwargs["_apc_image_hash"] = _apc.hash_image_payload(
                pixel_values=pixel_values
            )
        return input_ids, gen_kwargs

    def _collect_pending_requests(
        self,
        *,
        active: bool,
        idle_timeout: float = 0.1,
        coalesce_s: float = 0.0,
    ):
        """Collect the first queued request, then drain immediately available peers."""
        pending = []
        should_stop = False

        def append_item(item):
            nonlocal should_stop
            if item is None:
                if self._stop and not pending:
                    should_stop = True
                return
            pending.append(item)

        try:
            if active:
                append_item(self.requests.get_nowait())
            else:
                append_item(self.requests.get(timeout=idle_timeout))
        except QueueEmpty:
            pass

        if pending and coalesce_s > 0:
            time.sleep(coalesce_s)

        while not should_stop:
            try:
                append_item(self.requests.get_nowait())
            except QueueEmpty:
                break

        return pending, should_stop

    def _run(self):
        """Single GPU thread: owns BatchGenerator, runs tight next() loop."""
        try:
            self._initialize_model()
        except Exception as e:
            self._load_error = e
            self._ready.set()
            print(f"Error loading model in generation thread: {e}")
            traceback.print_exc()
            return

        self._ready.set()

        if self.draft_model is not None:
            self._run_speculative()
            return

        generation_stream = mx.default_stream(mx.default_device())

        batch_gen = None
        # uid -> {rqueue, tokens, gen_kwargs}
        active: dict = {}

        while not self._stop:
            try:
                # Poll the request queue — non-blocking when generating, short
                # blocking wait when idle so we don't spin.
                new_items, should_stop = self._collect_pending_requests(
                    active=bool(active)
                )
                if should_stop:
                    break

                # Drop abandoned requests before doing more work.
                cancelled = self._drain_cancellations()
                if cancelled and batch_gen is not None:
                    for uid in cancelled:
                        if uid in active:
                            batch_gen.remove(uid)
                            info = active.pop(uid)
                            try:
                                info["rqueue"].put(None)
                            except Exception:
                                pass

                for rqueue, raw_inputs, prompt_tokens, args, images in new_items:
                    if batch_gen is None:
                        batch_gen = BatchGenerator(
                            self.model.language_model,
                            self.processor,
                            stop_tokens=self.stop_tokens,
                            sampler=self._make_sampler(args),
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            kv_quant_scheme=self.kv_quant_scheme,
                            quantized_kv_start=self.quantized_kv_start,
                            top_logprobs_k=self.top_logprobs_k,
                            stream=generation_stream,
                            apc_manager=self.apc_manager,
                        )

                    # Vision encoder runs on the GPU thread; text tokenization
                    # already happened on the caller thread.
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    has_embeds = bool(gen_kwargs.get("inputs_embeds") is not None)
                    # Per-tenant APC salt: keep this out of the model forward
                    # by namespacing under "_apc_tenant"; BatchGenerator strips
                    # it before merging kwargs for the language model.
                    if getattr(args, "tenant_id", None):
                        gen_kwargs["_apc_tenant"] = args.tenant_id

                    # Drain pending text-only prompts before inserting an
                    # embed-bearing request — multi-row PromptProcessingBatch
                    # admission expects all rows to carry inputs_embeds (the
                    # mixed APC path concatenates them per-row).
                    if has_embeds and any(
                        not (s[3] and s[3].get("inputs_embeds") is not None)
                        for s in batch_gen.unprocessed_prompts
                    ):
                        self._flush(batch_gen, active)

                    try:
                        (uid,) = batch_gen.insert(
                            [input_ids.squeeze(0).tolist()],
                            max_tokens=args.max_tokens,
                            prompt_kwargs=[gen_kwargs],
                            logits_processors=[args.logits_processors],
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    active[uid] = {
                        "rqueue": rqueue,
                        "streamer": _ServerTokenStreamer(
                            self.tokenizer,
                            make_streaming_detokenizer(self.processor),
                        ),
                        "gen_kwargs": gen_kwargs if has_embeds else None,
                        "prompt_tps": None,
                    }

                if not active or batch_gen is None:
                    continue

                self._step(batch_gen, active)

            except Exception as e:
                logger.exception("Error in generation thread")
                for info in list(active.values()):
                    try:
                        info["rqueue"].put(e)
                        info["rqueue"].put(None)
                    except Exception:
                        pass
                active.clear()
                batch_gen = None
                mx.clear_cache()
                gc.collect()

    def _run_speculative(self):
        """GPU thread loop with DFlash, EAGLE-3, or MTP speculative decoding.

        Collects incoming requests, prefills them as a batch with the
        per-family hooks, then runs the matching round-loop for decode.
        Finished sequences are filtered out automatically by the round-loop's
        ``stop_check`` callback.
        """
        from mlx_lm.sample_utils import make_sampler as _make_sampler

        generation_stream = mx.default_stream(mx.default_device())

        lm = self.model.language_model
        drafter = self.draft_model
        draft_kind = self.draft_kind
        is_mtp = draft_kind == "mtp"
        prefill_kwargs = speculative_prefill_kwargs(draft_kind, drafter)
        eos_set = set(self.stop_tokens) if is_mtp else None
        sampler = _make_sampler(temp=0)
        draft_block_size = _get_draft_block_size_from_env()

        while not self._stop:
            pending = []
            rqueues = {}
            try:
                # --- Phase 1: collect pending requests ---
                pending, should_stop = self._collect_pending_requests(
                    active=False,
                    coalesce_s=get_speculative_batch_coalesce_s(),
                )
                if should_stop:
                    break

                if not pending:
                    continue
                # --- Phase 2: prefill new batch ---
                uids = []
                rqueues = {}
                token_lists = {}
                stream_infos = {}
                max_tokens_map = {}
                prompt_tokens_map = {}
                prompt_tps_map = {}
                all_input_ids = []
                prompt_kwargs_list = []

                if hasattr(lm, "_position_ids"):
                    lm._position_ids = None
                if hasattr(lm, "_rope_deltas"):
                    lm._rope_deltas = None

                for rqueue, raw_inputs, prompt_tokens, args, images in pending:
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    uid = id(rqueue)
                    uids.append(uid)
                    rqueues[uid] = rqueue
                    token_lists[uid] = []
                    stream_infos[uid] = {
                        "streamer": _ServerTokenStreamer(
                            self.tokenizer,
                            make_streaming_detokenizer(self.processor),
                        )
                    }
                    max_tokens_map[uid] = args.max_tokens
                    prompt_tokens_map[uid] = prompt_tokens
                    all_input_ids.append(input_ids.squeeze(0).tolist())
                    prompt_kwargs_list.append(gen_kwargs)
                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    sampler = self._make_sampler(args) or _make_sampler(temp=0)

                B = len(uids)
                max_len = max(len(ids) for ids in all_input_ids)
                left_padding = [max_len - len(ids) for ids in all_input_ids]
                padded = [
                    [0] * left_padding[i] + ids for i, ids in enumerate(all_input_ids)
                ]
                input_mx = mx.array(padded, dtype=mx.int32)

                inputs_embeds_mx, prompt_kwargs = _merge_prefill_prompt_kwargs(
                    prompt_kwargs_list, all_input_ids
                )

                prompt_cache = make_speculative_prompt_cache(
                    lm,
                    draft_kind=draft_kind,
                    batch_size=B,
                    left_padding=left_padding,
                    make_cache=_make_cache,
                )

                lm_call_kwargs = {**prefill_kwargs, **prompt_kwargs}
                lm_call_kwargs["inputs_embeds"] = inputs_embeds_mx

                prompt_started = time.perf_counter()
                with mx.stream(generation_stream):
                    out = lm(input_mx, cache=prompt_cache, **lm_call_kwargs)
                hidden = speculative_hidden_state(draft_kind, out)
                shared_kv_states = out.shared_kv_states if is_mtp else None
                first_bonus = sampler(out.logits[:, -1:]).squeeze(-1)
                mx.eval(first_bonus, hidden, out.logits)
                prompt_elapsed = time.perf_counter() - prompt_started
                for uid in uids:
                    prompt_tokens = prompt_tokens_map[uid]
                    prompt_tps_map[uid] = (
                        prompt_tokens / prompt_elapsed
                        if prompt_tokens > 0 and prompt_elapsed > 0
                        else None
                    )

                finished_uids = set()

                # Send first bonus tokens to clients
                fb_list = first_bonus.tolist()
                for j, uid in enumerate(uids):
                    tok = int(fb_list[j])
                    token_lists[uid].append(tok)
                    is_stop = tok in self.stop_tokens
                    is_max = len(token_lists[uid]) >= max_tokens_map[uid]
                    finish = "stop" if is_stop else "length" if is_max else None
                    text = self._stream_text(stream_infos[uid], tok, finish)
                    rqueues[uid].put(
                        StreamingToken(
                            text=text,
                            token=tok,
                            logprobs=0.0,
                            finish_reason=finish,
                            peak_memory=mx.get_peak_memory() / 1e9,
                            prompt_tps=prompt_tps_map.get(uid),
                        )
                    )
                    if finish is not None:
                        rqueues[uid].put(None)
                        finished_uids.add(uid)

                if len(finished_uids) == len(uids):
                    continue

                # --- Phase 3: speculative decode rounds ---
                max_tok = max(max_tokens_map[u] for u in uids)

                def stop_check(seq_idx, token_id):
                    uid = uids[seq_idx]
                    if uid in finished_uids:
                        return True
                    if token_id in self.stop_tokens:
                        return True
                    if len(token_lists[uid]) >= max_tokens_map[uid]:
                        return True
                    return False

                rounds_iter = run_speculative_server_rounds(
                    self.model,
                    drafter,
                    prompt_cache,
                    hidden,
                    draft_kind=draft_kind,
                    first_bonus=first_bonus,
                    max_tokens=max_tok,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=mx.int32,
                    stop_check=stop_check,
                    greedy_sampling=all(
                        pending_args.temperature == 0
                        for _, _, _, pending_args, _ in pending
                    ),
                    shared_kv_states=shared_kv_states,
                    eos_token_ids=eos_set,
                    prompt_tokens=input_mx,
                )
                for tok_list, _ in rounds_iter:
                    for j, tok in enumerate(tok_list):
                        if tok is None:
                            continue
                        uid = uids[j]
                        if uid in finished_uids:
                            continue

                        token_lists[uid].append(tok)
                        tokens = token_lists[uid]

                        is_stop = tok in self.stop_tokens
                        is_max = len(tokens) >= max_tokens_map[uid]
                        finish = "stop" if is_stop else "length" if is_max else None
                        text = self._stream_text(stream_infos[uid], tok, finish)

                        rqueues[uid].put(
                            StreamingToken(
                                text=text,
                                token=tok,
                                logprobs=0.0,
                                finish_reason=finish,
                                peak_memory=mx.get_peak_memory() / 1e9,
                                prompt_tps=prompt_tps_map.get(uid),
                            )
                        )

                        if finish is not None:
                            rqueues[uid].put(None)
                            finished_uids.add(uid)
                    if len(finished_uids) == len(uids):
                        break

                # Log acceptance stats
                al = drafter.accept_lens
                if al:
                    mean_a = (sum(al) + len(al)) / len(al)
                    print(
                        f"[{draft_kind.upper()}] batch={B} "
                        f"tokens={sum(len(token_lists[u]) for u in uids)} "
                        f"accept={mean_a:.2f} rounds={len(al)}"
                    )

                # Finalize any remaining
                for uid in uids:
                    if uid not in finished_uids:
                        text = stream_infos[uid]["streamer"].finalize()
                        rqueues[uid].put(
                            StreamingToken(
                                text=text,
                                token=0,
                                logprobs=0.0,
                                finish_reason="length",
                                peak_memory=mx.get_peak_memory() / 1e9,
                                prompt_tps=prompt_tps_map.get(uid),
                            )
                        )
                        rqueues[uid].put(None)

            except Exception as e:
                print(f"Error in speculative generation thread: {e}")
                traceback.print_exc()
                error_queues = {id(rqueue): rqueue for rqueue in rqueues.values()}
                error_queues.update({id(rqueue): rqueue for rqueue, *_ in pending})
                for rqueue in error_queues.values():
                    rqueue.put(e)
                    rqueue.put(None)

    def _step(self, batch_gen, active, gen_kwargs=None):
        """One batch generation step: prefill + decode."""
        kwargs = gen_kwargs or {}
        prompt_responses, responses = batch_gen.next(**kwargs)
        for prompt_response in prompt_responses:
            if prompt_response.uid in active:
                active[prompt_response.uid]["prompt_tps"] = prompt_response.prompt_tps
        if not responses:
            return

        for r in responses:
            if r.uid not in active:
                continue

            info = active[r.uid]
            rqueue = info["rqueue"]

            tok = r.token
            if hasattr(tok, "item"):
                tok = tok.item()

            text = self._stream_text(info, tok, r.finish_reason)

            lp = r.token_logprob

            rqueue.put(
                StreamingToken(
                    text=text,
                    token=tok,
                    logprobs=lp,
                    finish_reason=r.finish_reason,
                    peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
                    prompt_tps=info.get("prompt_tps"),
                    top_logprobs=getattr(r, "top_logprobs", None),
                )
            )

            if r.finish_reason is not None:
                rqueue.put(None)
                del active[r.uid]

    def _stream_text(self, info: dict, token: int, finish_reason: Optional[str]) -> str:
        """Convert one generated token into a streaming text segment."""
        return info["streamer"].advance(token, finish_reason)

    def _flush(self, batch_gen, active):
        """Drain all pending text-only prompts before inserting an image request."""
        while batch_gen.has_pending_prompts:
            self._step(batch_gen, active)

    def validate_context_budget(
        self,
        prompt: str,
        images: Optional[List] = None,
        audio: Optional[List] = None,
        args: Optional[GenerationArguments] = None,
    ):
        """Validate request size before opening a streaming response."""
        if get_configured_context_limit() is None:
            return
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        raw_inputs = self._cpu_preprocess(prompt, images, audio)
        _check_configured_context_budget(
            _count_prompt_tokens(raw_inputs), args.max_tokens
        )


def suppress_tool_call_content(
    full_output: str,
    in_tool_call: bool,
    tc_start: Optional[str],
    delta_content: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Suppress tool-call markup from streamed delta.content.

    Returns updated (in_tool_call, delta_content).
    """
    if not tc_start:
        return in_tool_call, delta_content
    if not in_tool_call:
        if tc_start in full_output:
            return True, None

        if any(full_output.endswith(tc_start[:j]) for j in range(2, len(tc_start))):
            return False, None
    else:
        return True, None
    return in_tool_call, delta_content


def process_tool_calls(model_output: str, tool_module, tools):
    """Parse tool calls from model output using the appropriate tool parser."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for i, match in enumerate(matches):
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    args = tool_call["arguments"]
                    called_tools.append(
                        {
                            "type": "function",
                            "index": i,
                            "id": str(uuid.uuid4()),
                            "function": {
                                "name": tool_call["name"].strip(),
                                "arguments": (
                                    args
                                    if isinstance(args, str)
                                    else json.dumps(args, ensure_ascii=False)
                                ),
                            },
                        }
                    )
                except Exception:
                    print(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


def _build_gen_args(
    request, processor=None, tenant_id: Optional[str] = None
) -> GenerationArguments:
    """Build GenerationArguments from an OpenAIRequest or ChatRequest."""
    max_tokens = getattr(request, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(request, "max_output_tokens", None)
    if max_tokens is None:
        max_tokens = get_server_max_tokens()
    logit_bias = getattr(request, "logit_bias", None)
    if logit_bias is not None and isinstance(logit_bias, dict):
        logit_bias = {int(k): v for k, v in logit_bias.items()}
    enable_thinking = _request_field_or_default(
        request,
        "enable_thinking",
        get_server_enable_thinking(),
    )
    args = GenerationArguments(
        max_tokens=max_tokens,
        temperature=getattr(request, "temperature", DEFAULT_TEMPERATURE),
        top_p=getattr(request, "top_p", DEFAULT_TOP_P),
        top_k=getattr(request, "top_k", 0),
        min_p=getattr(request, "min_p", 0.0),
        repetition_penalty=getattr(request, "repetition_penalty", None),
        logit_bias=logit_bias,
        enable_thinking=enable_thinking,
        thinking_budget=getattr(request, "thinking_budget", None),
        thinking_start_token=getattr(request, "thinking_start_token", None),
        tenant_id=tenant_id,
    )
    if processor is not None:
        args.logits_processors = _build_structured_logits_processors(request, processor)
    return args


def _request_field_or_default(request, field_name: str, default):
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None and field_name not in fields_set:
        return default
    value = getattr(request, field_name, default)
    return default if value is None else value


def _read_tenant_id(http_request) -> Optional[str]:
    """Pull a per-tenant APC salt from the request headers.

    Honoured headers (in order): ``X-APC-Tenant``, ``X-Tenant-Id``.
    """
    if http_request is None or not hasattr(http_request, "headers"):
        return None
    h = http_request.headers
    return h.get("x-apc-tenant") or h.get("x-tenant-id") or None


async def _preflight_stream_context_budget(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    images: Optional[List] = None,
    audio: Optional[List] = None,
    args: GenerationArguments,
):
    """Reject over-budget streaming requests before the HTTP stream starts."""
    if response_generator is None:
        return
    try:
        await asyncio.to_thread(
            response_generator.validate_context_budget,
            prompt,
            images,
            audio,
            args,
        )
    except PromptTooLongError as e:
        server_metrics.record_failure(
            endpoint=endpoint,
            model=model,
            stream=True,
            error=str(e),
        )
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=400, detail=str(e))


def _as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _extract_response_format_schema(request) -> Optional[Union[str, dict]]:
    response_format = _as_plain_dict(getattr(request, "response_format", None))

    text_config = _as_plain_dict(getattr(request, "text", None))
    if response_format is None and isinstance(text_config, dict):
        response_format = _as_plain_dict(text_config.get("format"))

    if response_format is None:
        return None

    format_type = response_format.get("type")
    if format_type in (None, "text"):
        return None
    if format_type != "json_schema":
        raise ValueError(f"Unsupported response_format type: {format_type!r}")

    json_schema = _as_plain_dict(response_format.get("json_schema"))
    if json_schema is None:
        # Responses API text.format places schema directly on the format object.
        json_schema = response_format

    schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
    if schema is None:
        raise ValueError("response_format json_schema must include a schema field")
    return schema


def _build_structured_logits_processors(request, processor):
    schema = _extract_response_format_schema(request)
    if schema is None:
        return None

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    logits_processor = build_json_schema_logits_processor(tokenizer, schema)
    return [logits_processor]


def _count_thinking_tag_tokens(text: str) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _split_thinking(text: str) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content)."""
    # Handle <|channel>thought...<channel|> format (gemma4)
    # Also handle partial tag: text starting with "thought\n" (continuation)
    if "<|channel>thought" in text or (
        "<channel|>" in text and text.lstrip().startswith("thought")
    ):
        parts = text.split("<channel|>", 1)
        if len(parts) == 2:
            reasoning = (
                parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
            )
            content = parts[1].strip()
            return reasoning or None, content
        reasoning = parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
        return reasoning or None, ""
    # Handle <think>...</think> format (qwen3.5 etc)
    # Also handle partial: output starts with thinking text + </think> (no opening tag)
    if "<think>" in text or "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
            return reasoning or None, content
        return parts[0].replace("<think>", "").strip(), ""
    return None, text


def _decode_token(tokenizer, token_id: int) -> Tuple[str, Optional[List[int]]]:
    """Decode a single token id to its string + UTF-8 bytes."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    try:
        token_bytes = list(text.encode("utf-8"))
    except Exception:
        token_bytes = None
    return text, token_bytes


def _make_logprob_content(
    tokenizer,
    token_id: int,
    logprob: float,
    top_logprobs: Optional[List[Tuple[int, float]]] = None,
    top_k: int = 0,
) -> "ChatLogprobContent":
    """Build an OpenAI-style logprob entry for a single token."""
    token_text, token_bytes = _decode_token(tokenizer, token_id)
    top_list: List[TopLogprob] = []
    if top_k > 0 and top_logprobs:
        for tid, lp in top_logprobs[:top_k]:
            t_text, t_bytes = _decode_token(tokenizer, tid)
            top_list.append(TopLogprob(token=t_text, logprob=float(lp), bytes=t_bytes))
    return ChatLogprobContent(
        token=token_text,
        logprob=float(logprob),
        bytes=token_bytes,
        top_logprobs=top_list,
    )


# Global response generator for continuous batching
response_generator: Optional[ResponseGenerator] = None

# Global APC manager (shared across requests for the loaded model)
apc_manager: Optional[_apc.APCManager] = None


# Loading/unloading utilities
model_cache = {}
server_metrics = ServerMetricsStore()


@asynccontextmanager
async def lifespan(app):
    model_path = os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
    if model_path:
        adapter_path = os.environ.pop("MLX_VLM_PRELOAD_ADAPTER", None)
        logger.info("Pre-loading model: %s", model_path)
        get_cached_model(model_path, adapter_path)
        kv_bits = os.environ.get("KV_BITS")
        kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
        if kv_bits:
            logger.info("KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme)
        logger.info("Model ready, continuous batching enabled.")
    yield


app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        print(f"Loading model from: {model_path}")
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        print("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


_INHERIT_ADAPTER = object()


def get_cached_model(model_path: str, adapter_path=_INHERIT_ADAPTER):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    Also creates/updates the ResponseGenerator for continuous batching.
    """
    global model_cache, response_generator, apc_manager

    if adapter_path is _INHERIT_ADAPTER:
        cached = model_cache.get("cache_key")
        adapter_path = cached[1] if cached and cached[0] == model_path else None

    cache_key = (model_path, adapter_path)

    # Return from cache if already loaded and matches the requested paths
    if model_cache.get("cache_key") == cache_key:
        print(f"Using cached model: {model_path}, Adapter: {adapter_path}")
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If cache exists but doesn't match, clear it
    if model_cache:
        print("New model request, clearing existing cache...")
        unload_model_sync()  # Use a synchronous version for internal call

    vision_cache_size = int(os.environ.get("MLX_VLM_VISION_CACHE_SIZE", "20"))
    vision_cache = VisionFeatureCache(max_size=vision_cache_size)

    # APC: build a shared block pool if opted in via env var.
    apc_manager = _apc.from_env(model_namespace=model_path)

    # KV cache quantization (uniform or TurboQuant)
    kv_bits = get_quantized_kv_bits(model_path)
    kv_group_size = get_kv_group_size()
    quantized_kv_start = get_quantized_kv_start()
    kv_quant_scheme = get_kv_quant_scheme()

    response_generator = ResponseGenerator(
        model_path=model_path,
        adapter_path=adapter_path,
        vision_cache=vision_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_quant_scheme=kv_quant_scheme,
        quantized_kv_start=quantized_kv_start,
        top_logprobs_k=get_top_logprobs_k(),
        apc_manager=apc_manager,
    )
    try:
        model, processor, config = response_generator.wait_until_ready()
    except Exception:
        response_generator.stop_and_join()
        response_generator = None
        vision_cache.clear()
        raise

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache, response_generator, apc_manager
    if not model_cache:
        return False

    print(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )

    # Stop the ResponseGenerator if running
    if response_generator is not None:
        print("Stopping ResponseGenerator...")
        response_generator.stop_and_join()
        response_generator = None

    # Drop APC blocks for the previous model
    if apc_manager is not None:
        apc_manager.clear()
        apc_manager = None

    # Clear vision cache before dropping references
    if "vision_cache" in model_cache:
        model_cache["vision_cache"].clear()
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


# OpenAI API Models

# Models for /responses endpoint


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["input_text", "text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field(
        "auto", description="The detail level of the image to be sent to the model."
    )
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`. Defaults to `auto`.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Required[str]
    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    """


class InputAudio(TypedDict, total=False):
    data: Required[str]
    format: Required[str]


class ResponseInputAudioParam(TypedDict, total=False):
    type: Required[
        Literal["input_audio"]
    ]  # The type of the input item. Always `input_audio`.
    input_audio: Required[InputAudio]


class ImageUrl(TypedDict, total=False):
    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    type: Required[
        Literal["image_url"]
    ]  # The type of the input item. Always`image_url`.
    image_url: Required[ImageUrl]


ResizeShapeInput: TypeAlias = Union[Tuple[int], Tuple[int, int]]

ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(FlexibleBaseModel):
    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ...,
        description="Role of the message sender.",
    )
    content: Union[
        str,
        None,
        ResponseInputMessageContentListParam,
        ResponseOutputMessageContentList,
    ] = Field(None, description="Content of the message.")
    reasoning: Optional[str] = Field(
        None, description="Thinking/reasoning content (when thinking is enabled)."
    )
    tool_calls: Optional[List[Any]] = Field(
        None, description="Tool calls made by the assistant."
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message is a response to."
    )
    name: Optional[str] = Field(None, description="Name of the tool/function.")


class OpenAIRequest(FlexibleBaseModel):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[Any]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: Optional[bool] = Field(
        None,
        description=(
            "Override server thinking mode for this request. If omitted, the "
            "server default set by --enable-thinking is used."
        ),
    )
    thinking_budget: Optional[int] = Field(None, description="Max thinking tokens.")
    thinking_start_token: Optional[str] = Field(
        None, description="Thinking start token."
    )
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )
    response_format: Optional[Any] = Field(
        None, description="OpenAI-compatible response_format for structured outputs."
    )
    text: Optional[Any] = Field(
        None, description="Responses API text format configuration."
    )
    instructions: Optional[str] = Field(
        None, description="System/developer instructions for this response."
    )
    previous_response_id: Optional[str] = Field(
        None,
        description="ID of a previous response whose input/output items should be included.",
    )
    tools: Optional[List[Any]] = Field(
        None, description="Responses API tool definitions."
    )
    tool_choice: Optional[Any] = Field(None, description="Tool choice policy.")
    store: Optional[bool] = Field(
        True, description="Whether to store this response for later retrieval."
    )


class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this Response")
    object: Literal["response"] = Field(
        ..., description="The object type of this resource - always set to response"
    )
    created_at: int = Field(
        ..., description="Unix timestamp (in seconds) of when this Response was created"
    )
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        ..., description="The status of the response generation"
    )
    error: Optional[OpenAIErrorObject] = Field(
        None,
        description="An error object returned when the model fails to generate a Response",
    )
    instructions: Optional[str] = Field(
        None,
        description="Inserts a system (or developer) message as the first item in the model's context",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a response",
    )
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(
        ..., description="An array of content items generated by the model"
    )
    output_text: Optional[str] = Field(
        None,
        description="SDK-only convenience property containing aggregated text output",
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature between 0 and 2"
    )
    top_p: Optional[float] = Field(
        None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    truncation: Union[Literal["auto", "disabled"], str] = Field(
        "disabled", description="The truncation strategy to use"
    )
    usage: OpenAIUsage = Field(
        ..., description="Token usage details"
    )  # we need the model to return stats
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )
    previous_response_id: Optional[str] = Field(
        None, description="ID of the previous response used for this response."
    )
    store: Optional[bool] = Field(
        True, description="Whether this response is stored for later retrieval."
    )


class BaseStreamEvent(BaseModel):
    type: str


class ContentPartOutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[str] = []


class MessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed"]
    role: str
    content: List[ContentPartOutputText] = []


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: OpenAIResponse


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: OpenAIResponse


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: Any


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: Any


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: OpenAIResponse


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
]


RESPONSE_STORE_LIMIT = int(os.environ.get("MLX_VLM_RESPONSE_STORE_LIMIT", "1024"))


@dataclass
class StoredResponse:
    response: Dict[str, Any]
    input_items: List[Dict[str, Any]]
    output_items: List[Dict[str, Any]]
    previous_response_id: Optional[str] = None


response_store: Dict[str, StoredResponse] = {}
response_store_order: deque = deque()
response_store_lock = Lock()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _sse_event(event_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=_jsonable)}\n\n"


def _normalize_response_input(input_value: Any) -> List[Dict[str, Any]]:
    if isinstance(input_value, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        ]
    if not isinstance(input_value, list):
        raise HTTPException(status_code=400, detail="Invalid input format.")

    items = []
    for item in input_value:
        item = _as_plain_dict(item)
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Invalid input format.")
        item_type = item.get("type")
        if item_type is None and item.get("role") is not None:
            item = {**item, "type": "message"}
        items.append(item)
    return items


def _response_call_to_chat_tool_call(item: Dict[str, Any]) -> Dict[str, Any]:
    call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
    name = item.get("name")
    arguments = item.get("arguments")
    if item.get("type") == "shell_call":
        name = name or "shell"
        action = item.get("action") or {}
        arguments = arguments or json.dumps(action, ensure_ascii=False)
    elif item.get("type") == "apply_patch_call":
        name = name or "apply_patch"
        arguments = arguments or item.get("patch") or item.get("input") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name or "tool", "arguments": arguments},
    }


def _append_response_item_to_prompt(
    item: Dict[str, Any],
    chat_messages: List[Dict[str, Any]],
    images: List[Any],
):
    item_type = item.get("type")
    if item_type == "message":
        role = item.get("role") or "user"
        content = item.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                part = _as_plain_dict(part)
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("input_text", "output_text", "text"):
                    text_parts.append(str(part.get("text", "")))
                elif part_type == "input_image":
                    image = part.get("image_url") or part.get("file_id")
                    if image:
                        images.append(image)
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    images.append(
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else image_url
                    )
            content = "\n".join(p for p in text_parts if p)
        chat_messages.append({"role": role, "content": content or ""})
        return

    if item_type in ("function_call", "shell_call", "apply_patch_call"):
        chat_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_response_call_to_chat_tool_call(item)],
            }
        )
        return

    if item_type in (
        "function_call_output",
        "shell_call_output",
        "apply_patch_call_output",
        "tool_result",
    ):
        output = item.get("output", item.get("content", ""))
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        chat_messages.append(
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("tool_call_id"),
                "content": output,
            }
        )


def _response_chain_items(previous_response_id: Optional[str]) -> List[Dict[str, Any]]:
    if not previous_response_id:
        return []
    chain: List[StoredResponse] = []
    seen = set()
    current_id = previous_response_id
    with response_store_lock:
        while current_id:
            if current_id in seen:
                break
            seen.add(current_id)
            stored = response_store.get(current_id)
            if stored is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Previous response not found: {current_id}",
                )
            chain.append(stored)
            current_id = stored.previous_response_id

    items: List[Dict[str, Any]] = []
    for stored in reversed(chain):
        items.extend(stored.input_items)
        items.extend(stored.output_items)
    return items


def _response_items_to_chat(
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    chat_messages: List[Dict[str, Any]] = []
    images: List[Any] = []
    for item in items:
        _append_response_item_to_prompt(item, chat_messages, images)
    return chat_messages, images


def _store_response(
    response: OpenAIResponse,
    input_items: List[Dict[str, Any]],
    output_items: List[Dict[str, Any]],
    previous_response_id: Optional[str],
):
    if getattr(response, "store", True) is False:
        return
    payload = response.model_dump(exclude_none=True)
    with response_store_lock:
        response_store[response.id] = StoredResponse(
            response=payload,
            input_items=input_items,
            output_items=output_items,
            previous_response_id=previous_response_id,
        )
        response_store_order.append(response.id)
        while len(response_store_order) > RESPONSE_STORE_LIMIT:
            old_id = response_store_order.popleft()
            response_store.pop(old_id, None)


def _response_tool_to_chat_tool(tool: Any) -> Optional[Dict[str, Any]]:
    tool = _as_plain_dict(tool)
    if not isinstance(tool, dict):
        return None
    tool_type = tool.get("type")
    if tool_type == "function" and isinstance(tool.get("function"), dict):
        return tool
    if tool_type == "function":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters") or {},
            },
        }
    if tool_type == "shell":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name") or "shell",
                "description": tool.get("description") or "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    if tool_type == "apply_patch":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name") or "apply_patch",
                "description": tool.get("description") or "Apply a patch to files.",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        }
    return None


def _response_tool_registry(
    tools: Optional[List[Any]],
) -> Tuple[List[Any], Dict[str, str]]:
    chat_tools = []
    registry: Dict[str, str] = {}
    for tool in tools or []:
        plain = _as_plain_dict(tool)
        chat_tool = _response_tool_to_chat_tool(plain)
        if chat_tool is None:
            continue
        chat_tools.append(chat_tool)
        function = chat_tool.get("function", {})
        name = function.get("name")
        if name:
            registry[name] = (plain or {}).get("type", "function")
    return chat_tools, registry


def _tool_call_to_response_item(
    call: Dict[str, Any],
    registry: Dict[str, str],
) -> Dict[str, Any]:
    function = call.get("function", {})
    name = function.get("name") or "tool"
    arguments = function.get("arguments") or "{}"
    call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
    tool_type = registry.get(name, "function")
    if tool_type == "shell":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed = {"command": arguments}
        command = parsed.get("command", parsed) if isinstance(parsed, dict) else parsed
        return {
            "id": f"sh_{uuid.uuid4().hex}",
            "type": "shell_call",
            "call_id": call_id,
            "status": "completed",
            "action": {"type": "exec", "command": command},
        }
    if tool_type == "apply_patch":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed = {"patch": arguments}
        patch = parsed.get("patch", parsed) if isinstance(parsed, dict) else parsed
        return {
            "id": f"apc_{uuid.uuid4().hex}",
            "type": "apply_patch_call",
            "call_id": call_id,
            "status": "completed",
            "patch": patch,
        }
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _response_output_items_from_text(
    full_text: str,
    message_id: str,
    tool_module: Any,
    chat_tools: List[Any],
    tool_registry: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], str, Optional[str], str]:
    reasoning, content = _split_thinking(full_text)
    if tool_module is not None and chat_tools:
        tc = process_tool_calls(full_text, tool_module, chat_tools)
        if tc["calls"]:
            items = [
                _tool_call_to_response_item(call, tool_registry) for call in tc["calls"]
            ]
            _, remaining = _split_thinking(tc.get("remaining_text") or "")
            remaining = re.sub(r"<\|[^>]+\|>|<[^>]+>", "", remaining).strip()
            return items, remaining, reasoning, "tool_calls"
    item = {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    if reasoning:
        item["reasoning"] = reasoning
    return [item], content, reasoning, "stop"


# Models for /chat/completion endpoint


class VLMRequest(FlexibleBaseModel):
    model: str = Field(
        DEFAULT_MODEL_PATH,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    max_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: Optional[bool] = Field(
        None,
        description=(
            "Override server thinking mode for this request. If omitted, the "
            "server default set by --enable-thinking is used."
        ),
    )
    thinking_budget: Optional[int] = Field(None, description="Max thinking tokens.")
    thinking_start_token: Optional[str] = Field(
        None, description="Thinking start token."
    )
    logprobs: Optional[bool] = Field(
        None,
        description="Return log-probabilities for each output token.",
    )
    top_logprobs: Optional[int] = Field(
        None,
        description=(
            "Number of most-likely tokens to return at each position "
            "(0-20). Requires logprobs=true. The server-side cap is set by "
            "the TOP_LOGPROBS_K env var; values above the cap are clamped."
        ),
    )
    resize_shape: Optional[ResizeShapeInput] = Field(
        None,
        description="Resize shape for the image. Provide one integer for square or two for (height, width).",
    )
    response_format: Optional[Any] = Field(
        None, description="OpenAI-compatible response_format for structured outputs."
    )

    @field_validator("resize_shape", mode="before")
    @classmethod
    def normalize_resize_shape_field(cls, value):
        return normalize_resize_shape(value)


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class UsageStats(BaseModel):
    """OpenAI-compatible usage statistics for chat completions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = PromptTokensDetails()
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


class ChatRequest(GenerationRequest):
    messages: List[ChatMessage]


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class ChatLogprobContent(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[TopLogprob] = []


class ChatLogprobs(BaseModel):
    content: List[ChatLogprobContent] = []


class ChatChoice(BaseModel):
    index: int = 0
    finish_reason: str = "stop"
    message: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = []
    usage: Optional[UsageStats] = None


class ChatStreamChoice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatStreamChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[ChatStreamChoice] = []
    usage: Optional[UsageStats] = None


# Models for Anthropic-compatible /v1/messages endpoint


class AnthropicMessageParam(FlexibleBaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Any]]


class AnthropicRequest(FlexibleBaseModel):
    model: str = Field(..., description="The model to use for generation.")
    messages: List[AnthropicMessageParam]
    max_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    system: Optional[Union[str, List[Any]]] = None
    stream: bool = False
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[Any] = None
    thinking: Optional[Any] = None
    output_config: Optional[Any] = None
    adapter_path: Optional[str] = None
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None
    response_format: Optional[Any] = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Any]
    model: str
    stop_reason: Optional[
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# Models for /models endpoint


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[ModelInfo]


def _anthropic_error_response(
    status_code: int, message: str, error_type: str = "invalid_request_error"
):
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
        headers={"request-id": f"req_{uuid.uuid4().hex}"},
    )


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _anthropic_system_text(system: Optional[Union[str, List[Any]]]) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    parts = []
    for item in system:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if text:
                parts.append(str(text))
        elif item is not None:
            parts.append(str(item))
    return "\n".join(parts).strip() or None


def _anthropic_image_source_to_ref(source: Any) -> Optional[str]:
    source = _as_plain_dict(source)
    if not isinstance(source, dict):
        return None
    source_type = source.get("type")
    if source_type == "url":
        return source.get("url")
    if source_type == "base64":
        media_type = source.get("media_type") or "image/png"
        data = source.get("data")
        if data:
            return f"data:{media_type};base64,{data}"
    return None


def _anthropic_tool_result_content_to_openai(
    content: Any, images: Optional[List[str]] = None
) -> Union[str, List[Dict[str, Any]]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        content_parts: List[Dict[str, Any]] = []
        saw_image = False

        def append_text(text: Any) -> None:
            if text:
                text_parts.append(str(text))
                content_parts.append({"type": "text", "text": str(text)})

        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    append_text(item.get("text"))
                elif item_type == "document":
                    source = _as_plain_dict(item.get("source"))
                    if isinstance(source, dict) and source.get("type") == "text":
                        append_text(source.get("data"))
                elif item_type == "image":
                    image_ref = _anthropic_image_source_to_ref(item.get("source"))
                    if image_ref:
                        saw_image = True
                        if images is not None:
                            images.append(image_ref)
                        content_parts.append({"type": "image"})
                elif item.get("content"):
                    append_text(item["content"])
            elif item is not None:
                append_text(item)
        if saw_image:
            return content_parts
        return "\n".join(text_parts).strip()
    return str(content)


def _anthropic_tool_to_openai(tool: Any) -> Optional[Dict[str, Any]]:
    tool = _as_plain_dict(tool)
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    input_schema = tool.get("input_schema")
    if not name or input_schema is None:
        # Anthropic server tools (web_search, code_execution, etc.) cannot be
        # executed by this local server, so they are accepted but not surfaced
        # to model chat templates.
        return None
    function = {
        "name": name,
        "description": tool.get("description", ""),
        "parameters": input_schema,
    }
    if tool.get("strict") is not None:
        function["strict"] = tool.get("strict")
    return {"type": "function", "function": function}


def _anthropic_tools_to_openai(
    tools: Optional[List[Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    converted = []
    for tool in tools:
        converted_tool = _anthropic_tool_to_openai(tool)
        if converted_tool is not None:
            converted.append(converted_tool)
    return converted or None


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Optional[Any]:
    tool_choice = _as_plain_dict(tool_choice)
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type in ("auto", "none"):
        return choice_type
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    return None


def _anthropic_tool_use_to_openai(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": block.get("id") or f"toolu_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
        },
    }


def _openai_tool_call_to_anthropic(call: Any) -> Dict[str, Any]:
    call = _as_plain_dict(call) or {}
    function = _as_plain_dict(call.get("function")) or {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}
    return {
        "type": "tool_use",
        "id": call.get("id") or f"toolu_{uuid.uuid4().hex}",
        "name": function.get("name", ""),
        "input": arguments if isinstance(arguments, dict) else {},
    }


def _anthropic_content_blocks_to_text_and_tools(
    role: str,
    content: Union[str, List[Any]],
    images: List[str],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content, [], []

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    for raw_item in content or []:
        item = _as_plain_dict(raw_item)
        if not isinstance(item, dict):
            if item is not None:
                text_parts.append(str(item))
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if text:
                text_parts.append(str(text))
        elif item_type == "image" and role == "user":
            image_ref = _anthropic_image_source_to_ref(item.get("source"))
            if image_ref:
                images.append(image_ref)
        elif item_type == "tool_use" and role == "assistant":
            tool_calls.append(_anthropic_tool_use_to_openai(item))
        elif item_type == "tool_result" and role == "user":
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("tool_use_id"),
                    "content": _anthropic_tool_result_content_to_openai(
                        item.get("content"), images
                    ),
                    "name": item.get("name"),
                }
            )
        elif item_type in ("thinking", "redacted_thinking"):
            continue
        elif item.get("text"):
            text_parts.append(str(item["text"]))

    return "\n".join(text_parts).strip(), tool_calls, tool_results


def _anthropic_messages_to_internal(
    request: AnthropicRequest,
) -> Tuple[
    List[Dict[str, Any]], List[str], Optional[List[Dict[str, Any]]], Optional[Any]
]:
    images: List[str] = []
    processed_messages: List[Dict[str, Any]] = []

    system_text = _anthropic_system_text(request.system)
    if system_text:
        processed_messages.append({"role": "system", "content": system_text})

    for message in request.messages:
        content_text, tool_calls, tool_results = (
            _anthropic_content_blocks_to_text_and_tools(
                message.role, message.content, images
            )
        )
        if content_text or tool_calls or not tool_results:
            msg: Dict[str, Any] = {"role": message.role, "content": content_text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                if not content_text:
                    msg["content"] = None
            processed_messages.append(msg)
        processed_messages.extend(tool_results)

    tools = _anthropic_tools_to_openai(request.tools)
    tool_choice = _anthropic_tool_choice_to_openai(request.tool_choice)
    return processed_messages, images, tools, tool_choice


def _anthropic_request_with_derived_fields(
    request: AnthropicRequest,
) -> AnthropicRequest:
    thinking = _as_plain_dict(request.thinking)
    if request.enable_thinking is None and isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type in ("enabled", "adaptive"):
            request.enable_thinking = True
        elif thinking_type == "disabled":
            request.enable_thinking = False
    if request.thinking_budget is None and isinstance(thinking, dict):
        budget = thinking.get("budget_tokens")
        if budget is not None:
            request.thinking_budget = int(budget)

    output_config = _as_plain_dict(request.output_config)
    if request.response_format is None and isinstance(output_config, dict):
        fmt = _as_plain_dict(output_config.get("format"))
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            request.response_format = {
                "type": "json_schema",
                "json_schema": {"schema": fmt.get("schema", {})},
            }
    return request


def _anthropic_stop_reason(
    finish_reason: Optional[str],
    tool_calls: bool = False,
    stop_sequence: Optional[str] = None,
) -> str:
    if tool_calls:
        return "tool_use"
    if stop_sequence is not None:
        return "stop_sequence"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"


def _apply_stop_sequences(
    text: str, stop_sequences: Optional[List[str]]
) -> Tuple[str, Optional[str]]:
    if not text or not stop_sequences:
        return text, None
    best_index = None
    best_sequence = None
    for sequence in stop_sequences:
        if not sequence:
            continue
        index = text.find(sequence)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
            best_sequence = sequence
    if best_index is None:
        return text, None
    return text[:best_index], best_sequence


def _anthropic_content_from_generation(
    full_text: str,
    parsed_tool_calls: Optional[List[Any]] = None,
    include_thinking: bool = False,
) -> List[Dict[str, Any]]:
    reasoning, content = _split_thinking(full_text)
    blocks: List[Dict[str, Any]] = []
    if include_thinking and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content:
        blocks.append({"type": "text", "text": content})
    if parsed_tool_calls:
        blocks.extend(
            _openai_tool_call_to_anthropic(call) for call in parsed_tool_calls
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


# Anthropic-compatible endpoints


@app.post("/messages")
@app.post("/v1/messages", include_in_schema=False)
async def anthropic_messages_endpoint(http_request: Request):
    request_start = time.perf_counter()
    try:
        body = await http_request.json()
        request = _anthropic_request_with_derived_fields(AnthropicRequest(**body))
    except Exception as e:
        return _anthropic_error_response(400, f"Invalid request body: {e}")

    try:
        adapter_path = (
            request.adapter_path
            if "adapter_path" in request.model_fields_set
            else _INHERIT_ADAPTER
        )
        model, processor, config = get_cached_model(request.model, adapter_path)

        processed_messages, images, tools, tool_choice = (
            _anthropic_messages_to_internal(request)
        )
        tool_parser_type = _infer_tool_parser_from_processor(processor)
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                request, processor, tenant_id=_read_tenant_id(http_request)
            )
        except Exception as e:
            return _anthropic_error_response(400, str(e))

        template_kwargs = gen_args.to_template_kwargs()
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            tools=tools,
            **template_kwargs,
        )

        logger.debug(
            "anthropic messages request: model=%s images=%d max_tokens=%s "
            "temp=%s stream=%s tools=%d",
            request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
            len(tools or []),
        )

        if request.stream:
            server_metrics.begin_request(
                endpoint="/v1/messages",
                model=request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/v1/messages",
                model=request.model,
                prompt=formatted_prompt,
                images=images if images else None,
                audio=None,
                args=gen_args,
            )

            async def stream_generator():
                token_iterator = None
                token_iter = None
                metrics_finalized = False
                token_times: List[float] = []
                prompt_tps = None
                generation_tps = None
                peak_memory = 0.0
                prompt_tokens = 0
                output_tokens = 0
                finish_reason = None
                message_id = f"msg_{uuid.uuid4().hex}"
                block_index = 0
                open_block_type = None
                full_output = ""
                text_output = ""
                in_thinking = False
                accumulated = ""
                in_tool_call = False
                tc_start = tool_module.tool_call_start if tool_module else None

                def close_open_block():
                    nonlocal open_block_type, block_index
                    if open_block_type is None:
                        return ""
                    event = _sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_index},
                    )
                    open_block_type = None
                    block_index += 1
                    return event

                def open_block(block_type: str):
                    nonlocal open_block_type
                    if open_block_type == block_type:
                        return ""
                    event = close_open_block()
                    content_block: Dict[str, Any]
                    if block_type == "thinking":
                        content_block = {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        }
                    else:
                        content_block = {"type": "text", "text": ""}
                    open_block_type = block_type
                    return event + _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": content_block,
                        },
                    )

                try:
                    if response_generator is not None:
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            None,
                            gen_args,
                        )
                        prompt_tokens = ctx.prompt_tokens

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        token_source = "continuous_batching"
                    else:
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            temperature=request.temperature,
                            max_tokens=gen_args.max_tokens,
                            top_p=request.top_p,
                            vision_cache=model_cache.get("vision_cache"),
                            logits_processors=gen_args.logits_processors,
                            apc_manager=apc_manager,
                            apc_tenant=gen_args.tenant_id,
                        )

                        def _next_token():
                            try:
                                return next(token_iterator)
                            except StopIteration:
                                return None

                        token_source = "generate"

                    start_message = AnthropicMessageResponse(
                        id=message_id,
                        content=[],
                        model=request.model,
                        stop_reason=None,
                        usage={
                            "input_tokens": prompt_tokens,
                            "output_tokens": 0,
                        },
                    )
                    yield _sse_event(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": start_message.model_dump(),
                        },
                    )

                    while True:
                        token = await asyncio.to_thread(_next_token)
                        if token is None:
                            break
                        if not hasattr(token, "text"):
                            continue

                        output_tokens += 1
                        delta = token.text
                        full_output += delta
                        accumulated += delta
                        token_times.append(time.perf_counter())
                        prompt_tps = getattr(token, "prompt_tps", prompt_tps)
                        generation_tps = getattr(
                            token, "generation_tps", generation_tps
                        )
                        peak_memory = max(
                            peak_memory,
                            float(getattr(token, "peak_memory", 0.0) or 0.0),
                        )
                        if prompt_tokens == 0:
                            prompt_tokens = int(getattr(token, "prompt_tokens", 0) or 0)

                        delta_reasoning = None
                        delta_content = None
                        if not in_thinking and (
                            "<|channel>thought" in accumulated
                            or "<think>" in accumulated
                        ):
                            in_thinking = True
                            accumulated = ""
                        elif in_thinking and (
                            "<channel|>" in accumulated or "</think>" in accumulated
                        ):
                            if open_block_type == "thinking":
                                yield _sse_event(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {
                                            "type": "signature_delta",
                                            "signature": "",
                                        },
                                    },
                                )
                            yield close_open_block()
                            in_thinking = False
                            accumulated = ""
                        elif in_thinking:
                            delta_reasoning = delta
                        elif not in_thinking and (
                            "<|channel>" in accumulated or "<think" in accumulated
                        ):
                            pass
                        else:
                            delta_content = delta

                        in_tool_call, delta_content = suppress_tool_call_content(
                            full_output, in_tool_call, tc_start, delta_content
                        )

                        if delta_reasoning is not None and gen_args.enable_thinking:
                            yield open_block("thinking")
                            yield _sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "thinking_delta",
                                        "thinking": delta_reasoning,
                                    },
                                },
                            )
                        elif delta_content:
                            text_output += delta_content
                            yield open_block("text")
                            yield _sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": delta_content,
                                    },
                                },
                            )

                        if getattr(token, "finish_reason", None):
                            finish_reason = token.finish_reason
                            break

                    yield close_open_block()

                    parsed_tool_calls = None
                    if tool_module is not None and tools:
                        tc = process_tool_calls(full_output, tool_module, tools)
                        if tc["calls"]:
                            parsed_tool_calls = tc["calls"]

                    if parsed_tool_calls:
                        for call in parsed_tool_calls:
                            tool_block = _openai_tool_call_to_anthropic(call)
                            input_json = json.dumps(
                                tool_block.get("input") or {}, ensure_ascii=False
                            )
                            yield _sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_block["id"],
                                        "name": tool_block["name"],
                                        "input": {},
                                    },
                                },
                            )
                            if input_json:
                                yield _sse_event(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": input_json,
                                        },
                                    },
                                )
                            yield _sse_event(
                                "content_block_stop",
                                {
                                    "type": "content_block_stop",
                                    "index": block_index,
                                },
                            )
                            block_index += 1

                    stop_sequence = None
                    if not parsed_tool_calls:
                        _, stop_sequence = _apply_stop_sequences(
                            text_output, request.stop_sequences
                        )

                    anth_stop_reason = _anthropic_stop_reason(
                        finish_reason,
                        tool_calls=bool(parsed_tool_calls),
                        stop_sequence=stop_sequence,
                    )
                    yield _sse_event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": anth_stop_reason,
                                "stop_sequence": stop_sequence,
                            },
                            "usage": {"output_tokens": output_tokens},
                        },
                    )
                    yield _sse_event("message_stop", {"type": "message_stop"})

                    completion_tokens = max(
                        0, output_tokens - _count_thinking_tag_tokens(full_output)
                    )
                    envelope = _build_metrics_envelope(
                        endpoint="/v1/messages",
                        model=request.model,
                        stream=True,
                        backend=token_source,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        generated_tokens=output_tokens,
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=token_times,
                        prompt_tps=prompt_tps,
                        generation_tps=generation_tps,
                        peak_memory_gb=peak_memory or None,
                        finish_reason=anth_stop_reason,
                        image_count=len(images),
                        structured_output=bool(gen_args.logits_processors),
                        thinking_enabled=bool(gen_args.enable_thinking),
                        tool_parser=tool_parser_type,
                        tool_calls=bool(parsed_tool_calls),
                    )
                    server_metrics.record_success(envelope)
                    metrics_finalized = True

                except Exception as e:
                    if not metrics_finalized:
                        server_metrics.record_failure(
                            endpoint="/v1/messages",
                            model=request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    yield _sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(e),
                            },
                        },
                    )
                finally:
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        server_metrics.record_failure(
                            endpoint="/v1/messages",
                            model=request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    mx.clear_cache()
                    gc.collect()

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "request-id": f"req_{uuid.uuid4().hex}",
                },
            )

        server_metrics.begin_request(
            endpoint="/v1/messages",
            model=request.model,
            stream=False,
        )
        try:
            full_text = ""
            prompt_tokens = 0
            output_tokens = 0
            peak_memory = 0.0
            token_times: List[float] = []
            prompt_tps = None
            generation_tps = None
            finish_reason = None

            if response_generator is not None:

                def _blocking_generate():
                    text = ""
                    ot = 0
                    tt: List[float] = []
                    ptps = None
                    pm = 0.0
                    fr = None
                    ctx, token_iter = response_generator.generate(
                        prompt=formatted_prompt,
                        images=images if images else None,
                        audio=None,
                        args=gen_args,
                    )
                    for tok in token_iter:
                        text += tok.text
                        ot += 1
                        tt.append(time.perf_counter())
                        ptps = getattr(tok, "prompt_tps", ptps)
                        pm = max(pm, float(getattr(tok, "peak_memory", 0.0) or 0.0))
                        if tok.finish_reason:
                            fr = tok.finish_reason
                            break
                    try:
                        token_iter.close()
                    except Exception:
                        pass
                    return ctx.prompt_tokens, text, ot, tt, ptps, pm, fr

                (
                    prompt_tokens,
                    full_text,
                    output_tokens,
                    token_times,
                    prompt_tps,
                    peak_memory,
                    finish_reason,
                ) = await asyncio.to_thread(_blocking_generate)
            else:
                result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    verbose=logger.isEnabledFor(logging.DEBUG),
                    vision_cache=model_cache.get("vision_cache"),
                    apc_manager=apc_manager,
                    **gen_args.to_generate_kwargs(),
                )
                full_text = result.text
                prompt_tokens = result.prompt_tokens
                output_tokens = result.generation_tokens
                prompt_tps = getattr(result, "prompt_tps", None)
                generation_tps = getattr(result, "generation_tps", None)
                peak_memory = float(getattr(result, "peak_memory", 0.0) or 0.0)
                finish_reason = "stop"

            parsed_tool_calls = None
            response_text = full_text
            if tool_module is not None and tools:
                tc = process_tool_calls(full_text, tool_module, tools)
                if tc["calls"]:
                    parsed_tool_calls = tc["calls"]
                    response_text = tc["remaining_text"] or ""

            response_text, stop_sequence = _apply_stop_sequences(
                response_text, request.stop_sequences
            )
            content_blocks = _anthropic_content_from_generation(
                response_text,
                parsed_tool_calls=parsed_tool_calls,
                include_thinking=bool(gen_args.enable_thinking),
            )
            stop_reason = _anthropic_stop_reason(
                finish_reason,
                tool_calls=bool(parsed_tool_calls),
                stop_sequence=stop_sequence,
            )
            response = AnthropicMessageResponse(
                id=f"msg_{uuid.uuid4().hex}",
                content=content_blocks,
                model=request.model,
                stop_reason=stop_reason,
                stop_sequence=stop_sequence,
                usage={
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                },
            )

            completion_tokens = max(
                0, output_tokens - _count_thinking_tag_tokens(full_text)
            )
            envelope = _build_metrics_envelope(
                endpoint="/v1/messages",
                model=request.model,
                stream=False,
                backend=(
                    "continuous_batching"
                    if response_generator is not None
                    else "generate"
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generated_tokens=output_tokens,
                request_elapsed_s=time.perf_counter() - request_start,
                request_started_s=request_start,
                token_times=token_times,
                prompt_tps=prompt_tps,
                generation_tps=generation_tps,
                peak_memory_gb=peak_memory or None,
                finish_reason=stop_reason,
                image_count=len(images),
                structured_output=bool(gen_args.logits_processors),
                thinking_enabled=bool(gen_args.enable_thinking),
                tool_parser=tool_parser_type,
                tool_calls=bool(parsed_tool_calls),
            )
            server_metrics.record_success(envelope)
            mx.clear_cache()
            gc.collect()
            return response

        except PromptTooLongError as e:
            server_metrics.record_failure(
                endpoint="/v1/messages",
                model=request.model,
                stream=False,
                error=str(e),
            )
            mx.clear_cache()
            gc.collect()
            return _anthropic_error_response(400, str(e))
        except Exception as e:
            server_metrics.record_failure(
                endpoint="/v1/messages",
                model=request.model,
                stream=False,
                error=str(e),
            )
            traceback.print_exc()
            mx.clear_cache()
            gc.collect()
            return _anthropic_error_response(
                500, f"Generation failed: {e}", "api_error"
            )

    except Exception as e:
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        return _anthropic_error_response(500, str(e), "api_error")


@app.post("/messages/count_tokens")
@app.post("/v1/messages/count_tokens", include_in_schema=False)
async def anthropic_count_tokens_endpoint(http_request: Request):
    try:
        body = await http_request.json()
        request = _anthropic_request_with_derived_fields(AnthropicRequest(**body))
        model, processor, config = get_cached_model(request.model)
        processed_messages, images, tools, tool_choice = (
            _anthropic_messages_to_internal(request)
        )
        gen_args = _build_gen_args(
            request, processor, tenant_id=_read_tenant_id(http_request)
        )
        template_kwargs = gen_args.to_template_kwargs()
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice
        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            tools=tools,
            **template_kwargs,
        )
        if response_generator is not None:
            raw_inputs = await asyncio.to_thread(
                response_generator._cpu_preprocess,
                formatted_prompt,
                images if images else None,
                None,
            )
        else:
            image_token_index = getattr(config, "image_token_index", None)
            raw_inputs = prepare_inputs(
                processor,
                images=images if images else None,
                prompts=formatted_prompt,
                image_token_index=image_token_index,
            )
        return {"input_tokens": _count_prompt_tokens(raw_inputs)}
    except Exception as e:
        return _anthropic_error_response(400, str(e))


# OpenAI compatile endpoints


@app.post("/responses/input_tokens")
@app.post("/v1/responses/input_tokens", include_in_schema=False)
async def responses_input_tokens_endpoint(request: Request):
    body = await request.json()
    openai_request = OpenAIRequest(**body)
    try:
        model, processor, config = get_cached_model(openai_request.model)
        del model
        current_input_items = _normalize_response_input(openai_request.input)
        prompt_items = (
            _response_chain_items(openai_request.previous_response_id)
            + current_input_items
        )
        chat_messages, images = _response_items_to_chat(prompt_items)
        if openai_request.instructions:
            chat_messages.insert(
                0, {"role": "system", "content": openai_request.instructions}
            )
        chat_tools, _ = _response_tool_registry(openai_request.tools)
        gen_args = _build_gen_args(
            openai_request, processor, tenant_id=_read_tenant_id(request)
        )
        template_kwargs = gen_args.to_template_kwargs()
        if openai_request.tool_choice is not None:
            template_kwargs["tool_choice"] = openai_request.tool_choice
        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            tools=chat_tools or None,
            **template_kwargs,
        )
        if response_generator is not None:
            raw_inputs = await asyncio.to_thread(
                response_generator._cpu_preprocess,
                formatted_prompt,
                images if images else None,
                None,
            )
        else:
            image_token_index = getattr(config, "image_token_index", None)
            raw_inputs = prepare_inputs(
                processor,
                images=images if images else None,
                prompts=formatted_prompt,
                image_token_index=image_token_index,
            )
        return {"input_tokens": _count_prompt_tokens(raw_inputs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/responses/{response_id}")
@app.get("/v1/responses/{response_id}", include_in_schema=False)
async def responses_retrieve_endpoint(response_id: str):
    with response_store_lock:
        stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    return stored.response


@app.delete("/responses/{response_id}")
@app.delete("/v1/responses/{response_id}", include_in_schema=False)
async def responses_delete_endpoint(response_id: str):
    with response_store_lock:
        existed = response_store.pop(response_id, None) is not None
    if not existed:
        raise HTTPException(status_code=404, detail="Response not found.")
    return {"id": response_id, "object": "response.deleted", "deleted": True}


@app.post("/responses/{response_id}/cancel")
@app.post("/v1/responses/{response_id}/cancel", include_in_schema=False)
async def responses_cancel_endpoint(response_id: str):
    with response_store_lock:
        stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    response = dict(stored.response)
    if response.get("status") == "in_progress":
        response["status"] = "cancelled"
    return response


@app.get("/responses/{response_id}/input_items")
@app.get("/v1/responses/{response_id}/input_items", include_in_schema=False)
async def responses_input_items_endpoint(response_id: str):
    with response_store_lock:
        stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    data = stored.input_items
    return {
        "object": "list",
        "data": data,
        "first_id": data[0].get("id") if data else None,
        "last_id": data[-1].get("id") if data else None,
        "has_more": False,
    }


@app.post("/responses")
@app.post("/v1/responses", include_in_schema=False)
async def responses_endpoint(request: Request):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)

        try :
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                print(response.output[0].content[0].text)
                print(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == 'response.completed':
                        print("\n--- Usage ---")
                        print(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    request_start = time.perf_counter()
    body = await request.json()
    openai_request = OpenAIRequest(**body)

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(openai_request.model)

        kwargs = {}

        if openai_request.input is None:
            print("no input")
            raise HTTPException(status_code=400, detail="Missing input.")

        current_input_items = _normalize_response_input(openai_request.input)
        prompt_items = (
            _response_chain_items(openai_request.previous_response_id)
            + current_input_items
        )
        chat_messages, images = _response_items_to_chat(prompt_items)
        instructions = openai_request.instructions
        if instructions:
            chat_messages.insert(0, {"role": "system", "content": instructions})
        elif chat_messages and chat_messages[0].get("role") in ("system", "developer"):
            instructions = chat_messages[0].get("content")

        chat_tools, tool_registry = _response_tool_registry(openai_request.tools)
        tool_parser_type = _infer_tool_parser_from_processor(processor)
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                openai_request, processor, tenant_id=_read_tenant_id(request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        template_kwargs = gen_args.to_template_kwargs()
        if openai_request.tool_choice is not None:
            template_kwargs["tool_choice"] = openai_request.tool_choice

        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            tools=chat_tools or None,
            **template_kwargs,
        )

        logger.debug(
            "responses request: model=%s images=%d max_tokens=%s temp=%s stream=%s",
            openai_request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            openai_request.stream,
        )

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            # Streaming response
            server_metrics.begin_request(
                endpoint="/responses",
                model=openai_request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/responses",
                model=openai_request.model,
                prompt=formatted_prompt,
                images=images if images else None,
                audio=None,
                args=gen_args,
            )

            async def stream_generator():
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                metrics_finalized = False
                token_times: List[float] = []
                prompt_tps = None
                generation_tps = None
                peak_memory = 0.0
                finish_reason = None
                try:
                    # Create base response object (to match the openai pipeline)
                    base_response = OpenAIResponse(
                        id=response_id,
                        object="response",
                        created_at=int(generated_at),
                        status="in_progress",
                        instructions=instructions,
                        max_output_tokens=openai_request.max_output_tokens,
                        model=openai_request.model,
                        output=[],
                        output_text="",
                        temperature=openai_request.temperature,
                        top_p=openai_request.top_p,
                        previous_response_id=openai_request.previous_response_id,
                        store=openai_request.store,
                        usage={
                            "input_tokens": 0,  # get prompt tokens
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )

                    # Send response.created event  (to match the openai pipeline)
                    yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"

                    # Send response.in_progress event  (to match the openai pipeline)
                    yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

                    # Send response.output_item.added event  (to match the openai pipeline)
                    message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="in_progress",
                        role="assistant",
                        content=[],
                    )
                    yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

                    # Send response.content_part.added event
                    content_part = ContentPartOutputText(
                        type="output_text", text="", annotations=[]
                    )
                    yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

                    # Stream text deltas using ResponseGenerator (continuous batching)
                    full_text = ""
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}
                    in_tool_call = False
                    tc_start = (
                        tool_module.tool_call_start
                        if tool_module is not None and chat_tools
                        else None
                    )

                    if response_generator is not None:
                        # generate() blocks on _cpu_preprocess + queue.get;
                        # offload so concurrent handlers preprocess in parallel.
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            None,  # audio
                            gen_args,
                        )

                        output_tokens = 0

                        def _next_token_resp_stream():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token_resp_stream)
                            if token is None:
                                break
                            output_tokens += 1
                            delta = token.text
                            full_text += delta
                            in_tool_call, delta = suppress_tool_call_content(
                                full_text, in_tool_call, tc_start, delta
                            )
                            token_times.append(time.perf_counter())
                            peak_memory = max(
                                peak_memory,
                                float(getattr(token, "peak_memory", 0.0) or 0.0),
                            )
                            prompt_tps = getattr(token, "prompt_tps", prompt_tps)
                            usage_stats = {
                                "input_tokens": ctx.prompt_tokens,
                                "output_tokens": output_tokens,
                            }

                            if delta is not None:
                                yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                                await asyncio.sleep(0.01)

                            if token.finish_reason:
                                finish_reason = token.finish_reason
                                break
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            temperature=openai_request.temperature,
                            max_tokens=gen_args.max_tokens,
                            top_p=openai_request.top_p,
                            vision_cache=model_cache.get("vision_cache"),
                            logits_processors=gen_args.logits_processors,
                            apc_manager=apc_manager,
                            apc_tenant=gen_args.tenant_id,
                            **kwargs,
                        )

                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            delta = chunk.text
                            full_text += delta
                            in_tool_call, delta = suppress_tool_call_content(
                                full_text, in_tool_call, tc_start, delta
                            )
                            token_times.append(time.perf_counter())
                            prompt_tps = getattr(chunk, "prompt_tps", prompt_tps)
                            generation_tps = getattr(
                                chunk, "generation_tps", generation_tps
                            )
                            peak_memory = max(
                                peak_memory,
                                float(getattr(chunk, "peak_memory", 0.0) or 0.0),
                            )
                            usage_stats = {
                                "input_tokens": chunk.prompt_tokens,
                                "output_tokens": chunk.generation_tokens,
                            }

                            if delta is not None:
                                yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                                await asyncio.sleep(0.01)

                    output_items, clean_text, _, output_finish_reason = (
                        _response_output_items_from_text(
                            full_text,
                            message_id,
                            tool_module,
                            chat_tools,
                            tool_registry,
                        )
                    )
                    tool_output_items = [
                        item for item in output_items if item.get("type") != "message"
                    ]

                    # Send response.output_text.done event (to match the openai pipeline)
                    yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=clean_text).model_dump_json()}\n\n"

                    # Send response.content_part.done event (to match the openai pipeline)
                    final_content_part = ContentPartOutputText(
                        type="output_text", text=clean_text, annotations=[]
                    )
                    yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

                    # Send response.output_item.done event (to match the openai pipeline)
                    final_message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[final_content_part] if clean_text else [],
                    )
                    yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_item).model_dump_json()}\n\n"

                    completed_output = []
                    if clean_text:
                        completed_output.append(final_message_item.model_dump())
                    completed_output.extend(tool_output_items)
                    for output_index, tool_item in enumerate(
                        tool_output_items, start=1
                    ):
                        yield _sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": tool_item,
                            },
                        )
                        yield _sse_event(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": output_index,
                                "item": tool_item,
                            },
                        )

                    # Send response.completed event (to match the openai pipeline)
                    finish_reason = (
                        "tool_calls"
                        if output_finish_reason == "tool_calls"
                        else finish_reason
                    ) or ("stop" if usage_stats["output_tokens"] > 0 else None)
                    envelope = _build_metrics_envelope(
                        endpoint="/responses",
                        model=openai_request.model,
                        stream=True,
                        backend=(
                            "continuous_batching"
                            if response_generator is not None
                            else "generate"
                        ),
                        prompt_tokens=usage_stats["input_tokens"],
                        completion_tokens=usage_stats["output_tokens"],
                        generated_tokens=usage_stats["output_tokens"],
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=token_times,
                        prompt_tps=prompt_tps,
                        generation_tps=generation_tps,
                        peak_memory_gb=peak_memory or None,
                        finish_reason=finish_reason,
                        image_count=len(images),
                        structured_output=bool(gen_args.logits_processors),
                        thinking_enabled=bool(gen_args.enable_thinking),
                    )
                    server_metrics.record_success(envelope)
                    metrics_finalized = True
                    completed_response = base_response.model_copy(
                        update={
                            "status": "completed",
                            "output": completed_output,
                            "output_text": clean_text,
                            "usage": OpenAIUsage(
                                input_tokens=usage_stats["input_tokens"],
                                output_tokens=usage_stats["output_tokens"],
                                total_tokens=usage_stats["input_tokens"]
                                + usage_stats["output_tokens"],
                            ),
                        }
                    )
                    _store_response(
                        completed_response,
                        current_input_items,
                        completed_output,
                        openai_request.previous_response_id,
                    )
                    yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

                except Exception as e:
                    if not metrics_finalized:
                        server_metrics.record_failure(
                            endpoint="/responses",
                            model=openai_request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        server_metrics.record_failure(
                            endpoint="/responses",
                            model=openai_request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    mx.clear_cache()
                    gc.collect()
                    print("Stream finished, cleared cache.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            server_metrics.begin_request(
                endpoint="/responses",
                model=openai_request.model,
                stream=False,
            )
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                token_times: List[float] = []
                prompt_tps = None
                generation_tps = None
                peak_memory = 0.0
                finish_reason = None

                if response_generator is not None:

                    def _blocking_resp():
                        ctx_, ti = response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            args=gen_args,
                        )
                        text = ""
                        ot = 0
                        tt: List[float] = []
                        ptps = None
                        pm = 0.0
                        fr = None
                        for tok in ti:
                            text += tok.text
                            ot += 1
                            tt.append(time.perf_counter())
                            ptps = getattr(tok, "prompt_tps", ptps)
                            pm = max(pm, float(getattr(tok, "peak_memory", 0.0) or 0.0))
                            if tok.finish_reason:
                                fr = tok.finish_reason
                                break
                        try:
                            ti.close()
                        except Exception:
                            pass
                        return ctx_.prompt_tokens, text, ot, tt, ptps, pm, fr

                    (
                        prompt_tokens,
                        full_text,
                        output_tokens,
                        token_times,
                        prompt_tps,
                        peak_memory,
                        finish_reason,
                    ) = await asyncio.to_thread(_blocking_resp)
                else:
                    result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=model_cache.get("vision_cache"),
                        apc_manager=apc_manager,
                        apc_tenant=gen_args.tenant_id,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = result.text
                    prompt_tokens = result.prompt_tokens
                    output_tokens = result.generation_tokens
                    prompt_tps = getattr(result, "prompt_tps", None)
                    generation_tps = getattr(result, "generation_tps", None)
                    peak_memory = float(getattr(result, "peak_memory", 0.0) or 0.0)
                    finish_reason = "stop"

                mx.clear_cache()
                gc.collect()

                output_items, content, reasoning, output_finish_reason = (
                    _response_output_items_from_text(
                        full_text,
                        message_id,
                        tool_module,
                        chat_tools,
                        tool_registry,
                    )
                )
                if output_finish_reason == "tool_calls":
                    finish_reason = "tool_calls"

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=output_items,
                    output_text=content,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    previous_response_id=openai_request.previous_response_id,
                    store=openai_request.store,
                    usage={
                        "input_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    },
                )
                _store_response(
                    response,
                    current_input_items,
                    output_items,
                    openai_request.previous_response_id,
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "responses done: prompt_tokens=%d output_tokens=%d "
                    "total_time=%.2fs",
                    prompt_tokens,
                    output_tokens,
                    elapsed,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                envelope = _build_metrics_envelope(
                    endpoint="/responses",
                    model=openai_request.model,
                    stream=False,
                    backend=(
                        "continuous_batching"
                        if response_generator is not None
                        else "generate"
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=output_tokens,
                    generated_tokens=output_tokens,
                    request_elapsed_s=elapsed,
                    request_started_s=request_start,
                    token_times=token_times,
                    prompt_tps=prompt_tps,
                    generation_tps=generation_tps,
                    peak_memory_gb=peak_memory or None,
                    finish_reason=finish_reason,
                    image_count=len(images),
                    structured_output=bool(gen_args.logits_processors),
                    thinking_enabled=bool(gen_args.enable_thinking),
                )
                server_metrics.record_success(envelope)

                return response

            except PromptTooLongError as e:
                server_metrics.record_failure(
                    endpoint="/responses",
                    model=openai_request.model,
                    stream=False,
                    error=str(e),
                )
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                server_metrics.record_failure(
                    endpoint="/responses",
                    model=openai_request.model,
                    stream=False,
                    error=str(e),
                )
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/chat/completions", response_model=None)
@app.post("/v1/chat/completions", response_model=None, include_in_schema=False)
async def chat_completions_endpoint(request: ChatRequest, http_request: Request):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    request_start = time.perf_counter()
    try:
        adapter_path = (
            request.adapter_path
            if "adapter_path" in request.model_fields_set
            else _INHERIT_ADAPTER
        )
        model, processor, config = get_cached_model(request.model, adapter_path)

        kwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        images = []
        audio = []
        processed_messages = []
        for message in request.messages:
            msg = {"role": message.role}

            if isinstance(message.content, str):
                msg["content"] = message.content
            elif isinstance(message.content, list):
                if message.role == "user":
                    for item in message.content:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        if item_type == "input_image":
                            images.append(item["image_url"])
                        elif item_type == "image_url":
                            images.append(item["image_url"]["url"])
                        elif item_type == "input_audio":
                            audio.append(item["input_audio"]["data"])
                msg["content"] = extract_text_from_content(message.content)
            else:
                msg["content"] = message.content

            # Preserve tool-calling metadata.
            # Ensure arguments are dicts (not JSON strings) for Jinja templates
            # that iterate them with |items (e.g. Qwen3.5).
            if message.tool_calls is not None:
                normalized_calls = []
                for tc in message.tool_calls:
                    tc = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc, dict) and "function" in tc:
                        fn = dict(tc["function"])
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                fn["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                fn["arguments"] = {}
                        tc["function"] = fn
                    normalized_calls.append(tc)
                msg["tool_calls"] = normalized_calls
            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id
            if message.name is not None:
                msg["name"] = message.name

            processed_messages.append(msg)

        # Detect tool parser from chat template
        tools = getattr(request, "tools", None)
        tool_parser_type = _infer_tool_parser_from_processor(processor)
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                request, processor, tenant_id=_read_tenant_id(http_request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            num_audios=len(audio),
            tools=tools,
            **gen_args.to_template_kwargs(),
        )

        logger.debug(
            "chat/completions request: model=%s images=%d audio=%d "
            "max_tokens=%s temp=%s stream=%s",
            request.model,
            len(images),
            len(audio),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
        )

        if request.stream:
            # Streaming response using ResponseGenerator for continuous batching
            server_metrics.begin_request(
                endpoint="/chat/completions",
                model=request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/chat/completions",
                model=request.model,
                prompt=formatted_prompt,
                images=images if images else None,
                audio=audio if audio else None,
                args=gen_args,
            )

            async def stream_generator():
                global response_generator
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                metrics_finalized = False
                token_times: List[float] = []
                prompt_tps = None
                generation_tps = None
                peak_memory = 0.0
                finish_reason = None
                try:
                    output_tokens = 0
                    full_output = ""
                    output_text = ""
                    stream_prompt_tokens = 0
                    tool_calls_made = False

                    # Use ResponseGenerator if available, otherwise fall back to stream_generate
                    if response_generator is not None:
                        # generate() does blocking Queue.get — run off event loop
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            audio if audio else None,
                            gen_args,
                        )

                        output_tokens = 0
                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        # Track thinking state for reasoning/content split
                        in_thinking = False
                        accumulated = ""
                        full_output = ""  # raw output for tool call parsing
                        # Track tool-call state to suppress markup from content
                        in_tool_call = False
                        tc_start = tool_module.tool_call_start if tool_module else None
                        tc_end = tool_module.tool_call_end if tool_module else None

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token)
                            if token is None:
                                break
                            output_tokens += 1
                            accumulated += token.text
                            full_output += token.text
                            token_times.append(time.perf_counter())
                            peak_memory = max(
                                peak_memory,
                                float(getattr(token, "peak_memory", 0.0) or 0.0),
                            )
                            prompt_tps = getattr(token, "prompt_tps", prompt_tps)

                            # Detect thinking boundaries
                            delta_reasoning = None
                            delta_content = None

                            if not in_thinking and (
                                "<|channel>thought" in accumulated
                                or "<think>" in accumulated
                            ):
                                in_thinking = True
                                accumulated = ""
                                # Don't emit opening tag tokens
                            elif in_thinking and (
                                "<channel|>" in accumulated or "</think>" in accumulated
                            ):
                                in_thinking = False
                                accumulated = ""
                                # Don't emit closing tag tokens
                            elif in_thinking:
                                delta_reasoning = token.text
                            elif not in_thinking and (
                                "<|channel>" in accumulated or "<think" in accumulated
                            ):
                                pass  # Partial tag, don't emit yet
                            else:
                                delta_content = token.text

                            # Suppress tool-call markup from content
                            in_tool_call, delta_content = suppress_tool_call_content(
                                full_output, in_tool_call, tc_start, delta_content
                            )

                            chunk_logprobs = None
                            if request.logprobs and token.finish_reason != "stop":
                                req_top_k = int(request.top_logprobs or 0)
                                chunk_logprobs = ChatLogprobs(
                                    content=[
                                        _make_logprob_content(
                                            response_generator.tokenizer,
                                            token.token,
                                            token.logprobs,
                                            top_logprobs=token.top_logprobs,
                                            top_k=req_top_k,
                                        )
                                    ]
                                )

                            # Skip empty deltas (e.g. suppressed tool-call tokens)
                            has_payload = (
                                delta_content is not None
                                or delta_reasoning is not None
                                or token.finish_reason is not None
                                or chunk_logprobs is not None
                            )
                            if has_payload:
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason=token.finish_reason,
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=delta_reasoning,
                                        ),
                                        logprobs=chunk_logprobs,
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    usage={
                                        "prompt_tokens": ctx.prompt_tokens,
                                        "completion_tokens": output_tokens,
                                        "total_tokens": ctx.prompt_tokens
                                        + output_tokens,
                                    },
                                    choices=choices,
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                            if token.finish_reason:
                                finish_reason = token.finish_reason
                                break

                        # Parse tool calls from full output and emit final chunk
                        if tool_module is not None:
                            tc = process_tool_calls(full_output, tool_module, tools)
                            if tc["calls"]:
                                tool_calls_made = True
                                finish_reason = "tool_calls"
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason="tool_calls",
                                        delta=ChatMessage(
                                            role="assistant",
                                            tool_calls=tc["calls"],
                                        ),
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            audio=audio,
                            temperature=request.temperature,
                            max_tokens=gen_args.max_tokens,
                            top_p=request.top_p,
                            vision_cache=model_cache.get("vision_cache"),
                            logits_processors=gen_args.logits_processors,
                            apc_manager=apc_manager,
                            apc_tenant=gen_args.tenant_id,
                            **kwargs,
                        )

                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        output_text = ""
                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            output_text += chunk.text
                            stream_prompt_tokens = chunk.prompt_tokens
                            token_times.append(time.perf_counter())
                            output_tokens = chunk.generation_tokens
                            prompt_tps = getattr(chunk, "prompt_tps", prompt_tps)
                            generation_tps = getattr(
                                chunk, "generation_tps", generation_tps
                            )
                            peak_memory = max(
                                peak_memory,
                                float(getattr(chunk, "peak_memory", 0.0) or 0.0),
                            )

                            choices = [
                                ChatStreamChoice(
                                    delta=ChatMessage(
                                        role="assistant", content=chunk.text
                                    )
                                )
                            ]
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                usage={
                                    "prompt_tokens": chunk.prompt_tokens,
                                    "completion_tokens": chunk.generation_tokens,
                                    "total_tokens": chunk.prompt_tokens
                                    + chunk.generation_tokens,
                                },
                                choices=choices,
                            )

                            yield f"data: {chunk_data.model_dump_json()}\n\n"
                            await asyncio.sleep(0.01)

                    metrics_text = full_output or output_text
                    completion_tokens = max(
                        0, output_tokens - _count_thinking_tag_tokens(metrics_text)
                    )
                    envelope = _build_metrics_envelope(
                        endpoint="/chat/completions",
                        model=request.model,
                        stream=True,
                        backend=(
                            "continuous_batching"
                            if response_generator is not None
                            else "generate"
                        ),
                        prompt_tokens=(
                            ctx.prompt_tokens
                            if response_generator is not None
                            else stream_prompt_tokens
                        ),
                        completion_tokens=completion_tokens,
                        generated_tokens=output_tokens,
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=token_times,
                        prompt_tps=prompt_tps,
                        generation_tps=generation_tps,
                        peak_memory_gb=peak_memory or None,
                        finish_reason=finish_reason
                        or ("stop" if output_tokens > 0 else None),
                        image_count=len(images),
                        audio_count=len(audio),
                        structured_output=bool(gen_args.logits_processors),
                        thinking_enabled=bool(gen_args.enable_thinking),
                        tool_parser=tool_parser_type,
                        tool_calls=tool_calls_made,
                    )
                    server_metrics.record_success(envelope)
                    metrics_finalized = True

                    # Signal stream end
                    yield "data: [DONE]\n\n"

                    elapsed = time.perf_counter() - request_start
                    logger.debug(
                        "chat/completions stream done: tokens=%d " "total_time=%.2fs",
                        output_tokens,
                        elapsed,
                    )

                except Exception as e:
                    if not metrics_finalized:
                        server_metrics.record_failure(
                            endpoint="/chat/completions",
                            model=request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    # Close the token iterator to trigger cleanup (important for ResponseGenerator)
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        server_metrics.record_failure(
                            endpoint="/chat/completions",
                            model=request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    mx.clear_cache()
                    gc.collect()
                    print("Stream finished, cleared cache.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            server_metrics.begin_request(
                endpoint="/chat/completions",
                model=request.model,
                stream=False,
            )
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                peak_memory = 0.0
                token_times: List[float] = []
                prompt_tps = None
                generation_tps = None
                finish_reason = None

                collected_logprobs: List[
                    Tuple[int, float, Optional[List[Tuple[int, float]]]]
                ] = []

                if response_generator is not None:

                    def _blocking_generate():
                        text = ""
                        pt = gt = 0
                        pm = 0.0
                        tt: List[float] = []
                        ptps = None
                        fr = None
                        ctx, token_iter = response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            audio=audio if audio else None,
                            args=gen_args,
                        )
                        pt = ctx.prompt_tokens
                        for token in token_iter:
                            text += token.text
                            gt += 1
                            tt.append(time.perf_counter())
                            ptps = getattr(token, "prompt_tps", ptps)
                            pm = token.peak_memory
                            if request.logprobs and token.finish_reason != "stop":
                                collected_logprobs.append(
                                    (token.token, token.logprobs, token.top_logprobs)
                                )
                            if token.finish_reason:
                                fr = token.finish_reason
                                break
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                        return text, pt, gt, ptps, pm, tt, fr

                    (
                        full_text,
                        prompt_tokens,
                        output_tokens,
                        prompt_tps,
                        peak_memory,
                        token_times,
                        finish_reason,
                    ) = await asyncio.to_thread(_blocking_generate)
                else:
                    gen_result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=model_cache.get("vision_cache"),
                        apc_manager=apc_manager,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = gen_result.text
                    prompt_tokens = gen_result.prompt_tokens
                    output_tokens = gen_result.generation_tokens
                    peak_memory = float(getattr(gen_result, "peak_memory", 0.0) or 0.0)
                    prompt_tps = getattr(gen_result, "prompt_tps", None)
                    generation_tps = getattr(gen_result, "generation_tps", None)
                    finish_reason = "stop"

                mx.clear_cache()
                gc.collect()

                reasoning, content = _split_thinking(full_text)

                # Count raw generated tokens minus thinking tag tokens
                completion_tokens = output_tokens - _count_thinking_tag_tokens(
                    full_text
                )

                usage_stats = UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tps=float(prompt_tps or 0.0),
                    generation_tps=float(generation_tps or 0.0),
                    peak_memory=peak_memory,
                )

                # Parse tool calls from generated output
                parsed_tool_calls = None
                if tool_module is not None:
                    tc = process_tool_calls(
                        model_output=full_text,
                        tool_module=tool_module,
                        tools=tools,
                    )
                    if tc["calls"]:
                        parsed_tool_calls = tc["calls"]
                        # Clean thinking tags and control tokens from remaining text
                        _, clean_remaining = _split_thinking(tc["remaining_text"] or "")
                        if clean_remaining:
                            # Strip model control tokens
                            clean_remaining = re.sub(
                                r"<\|[^>]+\|>|<[^>]+>", "", clean_remaining
                            ).strip()
                        content = clean_remaining or None

                response_logprobs = None
                if request.logprobs and collected_logprobs:
                    tokenizer = (
                        processor.tokenizer
                        if hasattr(processor, "tokenizer")
                        else processor
                    )
                    req_top_k = int(request.top_logprobs or 0)
                    response_logprobs = ChatLogprobs(
                        content=[
                            _make_logprob_content(
                                tokenizer,
                                tid,
                                lp,
                                top_logprobs=top_lps,
                                top_k=req_top_k,
                            )
                            for tid, lp, top_lps in collected_logprobs
                        ]
                    )

                choices = [
                    ChatChoice(
                        finish_reason="tool_calls" if parsed_tool_calls else "stop",
                        message=ChatMessage(
                            role="assistant",
                            content=content if content else None,
                            reasoning=reasoning,
                            tool_calls=parsed_tool_calls,
                        ),
                        logprobs=response_logprobs,
                    )
                ]
                result = ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    created=int(time.time()),
                    model=request.model,
                    usage=usage_stats,
                    choices=choices,
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "chat/completions done: prompt_tokens=%d completion_tokens=%d "
                    "total_time=%.2fs peak_memory=%.2fGB",
                    prompt_tokens,
                    completion_tokens,
                    elapsed,
                    peak_memory,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                envelope = _build_metrics_envelope(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    backend=(
                        "continuous_batching"
                        if response_generator is not None
                        else "generate"
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    generated_tokens=output_tokens,
                    request_elapsed_s=elapsed,
                    request_started_s=request_start,
                    token_times=token_times,
                    prompt_tps=prompt_tps,
                    generation_tps=generation_tps,
                    peak_memory_gb=peak_memory or None,
                    finish_reason=(
                        "tool_calls" if parsed_tool_calls else finish_reason or "stop"
                    ),
                    image_count=len(images),
                    audio_count=len(audio),
                    structured_output=bool(gen_args.logits_processors),
                    thinking_enabled=bool(gen_args.enable_thinking),
                    tool_parser=tool_parser_type,
                    tool_calls=bool(parsed_tool_calls),
                )
                server_metrics.record_success(envelope)

                return result

            except PromptTooLongError as e:
                server_metrics.record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    error=str(e),
                )
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                server_metrics.record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    error=str(e),
                )
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    required_files = {"config.json", "tokenizer_config.json"}

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        has_weights = "model.safetensors.index.json" in file_names or any(
            file_name.endswith(".safetensors") for file_name in file_names
        )
        return required_files.issubset(file_names) and has_weights

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.middleware("http")
async def add_server_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Server"] = f"mlx_vlm/{__version__}"
    return response


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    runtime = _server_runtime_snapshot()
    return {
        "status": "healthy",
        "loaded_model": runtime["loaded_model"],
        "loaded_adapter": runtime["loaded_adapter"],
        "loaded_context_size": runtime["loaded_context_size"],
        "configured_context_limit": runtime["configured_context_limit"],
        "effective_context_limit": runtime["effective_context_limit"],
        "loaded_tool_parser": runtime["loaded_tool_parser"],
        "continuous_batching_enabled": runtime["continuous_batching_enabled"],
        "apc_enabled": runtime["apc"]["enabled"],
    }


@app.get("/metrics")
@app.get("/v1/metrics", include_in_schema=False)
async def metrics_endpoint():
    payload = server_metrics.snapshot()
    payload["server"] = _server_runtime_snapshot()
    return payload


@app.get("/v1/cache/stats")
@app.get("/cache/stats", include_in_schema=False)
async def apc_cache_stats():
    """Return Automatic Prefix Cache statistics (or ``enabled=false``)."""
    if apc_manager is None:
        return {"enabled": False}
    snap = apc_manager.stats_snapshot()
    snap["enabled"] = True
    return snap


@app.post("/v1/cache/reset")
@app.post("/cache/reset", include_in_schema=False)
async def apc_cache_reset():
    if apc_manager is None:
        return {"enabled": False}
    apc_manager.clear()
    return {"enabled": True, "status": "cleared"}


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_server_max_tokens(),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Enable thinking mode by default for requests that do not set "
            "enable_thinking explicitly."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, google/gemma-4-31B-it-assistant)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp"],
        help="Drafter family — 'dflash', 'eagle3', or 'mtp' (Gemma 4). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["MLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    os.environ["MLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["MLX_VLM_DRAFT_MODEL"] = args.draft_model
        if args.draft_kind is not None:
            os.environ["MLX_VLM_DRAFT_KIND"] = args.draft_kind
        if args.draft_block_size is not None:
            os.environ["MLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["MLX_VLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["MLX_VLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )


if __name__ == "__main__":
    main()
