import gc
import logging
import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Generator, List, Optional, Tuple

import mlx.core as mx
from fastapi import HTTPException

from .. import apc as _apc
from ..generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
    BatchGenerator,
    _chunked_prefill_enabled,
    _make_cache,
    _merge_prefill_prompt_kwargs,
)
from ..generate.diffusion import (
    is_diffusion_model,
    stream_diffusion_generate_from_kwargs,
)
from ..sample_utils import make_logits_processors, make_sampler, top_p_sampling
from ..speculative.utils import (
    make_speculative_prompt_cache,
    run_speculative_server_rounds,
    speculative_hidden_state,
    speculative_prefill_kwargs,
)
from ..structured import ThinkingAwareLogitsProcessor
from ..tokenizer_utils import _ServerTokenStreamer, make_streaming_detokenizer
from ..utils import ThinkingBudgetCriteria, load, prepare_inputs
from .runtime import runtime

logger = logging.getLogger("mlx_vlm.server")

DEFAULT_TOKEN_QUEUE_TIMEOUT = 600.0
DEFAULT_SPECULATIVE_BATCH_COALESCE_MS = 5.0
DEFAULT_LOG_PROGRESS_INTERVAL = 32
DEFAULT_ENABLE_THINKING = False
METRICS_HISTORY_LIMIT = 100
METRICS_RECENT_LIMIT = 32


class PromptTooLongError(ValueError):
    """Raised when a request exceeds the configured server context budget."""


def _get_draft_block_size_from_env():
    draft_block_size_str = os.environ.get("MLX_VLM_DRAFT_BLOCK_SIZE")
    return int(draft_block_size_str) if draft_block_size_str else None


def _notify_queues(queues, *items):
    for queue in queues:
        for item in items:
            try:
                queue.put(item)
            except Exception:
                break


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


def get_log_progress_interval():
    """Number of decoded tokens between INFO progress messages (0 disables)."""
    raw = os.environ.get(
        "MLX_VLM_LOG_PROGRESS_INTERVAL", str(DEFAULT_LOG_PROGRESS_INTERVAL)
    )
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid MLX_VLM_LOG_PROGRESS_INTERVAL=%r; falling back to %d.",
            raw,
            DEFAULT_LOG_PROGRESS_INTERVAL,
        )
        return DEFAULT_LOG_PROGRESS_INTERVAL


def _sequence_aligned_prefill_keys(
    prompt_kwargs: dict, *, batch_size: int, sequence_length: int
) -> List[str]:
    return [
        k
        for k, v in (prompt_kwargs or {}).items()
        if isinstance(v, mx.array)
        and v.ndim >= 2
        and v.shape[0] == batch_size
        and v.shape[1] == sequence_length
    ]


def _slice_prefill_kwargs(prompt_kwargs: dict, keys: List[str], n: int) -> dict:
    if not keys:
        return prompt_kwargs
    out = dict(prompt_kwargs)
    for key in keys:
        if key in out:
            out[key] = out[key][:, :n, ...]
    return out


def _drop_prefill_kwargs(prompt_kwargs: dict, keys: List[str], n: int) -> dict:
    if not keys:
        return prompt_kwargs
    out = dict(prompt_kwargs)
    for key in keys:
        if key in out:
            out[key] = out[key][:, n:, ...]
    return out


def _run_chunked_speculative_prefill(
    lm,
    input_ids: mx.array,
    inputs_embeds: mx.array,
    prompt_cache,
    prompt_kwargs: dict,
    speculative_kwargs: dict,
    *,
    prefill_step_size: Optional[int],
    generation_stream,
) -> Tuple[object, mx.array]:
    """Prefill target cache in chunks, capturing speculative state only at end."""
    remaining_input_ids = input_ids
    remaining_embeds = inputs_embeds
    remaining_kwargs = dict(prompt_kwargs or {})
    sequence_keys = _sequence_aligned_prefill_keys(
        remaining_kwargs,
        batch_size=input_ids.shape[0],
        sequence_length=inputs_embeds.shape[1],
    )

    if (
        prefill_step_size is not None
        and prefill_step_size > 0
        and remaining_embeds.shape[1] > prefill_step_size
    ):
        while remaining_embeds.shape[1] > 1:
            n_to_process = min(prefill_step_size, remaining_embeds.shape[1] - 1)
            chunk_kwargs = _slice_prefill_kwargs(
                remaining_kwargs, sequence_keys, n_to_process
            )
            with mx.stream(generation_stream):
                lm(
                    remaining_input_ids[:, :n_to_process],
                    cache=prompt_cache,
                    inputs_embeds=remaining_embeds[:, :n_to_process],
                    n_to_process=n_to_process,
                    **chunk_kwargs,
                )
            mx.eval([c.state for c in prompt_cache])
            remaining_input_ids = remaining_input_ids[:, n_to_process:]
            remaining_embeds = remaining_embeds[:, n_to_process:]
            remaining_kwargs = _drop_prefill_kwargs(
                remaining_kwargs, sequence_keys, n_to_process
            )
            mx.clear_cache()

    final_kwargs = {**remaining_kwargs, **speculative_kwargs}
    final_kwargs["inputs_embeds"] = remaining_embeds
    with mx.stream(generation_stream):
        out = lm(remaining_input_ids, cache=prompt_cache, **final_kwargs)
    return out, remaining_input_ids


def _position_seed(seed: int, row_id: int, position: int) -> int:
    x = (int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    x = (x + (int(row_id) + 1) * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ ((int(position) + 1) * 0xC2B2AE35)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return int(x & 0xFFFFFFFF)


def _position_keys(seed: int, row_ids: List[int], positions: List[int]) -> mx.array:
    return mx.stack(
        [
            mx.random.key(_position_seed(seed, row, pos))
            for row, pos in zip(row_ids, positions)
        ]
    )


class _PositionedTargetSampler:
    """Server sampler with stateless target draws for ragged verification."""

    def __init__(self, *, temperature: float, top_p: float, seed: Optional[int]):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = DEFAULT_SEED if seed is None else int(seed)

    def __call__(self, logprobs: mx.array) -> mx.array:
        if self.top_p > 0 and self.top_p < 1.0:
            return top_p_sampling(logprobs, self.top_p, self.temperature)
        return mx.random.categorical(logprobs * (1 / self.temperature))

    def sample_target(
        self,
        logprobs: mx.array,
        *,
        row_ids: List[int],
        positions: List[int],
    ) -> mx.array:
        if logprobs.shape[0] != len(row_ids) or len(row_ids) != len(positions):
            raise ValueError("row_ids and positions must match logprobs batch size.")
        keys = _position_keys(self.seed, row_ids, positions)
        if self.top_p > 0 and self.top_p < 1.0:
            return mx.vmap(self._sample_top_p_one, in_axes=(0, 0))(logprobs, keys)
        return mx.vmap(self._sample_one, in_axes=(0, 0))(logprobs, keys)

    def _sample_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        return mx.random.categorical(logprobs * (1 / self.temperature), key=key)

    def _sample_top_p_one(self, logprobs: mx.array, key: mx.array) -> mx.array:
        if logprobs.dtype == mx.bfloat16:
            logprobs = logprobs.astype(mx.float32)
        probs = mx.softmax(logprobs / self.temperature, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        top_probs = mx.where(
            cumulative_probs > 1 - self.top_p,
            sorted_probs,
            mx.zeros_like(sorted_probs),
        )
        sampled_pos = mx.random.categorical(mx.log(top_probs), key=key)
        return mx.take_along_axis(sorted_indices, sampled_pos[..., None], axis=-1)[0]


def _sample_last_token(
    logits: mx.array,
    sampler: Callable[[mx.array], mx.array],
    *,
    row_ids: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
):
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    sample_target = getattr(sampler, "sample_target", None)
    if callable(sample_target) and row_ids is not None and positions is not None:
        return sample_target(logprobs, row_ids=row_ids, positions=positions)
    return sampler(logprobs)


def get_server_enable_thinking():
    raw = os.environ.get("MLX_VLM_ENABLE_THINKING")
    if raw is None:
        return DEFAULT_ENABLE_THINKING
    return raw.lower() in ("1", "true", "yes", "on")


def get_server_thinking_budget():
    raw = os.environ.get("MLX_VLM_THINKING_BUDGET")
    return None if raw is None else int(raw)


def get_server_thinking_start_token():
    return os.environ.get("MLX_VLM_THINKING_START_TOKEN")


def get_server_thinking_end_token():
    return os.environ.get("MLX_VLM_THINKING_END_TOKEN")


def get_quantized_kv_bits(model: str):
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        logger.info(
            "Model %s is quantization aware; KV cache will not be quantized.", model
        )
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
        logger.warning("Model %s uses QuantizedKVCache; MAX_KV_SIZE is ignored.", model)
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
        with self._lock:
            self._requests_started += 1
            self._in_flight += 1
            if stream:
                self._streaming_requests += 1
            in_flight = self._in_flight
        logger.info(
            "Request started: endpoint=%s model=%s stream=%s in_flight=%d",
            endpoint,
            model,
            stream,
            in_flight,
        )

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
            in_flight = self._in_flight
        logger.info(
            "Request completed: endpoint=%s model=%s stream=%s backend=%s "
            "prompt_tokens=%d generated_tokens=%d elapsed=%.3fs "
            "prefill=%.1f tok/s decode=%.1f tok/s finish_reason=%s in_flight=%d",
            payload.get("endpoint"),
            payload.get("model"),
            payload.get("stream"),
            payload.get("backend"),
            int(payload.get("prompt_tokens") or 0),
            int(payload.get("generated_tokens") or 0),
            float(payload.get("request_elapsed_s") or 0.0),
            float(payload.get("prefill_tok_s") or 0.0),
            float(payload.get("decode_tok_s") or 0.0),
            payload.get("finish_reason"),
            in_flight,
        )

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
            in_flight = self._in_flight
        logger.warning(
            "Request failed: endpoint=%s model=%s stream=%s error=%s in_flight=%d",
            endpoint,
            model,
            stream,
            error,
            in_flight,
        )

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
        "apc_enabled": runtime.apc_manager is not None,
    }


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        logger.info("Loading model: %s", model_path)
        if adapter_path:
            logger.info("Loading adapter: %s", adapter_path)
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        logger.info("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        logger.exception("Error loading model %s: %s", model_path, e)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


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
    logprobs: bool = False
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    presence_penalty: Optional[float] = None
    presence_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    frequency_penalty: Optional[float] = None
    frequency_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE
    max_denoising_steps: Optional[int] = None
    block_length: Optional[int] = None
    num_to_transfer: Optional[int] = None
    max_transfer_per_step: Optional[int] = None
    editing_threshold: Optional[float] = None
    max_post_steps: Optional[int] = None
    stability_steps: Optional[int] = None
    diffusion_full_canvas: Optional[bool] = None
    diffusion_min_canvas_length: Optional[int] = None
    diffusion_max_canvas_length: Optional[int] = None
    diffusion_sampler: Optional[str] = None
    threshold: Optional[float] = None
    min_threshold: Optional[float] = None
    logit_bias: Optional[dict] = None
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None
    thinking_end_token: Optional[str] = None
    skip_special_tokens: bool = True
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None
    # Per-tenant salt for APC. When set, it's mixed into ``extra_hash`` so
    # cached blocks from one tenant can't be reused (or detected via timing)
    # by another. None = no salt = single-tenant behaviour.
    tenant_id: Optional[str] = None

    def diffusion_kwargs(self) -> dict:
        """Diffusion-only generation kwargs explicitly supplied by a request."""
        kw = {}
        for key in (
            "max_denoising_steps",
            "block_length",
            "num_to_transfer",
            "max_transfer_per_step",
            "editing_threshold",
            "max_post_steps",
            "stability_steps",
            "diffusion_full_canvas",
            "diffusion_min_canvas_length",
            "diffusion_max_canvas_length",
            "diffusion_sampler",
            "threshold",
            "min_threshold",
        ):
            value = getattr(self, key)
            if value is not None:
                kw[key] = value
        return kw

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
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        if self.repetition_context_size is not None:
            kw["repetition_context_size"] = self.repetition_context_size
        if self.presence_penalty is not None:
            kw["presence_penalty"] = self.presence_penalty
        if self.presence_context_size is not None:
            kw["presence_context_size"] = self.presence_context_size
        if self.frequency_penalty is not None:
            kw["frequency_penalty"] = self.frequency_penalty
        if self.frequency_context_size is not None:
            kw["frequency_context_size"] = self.frequency_context_size
        if self.logit_bias is not None:
            kw["logit_bias"] = self.logit_bias
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        if self.logits_processors is not None:
            kw["logits_processors"] = self.logits_processors
        if self.tenant_id is not None:
            kw["apc_tenant"] = self.tenant_id
        kw.update(self.diffusion_kwargs())
        return kw

    def to_template_kwargs(self) -> dict:
        """Convert to kwargs for apply_chat_template()."""
        kw = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        return kw


@dataclass
class GenerationContext:
    """Context returned when a request is queued."""

    uid: int
    prompt_tokens: int


@dataclass
class QueuedGenerationRequest:
    """Preprocessed generation request waiting for the GPU worker."""

    rqueue: Queue
    raw_inputs: dict
    prompt_tokens: int
    args: GenerationArguments
    thinking_budget_criteria: Optional[ThinkingBudgetCriteria] = None
    images: Optional[List] = None
    videos: Optional[List] = None
    audio: Optional[List] = None
    request_id: Optional[str] = None
    queued_at: float = field(default_factory=time.perf_counter)


@dataclass
class GenerationMetrics:
    """Runtime metrics collected while consuming generation output."""

    token_times: List[float] = field(default_factory=list)
    peak_memory: float = 0.0
    cached_tokens: int = 0
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None

    def record_chunk(self, chunk) -> None:
        self.token_times.append(time.perf_counter())
        self.record_result(chunk)

    def record_result(self, result) -> None:
        self.peak_memory = max(
            self.peak_memory, float(getattr(result, "peak_memory", 0.0) or 0.0)
        )
        prompt_tps = getattr(result, "prompt_tps", None)
        if prompt_tps is not None:
            self.prompt_tps = prompt_tps
        generation_tps = getattr(result, "generation_tps", None)
        if generation_tps is not None:
            self.generation_tps = generation_tps
        cached_tokens = getattr(result, "cached_tokens", None)
        if cached_tokens is not None:
            self.cached_tokens = max(self.cached_tokens, int(cached_tokens))


@dataclass
class StreamingToken:
    """A single token response during streaming generation.

    Diffusion models stream block-by-block: one StreamingToken per denoised
    block, with ``token_count`` carrying the number of tokens in the block.
    """

    text: str
    token: int
    logprobs: float
    finish_reason: Optional[str]
    peak_memory: float = 0.0
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    top_logprobs: Optional[List[Tuple[int, float]]] = None
    cached_tokens: int = 0
    token_count: int = 1


class _DiffusionBlockEmitter:
    """Incrementally group diffusion results into block streaming tokens."""

    def __init__(self):
        self.block_text: List[str] = []
        self.last_token = 0
        self.emitted_tokens = 0

    def feed(self, result) -> "Generator[StreamingToken, None, None]":
        if result.is_draft:
            return
        if result.text:
            self.block_text.append(result.text)
        if result.token is not None:
            self.last_token = int(result.token)
        if not result.diffusion_block_complete and not result.finish_reason:
            return
        if result.finish_reason or self.block_text:
            token_count = max(result.generation_tokens - self.emitted_tokens, 0)
            self.emitted_tokens = result.generation_tokens
            yield StreamingToken(
                text="".join(self.block_text),
                token=self.last_token,
                logprobs=None,
                finish_reason=result.finish_reason,
                peak_memory=result.peak_memory,
                prompt_tps=result.prompt_tps,
                generation_tps=result.generation_tps,
                token_count=token_count,
            )
            self.block_text = []


def _diffusion_block_chunks(results) -> "Generator[StreamingToken, None, None]":
    """Group diffusion engine results into block-by-block streaming tokens.

    The diffusion engine emits a canvas's tokens right after that canvas
    finishes denoising, followed by a block-boundary marker. Each completed
    block becomes one StreamingToken; the final token carries the finish
    reason (plus any text flushed by detokenizer finalization).
    """
    emitter = _DiffusionBlockEmitter()
    for result in results:
        for chunk in emitter.feed(result):
            yield chunk
            if chunk.finish_reason:
                return


class _TokenIterator:
    """Closeable iterator over queued tokens for one generation request.

    close() cancels unfinished requests and is safe while another thread is
    blocked in __next__ waiting for the next token.
    """

    def __init__(self, rqueue, uid, cancel_fn, queue_timeout):
        self._rqueue = rqueue
        self._uid = uid
        self._cancel_fn = cancel_fn
        self._queue_timeout = queue_timeout
        self._ended = False
        self._closed = False
        self._lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        if self._ended:
            raise StopIteration
        try:
            item = self._rqueue.get(timeout=self._queue_timeout)
        except QueueEmpty as exc:
            # Consumer is stalled or upstream is wedged — treat as cancel.
            self.close()
            label = (
                "without a timeout"
                if self._queue_timeout is None
                else f"for {self._queue_timeout:g}s"
            )
            raise RuntimeError(
                "Timed out waiting "
                f"{label} for the next generated token. "
                "Increase MLX_VLM_TOKEN_QUEUE_TIMEOUT for long "
                "prefills, or reduce the prompt size."
            ) from exc
        if item is None:
            self._ended = True
            raise StopIteration
        if isinstance(item, Exception):
            self._ended = True
            raise item
        if getattr(item, "finish_reason", None):
            self._ended = True
        return item

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if not self._ended:
            self._cancel_fn(self._uid)

    def __del__(self):
        # Mirror generator semantics: implicit close on GC.
        try:
            self.close()
        except Exception:
            pass


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
        self._tokenizer_lock = Lock()
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
            from ..speculative.drafters import (
                load_drafter,
                validate_drafter_compatibility,
            )

            logger.info(
                "Loading speculative drafter (%s): %s",
                draft_kind or "auto",
                draft_model_path,
            )
            draft_model, resolved_kind = load_drafter(draft_model_path, kind=draft_kind)
            if draft_kind is None:
                logger.info("Auto-detected speculative draft kind: %s", resolved_kind)
            elif resolved_kind != draft_kind:
                logger.warning(
                    "Drafter requires draft kind %s; using it instead of %s.",
                    resolved_kind,
                    draft_kind,
                )
            draft_kind = resolved_kind
            try:
                validate_drafter_compatibility(model, draft_model, draft_kind)
            except ValueError as e:
                logger.warning(
                    "Speculative drafter is incompatible with the target model; "
                    "falling back to autoregressive generation: %s",
                    e,
                )
                draft_model = None
                draft_kind = None
            else:
                logger.info("Drafter ready; speculative decoding enabled.")

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
        videos: Optional[List] = None,
    ) -> Tuple[GenerationContext, "_TokenIterator"]:
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        if self.draft_model is not None and args.logits_processors is not None:
            raise ValueError(
                "Structured response_format is not supported with speculative decoding."
            )
        if self.draft_model is not None and args.thinking_budget is not None:
            raise ValueError(
                "thinking_budget is not supported with speculative decoding in the server."
            )
        rqueue: Queue = Queue()
        request_started_at = time.perf_counter()

        # CPU preprocessing and thinking-token resolution share tokenizer state.
        # Keep both on the caller side and serialize them so the GPU worker never
        # races request threads through the mutable fast-tokenizer backend.
        tokenizer_lock = getattr(self, "_tokenizer_lock", None)
        with tokenizer_lock if tokenizer_lock is not None else nullcontext():
            raw_inputs = self._preprocess_request(prompt, images, audio, videos)
            thinking_budget_criteria = self._make_thinking_budget_criteria(
                args, raw_inputs.get("input_ids")
            )
        prompt_tokens = _count_prompt_tokens(raw_inputs)
        _check_configured_context_budget(prompt_tokens, args.max_tokens)

        request_id = f"{id(rqueue):x}"
        queued_request = QueuedGenerationRequest(
            rqueue=rqueue,
            raw_inputs=raw_inputs,
            prompt_tokens=prompt_tokens,
            args=args,
            thinking_budget_criteria=thinking_budget_criteria,
            images=images,
            videos=videos,
            audio=audio,
            request_id=request_id,
            queued_at=request_started_at,
        )
        logger.info(
            "Generation queued: request=%s prompt_tokens=%d max_tokens=%d "
            "images=%d audio=%d videos=%d",
            request_id,
            prompt_tokens,
            args.max_tokens,
            len(images or []),
            len(audio or []),
            len(videos or []),
        )
        self.requests.put(queued_request)

        # Block until the GPU thread sends back the context
        ctx = rqueue.get()
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _TokenIterator(
            rqueue, ctx.uid, self._cancel, get_token_queue_timeout()
        )

    def _cpu_preprocess(self, prompt, images=None, audio=None, videos=None) -> dict:
        """CPU-only: tokenize text, load/resize images. Thread-safe."""
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if self.model.config.model_type
            in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)
        return prepare_inputs(
            self.processor,
            images=images,
            audio=audio,
            videos=videos,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )

    def _preprocess_request(self, prompt, images=None, audio=None, videos=None) -> dict:
        if videos is None:
            return self._cpu_preprocess(prompt, images, audio)
        return self._cpu_preprocess(prompt, images, audio, videos)

    # -- internals --

    @staticmethod
    def _request_log_id(request: QueuedGenerationRequest) -> str:
        if request.request_id is None:
            request.request_id = f"{id(request.rqueue):x}"
        return request.request_id

    def _log_prefill_started(
        self, request: QueuedGenerationRequest, *, backend: str
    ) -> dict:
        request_id = self._request_log_id(request)
        now = time.perf_counter()
        logger.info(
            "Prefill started: request=%s backend=%s prompt_tokens=%d "
            "images=%d audio=%d videos=%d",
            request_id,
            backend,
            request.prompt_tokens,
            len(request.images or []),
            len(request.audio or []),
            len(request.videos or []),
        )
        return {
            "request_id": request_id,
            "queued_at": request.queued_at,
            "prefill_started_at": now,
            "prefill_processed": -1,
            "generated_tokens": 0,
            "decode_started_at": None,
        }

    def _log_prefill_progress(self, batch_gen, active: dict) -> None:
        """Report each chunked-prefill step without changing generation output."""
        prompt_batch = getattr(batch_gen, "_prompt_batch", None)
        if prompt_batch is None:
            return
        processed_columns = int(
            getattr(prompt_batch, "_processed_prompt_columns", 0) or 0
        )
        remaining = getattr(prompt_batch, "_inputs_embeds", None)
        remaining_columns = (
            int(remaining.shape[1]) if getattr(remaining, "ndim", 0) >= 2 else 0
        )
        total_columns = processed_columns + remaining_columns
        uids = list(getattr(prompt_batch, "uids", []))
        suffix_lengths = list(getattr(prompt_batch, "_suffix_lens", []))
        cached_tokens = list(getattr(prompt_batch, "_cached_tokens_per_row", []))
        left_padding = list(getattr(prompt_batch, "_left_padding_per_row", []))
        right_padding = getattr(prompt_batch, "_right_pad_per_row", None)

        for index, uid in enumerate(uids):
            info = active.get(uid)
            if info is None:
                continue
            suffix_len = (
                int(suffix_lengths[index])
                if index < len(suffix_lengths)
                else total_columns
            )
            cached = int(cached_tokens[index]) if index < len(cached_tokens) else 0
            if right_padding is not None:
                suffix_processed = min(suffix_len, processed_columns)
            else:
                pad = int(left_padding[index]) if index < len(left_padding) else 0
                suffix_processed = min(suffix_len, max(0, processed_columns - pad))
            processed = cached + suffix_processed
            total = cached + suffix_len
            if processed <= int(info.get("prefill_processed", -1)):
                continue
            info["prefill_processed"] = processed
            percent = 100.0 * processed / total if total > 0 else 100.0
            logger.info(
                "Prefill progress: request=%s tokens=%d/%d (%.1f%%)",
                info.get("request_id", uid),
                processed,
                total,
                percent,
            )

    @staticmethod
    def _log_prefill_completed(uid, info: dict, prompt_response) -> None:
        prompt_tokens = int(getattr(prompt_response, "prompt_tokens", 0) or 0)
        cached_tokens = int(getattr(prompt_response, "cached_tokens", 0) or 0)
        prompt_tps = float(getattr(prompt_response, "prompt_tps", 0.0) or 0.0)
        prompt_time = float(getattr(prompt_response, "prompt_time", 0.0) or 0.0)
        if prompt_time <= 0 and prompt_tps > 0:
            prompt_time = prompt_tokens / prompt_tps
        info["prefill_processed"] = prompt_tokens
        logger.info(
            "Prefill completed: request=%s prompt_tokens=%d cached_tokens=%d "
            "elapsed=%.3fs rate=%.1f tok/s",
            info.get("request_id", uid),
            prompt_tokens,
            cached_tokens,
            prompt_time,
            prompt_tps,
        )

    @staticmethod
    def _log_decode_progress(
        uid,
        info: dict,
        *,
        token: int,
        text: str,
        finish_reason: Optional[str],
        token_count: int = 1,
    ) -> None:
        now = time.perf_counter()
        previous_tokens = int(info.get("generated_tokens", 0) or 0)
        generated_tokens = previous_tokens + max(0, int(token_count or 0))
        info["generated_tokens"] = generated_tokens
        request_id = info.get("request_id", uid)

        decode_started_at = info.get("decode_started_at")
        if decode_started_at is None:
            decode_started_at = now
            info["decode_started_at"] = now
            queued_at = float(info.get("queued_at", now) or now)
            logger.info(
                "Decode started: request=%s time_to_first_token=%.3fs",
                request_id,
                max(0.0, now - queued_at),
            )

        elapsed = max(0.0, now - decode_started_at)
        rate = generated_tokens / elapsed if elapsed > 0 else 0.0
        interval = get_log_progress_interval()
        crossed_interval = interval > 0 and (
            generated_tokens // interval > previous_tokens // interval
        )
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            logger.debug(
                "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%.1f tok/s token_number=%d token_id=%s text=%r",
                request_id,
                generated_tokens,
                elapsed,
                rate,
                generated_tokens,
                token,
                text,
            )

        if finish_reason is not None:
            logger.info(
                "Decode completed: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%.1f tok/s finish_reason=%s",
                request_id,
                generated_tokens,
                elapsed,
                rate,
                finish_reason,
            )
        elif crossed_interval and not debug_enabled:
            logger.info(
                "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
                "rate=%.1f tok/s",
                request_id,
                generated_tokens,
                elapsed,
                rate,
            )

    def _make_sampler(self, args: GenerationArguments) -> Optional[Callable]:
        if args.temperature == 0:
            return None
        return _PositionedTargetSampler(
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )

    def _make_logits_processors(
        self, args: GenerationArguments, input_ids: Optional[mx.array] = None
    ) -> List[Callable[[mx.array, mx.array], mx.array]]:
        processors = make_logits_processors(
            args.logit_bias,
            args.repetition_penalty,
            args.repetition_context_size,
            args.presence_penalty,
            args.presence_context_size,
            args.frequency_penalty,
            args.frequency_context_size,
        )
        if args.logits_processors is not None:
            request_processors = args.logits_processors
            if input_ids is not None and self._prompt_has_open_thinking(
                args, input_ids
            ):
                request_processors = self._wrap_processors_until_thinking_done(
                    args, request_processors
                )
            processors.extend(request_processors)
        return processors

    def _thinking_token_ids(self, args: GenerationArguments) -> Tuple[int, int]:
        tokenizer = self.tokenizer
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]
        return thinking_start_token_id, thinking_end_token_id

    def _prompt_has_open_thinking(
        self, args: GenerationArguments, input_ids: mx.array
    ) -> bool:
        if not args.enable_thinking:
            return False
        thinking_start_token_id, thinking_end_token_id = self._thinking_token_ids(args)
        tokens = input_ids.flatten().tolist()
        try:
            last_start = len(tokens) - 1 - tokens[::-1].index(thinking_start_token_id)
        except ValueError:
            return False
        try:
            last_end = len(tokens) - 1 - tokens[::-1].index(thinking_end_token_id)
        except ValueError:
            last_end = -1
        return last_start > last_end

    def _wrap_processors_until_thinking_done(
        self,
        args: GenerationArguments,
        processors: List[Callable[[mx.array, mx.array], mx.array]],
    ) -> List[Callable[[mx.array, mx.array], mx.array]]:
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        return [
            ThinkingAwareLogitsProcessor(
                processor=processor,
                tokenizer=self.tokenizer,
                thinking_start_token=thinking_start_token,
                thinking_end_token=thinking_end_token,
                enable_thinking=True,
            )
            for processor in processors
        ]

    def _make_thinking_budget_criteria(
        self, args: GenerationArguments, input_ids: mx.array
    ) -> Optional[ThinkingBudgetCriteria]:
        if args.thinking_budget is None:
            return None
        tokenizer = self.tokenizer
        thinking_start_token = args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        thinking_end_token = args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        enable_thinking = self._prompt_has_open_thinking(args, input_ids)
        return ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=args.thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
        )

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
        gen_kwargs = {
            **data_kwargs,
            **{k: v for k, v in embed.to_dict().items() if v is not None},
        }
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
            logger.exception("Error loading model in generation thread: %s", e)
            return

        self._ready.set()

        # Diffusion models cannot run through the AR batch generator.
        if is_diffusion_model(self.model):
            self._run_diffusion()
            return

        if self.draft_model is not None and self.draft_kind != "mtp":
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
                active_batch = bool(active)
                coalesce_s = (
                    get_speculative_batch_coalesce_s()
                    if (
                        not active_batch
                        and self.draft_model is not None
                        and self.draft_kind == "mtp"
                    )
                    else 0.0
                )
                new_items, should_stop = self._collect_pending_requests(
                    active=active_batch,
                    coalesce_s=coalesce_s,
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
                            logger.info(
                                "Generation cancelled: request=%s generated_tokens=%d",
                                info.get("request_id", uid),
                                int(info.get("generated_tokens", 0) or 0),
                            )
                            try:
                                info["rqueue"].put(None)
                            except Exception:
                                pass

                if new_items and batch_gen is not None and not active:
                    if not batch_gen.has_work:
                        batch_gen.close()
                        batch_gen = None

                for request in new_items:
                    rqueue = request.rqueue
                    raw_inputs = request.raw_inputs
                    prompt_tokens = request.prompt_tokens
                    args = request.args
                    images = request.images
                    log_state = self._log_prefill_started(
                        request, backend="continuous_batching"
                    )
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
                            compute_logprobs=bool(args.logprobs),
                            top_logprobs_k=self.top_logprobs_k if args.logprobs else 0,
                            stream=generation_stream,
                            apc_manager=self.apc_manager,
                            draft_model=self.draft_model,
                            draft_kind=self.draft_kind,
                            draft_block_size=_get_draft_block_size_from_env(),
                            greedy_sampling=args.temperature == 0,
                            prefill_step_size=get_prefill_step_size(),
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
                        thinking_budget_criteria = request.thinking_budget_criteria
                        (uid,) = batch_gen.insert(
                            [input_ids.squeeze(0).tolist()],
                            max_tokens=args.max_tokens,
                            prompt_kwargs=[gen_kwargs],
                            logits_processors=[
                                self._make_logits_processors(args, input_ids)
                            ],
                            thinking_budget_criteria=[thinking_budget_criteria],
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
                        "cached_tokens": 0,
                        **log_state,
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

        if batch_gen is not None and callable(getattr(batch_gen, "close", None)):
            batch_gen.close()

    def _run_diffusion(self):
        """GPU thread loop for diffusion models.

        Diffusion generation runs one request at a time (batch size 1).
        Output streams back block-by-block: one queue item per denoised
        block, plus a final item carrying the finish reason. Non-streaming
        endpoints aggregate the same items into a single response.
        """
        uid_counter = 0
        cancelled: set = set()
        while not self._stop:
            try:
                new_items, should_stop = self._collect_pending_requests(active=False)
                if should_stop:
                    break
                cancelled |= self._drain_cancellations()
                for request in new_items:
                    rqueue = request.rqueue
                    raw_inputs = request.raw_inputs
                    prompt_tokens = request.prompt_tokens
                    args = request.args
                    log_state = self._log_prefill_started(request, backend="diffusion")
                    uid_counter += 1
                    uid = uid_counter
                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    try:
                        self._generate_diffusion(
                            uid, rqueue, raw_inputs, args, cancelled, log_state
                        )
                        rqueue.put(None)
                    except Exception as e:
                        logger.exception("Error in diffusion generation")
                        try:
                            rqueue.put(e)
                            rqueue.put(None)
                        except Exception:
                            pass
                    mx.clear_cache()
            except Exception:
                logger.exception("Error in diffusion generation thread")
                mx.clear_cache()
                gc.collect()

    def _generate_diffusion(
        self, uid, rqueue, raw_inputs, args, cancelled, log_state=None
    ):
        log_state = log_state or {"request_id": uid, "generated_tokens": 0}
        input_ids = raw_inputs.get("input_ids")
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids[None]
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(self.config.eos_token_id)
        skip_special_token_ids = (
            set(getattr(tokenizer, "all_special_ids", None) or [])
            if args.skip_special_tokens
            else set()
        )

        stream_kwargs = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "mm_token_type_ids": raw_inputs.get("mm_token_type_ids"),
        }
        prefill_step_size = get_prefill_step_size()
        if prefill_step_size > 0:
            stream_kwargs["prefill_step_size"] = prefill_step_size
        if args.seed is not None:
            stream_kwargs["seed"] = args.seed
        if args.logits_processors is not None:
            stream_kwargs["logits_processors"] = args.logits_processors
        stream_kwargs.update(args.diffusion_kwargs())

        emitter = _DiffusionBlockEmitter()
        prefill_logged = False

        def on_result(result):
            nonlocal prefill_logged
            if not prefill_logged and getattr(result, "prompt_tps", None) is not None:
                prompt_tps = float(result.prompt_tps or 0.0)
                prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
                logger.info(
                    "Prefill completed: request=%s prompt_tokens=%d cached_tokens=%d "
                    "elapsed=%.3fs rate=%.1f tok/s",
                    log_state.get("request_id", uid),
                    prompt_tokens,
                    int(getattr(result, "cached_tokens", 0) or 0),
                    prompt_tokens / prompt_tps if prompt_tps > 0 else 0.0,
                    prompt_tps,
                )
                prefill_logged = True
            for chunk in emitter.feed(result):
                self._log_decode_progress(
                    uid,
                    log_state,
                    token=chunk.token,
                    text=chunk.text,
                    finish_reason=chunk.finish_reason,
                    token_count=chunk.token_count,
                )
                rqueue.put(chunk)
                if chunk.finish_reason:
                    return False
            cancelled.update(self._drain_cancellations())
            if uid in cancelled:
                cancelled.discard(uid)
                return False
            return True

        results = stream_diffusion_generate_from_kwargs(
            self.model,
            self.processor,
            tokenizer,
            input_ids,
            raw_inputs.get("pixel_values"),
            raw_inputs.get("attention_mask"),
            skip_special_token_ids,
            stream_kwargs,
            skip_special_tokens=args.skip_special_tokens,
            on_result=on_result,
        )
        try:
            for _ in results:
                pass
        finally:
            results.close()

    def _run_speculative(self):
        """GPU thread loop with DFlash, EAGLE-3, or MTP speculative decoding.

        Collects incoming requests, prefills them as a batch with the
        per-family hooks, then runs the matching round-loop for decode.
        Finished sequences are filtered out automatically by the round-loop's
        ``stop_check`` callback.
        """
        generation_stream = mx.default_stream(mx.default_device())

        lm = self.model.language_model
        drafter = self.draft_model
        draft_kind = self.draft_kind
        is_mtp = draft_kind == "mtp"
        prefill_kwargs = speculative_prefill_kwargs(draft_kind, drafter)
        eos_set = set(self.stop_tokens) if is_mtp else None
        sampler = make_sampler(temp=0)
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

                for request in pending:
                    rqueue = request.rqueue
                    raw_inputs = request.raw_inputs
                    prompt_tokens = request.prompt_tokens
                    args = request.args
                    images = request.images
                    log_state = self._log_prefill_started(
                        request, backend=f"speculative_{draft_kind}"
                    )
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    uid = id(rqueue)
                    uids.append(uid)
                    rqueues[uid] = rqueue
                    token_lists[uid] = []
                    stream_infos[uid] = {
                        "streamer": _ServerTokenStreamer(
                            self.tokenizer,
                            make_streaming_detokenizer(self.processor),
                        ),
                        **log_state,
                    }
                    max_tokens_map[uid] = args.max_tokens
                    prompt_tokens_map[uid] = prompt_tokens
                    all_input_ids.append(input_ids.squeeze(0).tolist())
                    prompt_kwargs_list.append(gen_kwargs)
                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    sampler = self._make_sampler(args) or make_sampler(temp=0)

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

                prefill_step_size = get_prefill_step_size()
                policy_kwargs = {**prompt_kwargs, **prefill_kwargs}
                if not _chunked_prefill_enabled(
                    self.model,
                    input_ids=input_mx,
                    inputs_embeds=inputs_embeds_mx,
                    prompt_cache=prompt_cache,
                    draft_model=drafter,
                    draft_kind=draft_kind,
                    prefill_kwargs=policy_kwargs,
                ):
                    prefill_step_size = None

                prompt_started = time.perf_counter()
                out, input_mx = _run_chunked_speculative_prefill(
                    lm,
                    input_mx,
                    inputs_embeds_mx,
                    prompt_cache,
                    prompt_kwargs,
                    prefill_kwargs,
                    prefill_step_size=prefill_step_size,
                    generation_stream=generation_stream,
                )
                hidden = speculative_hidden_state(draft_kind, out)
                shared_kv_states = out.shared_kv_states if is_mtp else None
                sample_row_ids = [0] * B
                first_bonus = _sample_last_token(
                    out.logits,
                    sampler,
                    row_ids=sample_row_ids,
                    positions=[0] * B,
                )
                mx.eval(first_bonus, hidden, out.logits)
                prompt_elapsed = time.perf_counter() - prompt_started
                for uid in uids:
                    prompt_tokens = prompt_tokens_map[uid]
                    prompt_tps_map[uid] = (
                        prompt_tokens / prompt_elapsed
                        if prompt_tokens > 0 and prompt_elapsed > 0
                        else None
                    )
                    logger.info(
                        "Prefill completed: request=%s prompt_tokens=%d "
                        "cached_tokens=0 elapsed=%.3fs rate=%.1f tok/s",
                        stream_infos[uid].get("request_id", uid),
                        prompt_tokens,
                        prompt_elapsed,
                        float(prompt_tps_map[uid] or 0.0),
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
                    self._log_decode_progress(
                        uid,
                        stream_infos[uid],
                        token=tok,
                        text=text,
                        finish_reason=finish,
                    )
                    rqueues[uid].put(
                        StreamingToken(
                            text=text,
                            token=tok,
                            logprobs=0.0,
                            finish_reason=finish,
                            peak_memory=mx.get_peak_memory() / 1e9 if finish else 0,
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
                        request.args.temperature == 0 for request in pending
                    ),
                    shared_kv_states=shared_kv_states,
                    eos_token_ids=eos_set,
                    prompt_tokens=input_mx,
                    row_ids=sample_row_ids,
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

                        self._log_decode_progress(
                            uid,
                            stream_infos[uid],
                            token=tok,
                            text=text,
                            finish_reason=finish,
                        )

                        rqueues[uid].put(
                            StreamingToken(
                                text=text,
                                token=tok,
                                logprobs=0.0,
                                finish_reason=finish,
                                peak_memory=mx.get_peak_memory() / 1e9 if finish else 0,
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
                    logger.info(
                        "Speculative decode: kind=%s batch=%d tokens=%d "
                        "accept=%.2f rounds=%d",
                        draft_kind,
                        B,
                        sum(len(token_lists[u]) for u in uids),
                        mean_a,
                        len(al),
                    )

                # Finalize any remaining
                for uid in uids:
                    if uid not in finished_uids:
                        text = stream_infos[uid]["streamer"].finalize()
                        self._log_decode_progress(
                            uid,
                            stream_infos[uid],
                            token=0,
                            text=text,
                            finish_reason="length",
                            token_count=0,
                        )
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
                logger.exception("Error in speculative generation thread: %s", e)
                error_queues = {id(rqueue): rqueue for rqueue in rqueues.values()}
                error_queues.update(
                    {id(request.rqueue): request.rqueue for request in pending}
                )
                _notify_queues(error_queues.values(), e, None)
                mx.clear_cache()
                gc.collect()

    def _step(self, batch_gen, active, gen_kwargs=None):
        """One batch generation step: prefill + decode."""
        kwargs = gen_kwargs or {}
        prompt_responses, responses = batch_gen.next(**kwargs)
        self._log_prefill_progress(batch_gen, active)
        for prompt_response in prompt_responses:
            if prompt_response.uid in active:
                info = active[prompt_response.uid]
                info["prompt_tps"] = prompt_response.prompt_tps
                info["cached_tokens"] = getattr(prompt_response, "cached_tokens", 0)
                self._log_prefill_completed(prompt_response.uid, info, prompt_response)
        if not responses:
            return

        for r in responses:
            if r.uid not in active:
                continue

            info = active[r.uid]
            rqueue = info["rqueue"]

            tok = r.token
            token_count = 0 if tok is None else 1
            if tok is None:
                text = info["streamer"].finalize()
                tok = 0
            elif hasattr(tok, "item"):
                tok = tok.item()
                text = self._stream_text(info, tok, r.finish_reason)
            else:
                text = self._stream_text(info, tok, r.finish_reason)

            lp = r.token_logprob

            self._log_decode_progress(
                r.uid,
                info,
                token=tok,
                text=text,
                finish_reason=r.finish_reason,
                token_count=token_count,
            )

            rqueue.put(
                StreamingToken(
                    text=text,
                    token=tok,
                    logprobs=lp,
                    finish_reason=r.finish_reason,
                    peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
                    prompt_tps=info.get("prompt_tps"),
                    top_logprobs=getattr(r, "top_logprobs", None),
                    cached_tokens=info.get("cached_tokens", 0),
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
        videos: Optional[List] = None,
    ):
        """Validate request size before opening a streaming response."""
        if get_configured_context_limit() is None:
            return
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        raw_inputs = self._preprocess_request(prompt, images, audio, videos)
        _check_configured_context_budget(
            _count_prompt_tokens(raw_inputs), args.max_tokens
        )
