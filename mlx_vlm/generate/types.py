from __future__ import annotations

from typing import Any, Callable, Protocol, TypedDict

import mlx.core as mx

try:
    from typing import Unpack as Unpack
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    from typing_extensions import Unpack as Unpack

__all__ = ["GenerateKwargs", "ProcessorLike", "Unpack"]


class ProcessorLike(Protocol):
    tokenizer: Any
    detokenizer: Any


class GenerateKwargs(TypedDict, total=False):
    max_tokens: int
    temperature: float
    repetition_penalty: float | None
    repetition_context_size: int | None
    presence_penalty: float | None
    presence_context_size: int | None
    frequency_penalty: float | None
    frequency_context_size: int | None
    top_p: float
    min_p: float
    top_k: int
    logit_bias: dict[int, float] | None
    prompt_cache: list[Any] | None
    max_kv_size: int | None
    kv_bits: float | None
    kv_group_size: int
    kv_quant_scheme: str
    quantized_kv_start: int
    sampler: Callable[[mx.array], mx.array] | None
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None
    prefill_step_size: int | None
    input_ids: mx.array
    pixel_values: Any
    mask: Any
    resize_shape: tuple[int, int] | None
    eos_tokens: list[int] | list[str] | None
    stopping_criteria: Any
    thinking_budget: int | None
    thinking_end_token: str
    thinking_start_token: str | None
    enable_thinking: bool
    skip_special_tokens: bool
    vision_cache: Any
    prompt_cache_state: Any
    apc_manager: Any
    apc_tenant: str | None
    seed: int | None
    verbose: bool
    video: str | list[str] | None
