from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import time
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm

from .. import apc as _apc
from ..models import cache
from ..prompt_utils import apply_chat_template
from ..sample_utils import top_p_sampling
from ..speculative.utils import (
    make_speculative_prompt_cache,
    run_speculative_rounds,
    run_speculative_server_rounds,
    speculative_hidden_state,
    speculative_prefill_kwargs,
)
from ..turboquant import BatchTurboQuantKVCache, turboquant_enabled
from ..utils import group_images_by_shape, prepare_inputs
from .common import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    _chunked_prefill_enabled,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
DEFAULT_PREFILL_STEP_SIZE = 2048
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_BATCH_SIZE = 8
DEFAULT_BATCH_CACHE_EVAL_INTERVAL = 50


def _get_batch_cache_eval_interval() -> int:
    raw = os.environ.get("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL")
    if raw is None:
        return DEFAULT_BATCH_CACHE_EVAL_INTERVAL
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=%r", raw)
        return DEFAULT_BATCH_CACHE_EVAL_INTERVAL


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
    """Sampler with stateless target draws keyed by generated-token position."""

    def __init__(self, *, temperature: float, top_p: float, seed: int):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)

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


def _generate_module_override(name: str, fallback):
    generate_module = sys.modules.get("mlx_vlm.generate")
    return getattr(generate_module, name, fallback) if generate_module else fallback


def normalize_resize_shape(values):
    if values is None:
        return None
    if not (
        not isinstance(values, (str, bytes))
        and len(values) in (1, 2)
        and all(type(value) is int for value in values)
    ):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    return (values[0], values[0]) if len(values) == 1 else tuple(values)


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    presence_penalty: Optional[float] = None,
    presence_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    top_p: float = DEFAULT_TOP_P,
    min_p: float = DEFAULT_MIN_P,
    top_k: int = DEFAULT_TOP_K,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[float] = None,
    kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    draft_model: Optional[nn.Module] = None,
    draft_kind: str = "dflash",
    draft_block_size: Optional[int] = None,
    prompt_cache_checkpoint: Optional[Callable[[int, List[Any]], None]] = None,
    prompt_cache_checkpoint_len: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty.
        presence_penalty (float, optional): Additive penalty for tokens that
          already appeared in recent generated context.
        presence_context_size (int, optional): The number of tokens to
          consider for presence penalty.
        frequency_penalty (float, optional): Additive penalty scaled by token
          frequency in recent generated context.
        frequency_context_size (int, optional): The number of tokens to
          consider for frequency penalty.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): Minimum probability threshold relative to the
          highest-probability token.
        top_k (int, optional): Restrict sampling to the top-k tokens.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (float, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for uniform KV cache quantization.
        kv_quant_scheme (str): KV cache quantization backend.
        quantized_kv_start (int): Start index for quantized KV cache.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunked prefill processes prompts in smaller chunks to reduce peak
          memory usage.
        draft_model (nn.Module, optional): A drafter for speculative decoding.
          When set, the decode loop is replaced by the drafter's speculative
          loop (e.g. DFlash block-diffusion). VLM prefill with image/audio
          is supported via the same ``get_input_embeddings`` path the normal
          decoder uses; decode itself is text-only. ``temperature`` and
          ``sampler`` are respected; ``logprobs`` is always ``None`` on the
          speculative path.
        draft_block_size (int, optional): Override the drafter's configured
          block size.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        _generate_module_override("maybe_quantize_kv_cache", maybe_quantize_kv_cache),
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
    )

    sampler_is_greedy = sampler is None and temperature == 0
    if sampler is None:
        if (
            seed is not None
            and temperature > 0
            and min_p == DEFAULT_MIN_P
            and top_k == DEFAULT_TOP_K
        ):
            sampler = _PositionedTargetSampler(
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        else:
            sampler = _generate_module_override("make_sampler", make_sampler)(
                temp=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
            )

    processors = _generate_module_override(
        "make_logits_processors", make_logits_processors
    )(
        logit_bias,
        repetition_penalty,
        repetition_context_size,
        presence_penalty,
        presence_context_size,
        frequency_penalty,
        frequency_context_size,
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)
    target_sample_position = 0

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    # Speculative decoding setup
    last_outputs = None
    speculative_prefill_capture_kwargs = {}
    if draft_model is not None:
        from ..speculative.drafters import validate_drafter_compatibility

        validate_drafter_compatibility(model, draft_model, draft_kind)
        speculative_prefill_capture_kwargs = speculative_prefill_kwargs(
            draft_kind, draft_model
        )
        # Reset stale mRoPE state from any previous generation.
        lm = model.language_model if hasattr(model, "language_model") else model
        if hasattr(lm, "_position_ids"):
            lm._position_ids = None
        if hasattr(lm, "_rope_deltas"):
            lm._rope_deltas = None

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs, last_outputs, target_sample_position

        step_kwargs = kwargs
        if speculative_prefill_capture_kwargs:
            step_kwargs = {**kwargs, **speculative_prefill_capture_kwargs}

        with mx.stream(generation_stream):
            if "decoder_input_ids" in step_kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **step_kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **step_kwargs,
                )

            last_outputs = outputs
            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            y = _sample_with_positions(
                sampler,
                logprobs,
                row_ids=[0] * logprobs.shape[0],
                positions=list(
                    range(
                        target_sample_position,
                        target_sample_position + logprobs.shape[0],
                    )
                ),
            )
            target_sample_position += logprobs.shape[0]

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs

    with mx.stream(generation_stream):
        # Get input embeddings (handles both multimodal and text-only)
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )

        inputs_embeds = embedding_output.inputs_embeds

        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )
        policy_kwargs = kwargs
        if speculative_prefill_capture_kwargs:
            policy_kwargs = {**kwargs, **speculative_prefill_capture_kwargs}
        if prefill_step_size is not None and not _chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            prompt_cache=prompt_cache,
            draft_model=draft_model,
            draft_kind=draft_kind,
            prefill_kwargs=policy_kwargs,
        ):
            prefill_step_size = None
        checkpoint_len = (
            int(prompt_cache_checkpoint_len)
            if prompt_cache_checkpoint is not None
            and prompt_cache_checkpoint_len is not None
            else None
        )
        checkpoint_done = False
        should_chunk = (
            prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size
        ) or (
            checkpoint_len is not None and 0 < checkpoint_len < inputs_embeds.shape[1]
        )
        if prefill_step_size is not None and should_chunk:
            # Chunked prefill with embeddings
            total_tokens = inputs_embeds.shape[1]
            processed_tokens = 0
            with tqdm(
                total=total_tokens, desc="Prefill", unit="tok", disable=not verbose
            ) as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    if (
                        checkpoint_len is not None
                        and not checkpoint_done
                        and processed_tokens < checkpoint_len
                        and processed_tokens + n_to_process > checkpoint_len
                    ):
                        n_to_process = checkpoint_len - processed_tokens
                    model.language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        n_to_process=n_to_process,
                        **kwargs,
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    processed_tokens += n_to_process
                    if (
                        checkpoint_len is not None
                        and not checkpoint_done
                        and processed_tokens == checkpoint_len
                    ):
                        prompt_cache_checkpoint(processed_tokens, prompt_cache)
                        checkpoint_done = True
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)

            input_ids = input_ids[:, -1:]

        y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

    mx.async_eval(y, logprobs)

    # Speculative decoding
    if draft_model is not None:
        yield from run_speculative_rounds(
            model,
            draft_model,
            prompt_cache,
            input_ids,
            y,
            logprobs,
            last_outputs,
            draft_kind=draft_kind,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            sampler_is_greedy=sampler_is_greedy,
        )
        return

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()

        if thinking_budget_criteria is not None:
            next_y = thinking_budget_criteria.apply_forced_token(next_y)
        y, logprobs = next_y, next_logprobs
        n += 1


@dataclass
class BatchGenerationResult:
    """
    Result of batch generation with optional image size tracking.

    Attributes:
        texts: Generated text for each sample
        tokens: Last generated token for each sample
        logprobs: Log probabilities for each sample
        prompt_tokens: Number of prompt tokens per sample
        generation_tokens: Number of generated tokens per sample
        total_tokens: Total tokens (prompt + generation) per sample
        prompt_tps: Prompt tokens per second per sample
        generation_tps: Generation tokens per second per sample
        peak_memory: Peak memory usage in GB
        image_sizes: Original (height, width) for each image (for tracking)
    """

    texts: List[str]
    tokens: List[Optional[int]]
    logprobs: List[Optional[List[float]]]
    prompt_tokens: List[int]
    generation_tokens: List[int]
    total_tokens: List[int]
    prompt_tps: List[float]
    generation_tps: List[float]
    peak_memory: float = 0.0
    image_sizes: Optional[List[Tuple[int, int]]] = None


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([list(p) + [0] * (max_length - len(p)) for p in prompts])


_SEQUENCE_ALIGNED_PROMPT_KWARGS = {
    "attention_mask",
    "decoder_inputs_embeds",
    "deepstack_visual_embeds",
    "visual_pos_masks",
    "per_layer_inputs",
    "full_text_row_masked_out_mask",
    "position_ids",
    "pos_hw",
    "mm_token_type_ids",
    "token_type_ids",
}

APC_PRIVATE_PROMPT_KEYS = ("_apc_tenant", "_apc_image_hash")


def _prompt_kwarg_row(v: mx.array, row_idx: int, batch_size: int) -> mx.array:
    if v.shape[0] == batch_size:
        return v[row_idx : row_idx + 1]
    return v[:1]


def _split_prompt_kwargs_per_row(prompt_kwargs: dict, batch_size: int) -> List[dict]:
    """Normalize batched prompt kwargs into one dict per batch row.

    ``model.get_input_embeddings()`` commonly returns batch-sized tensors
    (notably ``inputs_embeds``). ``BatchGenerator.insert()`` stores prompt
    kwargs per sequence, so passing the same batched dict for every row causes
    the prompt builder to concatenate those batched tensors ``batch_size``
    times, effectively squaring the batch dimension.
    """
    if batch_size <= 1:
        return [prompt_kwargs or {}]

    rows = [{} for _ in range(batch_size)]
    for k, v in (prompt_kwargs or {}).items():
        if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
            for i in range(batch_size):
                rows[i][k] = _prompt_kwarg_row(v, i, batch_size)
        else:
            for row in rows:
                row[k] = v
    return rows


def _is_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, sequence_length: int
) -> bool:
    return (
        key in _SEQUENCE_ALIGNED_PROMPT_KWARGS
        and v.ndim >= 2
        and v.shape[1] == sequence_length
    )


def _pad_sequence_aligned_prompt_kwarg(
    v: mx.array, target_length: int, *, left: bool
) -> mx.array:
    pad = target_length - v.shape[1]
    if pad <= 0:
        return v
    pad_shape = (v.shape[0], pad) + tuple(v.shape[2:])
    pad_v = mx.zeros(pad_shape, dtype=v.dtype)
    parts = [pad_v, v] if left else [v, pad_v]
    return mx.concatenate(parts, axis=1)


def _merge_prefill_prompt_kwargs(
    prompt_kwargs_list: List[Optional[dict]],
    input_ids: List[List[int]],
) -> Tuple[mx.array, dict]:
    """Batch per-row prompt kwargs for a left-padded prefill forward."""
    lengths = [len(ids) for ids in input_ids]
    max_length = max(lengths)

    row_embeds: List[mx.array] = []
    embed_dtype = None
    embed_dim = None
    for kw, length in zip(prompt_kwargs_list, lengths):
        if not kw or kw.get("inputs_embeds") is None:
            raise ValueError("inputs_embeds is required")
        embeds = kw["inputs_embeds"]  # [1, length, D]
        embed_dtype = embeds.dtype
        embed_dim = embeds.shape[-1]
        if length < max_length:
            pad = mx.zeros(
                (embeds.shape[0], max_length - length, embed_dim),
                dtype=embed_dtype,
            )
            embeds = mx.concatenate([pad, embeds], axis=1)
        row_embeds.append(embeds)
    inputs_embeds = mx.concatenate(row_embeds, axis=0)

    merged_kwargs: dict = {}
    per_row_keys: dict = {}
    batch_size = len(prompt_kwargs_list)
    for i, (kw, length) in enumerate(zip(prompt_kwargs_list, lengths)):
        if not kw:
            continue
        for k, v in kw.items():
            if k == "inputs_embeds" or k in APC_PRIVATE_PROMPT_KEYS:
                continue
            if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
                row_v = _prompt_kwarg_row(v, i, batch_size)
                if _is_sequence_aligned_prompt_kwarg(k, row_v, length):
                    row_v = _pad_sequence_aligned_prompt_kwarg(
                        row_v, max_length, left=True
                    )
                per_row_keys.setdefault(k, []).append(row_v)
            elif k not in merged_kwargs:
                merged_kwargs[k] = v
    for k, vs in per_row_keys.items():
        merged_kwargs[k] = mx.concatenate(vs, axis=0)

    return inputs_embeds, merged_kwargs


def _extend_cache(cache_a, cache_b):
    """Extend cache_a with cache_b along the batch dimension."""
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    extended = []
    for ca, cb in zip(cache_a, cache_b):
        if not hasattr(ca, "left_padding") and hasattr(ca.__class__, "merge"):
            ca = ca.__class__.merge([ca])
        if not hasattr(cb, "left_padding") and hasattr(cb.__class__, "merge"):
            cb = cb.__class__.merge([cb])
        ca.extend(cb)
        extended.append(ca)
    return extended


def _make_cache(
    model,
    left_padding,
    kv_bits=None,
    kv_group_size=64,
    kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.

    When *kv_bits* is set, a quantized batch cache is used instead of
    ``BatchKVCache`` so that KV states are quantized on-the-fly during
    generation, reducing memory usage for long sequences.

    *kv_quant_scheme* selects the quantization backend:
    - ``"uniform"`` → ``BatchQuantizedKVCache`` (``mx.quantize``)
    - ``"turboquant"`` or fractional *kv_bits* → ``BatchTurboQuantKVCache``
    """
    use_turbo = kv_bits is not None and turboquant_enabled(kv_bits, kv_quant_scheme)

    def _make_quant_cache(lp):
        if use_turbo:
            return BatchTurboQuantKVCache(lp, bits=kv_bits)
        return cache.BatchQuantizedKVCache(
            lp, group_size=kv_group_size, bits=int(kv_bits)
        )

    def to_batch_cache(c, quantize=True):
        if isinstance(c, cache.KVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ChunkedKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.SimpleKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        elif isinstance(c, tuple):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        n = len(model_cache)
        # Skip quantizing the last layer — it's sensitive to quantization
        return [
            to_batch_cache(c, quantize=(i < n - 1 if n > 2 else True))
            for i, c in enumerate(model_cache)
        ]
    else:
        if kv_bits is not None:
            n = len(model.layers)
            return [
                (
                    _make_quant_cache(left_padding)
                    if i < n - 1 or n <= 2
                    else cache.BatchKVCache(left_padding)
                )
                for i in range(n)
            ]
        return [cache.BatchKVCache(left_padding) for _ in model.layers]


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        image_sizes: (Optional[List[Tuple[int, int]]]): Original (height, width)
            for each image. Useful for tracking which images produced which responses
            and for debugging padding/batching behavior.
    """

    texts: List[str]
    stats: BatchStats
    image_sizes: Optional[List[Tuple[int, int]]] = None


@dataclass
class PromptProgress:
    """Per-request prompt processing metrics for continuous batching."""

    uid: int
    prompt_tokens: int
    prompt_tps: float = 0.0
    prompt_time: float = 0.0
    cached_tokens: int = 0


def _sample_with_positions(
    sampler: Callable[[mx.array], mx.array],
    logprobs: mx.array,
    *,
    row_ids: Optional[List[int]] = None,
    positions: Optional[List[int]] = None,
) -> mx.array:
    sample_target = getattr(sampler, "sample_target", None)
    if callable(sample_target) and row_ids is not None and positions is not None:
        return sample_target(logprobs, row_ids=row_ids, positions=positions)
    return sampler(logprobs)


class GenerationBatch:
    """
    Batched token generator with double-buffered pipelining.

    Manages the generation phase after prompt processing, with KV caches,
    sampling, and stop detection for multiple sequences. Uses async_eval
    to overlap GPU computation with CPU processing (decode-ahead pattern).
    """

    @dataclass
    class Response:
        uid: int
        token: int
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[List[Tuple[int, float]]] = None

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: List[int],
        top_logprobs_k: int = 0,
        greedy_sampling: bool = False,
        token_context: Optional[List[List[int]]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
    ):
        self.model = model
        self._language_model = getattr(model, "language_model", model)
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = max_tokens
        self._num_tokens = [0] * len(uids)
        self.compute_logprobs = True
        self.top_logprobs_k = top_logprobs_k
        self.greedy_sampling = greedy_sampling
        self.logits_processors = logits_processors or []
        self.thinking_budget_criteria = thinking_budget_criteria or []
        self.token_context = [list(ctx) for ctx in (token_context or [])]
        self._ensure_token_context()

        self._current_tokens = None
        self._current_lps = None
        self._next_tokens = inputs
        self._next_lps = None
        self._next_top_idx = None
        self._next_top_lp = None

        # Per-sequence MRoPE delta
        self._rope_deltas = None

    def __len__(self):
        return len(self.uids)

    def cache_states(self):
        return [c.state for c in self.prompt_cache if hasattr(c, "state")]

    def _ensure_token_context(self, *, force: bool = False):
        if not (force or (self.logits_processors and any(self.logits_processors))):
            if not self.logits_processors:
                self.token_context = []
            return
        if len(self.token_context) < len(self.uids):
            missing = len(self.uids) - len(self.token_context)
            self.token_context.extend([[] for _ in range(missing)])
        elif len(self.token_context) > len(self.uids):
            self.token_context = self.token_context[: len(self.uids)]

    def _greedy_argmax_step(self, inputs: mx.array, fwd_kwargs: dict):
        if (
            not self.greedy_sampling
            or self.compute_logprobs
            or self.top_logprobs_k > 0
            or (self.logits_processors and any(self.logits_processors))
        ):
            return None

        argmax_from_hidden = getattr(
            self._language_model, "speculative_argmax_from_hidden", None
        )
        if not callable(argmax_from_hidden):
            return None

        output = self._language_model(
            inputs[:, None],
            cache=self.prompt_cache,
            return_hidden=True,
            skip_logits=True,
            **fwd_kwargs,
        )
        hidden = output.hidden_states[-1]
        sampled = argmax_from_hidden(hidden)
        if sampled is None:
            return None
        if sampled.ndim == 2 and sampled.shape[1] == 1:
            sampled = sampled[:, 0]
        return sampled

    def _step(self):
        """Perform one generation step with double buffering."""
        self._current_tokens = self._next_tokens
        self._current_lps = self._next_lps
        inputs = self._current_tokens

        fwd_kwargs = {}
        if self._rope_deltas is not None:
            fwd_kwargs["rope_deltas"] = self._rope_deltas

        sampled = self._greedy_argmax_step(inputs, fwd_kwargs)
        if sampled is not None:
            self._next_tokens = sampled
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            mx.async_eval(self._next_tokens)
            mx.eval(inputs)
            return inputs.tolist(), None, None, None

        output = self._language_model(
            inputs[:, None], cache=self.prompt_cache, **fwd_kwargs
        )
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        if self.logits_processors and any(self.logits_processors):
            last_tokens = inputs.tolist()
            self._ensure_token_context()
            for i, token in enumerate(last_tokens):
                self.token_context[i].append(token)

            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    if hasattr(processor, "process_last_token"):
                        sample_logits = processor.process_last_token(
                            last_tokens[i], sample_logits
                        )
                    else:
                        sample_logits = processor(
                            mx.array(self.token_context[i]), sample_logits
                        )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = _sample_with_positions(
            self.sampler,
            logprobs,
            row_ids=[0] * len(self.uids),
            positions=[n + 1 for n in self._num_tokens],
        )

        self._next_tokens = sampled
        prev_top_idx = self._next_top_idx
        prev_top_lp = self._next_top_lp

        eval_targets = [self._next_tokens]
        if self.compute_logprobs:
            self._next_lps = logprobs[mx.arange(sampled.shape[0]), sampled]
            eval_targets.append(self._next_lps)
        else:
            self._next_lps = None

        k = self.top_logprobs_k
        if k > 0:
            # argsort ascending; take last K columns and reverse for descending.
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            self._next_top_idx = top_idx
            self._next_top_lp = top_lp
            eval_targets.extend([top_idx, top_lp])
        else:
            self._next_top_idx = None
            self._next_top_lp = None

        mx.async_eval(*eval_targets)

        if self._current_lps is not None:
            to_eval = [inputs, self._current_lps]
            if prev_top_idx is not None:
                to_eval.extend([prev_top_idx, prev_top_lp])
            mx.eval(*to_eval)
            top_idx_list = prev_top_idx.tolist() if prev_top_idx is not None else None
            top_lp_list = prev_top_lp.tolist() if prev_top_lp is not None else None
            return (
                inputs.tolist(),
                self._current_lps.tolist(),
                top_idx_list,
                top_lp_list,
            )
        else:
            mx.eval(inputs)
            return inputs.tolist(), None, None, None

    def _eval_pending_state(self):
        """Materialize lazy decode outputs before mutating batch-owned state."""
        targets = []

        def append_arrays(value):
            if isinstance(value, mx.array):
                targets.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    append_arrays(item)

        append_arrays(
            (
                self._current_tokens,
                self._current_lps,
                self._next_tokens,
                self._next_lps,
                self._next_top_idx,
                self._next_top_lp,
                self._rope_deltas,
            )
        )
        for c in self.prompt_cache:
            try:
                append_arrays(c.state)
            except (AttributeError, TypeError):
                pass

        if targets:
            mx.eval(*targets)

    def extend(self, other: "GenerationBatch"):
        """Extend this batch with another generation batch."""
        self_was_empty = len(self.uids) == 0
        if not self_was_empty and len(other.uids) > 0:
            self._eval_pending_state()
            other._eval_pending_state()

        self_has_processors = self.logits_processors and any(self.logits_processors)
        other_has_processors = other.logits_processors and any(other.logits_processors)
        if self_has_processors or other_has_processors:
            self._ensure_token_context(force=bool(other_has_processors))
            other._ensure_token_context(force=bool(self_has_processors))
        else:
            self.token_context = []
            other.token_context = []
            self.logits_processors = []
            other.logits_processors = []

        self.uids.extend(other.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, other.prompt_cache)
        self.max_tokens.extend(other.max_tokens)
        self._num_tokens.extend(other._num_tokens)
        self.token_context.extend(other.token_context)
        self.logits_processors.extend(other.logits_processors)
        self.thinking_budget_criteria.extend(other.thinking_budget_criteria)
        self._ensure_token_context()

        if self._current_tokens is None:
            self._current_tokens = other._current_tokens
            self._current_lps = other._current_lps
        elif other._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, other._current_tokens]
            )
            if self._current_lps is not None and other._current_lps is not None:
                self._current_lps = mx.concatenate(
                    [self._current_lps, other._current_lps]
                )

        if self._next_tokens is None:
            self._next_tokens = other._next_tokens
            self._next_lps = other._next_lps
            self._next_top_idx = other._next_top_idx
            self._next_top_lp = other._next_top_lp
        elif other._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, other._next_tokens])
            if self._next_lps is not None and other._next_lps is not None:
                self._next_lps = mx.concatenate([self._next_lps, other._next_lps])

            if (
                self._next_top_idx is not None
                and other._next_top_idx is not None
                and self._next_top_idx.shape[-1] == other._next_top_idx.shape[-1]
            ):
                self._next_top_idx = mx.concatenate(
                    [self._next_top_idx, other._next_top_idx]
                )
                self._next_top_lp = mx.concatenate(
                    [self._next_top_lp, other._next_top_lp]
                )
            else:
                self._next_top_idx = None
                self._next_top_lp = None

        if self_was_empty:
            self._rope_deltas = other._rope_deltas
        elif (self._rope_deltas is None) != (other._rope_deltas is None):
            raise RuntimeError(
                "extend() mixes MRoPE and non-MRoPE batches; both sides must "
                "carry rope_deltas or neither side may."
            )
        elif self._rope_deltas is not None:
            self._rope_deltas = mx.concatenate([self._rope_deltas, other._rope_deltas])

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        if len(keep) < len(self.uids):
            self._eval_pending_state()

        self.uids = [self.uids[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        if self.token_context:
            self.token_context = [self.token_context[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        if self.thinking_budget_criteria:
            self.thinking_budget_criteria = [
                self.thinking_budget_criteria[idx] for idx in keep
            ]

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_lps = None
            self._next_tokens = None
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            self._rope_deltas = None
            self.token_context = []
            self.logits_processors = []
            self.thinking_budget_criteria = []
        else:
            keep_arr = mx.array(keep, mx.int32)
            for c in self.prompt_cache:
                c.filter(keep_arr)
            if self._next_tokens is not None:
                self._next_tokens = self._next_tokens[keep_arr]
            if self._next_lps is not None:
                self._next_lps = self._next_lps[keep_arr]
            if self._next_top_idx is not None:
                self._next_top_idx = self._next_top_idx[keep_arr]
                self._next_top_lp = self._next_top_lp[keep_arr]
            if self._rope_deltas is not None:
                self._rope_deltas = self._rope_deltas[keep_arr]

    def next(self) -> List[Response]:
        """Generate the next batch of tokens."""
        if not self.uids:
            return []

        tokens, lp_list, top_idx_list, top_lp_list = self._step()

        keep = []
        responses = []
        forced_next_tokens = None
        for i in range(len(self.uids)):
            finish_reason = None
            self._num_tokens[i] += 1
            tok = tokens[i]
            if (
                i < len(self.thinking_budget_criteria)
                and self.thinking_budget_criteria[i] is not None
            ):
                criteria = self.thinking_budget_criteria[i]
                criteria(tok)
                if forced_next_tokens is None:
                    mx.eval(self._next_tokens)
                    forced_next_tokens = self._next_tokens.tolist()
                next_y = criteria.apply_forced_token(
                    mx.array([forced_next_tokens[i]], dtype=mx.int32)
                )
                next_token = int(next_y.item())
                if next_token != forced_next_tokens[i]:
                    forced_next_tokens[i] = next_token

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            top_lp = None
            if top_idx_list is not None:
                top_lp = list(zip(top_idx_list[i], top_lp_list[i]))

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=tok,
                    token_logprob=lp_list[i] if lp_list is not None else 0.0,
                    finish_reason=finish_reason,
                    top_logprobs=top_lp,
                )
            )

        if forced_next_tokens is not None:
            self._next_tokens = mx.array(forced_next_tokens, dtype=mx.int32)
            mx.async_eval(self._next_tokens)

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model,
        sampler,
        stop_criteria,
        compute_logprobs=True,
        top_logprobs_k=0,
        greedy_sampling: bool = False,
    ):
        """Create an empty generation batch."""
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.compute_logprobs = compute_logprobs
        batch.top_logprobs_k = top_logprobs_k
        batch.greedy_sampling = greedy_sampling
        batch.token_context = []
        batch.logits_processors = []
        batch.thinking_budget_criteria = []
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch._next_lps = None
        batch._next_top_idx = None
        batch._next_top_lp = None
        batch._rope_deltas = None
        return batch


class SpeculativeGenerationBatch:
    """GenerationBatch-compatible wrapper for server-side MTP decode."""

    is_speculative = True
    Response = GenerationBatch.Response

    def __init__(
        self,
        model: nn.Module,
        draft_model: nn.Module,
        draft_kind: str,
        uids: List[int],
        first_tokens: mx.array,
        prompt_cache: List[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: List[int],
        hidden: mx.array,
        shared_kv_states: Optional[dict],
        prompt_tokens: mx.array,
        *,
        draft_block_size: Optional[int] = None,
        token_dtype: mx.Dtype = mx.int32,
        greedy_sampling: bool = False,
    ):
        self.model = model
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.uids = list(uids)
        self._all_uids = list(uids)
        self.first_tokens = first_tokens
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = list(max_tokens)
        self.hidden = hidden
        self.shared_kv_states = shared_kv_states
        self.prompt_tokens = prompt_tokens
        self.draft_block_size = draft_block_size
        self.token_dtype = token_dtype
        self.greedy_sampling = greedy_sampling
        self._num_tokens = [0] * len(uids)
        self._finished = [False] * len(uids)
        self._sent_first = False
        self._rounds_iter = None

    def __len__(self):
        return sum(not done for done in self._finished)

    def _refresh_uids(self):
        self.uids = [
            uid for uid, done in zip(self._all_uids, self._finished) if not done
        ]

    def extend(self, other: "SpeculativeGenerationBatch"):
        if len(self) == 0:
            self.__dict__.update(other.__dict__)
            return
        raise RuntimeError("Cannot extend an active speculative generation batch.")

    def filter(self, keep: List[int]):
        keep_uids = {self.uids[idx] for idx in keep}
        for i, uid in enumerate(self._all_uids):
            if uid not in keep_uids:
                self._finished[i] = True
        self._refresh_uids()

    def cache_states(self):
        return [c.state for c in self.prompt_cache if hasattr(c, "state")]

    def _finish_reason(self, row: int, token: int) -> Optional[str]:
        if self.stop_criteria(token):
            return "stop"
        if self._num_tokens[row] >= self.max_tokens[row]:
            return "length"
        return None

    def _append_token_responses(
        self,
        responses: List[GenerationBatch.Response],
        tok_list: List[Optional[int]],
    ) -> None:
        for row, token in enumerate(tok_list):
            if token is None or self._finished[row]:
                continue
            token = int(token)
            self._num_tokens[row] += 1
            finish_reason = self._finish_reason(row, token)
            if finish_reason is not None:
                self._finished[row] = True
            responses.append(
                self.Response(
                    uid=self._all_uids[row],
                    token=token,
                    token_logprob=0.0,
                    finish_reason=finish_reason,
                )
            )

    def _start_rounds(self):
        if self._rounds_iter is not None:
            return

        def stop_check(seq_idx, token_id):
            return (
                self._finished[seq_idx]
                or self.stop_criteria(token_id)
                or self._num_tokens[seq_idx] >= self.max_tokens[seq_idx]
            )

        self._rounds_iter = run_speculative_server_rounds(
            self.model,
            self.draft_model,
            self.prompt_cache,
            self.hidden,
            draft_kind=self.draft_kind,
            first_bonus=self.first_tokens,
            max_tokens=max(self.max_tokens) if self.max_tokens else 0,
            sampler=self.sampler,
            draft_block_size=self.draft_block_size,
            token_dtype=self.token_dtype,
            stop_check=stop_check,
            greedy_sampling=self.greedy_sampling,
            shared_kv_states=self.shared_kv_states,
            eos_token_ids=None,
            prompt_tokens=self.prompt_tokens,
            row_ids=[0] * len(self._all_uids),
        )

    def next(self) -> List[GenerationBatch.Response]:
        if len(self) == 0:
            return []

        responses: List[GenerationBatch.Response] = []
        if not self._sent_first:
            self._sent_first = True
            mx.eval(self.first_tokens)
            for row, token in enumerate(self.first_tokens.tolist()):
                if self._finished[row]:
                    continue
                token = int(token)
                self._num_tokens[row] += 1
                finish_reason = self._finish_reason(row, token)
                if finish_reason is not None:
                    self._finished[row] = True
                responses.append(
                    self.Response(
                        uid=self._all_uids[row],
                        token=token,
                        token_logprob=0.0,
                        finish_reason=finish_reason,
                    )
                )
            self._refresh_uids()
            return responses

        self._start_rounds()
        try:
            tok_list, round_meta = next(self._rounds_iter)
        except StopIteration:
            for row, done in enumerate(self._finished):
                if not done:
                    self._finished[row] = True
                    responses.append(
                        self.Response(
                            uid=self._all_uids[row],
                            token=None,
                            token_logprob=0.0,
                            finish_reason="length",
                        )
                    )
            self._refresh_uids()
            return responses

        self._append_token_responses(responses, tok_list)
        while isinstance(round_meta, dict) and int(
            round_meta.get("round_pos", 0)
        ) + 1 < int(round_meta.get("round_len", 1)):
            try:
                tok_list, round_meta = next(self._rounds_iter)
            except StopIteration:
                break
            self._append_token_responses(responses, tok_list)

        self._refresh_uids()
        return responses


class PromptProcessingBatch:
    """
    Handles VLM prompt processing with inputs_embeds and chunked prefill.

    Processes prompt tokens incrementally (one chunk per step) to allow
    interleaving with generation for continuous batching. Transitions to
    a GenerationBatch when prompt processing is complete.
    """

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        input_ids: List[List[int]],
        max_tokens: List[int],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        warm_cache: Optional[List[Any]] = None,
        apc_meta: Optional[List[dict]] = None,
        apc_manager: Optional["_apc.APCManager"] = None,
        right_pad_per_row: Optional[List[int]] = None,
        suffix_lens: Optional[List[int]] = None,
        apc_mode: Optional[str] = None,
        draft_model: Optional[nn.Module] = None,
        draft_kind: Optional[str] = None,
        draft_block_size: Optional[int] = None,
        greedy_sampling: bool = False,
    ):
        self.model = model
        self.uids = uids
        self._prompt_uids = list(uids)
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.draft_block_size = draft_block_size
        self.greedy_sampling = greedy_sampling

        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        # ``input_ids`` here are the per-row prefill inputs — for warm-start
        # rows this is the suffix, for cold rows the full prompt. When
        # ``right_pad_per_row`` is set the rows are right-padded (used in
        # mixed warm/cold prefill so suffix RoPE positions align). Otherwise
        # we left-pad as before.
        self._right_pad_per_row = right_pad_per_row
        self._suffix_lens = suffix_lens or lengths
        self._left_padding_per_row: List[int]

        if right_pad_per_row is not None:
            # Right-pad each row to max_length (so the last `pad[i]` cells are
            # right-pad and need to be rolled into left-pad by finalize()).
            left_padding = [0] * len(input_ids)
            self._input_ids = _right_pad_prompts(input_ids, max_length=max_length)
        else:
            left_padding = [max_length - l for l in lengths]
            self._input_ids = _left_pad_prompts(input_ids, max_length=max_length)
        self._left_padding_per_row = list(left_padding)
        self._total_prompt_tokens = sum(lengths)
        self._processed_prompt_columns = 0

        self.logits_processors = logits_processors or []
        self.thinking_budget_criteria = thinking_budget_criteria or []
        self._token_context = (
            [list(ids) for ids in input_ids]
            if self.logits_processors and any(self.logits_processors)
            else []
        )
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs or {}
        self._prompt_length_aware_keys: List[str] = []
        if self._prompt_kwargs and self._inputs_embeds is not None:
            prompt_batch = self._inputs_embeds.shape[0]
            prompt_len = self._inputs_embeds.shape[1]
            for k, v in self._prompt_kwargs.items():
                if (
                    isinstance(v, mx.array)
                    and v.ndim >= 2
                    and v.shape[0] == prompt_batch
                    and v.shape[1] == prompt_len
                ):
                    self._prompt_length_aware_keys.append(k)

        # APC metadata used for post-prefill block harvest (per-row).
        self._apc_meta = apc_meta or []
        self._apc_manager = apc_manager
        self._apc_mode = apc_mode
        self._apc_harvest_enabled = True
        self._prompt_time_s = 0.0
        self._prompt_tokens_per_row: List[int] = []
        self._cached_tokens_per_row: List[int] = []
        for idx, suffix_len in enumerate(lengths):
            full_input_ids = None
            prefix_len = 0
            if idx < len(self._apc_meta) and self._apc_meta[idx] is not None:
                full_input_ids = self._apc_meta[idx].get("full_input_ids")
                prefix_len = int(self._apc_meta[idx].get("prefix_len") or 0)
            self._prompt_tokens_per_row.append(
                len(full_input_ids) if full_input_ids is not None else suffix_len
            )
            self._cached_tokens_per_row.append(prefix_len)

        if warm_cache is not None:
            self.prompt_cache = warm_cache
        elif draft_model is not None and draft_kind is not None:
            self.prompt_cache = make_speculative_prompt_cache(
                model,
                draft_kind=draft_kind,
                batch_size=len(input_ids),
                left_padding=left_padding,
                make_cache=lambda lm, lp: _make_cache(
                    lm,
                    lp,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    kv_quant_scheme=kv_quant_scheme,
                ),
            )
        elif (
            len(input_ids) == 1
            and right_pad_per_row is None
            and kv_bits is None
            and hasattr(model, "make_cache")
        ):
            self.prompt_cache = cache.make_prompt_cache(model)
        else:
            self.prompt_cache = _make_cache(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                kv_quant_scheme=kv_quant_scheme,
            )

        # Declare per-row right-padding on each cache so finalize() can roll
        # it into left-padding once the prefill forward pass is complete.
        if right_pad_per_row is not None and any(right_pad_per_row):
            for c in self.prompt_cache:
                prepare = getattr(c, "prepare", None)
                if not callable(prepare):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with prepare()"
                    )
                prepare(right_padding=right_pad_per_row, lengths=self._suffix_lens)

        if self.prefill_step_size is not None:
            policy_kwargs = dict(self._prompt_kwargs)
            if draft_model is not None and draft_kind is not None:
                policy_kwargs.update(
                    speculative_prefill_kwargs(draft_kind, draft_model)
                )
            if not _chunked_prefill_enabled(
                self.model,
                input_ids=self._input_ids,
                inputs_embeds=self._inputs_embeds,
                prompt_cache=self.prompt_cache,
                draft_model=draft_model,
                draft_kind=draft_kind,
                prefill_kwargs=policy_kwargs,
            ):
                self.prefill_step_size = None

    def __len__(self):
        return len(self.uids)

    def _release_apc_meta_blocks(self):
        if self._apc_manager is None:
            return
        for meta in self._apc_meta:
            if meta is not None:
                self._apc_manager.release(meta.get("apc_blocks", []))

    def needs_processing(self):
        """True if prompt needs chunked processing before generate()."""
        if self._inputs_embeds is None or self.prefill_step_size is None:
            return self._next_apc_checkpoint_column() is not None
        if self._next_apc_checkpoint_column() is not None:
            return True
        return self._inputs_embeds.shape[1] > self.prefill_step_size

    def _apc_checkpoint_column_for_meta(
        self, batch_idx: int, meta: dict
    ) -> Optional[int]:
        checkpoint_len = int(meta.get("checkpoint_len") or 0)
        if (
            self._apc_mode != "exact"
            or checkpoint_len <= 0
            or meta.get("checkpoint_done")
        ):
            return None
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if checkpoint_len <= prefix_len:
            meta["checkpoint_done"] = True
            return None
        if self._right_pad_per_row is not None:
            suffix_checkpoint = checkpoint_len - prefix_len
            if suffix_checkpoint >= self._suffix_lens[batch_idx]:
                return None
            return suffix_checkpoint
        return self._left_padding_per_row[batch_idx] + checkpoint_len

    def _next_apc_checkpoint_column(self) -> Optional[int]:
        if (
            self._apc_manager is None
            or self._apc_mode != "exact"
            or not self._apc_meta
            or self._inputs_embeds is None
        ):
            return None
        start = self._processed_prompt_columns
        end = start + self._inputs_embeds.shape[1]
        next_col: Optional[int] = None
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            col = self._apc_checkpoint_column_for_meta(batch_idx, meta)
            if col is None or col <= start or col >= end:
                continue
            next_col = col if next_col is None else min(next_col, col)
        return next_col

    def _row_real_tokens_processed(self, batch_idx: int) -> int:
        meta = self._apc_meta[batch_idx]
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if self._right_pad_per_row is not None:
            suffix_done = min(
                self._suffix_lens[batch_idx],
                max(0, self._processed_prompt_columns),
            )
            return prefix_len + suffix_done
        real_done = (
            self._processed_prompt_columns - self._left_padding_per_row[batch_idx]
        )
        return prefix_len + min(self._suffix_lens[batch_idx], max(0, real_done))

    def _apc_prompt_cache_for_store(self, batch_idx: int) -> Optional[List[Any]]:
        # Single-request cold batches use an unbatched cache that is already
        # row-specific, so there is nothing to extract.
        if batch_idx == 0 and len(self.uids) == 1 and self._right_pad_per_row is None:
            return self.prompt_cache
        return _apc.extract_prompt_cache_from_batch(self.prompt_cache, batch_idx)

    def _store_apc_exact_checkpoints(self) -> None:
        if self._apc_manager is None or self._apc_mode != "exact":
            return
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None or meta.get("checkpoint_done"):
                continue
            checkpoint_len = int(meta.get("checkpoint_len") or 0)
            if checkpoint_len <= 0:
                continue
            if self._row_real_tokens_processed(batch_idx) != checkpoint_len:
                continue
            prompt_cache = self._apc_prompt_cache_for_store(batch_idx)
            if prompt_cache is None:
                continue
            self._apc_manager.store_exact_cache(
                meta["full_input_ids"][:checkpoint_len],
                prompt_cache,
                extra_hash=meta.get("extra_hash", 0),
            )
            meta["checkpoint_done"] = True

    def _prompt_kwargs_for_step(self, n: Optional[int] = None) -> dict:
        if n is None or not self._prompt_length_aware_keys:
            return self._prompt_kwargs
        out = dict(self._prompt_kwargs)
        for k in self._prompt_length_aware_keys:
            out[k] = out[k][:, :n, ...]
        return out

    def prompt_step(self) -> int:
        """Process one chunk of the prompt. Returns tokens processed."""
        if not self.needs_processing():
            return 0

        step = self.prefill_step_size or self._inputs_embeds.shape[1]
        n = min(step, self._inputs_embeds.shape[1] - 1)
        checkpoint_col = self._next_apc_checkpoint_column()
        if checkpoint_col is not None:
            n = min(n, checkpoint_col - self._processed_prompt_columns)
        if n <= 0:
            return 0
        prompt_kwargs = self._prompt_kwargs_for_step(n)
        self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **prompt_kwargs,
        )
        mx.eval([c.state for c in self.prompt_cache])
        self._processed_prompt_columns += n
        self._store_apc_exact_checkpoints()
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            self._prompt_kwargs[k] = self._prompt_kwargs[k][:, n:, ...]
        mx.clear_cache()
        return n

    def record_prompt_time(self, elapsed_s: float) -> None:
        self._prompt_time_s += max(0.0, float(elapsed_s))

    def prompt_progress(self) -> List[PromptProgress]:
        if self._prompt_time_s <= 0:
            return []
        return [
            PromptProgress(
                uid=uid,
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tokens / self._prompt_time_s,
                prompt_time=self._prompt_time_s,
                cached_tokens=cached_tokens,
            )
            for uid, prompt_tokens, cached_tokens in zip(
                self._prompt_uids,
                self._prompt_tokens_per_row,
                self._cached_tokens_per_row,
            )
        ]

    def generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ) -> GenerationBatch:
        """Process final tokens and transition to GenerationBatch."""
        call_kwargs = dict(self._prompt_kwargs)
        if self.draft_model is not None and self.draft_kind is not None:
            call_kwargs.update(
                speculative_prefill_kwargs(self.draft_kind, self.draft_model)
            )

        output = self.model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **call_kwargs,
        )
        logits = output.logits if hasattr(output, "logits") else output
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            # Per-row last *real* token sits at index (seq - 1 - right_pad[i]).
            seq = logits.shape[1]
            last_idx = mx.array(
                [seq - 1 - p for p in self._right_pad_per_row], dtype=mx.int32
            )[:, None, None]
            last_idx = mx.broadcast_to(last_idx, (logits.shape[0], 1, logits.shape[-1]))
            logits = mx.take_along_axis(logits, last_idx, axis=1).squeeze(1)
        else:
            logits = logits[:, -1, :]
        if self.logits_processors and any(self.logits_processors):
            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    sample_logits = processor(
                        mx.array(self._token_context[i]), sample_logits
                    )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = _sample_with_positions(
            sampler,
            logprobs,
            row_ids=[0] * len(self.uids),
            positions=[0] * len(self.uids),
        )

        mx.async_eval(first_tokens)

        # Roll any right-padding into left-padding so the cache decoded by
        # GenerationBatch sees a canonical layout.
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            for c in self.prompt_cache:
                finalize = getattr(c, "finalize", None)
                if not callable(finalize):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with finalize()"
                    )
                finalize()
        if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
            c0 = self.prompt_cache[0] if self.prompt_cache else None
            if c0 is not None:
                off = getattr(c0, "offset", None)
                lp = getattr(c0, "left_padding", None)
                logger.warning(
                    "post-prefill cache[0]: _idx=%s offset=%s left_padding=%s right_pad_per_row=%s suffix_lens=%s",
                    getattr(c0, "_idx", None),
                    off.tolist() if hasattr(off, "tolist") else off,
                    lp.tolist() if hasattr(lp, "tolist") else lp,
                    self._right_pad_per_row,
                    self._suffix_lens,
                )

        if self.draft_model is not None and self.draft_kind is not None:
            gen_batch = SpeculativeGenerationBatch(
                model=self.model,
                draft_model=self.draft_model,
                draft_kind=self.draft_kind,
                uids=list(self.uids),
                first_tokens=first_tokens,
                prompt_cache=self.prompt_cache,
                sampler=sampler,
                stop_criteria=stop_criteria,
                max_tokens=list(self.max_tokens),
                hidden=speculative_hidden_state(self.draft_kind, output),
                shared_kv_states=(
                    output.shared_kv_states if self.draft_kind == "mtp" else None
                ),
                prompt_tokens=self._input_ids,
                draft_block_size=self.draft_block_size,
                token_dtype=self._input_ids.dtype,
                greedy_sampling=self.greedy_sampling,
            )
            compute_logprobs = False
        else:
            gen_batch = GenerationBatch(
                model=self.model,
                uids=list(self.uids),
                inputs=first_tokens,
                prompt_cache=self.prompt_cache,
                sampler=sampler,
                stop_criteria=stop_criteria,
                max_tokens=list(self.max_tokens),
                top_logprobs_k=top_logprobs_k,
                greedy_sampling=self.greedy_sampling,
                token_context=[list(ctx) for ctx in self._token_context],
                logits_processors=list(self.logits_processors),
                thinking_budget_criteria=list(self.thinking_budget_criteria),
            )
        gen_batch.compute_logprobs = compute_logprobs

        if compute_logprobs and isinstance(gen_batch, GenerationBatch):
            gen_batch._next_lps = logprobs[
                mx.arange(first_tokens.shape[0]), first_tokens
            ]

        # Prime top-K buffers so the first token can emit top_logprobs too.
        if top_logprobs_k > 0 and isinstance(gen_batch, GenerationBatch):
            k = top_logprobs_k
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            gen_batch._next_top_idx = top_idx
            gen_batch._next_top_lp = top_lp

        language_model = getattr(self.model, "language_model", self.model)
        rope_deltas = self._capture_rope_deltas(language_model, len(gen_batch.uids))
        if rope_deltas is not None:
            # Normalize to shape (B, 1) so extend/filter stay consistent.
            if rope_deltas.ndim == 0:
                rope_deltas = rope_deltas.reshape(1, 1)
            elif rope_deltas.ndim == 1:
                rope_deltas = rope_deltas[:, None]
            # When a warm-start batch reuses the model's cached _rope_deltas
            # (computed during a previous prefill with a smaller batch), the
            # batch dim won't match this prompt batch's row count. Realign
            # so extend()/filter() down the line stay consistent with the
            # generation batch's row count.
            target_b = first_tokens.shape[0]
            if rope_deltas.shape[0] != target_b:
                if rope_deltas.shape[0] == 1:
                    rope_deltas = mx.broadcast_to(
                        rope_deltas, (target_b, rope_deltas.shape[1])
                    )
                elif rope_deltas.shape[0] < target_b:
                    pad = target_b - rope_deltas.shape[0]
                    rope_deltas = mx.concatenate(
                        [
                            rope_deltas,
                            mx.broadcast_to(
                                rope_deltas[-1:],
                                (pad, rope_deltas.shape[1]),
                            ),
                        ],
                        axis=0,
                    )
                else:
                    rope_deltas = rope_deltas[:target_b]
            gen_batch._rope_deltas = rope_deltas

        # Final prefill produces the first generated token and mutates the
        # prompt cache. Materialize that boundary before the decode loop so
        # the first decode step does not inherit the full lazy prefill graph.
        cache_states = []
        for c in self.prompt_cache:
            try:
                cache_states.append(c.state)
            except (AttributeError, TypeError):
                pass
        eval_targets = [first_tokens]
        if cache_states:
            eval_targets.append(cache_states)
        if compute_logprobs and isinstance(gen_batch, GenerationBatch):
            eval_targets.append(gen_batch._next_lps)
        if top_logprobs_k > 0 and isinstance(gen_batch, GenerationBatch):
            eval_targets.extend([gen_batch._next_top_idx, gen_batch._next_top_lp])
        if rope_deltas is not None:
            eval_targets.append(rope_deltas)
        mx.eval(*eval_targets)

        # APC: harvest the post-prefill K/V into hashed blocks. Done after the
        # final prefill forward but before the cache references are released
        # so the block tensors snapshot the prompt prefix.
        if (
            self._apc_manager is not None
            and self._apc_meta
            and self._apc_harvest_enabled
        ):
            try:
                for batch_idx, meta in enumerate(self._apc_meta):
                    if meta is None:
                        continue
                    if self._apc_mode == "exact":
                        prompt_cache = self._apc_prompt_cache_for_store(batch_idx)
                        if prompt_cache is not None:
                            self._apc_manager.store_exact_cache(
                                meta["full_input_ids"],
                                prompt_cache,
                                extra_hash=meta.get("extra_hash", 0),
                            )
                        self._apc_manager.release(meta.get("apc_blocks", []))
                    else:
                        new_blocks = _apc.harvest_blocks_from_batch_cache(
                            self._apc_manager,
                            self.prompt_cache,
                            batch_idx,
                            meta["full_input_ids"],
                            extra_hash=meta.get("extra_hash", 0),
                            skip_first_n_tokens=meta.get("prefix_len", 0),
                        )
                        self._apc_manager.release(
                            meta.get("apc_blocks", []) + new_blocks
                        )
            except Exception as e:
                logger.warning("APC harvest failed during batched prefill: %s", e)
                # Best effort — release any acquired prefix blocks.
                for meta in self._apc_meta:
                    if meta is not None:
                        self._apc_manager.release(meta.get("apc_blocks", []))

        self.uids = []
        self.prompt_cache = []
        self._token_context = []
        self.logits_processors = []
        self._apc_meta = []
        return gen_batch

    @property
    def total_prompt_tokens(self):
        return self._total_prompt_tokens

    @staticmethod
    def _capture_rope_deltas(language_model, B: int):
        if not hasattr(language_model, "_rope_deltas"):
            return None
        rope_deltas = language_model._rope_deltas
        if rope_deltas is None:
            return mx.zeros((B, 1), dtype=mx.int32)
        if rope_deltas.ndim == 0:
            rope_deltas = rope_deltas.reshape(1, 1)
        elif rope_deltas.ndim == 1:
            rope_deltas = rope_deltas[:, None]
        # Falcon OCR emits a singleton meant to broadcast across rows.
        if rope_deltas.shape[0] == 1 and B > 1:
            rope_deltas = mx.broadcast_to(rope_deltas, (B, 1))
        if rope_deltas.shape[0] != B:
            if rope_deltas.shape[0] > B:
                rope_deltas = rope_deltas[:B]
            else:
                pad = B - rope_deltas.shape[0]
                rope_deltas = mx.concatenate(
                    [
                        rope_deltas,
                        mx.broadcast_to(rope_deltas[-1:], (pad, rope_deltas.shape[1])),
                    ],
                    axis=0,
                )
        return rope_deltas


class BatchGenerator:
    """
    Continuous batching with separate prompt processing and generation phases.

    next() returns (prompt_responses, generation_responses) where:
    - prompt_responses contains completed prompt-batch timing stats
    - generation_responses is a list of GenerationBatch.Response objects
    """

    def __init__(
        self,
        model,
        processor,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        prompt_cache=None,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        stream=None,
        apc_manager: Optional["_apc.APCManager"] = None,
        draft_model: Optional[nn.Module] = None,
        draft_kind: Optional[str] = None,
        draft_block_size: Optional[int] = None,
        greedy_sampling: bool = False,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.processor = processor
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.draft_block_size = draft_block_size
        self.greedy_sampling = greedy_sampling or sampler is None
        if self.draft_model is not None:
            apc_manager = None
            compute_logprobs = False
            top_logprobs_k = 0
            self.compute_logprobs = False
            self.top_logprobs_k = 0
        # APC: opt-out for KV-quantized caches. Plain KV models use block APC;
        # mixed/custom cache models use exact prompt-cache snapshots.
        self.apc_mode = None
        if apc_manager is not None and kv_bits is not None:
            apc_manager = None
        if apc_manager is not None:
            self.apc_mode = _apc.model_apc_mode(model)
            if self.apc_mode is None:
                apc_manager = None
        self.apc_manager = apc_manager
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size

        self._stream = stream or generation_stream

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.tokenizer.stopping_criteria,
            compute_logprobs=self.compute_logprobs,
            top_logprobs_k=self.top_logprobs_k,
            greedy_sampling=self.greedy_sampling,
        )
        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0
        self._cache_eval_interval = _get_batch_cache_eval_interval()

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [self._stream]))

    # ---------------- APC integration helpers ----------------
    # Keys that are APC-only metadata; stripped from ``prompt_kwargs`` before
    # the merged kwargs are passed to the language model forward.
    _APC_PRIVATE_KEYS = APC_PRIVATE_PROMPT_KEYS

    def _apc_extra_hash(self, prompt_kwargs: dict) -> int:
        """Salt for the APC hash chain."""
        if self.apc_manager is None:
            return 0
        if prompt_kwargs is None:
            prompt_kwargs = {}
        img = prompt_kwargs.get("_apc_image_hash")
        if img is None:
            pixel_values = prompt_kwargs.get("pixel_values")
            img = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=None)
        tenant = prompt_kwargs.get("_apc_tenant")
        return _apc.tenant_scoped_hash(tenant, img)

    def _apc_media_token_ids(self) -> set[int]:
        return _apc.multimodal_token_ids_from_config(self.model.config)

    def _apc_safe_prefix_lookup_min(self, ids_list: List[int]) -> int:
        safe_min = _apc.media_safe_prefix_min(ids_list, self._apc_media_token_ids())
        return max(0, safe_min - 1)

    def _apc_suffix_is_text_only(self, ids_list: List[int], prefix_len: int) -> bool:
        return _apc.prefix_leaves_text_only_suffix(
            ids_list,
            prefix_len,
            self._apc_media_token_ids(),
        )

    def _apc_prefix_has_media_tokens(
        self, ids_list: List[int], prefix_len: int
    ) -> bool:
        return _apc.prefix_contains_media_tokens(
            ids_list,
            prefix_len,
            self._apc_media_token_ids(),
        )

    def _apc_exact_checkpoint_len(self, ids_list: List[int]) -> int:
        if self.apc_manager is None or getattr(self, "apc_mode", "block") != "exact":
            return 0
        return _apc.adjust_prefix_to_text_suffix_boundary(
            ids_list,
            len(ids_list) - self.apc_manager.exact_cache_guard_tokens,
            self._apc_media_token_ids(),
            max_prefix_tokens=len(ids_list) - 1,
        )

    def _apc_pick_for(self, sequence) -> Optional[dict]:
        """Look up an APC prefix for ``sequence``. Returns dict with matched
        blocks + suffix metadata when there is a usable hit, else None.
        """
        if self.apc_manager is None:
            return None
        uid, ids_list, max_toks, prompt_kwargs, lps, criteria = sequence
        if not ids_list or len(ids_list) < 2:
            return None
        safe_lookup_min = self._apc_safe_prefix_lookup_min(ids_list)
        extra_hash = self._apc_extra_hash(prompt_kwargs or {})
        apc_mode = getattr(self, "apc_mode", "block")
        if apc_mode == "exact":
            exact_cache, exact_prefix_len = self.apc_manager.lookup_exact_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=safe_lookup_min,
            )
            if (
                exact_cache is not None
                and exact_prefix_len > 0
                and exact_prefix_len < len(ids_list)
            ):
                if not self._apc_suffix_is_text_only(ids_list, exact_prefix_len):
                    return None
                return {
                    "matched_blocks": [],
                    "warm_cache": exact_cache,
                    "prefix_len": exact_prefix_len,
                    "extra_hash": extra_hash,
                    "full_input_ids": list(ids_list),
                }
            return None
        matched, prefix_len = self.apc_manager.lookup_prefix(
            ids_list, extra_hash=extra_hash
        )
        if prefix_len > 0 and self._apc_prefix_has_media_tokens(ids_list, prefix_len):
            self.apc_manager.release(matched)
            matched = []
            prefix_len = 0
        exact_cache = None
        exact_prefix_len = 0
        if prefix_len < len(ids_list):
            exact_cache, exact_prefix_len = self.apc_manager.lookup_exact_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=max(prefix_len, safe_lookup_min),
            )
        warm_cache = None
        disk_prefix_len = 0
        if max(prefix_len, exact_prefix_len) < len(ids_list):
            warm_cache, disk_prefix_len = self.apc_manager.lookup_prefix_disk_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=max(prefix_len, exact_prefix_len, safe_lookup_min),
                allow_memory_overlap=max(prefix_len, exact_prefix_len) > 0,
            )
        if disk_prefix_len > max(
            prefix_len, exact_prefix_len
        ) and disk_prefix_len < len(ids_list):
            if matched:
                self.apc_manager.release(matched)
            if not self._apc_suffix_is_text_only(ids_list, disk_prefix_len):
                return None
            return {
                "matched_blocks": [],
                "warm_cache": warm_cache,
                "prefix_len": disk_prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if exact_prefix_len > prefix_len and exact_prefix_len < len(ids_list):
            if matched:
                self.apc_manager.release(matched)
            if not self._apc_suffix_is_text_only(ids_list, exact_prefix_len):
                return None
            return {
                "matched_blocks": [],
                "warm_cache": exact_cache,
                "prefix_len": exact_prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if prefix_len > 0 and prefix_len < len(ids_list):
            if not self._apc_suffix_is_text_only(ids_list, prefix_len):
                self.apc_manager.release(matched)
                return None
            return {
                "matched_blocks": matched,
                "prefix_len": prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if matched:
            self.apc_manager.release(matched)
        return None

    def _build_mixed_prompt_batch(
        self, sequences: List[tuple]
    ) -> Optional["PromptProcessingBatch"]:
        """Build a multi-row PromptProcessingBatch admitting ``sequences``.

        Each row is independently looked up in APC. Warm rows have their
        suffixes prefilled against pre-populated K/V; cold rows prefill from
        scratch in the same batch. Right-padding aligns RoPE positions
        across rows with different prefix/suffix lengths.

        Returns ``None`` if APC is disabled (in which case the caller should
        use the cold-only path).
        """
        if self.apc_manager is None:
            return None

        picks: List[Optional[dict]] = [self._apc_pick_for(s) for s in sequences]
        any_warm = any(p is not None for p in picks)
        if not any_warm:
            return None  # caller falls back to cold-only path

        uids = [s[0] for s in sequences]
        full_ids = [list(s[1]) for s in sequences]
        max_tokens_list = [s[2] for s in sequences]
        prompt_kwargs_list = [s[3] for s in sequences]
        logits_processors = [s[4] for s in sequences]
        thinking_budget_criteria = [s[5] for s in sequences]

        # Per-row prefix length and suffix tokens
        prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
        suffix_ids_list = [full_ids[i][prefix_lens[i] :] for i in range(len(sequences))]
        suffix_lens = [len(s) for s in suffix_ids_list]

        max_suffix_len = max(suffix_lens)
        right_pad_per_row = [max_suffix_len - s for s in suffix_lens]

        # Source inputs_embeds: every row's prompt_kwargs holds the full-prompt
        # embeddings. Slice to suffix per-row, right-pad to max_suffix_len, stack.
        suffix_embeds_per_row: List[mx.array] = []
        for i, kw in enumerate(prompt_kwargs_list):
            if kw is None or kw.get("inputs_embeds") is None:
                raise ValueError("APC mixed prefill requires precomputed inputs_embeds")
            full = kw["inputs_embeds"]  # [1, full_len, D]
            suff = full[:, prefix_lens[i] :, :]
            pad = right_pad_per_row[i]
            if pad > 0:
                pad_emb = mx.zeros(
                    (suff.shape[0], pad, suff.shape[-1]), dtype=suff.dtype
                )
                suff = mx.concatenate([suff, pad_emb], axis=1)
            suffix_embeds_per_row.append(suff)
        inputs_embeds = mx.concatenate(suffix_embeds_per_row, axis=0)

        # Merge prompt-side kwargs (excluding inputs_embeds, which we've just
        # rebuilt). Per-batch tensors get concatenated across rows; scalars
        # take the first row's value (matches the existing cold-only path).
        # APC-private keys (e.g. tenant salt) are dropped — they're consumed
        # in _apc_extra_hash, never forwarded to the model.
        merged_kwargs: dict = {}
        per_row_keys: dict = {}
        batch_size = len(prompt_kwargs_list)
        for i, kw in enumerate(prompt_kwargs_list):
            if not kw:
                continue
            full_len = len(full_ids[i])
            prefix_len = prefix_lens[i]
            right_pad = right_pad_per_row[i]
            for k, v in kw.items():
                if k == "inputs_embeds" or k in self._APC_PRIVATE_KEYS:
                    continue
                if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
                    row_v = _prompt_kwarg_row(v, i, batch_size)
                    if _is_sequence_aligned_prompt_kwarg(k, row_v, full_len):
                        row_v = row_v[:, prefix_len:, ...]
                        row_v = _pad_sequence_aligned_prompt_kwarg(
                            row_v,
                            max_suffix_len,
                            left=False,
                        )
                    per_row_keys.setdefault(k, []).append(row_v)
                elif k not in merged_kwargs:
                    merged_kwargs[k] = v
        for k, vs in per_row_keys.items():
            merged_kwargs[k] = mx.concatenate(vs, axis=0)

        apc_mode = getattr(self, "apc_mode", "block")
        if apc_mode == "exact":
            row_caches = [
                p["warm_cache"] if p is not None else self.model.make_cache()
                for p in picks
            ]
            warm_cache, _ = _apc.make_warm_batch_exact_cache_multi(
                row_caches,
                prefix_lens,
            )
            if warm_cache is None:
                return None
        else:
            # Build the multi-row warm cache (zeros for cold rows, K/V for warm).
            num_layers = (
                len(self.model.make_cache())
                if hasattr(self.model, "make_cache")
                else len(self.model.layers)
            )
            warm_cache, _ = _apc.make_warm_batch_kv_cache_multi(
                picks, num_layers=num_layers
            )

        apc_meta = [
            {
                "full_input_ids": full_ids[i],
                "prefix_len": prefix_lens[i],
                "extra_hash": (
                    picks[i]["extra_hash"]
                    if picks[i]
                    else self._apc_extra_hash(prompt_kwargs_list[i] or {})
                ),
                "apc_blocks": picks[i].get("matched_blocks", []) if picks[i] else [],
                "checkpoint_len": self._apc_exact_checkpoint_len(full_ids[i]),
            }
            for i in range(len(sequences))
        ]

        prompt_batch_cls = _generate_module_override(
            "PromptProcessingBatch", PromptProcessingBatch
        )
        return prompt_batch_cls(
            model=self.model,
            uids=uids,
            input_ids=suffix_ids_list,
            max_tokens=max_tokens_list,
            inputs_embeds=inputs_embeds,
            prompt_kwargs=merged_kwargs,
            logits_processors=logits_processors,
            thinking_budget_criteria=thinking_budget_criteria,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            kv_quant_scheme=self.kv_quant_scheme,
            warm_cache=warm_cache,
            apc_meta=apc_meta,
            apc_manager=self.apc_manager,
            right_pad_per_row=right_pad_per_row,
            suffix_lens=suffix_lens,
            apc_mode=apc_mode,
            draft_model=getattr(self, "draft_model", None),
            draft_kind=getattr(self, "draft_kind", None),
            draft_block_size=getattr(self, "draft_block_size", None),
            greedy_sampling=getattr(self, "greedy_sampling", False),
        )

    def _build_apc_meta_for_cold(
        self,
        input_ids_list: List[List[int]],
        prompt_kwargs_list: List[Optional[dict]],
    ) -> Optional[List[Optional[dict]]]:
        """Build per-row harvest metadata for a cold-prefill batch so the
        produced K/V are added to APC after prefill.
        """
        if self.apc_manager is None:
            return None
        meta: List[Optional[dict]] = []
        for ids_list, kw in zip(input_ids_list, prompt_kwargs_list):
            extra_hash = self._apc_extra_hash(kw or {})
            meta.append(
                {
                    "full_input_ids": list(ids_list),
                    "prefix_len": 0,
                    "extra_hash": extra_hash,
                    "apc_blocks": [],
                    "checkpoint_len": self._apc_exact_checkpoint_len(list(ids_list)),
                }
            )
        return meta

    @property
    def stream(self):
        return self._stream

    def close(self):
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        self.close()

    def insert(
        self,
        prompts,
        max_tokens: Union[List[int], int, None] = None,
        prompt_kwargs: Optional[List[dict]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        thinking_budget_criteria: Optional[List[Any]] = None,
    ):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if prompt_kwargs is None:
            prompt_kwargs = [{}] * len(prompts)
        if logits_processors is None:
            logits_processors = [self.logits_processors] * len(prompts)
        elif len(logits_processors) != len(prompts):
            raise ValueError("Insufficient number of logits_processors provided")
        if thinking_budget_criteria is None:
            thinking_budget_criteria = [None] * len(prompts)
        elif len(thinking_budget_criteria) != len(prompts):
            raise ValueError("Insufficient number of thinking_budget_criteria provided")

        for p, m, kw, lp, tc in zip(
            prompts,
            max_tokens,
            prompt_kwargs,
            logits_processors,
            thinking_budget_criteria,
        ):
            self._unprocessed_sequences.append((self.uid_count, p, m, kw, lp, tc))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda x: len(x[1])
        )
        return uids

    def remove(self, uid) -> bool:
        """Remove a sequence from the batch by uid."""
        with mx.stream(self._stream):
            # Waiting in the queue.
            for i, (seq_uid, _, _, _, _, _) in enumerate(self._unprocessed_sequences):
                if seq_uid == uid:
                    self._unprocessed_sequences.pop(i)
                    return True

            # Being prefilled
            if self._prompt_batch is not None and uid in self._prompt_batch.uids:
                if len(self._prompt_batch.uids) == 1:
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            # Already decoding.
            if uid in self._generation_batch.uids:
                idx = self._generation_batch.uids.index(uid)
                keep = [i for i in range(len(self._generation_batch.uids)) if i != idx]
                self._generation_batch.filter(keep)
                return True

            return False

    @property
    def unprocessed_prompts(self):
        """Backward-compatible alias for server flush logic."""
        return self._unprocessed_sequences

    @property
    def has_pending_prompts(self):
        """True if there are prompts waiting or being processed."""
        return len(self._unprocessed_sequences) > 0 or self._prompt_batch is not None

    @property
    def has_work(self):
        """True if there is any remaining work."""
        return (
            len(self._generation_batch) > 0
            or self._prompt_batch is not None
            or len(self._unprocessed_sequences) > 0
        )

    def stats(self):
        """Return accumulated batch statistics."""
        stats = BatchStats()
        stats.prompt_tokens = self._prompt_tokens_counter
        stats.prompt_time = self._prompt_time_counter
        stats.prompt_tps = (
            self._prompt_tokens_counter / self._prompt_time_counter
            if self._prompt_time_counter > 0
            else 0
        )
        stats.generation_tokens = self._gen_tokens_counter
        stats.peak_memory = mx.get_peak_memory() / 1e9
        return stats

    @staticmethod
    def _record_prompt_batch_time(prompt_batch, elapsed_s: float) -> None:
        recorder = getattr(prompt_batch, "record_prompt_time", None)
        if callable(recorder):
            recorder(elapsed_s)

    @staticmethod
    def _prompt_batch_progress(prompt_batch) -> List[PromptProgress]:
        progress = getattr(prompt_batch, "prompt_progress", None)
        if callable(progress):
            return progress()
        return []

    def _extend_generation_batch(self, gen_batch) -> None:
        if len(self._generation_batch) == 0:
            self._generation_batch = gen_batch
        else:
            self._generation_batch.extend(gen_batch)

    def _next(self, **kwargs):
        generation_responses = []
        prompt_responses = []

        # Decode-first: always emit a generation step before touching prefill.
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if (
                self._cache_eval_interval > 0
                and self._steps_counter % self._cache_eval_interval == 0
            ):
                cache_states = getattr(self._generation_batch, "cache_states", None)
                if callable(cache_states):
                    mx.eval(cache_states())
                else:
                    mx.eval([c.state for c in self._generation_batch.prompt_cache])
                mx.clear_cache()

        if (
            getattr(self._generation_batch, "is_speculative", False)
            and len(self._generation_batch) > 0
        ):
            return prompt_responses, generation_responses

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
                self._prompt_tokens_counter += n
                return prompt_responses, generation_responses

            tic = time.perf_counter()
            gen_batch = self._prompt_batch.generate(
                self.sampler,
                self.tokenizer.stopping_criteria,
                compute_logprobs=self.compute_logprobs,
                top_logprobs_k=self.top_logprobs_k,
            )
            elapsed = time.perf_counter() - tic
            self._prompt_time_counter += elapsed
            self._record_prompt_batch_time(self._prompt_batch, elapsed)
            prompt_responses = self._prompt_batch_progress(self._prompt_batch)
            self._extend_generation_batch(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= self.prefill_batch_size:
            # Take up to prefill_batch_size pending sequences. If APC is on
            # and at least one of them has a prefix hit, build a mixed
            # warm/cold PromptProcessingBatch with right-padded suffixes so
            # warm and cold rows prefill in a single forward pass.
            n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
            sequences = self._unprocessed_sequences[:n]
            if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
                logger.warning(
                    "APC admit n=%d (pending=%d)",
                    n,
                    len(self._unprocessed_sequences),
                )
            mixed = self._build_mixed_prompt_batch(sequences)
            if mixed is not None:
                self._unprocessed_sequences = self._unprocessed_sequences[n:]
                self._prompt_batch = mixed
                self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens
                if self._prompt_batch.needs_processing():
                    tic = time.perf_counter()
                    nstep = self._prompt_batch.prompt_step()
                    elapsed = time.perf_counter() - tic
                    self._prompt_time_counter += elapsed
                    self._record_prompt_batch_time(self._prompt_batch, elapsed)
                else:
                    tic = time.perf_counter()
                    gen_batch = self._prompt_batch.generate(
                        self.sampler,
                        self.tokenizer.stopping_criteria,
                        compute_logprobs=self.compute_logprobs,
                        top_logprobs_k=self.top_logprobs_k,
                    )
                    elapsed = time.perf_counter() - tic
                    self._prompt_time_counter += elapsed
                    self._record_prompt_batch_time(self._prompt_batch, elapsed)
                    prompt_responses = self._prompt_batch_progress(self._prompt_batch)
                    self._extend_generation_batch(gen_batch)
                    self._prompt_batch = None
                    mx.clear_cache()
                return prompt_responses, generation_responses

            self._unprocessed_sequences = self._unprocessed_sequences[n:]

            uids = [s[0] for s in sequences]
            input_ids = [s[1] for s in sequences]
            max_tokens_list = [s[2] for s in sequences]
            prompt_kwargs_list = [s[3] for s in sequences]
            logits_processors = [s[4] for s in sequences]
            thinking_budget_criteria = [s[5] for s in sequences]

            inputs_embeds, merged_kwargs = _merge_prefill_prompt_kwargs(
                prompt_kwargs_list, input_ids
            )

            # APC: also harvest cold-prefill prefixes so future requests hit.
            apc_meta = self._build_apc_meta_for_cold(input_ids, prompt_kwargs_list)

            prompt_batch_cls = _generate_module_override(
                "PromptProcessingBatch", PromptProcessingBatch
            )
            self._prompt_batch = prompt_batch_cls(
                model=self.model,
                uids=uids,
                input_ids=input_ids,
                max_tokens=max_tokens_list,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=merged_kwargs,
                logits_processors=logits_processors,
                thinking_budget_criteria=thinking_budget_criteria,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                kv_quant_scheme=self.kv_quant_scheme,
                apc_meta=apc_meta,
                apc_manager=self.apc_manager,
                apc_mode=self.apc_mode,
                draft_model=getattr(self, "draft_model", None),
                draft_kind=getattr(self, "draft_kind", None),
                draft_block_size=getattr(self, "draft_block_size", None),
                greedy_sampling=getattr(self, "greedy_sampling", False),
            )
            self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens

            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
            else:
                tic = time.perf_counter()
                gen_batch = self._prompt_batch.generate(
                    self.sampler,
                    self.tokenizer.stopping_criteria,
                    compute_logprobs=self.compute_logprobs,
                    top_logprobs_k=self.top_logprobs_k,
                )
                elapsed = time.perf_counter() - tic
                self._prompt_time_counter += elapsed
                self._record_prompt_batch_time(self._prompt_batch, elapsed)
                prompt_responses = self._prompt_batch_progress(self._prompt_batch)
                self._extend_generation_batch(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses

    def next(self, **kwargs):
        with mx.stream(self._stream):
            return self._next(**kwargs)


def batch_generate(
    model,
    processor,
    images: Union[str, List[str]] = None,
    audios: Union[str, List[str]] = None,
    prompts: List[str] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    group_by_shape: bool = True,
    track_image_sizes: bool = True,
    **kwargs,
):
    """
    Generate responses for the given batch of prompts with variable-sized images.

    This function implements the transformers-style approach to batching:
    1. Group images with the same shape for efficient batch processing
    2. Process each group as a batch (no padding waste within groups)
    3. Track original image sizes for proper attention masking
    4. Restore results to original batch order

    Key insight: Instead of padding all images to the same spatial dimensions
    (which wastes computation and may hurt accuracy), we group same-sized
    images together so there's zero padding within each group.

    Args:
       model (nn.Module): The language model.
       processor (PreTrainedTokenizer): The tokenizer/processor.
       images (Union[str, List[str]]): Images (paths, URLs, or PIL images).
       audios (Union[str, List[str]]): Audio files (not yet supported for batching).
       prompts (List[str]): The input prompts.
       max_tokens (Union[int, List[int]]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       verbose (bool): If ``True``, print tokens and timing information.
       group_by_shape (bool): If ``True``, group same-shaped images for efficient
          batch processing.
       track_image_sizes (bool): If ``True``, track and return original image sizes.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.

    Returns:
        BatchResponse with generated texts, statistics, and optionally image_sizes.
    """
    from PIL import Image

    from ..utils import process_image

    processor.detokenizer.reset()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Handle single image case
    if isinstance(images, str):
        images = [images]

    # Handle no images case
    if images is None:
        texts, stats = _generate_batch(
            model, processor, prompts, None, max_tokens, verbose, **kwargs
        )
        return BatchResponse(texts, stats)

    # Load and preprocess images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )

    processed_images = []
    image_sizes_original = []
    for img in images:
        if isinstance(img, str):
            pil_img = process_image(img, None, image_processor)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = img
        processed_images.append(pil_img)
        # Track original size
        if hasattr(pil_img, "height"):
            image_sizes_original.append((pil_img.height, pil_img.width))
        else:
            image_sizes_original.append((0, 0))

    # Group images by shape for efficient processing (no padding within groups)
    if group_by_shape and len(processed_images) > 1:
        grouped_images, grouped_indices = group_images_by_shape(processed_images)

        if verbose:
            print(f"[batch_generate] Found {len(grouped_images)} unique image shapes")
    else:
        # Single image or grouping disabled - treat as one group
        shape = (
            (processed_images[0].height, processed_images[0].width)
            if processed_images
            else (0, 0)
        )
        grouped_images = {shape: processed_images}
        grouped_indices = {shape: list(range(len(processed_images)))}

    # Process each shape group
    all_texts = [None] * len(prompts)
    all_image_sizes = [None] * len(prompts)
    total_stats = BatchStats()

    for shape, indices in grouped_indices.items():
        # Get images and prompts for this shape group
        group_images = [processed_images[i] for i in indices]
        group_prompts = [prompts[i] for i in indices]
        group_sizes = [image_sizes_original[i] for i in indices]

        # Handle per-sample max_tokens
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs = dict(kwargs)
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None and isinstance(logits_processors, list):
            if not logits_processors or all(callable(p) for p in logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in indices
                ]

        # Process the entire group at once (same shape = no padding needed)
        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            group_images,
            group_max_tokens,
            **group_kwargs,
        )

        # Store results in original order
        for j, orig_idx in enumerate(indices):
            all_texts[orig_idx] = chunk_texts[j]
            all_image_sizes[orig_idx] = group_sizes[j]

        # Accumulate stats
        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    text_only_indices = list(range(len(processed_images), len(prompts)))
    if text_only_indices:
        group_prompts = [prompts[i] for i in text_only_indices]
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in text_only_indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs = dict(kwargs)
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None and isinstance(logits_processors, list):
            if not logits_processors or all(callable(p) for p in logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in text_only_indices
                ]

        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            None,
            group_max_tokens,
            **group_kwargs,
        )

        for j, orig_idx in enumerate(text_only_indices):
            all_texts[orig_idx] = chunk_texts[j]

        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    mx.clear_cache()

    # Compute final stats
    if total_stats.prompt_time > 0:
        total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
    if total_stats.generation_time > 0:
        total_stats.generation_tps = (
            total_stats.generation_tokens / total_stats.generation_time
        )
    total_stats.peak_memory = mx.get_peak_memory() / 1e9

    if verbose:
        print(f"[batch_generate] Finished processing {len(prompts)} samples")
        print(
            f"[batch_generate] Prompt: {total_stats.prompt_tokens} tokens, {total_stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {total_stats.generation_tokens} tokens, "
            f"{total_stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {total_stats.peak_memory:.3f} GB")

    response = BatchResponse(all_texts, total_stats)
    if track_image_sizes:
        response.image_sizes = all_image_sizes
    return response


def _clone_or_share_logits_processor(processor):
    if hasattr(processor, "clone"):
        return processor.clone()
    warnings.warn(
        "Sharing logits processor across batch entries because it does not "
        "implement clone(). Stateful logits processors should implement clone() "
        "to avoid shared state across sequences.",
        RuntimeWarning,
        stacklevel=2,
    )
    return processor


def _generate_batch(
    model,
    processor,
    prompts: List[str],
    images: List = None,
    max_tokens: Union[int, List[int]] = 100,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[str], BatchStats]:

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    batch_size = len(prompts)
    logits_processors = kwargs.pop("logits_processors", None)

    num_images_list = [
        1 if i < (len(images) if images is not None else 0) else 0
        for i in range(len(prompts))
    ]
    formatted_prompts = [
        apply_chat_template(
            processor,
            model.config,
            p,
            num_images=num_images_list[i],
        )
        for i, p in enumerate(prompts)
    ]

    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)

    inputs = prepare_inputs(
        processor,
        images=images,
        audio=None,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,  # Since images are pre-grouped by shape, they're already uniform size
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **data_kwargs
    )

    gen_kwargs = {**data_kwargs, **embedding_output.to_dict()}

    if kwargs.get("prefill_step_size", DEFAULT_PREFILL_STEP_SIZE) is not None:
        policy_kwargs = dict(gen_kwargs)
        draft_model = kwargs.get("draft_model")
        draft_kind = kwargs.get("draft_kind")
        if draft_model is not None and draft_kind is not None:
            policy_kwargs.update(speculative_prefill_kwargs(draft_kind, draft_model))
        if not _chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=embedding_output.inputs_embeds,
            draft_model=draft_model,
            draft_kind=draft_kind,
            prefill_kwargs=policy_kwargs,
        ):
            kwargs.pop("prefill_step_size", None)
            kwargs["prefill_step_size"] = None

    # Use batch_size for prefill and completion to ensure consistent processing
    gen = BatchGenerator(
        model.language_model,
        processor,
        prefill_batch_size=batch_size,
        completion_batch_size=batch_size,
        compute_logprobs=False,
        **kwargs,
    )

    if logits_processors and all(
        callable(processor) for processor in logits_processors
    ):
        logits_processors = [
            [_clone_or_share_logits_processor(p) for p in logits_processors]
            for _ in range(batch_size)
        ]

    uids = gen.insert(
        input_ids.tolist(),
        max_tokens,
        prompt_kwargs=_split_prompt_kwargs_per_row(gen_kwargs, batch_size),
        logits_processors=logits_processors,
    )
    results = {uid: [] for uid in uids}

    tic = time.perf_counter()
    while gen.has_work:
        _, generation_responses = gen.next()
        for r in generation_responses:
            if r.finish_reason != "stop":
                results[r.uid].append(r.token)
    total_time = time.perf_counter() - tic

    gen.close()

    detokenizer = processor.detokenizer
    texts = []
    for uid in uids:
        detokenizer.reset()
        for t in results[uid]:
            detokenizer.add_token(t)
        detokenizer.finalize()
        texts.append(detokenizer.text)

    stats = gen.stats()
    stats.generation_time = total_time - stats.prompt_time
    if stats.generation_time > 0:
        stats.generation_tps = stats.generation_tokens / stats.generation_time
    return texts, stats
