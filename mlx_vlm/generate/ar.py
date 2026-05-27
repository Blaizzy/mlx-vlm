from __future__ import annotations

import functools
import sys
from collections.abc import Generator
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm

from ..models import cache
from ..speculative.utils import run_speculative_rounds
from .common import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    generation_stream,
    maybe_quantize_kv_cache,
)

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
DEFAULT_PREFILL_STEP_SIZE = 2048


def _generate_module_override(name: str, fallback):
    generate_module = sys.modules.get("mlx_vlm.generate")
    return getattr(generate_module, name, fallback) if generate_module else fallback


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

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    # Speculative decoding setup
    last_outputs = None
    if draft_model is not None:
        from ..speculative.drafters import validate_drafter_compatibility

        validate_drafter_compatibility(model, draft_model, draft_kind)
        if draft_kind == "mtp":
            # MTP drafter consumes target's last-layer hidden + shared K/V
            # (per layer-type) rather than per-layer hidden captures.
            kwargs["return_hidden"] = True
            kwargs["return_shared_kv"] = True
        elif draft_kind == "eagle3":
            kwargs["capture_layer_ids"] = list(
                getattr(
                    draft_model.config,
                    "capture_layer_ids",
                    draft_model.config.target_layer_ids,
                )
            )
        else:
            kwargs["capture_layer_ids"] = list(draft_model.config.target_layer_ids)
        prefill_step_size = None
        # Reset stale mRoPE state from any previous generation.
        lm = model.language_model if hasattr(model, "language_model") else model
        if hasattr(lm, "_position_ids"):
            lm._position_ids = None
        if hasattr(lm, "_rope_deltas"):
            lm._rope_deltas = None

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs, last_outputs

        with mx.stream(generation_stream):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **kwargs,
                )

            last_outputs = outputs
            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            y = sampler(logprobs)

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
        if getattr(model, "no_chunked_prefill", False):
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
            with tqdm(total=total_tokens, desc="Prefill", unit="tok") as pbar:
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

