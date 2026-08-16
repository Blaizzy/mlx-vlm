from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Callable, Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from ..models.diffusion_visualizer import (
    clip_display_width,
    display_width,
    escape_carriage_returns,
)
from ..tokenizer_utils import make_streaming_detokenizer
from .common import (
    GenerationResult,
    _chunked_prefill_enabled,
    generation_stream,
    wired_limit,
)

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_TEMPERATURE = 0.0
DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH = 64
DEFAULT_DIFFUSION_MAX_DENOISING_STEPS = 48
DEFAULT_DIFFUSION_UNMASKING_WIDTH = 0
DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD = 0.9


def _diffusion_display_limit(requested_width: Optional[int] = None) -> Optional[int]:
    terminal_width = shutil.get_terminal_size((120, 20)).columns
    requested_width = requested_width or DEFAULT_DIFFUSION_UNMASKING_WIDTH
    if requested_width <= 0:
        return None
    return max(1, min(terminal_width - 1, requested_width))


def _diffusion_draft_width(requested_width: Optional[int] = None) -> int:
    limit = _diffusion_display_limit(requested_width)
    if limit is not None:
        return limit
    return max(1, shutil.get_terminal_size((120, 20)).columns - 1)


def _supports_in_place_output() -> bool:
    return os.isatty(1) and os.environ.get("TERM", "xterm") != "dumb"


def _format_diffusion_draft_line(
    response: GenerationResult,
    requested_width: Optional[int] = None,
) -> str:
    text = escape_carriage_returns(response.draft_text)
    width = _diffusion_display_limit(requested_width)
    if width is None:
        return text
    return clip_display_width(text, width)


def _print_diffusion_draft(
    response: GenerationResult,
    requested_width: Optional[int] = None,
) -> None:
    line = _format_diffusion_draft_line(response, requested_width)
    print("\r\033[2K" + line, end="", flush=True)


def _format_diffusion_live_text(
    text: str,
    requested_width: Optional[int] = None,
    *,
    preserve_newlines: bool = True,
) -> str:
    width = _diffusion_display_limit(requested_width)
    text = escape_carriage_returns(text)
    if not preserve_newlines:
        text = text.replace("\n", "\\n")
    if width is None:
        return text
    return clip_display_width(text, width)


def _print_diffusion_live_text(
    text: str,
    requested_width: Optional[int] = None,
) -> None:
    print(
        "\r\033[2K" + _format_diffusion_live_text(text, requested_width),
        end="",
        flush=True,
    )


def _clear_diffusion_draft_line() -> None:
    print("\r\033[2K", end="", flush=True)


def _terminal_rows_for_text(text: str, columns: Optional[int] = None) -> int:
    if not text:
        return 0
    columns = columns or shutil.get_terminal_size((120, 20)).columns
    columns = max(1, columns)
    rows = 0
    for line in text.split("\n"):
        width = display_width(line)
        rows += max(1, (width + columns - 1) // columns)
    return rows


class _DiffusionRedrawer:
    def __init__(self):
        self.rows = 0

    def clear(self) -> None:
        if self.rows <= 0:
            return

        controls = ["\r\033[2K"]
        for _ in range(self.rows - 1):
            controls.append("\033[1A\r\033[2K")
        print("".join(controls), end="", flush=True)
        self.rows = 0

    def draw(self, text: str) -> None:
        self.clear()
        print(text, end="", flush=True)
        self.rows = _terminal_rows_for_text(text)


def _has_engine_diffusion_config(config: Any) -> bool:
    # Engine-driven diffusion models declare the canvas length the denoising
    # loop operates on; that trait is what the shared engine drives, so
    # detection is not tied to a hardcoded model type.
    return getattr(config, "canvas_length", None) is not None


def _has_model_diffusion_generator(model: nn.Module) -> bool:
    """True for diffusion models that expose generation on the language model."""
    config = getattr(model, "config", None)
    language_model = getattr(model, "language_model", None)
    return (
        _has_engine_diffusion_config(config)
        or getattr(config, "mask_token_id", None) is not None
    ) and callable(getattr(language_model, "generate", None))


def _uses_model_diffusion_generator(
    model: nn.Module,
    kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    if not _has_model_diffusion_generator(model):
        return False

    config = getattr(model, "config", None)
    if getattr(config, "default_generation_mode", None) != "ar":
        return True

    generation_mode = (kwargs or {}).get("generation_mode")
    if generation_mode is not None:
        return generation_mode != "ar"

    return False


def is_diffusion_model(
    model: nn.Module,
    kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when this request should use the unified diffusion path."""
    return _diffusion_stream_strategy(model, kwargs) is not None


def diffusion_generation_family(
    model: nn.Module,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Backward-compatible diffusion router.

    New callers should use :func:`is_diffusion_model` and the unified stream
    adapter directly.
    """
    if is_diffusion_model(model, kwargs):
        return "diffusion"
    return None


def diffusion_kwargs_from_args(args: Any, config: Any) -> Dict[str, Any]:
    if not (
        _has_engine_diffusion_config(config)
        or getattr(config, "mask_token_id", None) is not None
    ):
        return {}

    kwargs = {}
    if args.max_denoising_steps is not None:
        kwargs["max_denoising_steps"] = args.max_denoising_steps
    if args.diffusion_full_canvas:
        kwargs["diffusion_full_canvas"] = True
    if args.diffusion_min_canvas_length is not None:
        kwargs["diffusion_min_canvas_length"] = args.diffusion_min_canvas_length
    if getattr(args, "diffusion_max_canvas_length", None) is not None:
        kwargs["diffusion_max_canvas_length"] = args.diffusion_max_canvas_length
    if args.diffusion_sampler != "confidence-threshold":
        kwargs["diffusion_sampler"] = args.diffusion_sampler
    if getattr(args, "block_length", None) is not None:
        kwargs["block_length"] = args.block_length
    if getattr(args, "num_to_transfer", None) is not None:
        kwargs["num_to_transfer"] = args.num_to_transfer
    if getattr(args, "max_transfer_per_step", None) is not None:
        kwargs["max_transfer_per_step"] = args.max_transfer_per_step
    if args.threshold is not None:
        kwargs["diffusion_threshold"] = args.threshold
        kwargs["threshold"] = args.threshold
    if getattr(args, "min_threshold", None) is not None:
        kwargs["min_threshold"] = args.min_threshold
    if getattr(args, "editing_threshold", None) is not None:
        kwargs["editing_threshold"] = args.editing_threshold
    if getattr(args, "max_post_steps", None) is not None:
        kwargs["max_post_steps"] = args.max_post_steps
    if getattr(args, "stability_steps", None) is not None:
        kwargs["stability_steps"] = args.stability_steps
    return kwargs


class DiffusionOutputHandler:
    def __init__(self, model: nn.Module, kwargs: Dict[str, Any], verbose: bool):
        self.verbose = verbose
        self.live_mode = bool(
            kwargs.get("diffusion_show_unmasking", False)
            and _has_model_diffusion_generator(model)
            and not callable(getattr(model, "make_unmasking_visualizer", None))
        )
        self.width = kwargs.get(
            "diffusion_unmasking_width", DEFAULT_DIFFUSION_UNMASKING_WIDTH
        )
        self.can_redraw = self.live_mode and _supports_in_place_output()
        self.redrawer = _DiffusionRedrawer() if self.can_redraw else None
        self.live_text = ""
        self.draft_active = False

    def handle_draft(self, response: GenerationResult) -> None:
        if self.verbose and self.redrawer is not None:
            self.redrawer.draw(_format_diffusion_draft_line(response, self.width))
            self.draft_active = True

    def handle_text(self, text: str) -> bool:
        if self.can_redraw and self.verbose:
            self.live_text += text
            if text:
                self.redrawer.draw(
                    _format_diffusion_live_text(
                        self.live_text,
                        self.width,
                        preserve_newlines=True,
                    )
                )
                self.draft_active = True
            return True

        if self.live_mode and self.verbose:
            return True

        if self.draft_active and self.verbose:
            _clear_diffusion_draft_line()
            self.draft_active = False
        return False

    def finish(self, text: str) -> None:
        if self.live_mode:
            if self.redrawer is not None and self.draft_active:
                self.redrawer.clear()
                self.draft_active = False
            if self.verbose and text:
                print(text, end="", flush=True)
        elif self.draft_active:
            _clear_diffusion_draft_line()
            self.draft_active = False


def _diffusion_config_dict(config: Any) -> Dict[str, Any]:
    return config if isinstance(config, dict) else {}


def _diffusion_initialize_canvas(
    batch_size: int,
    canvas_length: int,
    vocab_size: int,
    dtype,
) -> mx.array:
    return mx.random.randint(0, vocab_size, (batch_size, canvas_length)).astype(dtype)


def _normalize_decoder_input_ids(
    decoder_input_ids,
    batch_size: int,
    dtype,
) -> Optional[mx.array]:
    if decoder_input_ids is None:
        return None

    decoder_input_ids = mx.array(decoder_input_ids)
    if decoder_input_ids.ndim != 2:
        raise ValueError(
            "decoder_input_ids must be a 2D array with shape "
            "(batch_size, sequence_length)."
        )
    if decoder_input_ids.shape[0] != batch_size:
        raise ValueError(
            "decoder_input_ids batch size must match input_ids batch size."
        )
    return decoder_input_ids.astype(dtype)


def _diffusion_initial_canvas(
    decoder_input_ids: Optional[mx.array],
    start_index: int,
    batch_size: int,
    canvas_length: int,
    vocab_size: int,
    dtype,
) -> mx.array:
    seed_canvas = None
    seed_length = 0
    if decoder_input_ids is not None and start_index < decoder_input_ids.shape[1]:
        end_index = min(start_index + canvas_length, decoder_input_ids.shape[1])
        seed_canvas = decoder_input_ids[:, start_index:end_index].astype(dtype)
        seed_length = seed_canvas.shape[1]
        if seed_length == canvas_length:
            return seed_canvas

    random_canvas = _diffusion_initialize_canvas(
        batch_size,
        canvas_length,
        vocab_size,
        dtype,
    )
    if seed_canvas is None:
        return random_canvas

    random_canvas[:, :seed_length] = seed_canvas
    return random_canvas


def _diffusion_linear_temperature(
    cur_step: int,
    max_denoising_steps: int,
    schedule_config: Optional[Dict[str, Any]],
) -> Optional[float]:
    if schedule_config is None:
        return None
    t_min = float(schedule_config.get("t_min", 0.4))
    t_max = float(schedule_config.get("t_max", 0.8))
    return t_min + ((t_max - t_min) * (cur_step / max_denoising_steps))


def _diffusion_sample_canvas(
    processed_logits: mx.array,
    dtype,
    temperature: float,
) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(dtype)
    if temperature != 1.0:
        logits = logits / temperature
    return mx.random.categorical(logits).astype(dtype)


def _diffusion_token_probability(
    processed_logits: mx.array,
    token_ids: mx.array,
) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    token_logits = mx.take_along_axis(
        logits,
        token_ids[..., None],
        axis=-1,
    ).squeeze(-1)
    return mx.exp(token_logits - mx.logsumexp(logits, axis=-1))


def _diffusion_token_entropy(processed_logits: mx.array) -> mx.array:
    logits = processed_logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    return -mx.sum(probs * log_probs, axis=-1)


def _diffusion_confidence_transfer_mask(
    confidence: mx.array,
    unrevealed_mask: mx.array,
    threshold: float,
    *,
    force_all: bool = False,
) -> mx.array:
    if force_all:
        return unrevealed_mask

    transfer_mask = unrevealed_mask & (confidence >= threshold)
    has_unrevealed = mx.any(unrevealed_mask, axis=-1)
    has_transfer = mx.any(transfer_mask, axis=-1)
    needs_force = has_unrevealed & (~has_transfer)
    masked_confidence = mx.where(unrevealed_mask, confidence, -mx.inf)
    best_index = mx.argmax(masked_confidence, axis=-1)
    positions = mx.arange(confidence.shape[-1])[None, :]
    forced = (positions == best_index[:, None]) & needs_force[:, None]
    return transfer_mask | forced


def _diffusion_entropy_transfer_mask(
    entropy: mx.array,
    entropy_bound: float,
) -> mx.array:
    sorted_indices = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, sorted_indices, axis=-1)
    cumulative_entropy = mx.cumsum(sorted_entropy, axis=-1)
    cumulative_maximum_entropy = mx.cummax(sorted_entropy, axis=-1)
    sorted_selection_mask = (
        cumulative_entropy - cumulative_maximum_entropy
    ) <= entropy_bound
    selection_mask = mx.zeros_like(sorted_selection_mask)
    return mx.put_along_axis(
        selection_mask,
        sorted_indices,
        sorted_selection_mask,
        axis=-1,
    )


def _diffusion_static_cache_length(
    prompt_length: int,
    max_new_tokens: int,
    model_canvas_length: int,
) -> int:
    cached_canvas_tokens = ((max_new_tokens - 1) // model_canvas_length) * (
        model_canvas_length
    )
    return prompt_length + cached_canvas_tokens


def _diffusion_stable_and_confident(
    accepted_canvas: mx.array,
    processed_logits: mx.array,
    history: List[mx.array],
    stopping_config: Optional[Dict[str, Any]],
) -> bool:
    if stopping_config is None:
        return False

    stability_threshold = int(stopping_config.get("stability_threshold", 1))
    confidence_threshold = float(stopping_config.get("confidence_threshold", 0.005))

    if len(history) == stability_threshold:
        stable = all(
            bool(mx.all(accepted_canvas == canvas).item()) for canvas in history
        )
    else:
        stable = False

    history.append(accepted_canvas)
    if len(history) > stability_threshold:
        history.pop(0)

    if not stable:
        return False

    token_entropy = _diffusion_token_entropy(processed_logits)
    confident = bool((mx.mean(token_entropy) < confidence_threshold).item())
    return stable and confident


def _make_diffusion_decoder_logits_fns(
    model: nn.Module,
    kv_cache,
    mask_mapping,
    *,
    compile_graph: bool,
):
    def without_self_conditioning(current_canvas):
        return model.diffusion_decoder_logits(
            current_canvas,
            cache=kv_cache,
            self_conditioning=None,
            decoder_attention_mask=mask_mapping,
        )

    def with_self_conditioning(current_canvas, self_conditioning):
        return model.diffusion_decoder_logits(
            current_canvas,
            cache=kv_cache,
            self_conditioning=self_conditioning,
            decoder_attention_mask=mask_mapping,
        )

    if compile_graph:
        return mx.compile(without_self_conditioning), mx.compile(with_self_conditioning)
    return without_self_conditioning, with_self_conditioning


def _decode_diffusion_masked_draft(
    tokenizer: PreTrainedTokenizer,
    token_ids: List[int],
    reveal_mask: List[bool],
    skip_special_token_ids,
    max_chars: Optional[int] = None,
) -> str:
    skip_ids = set(skip_special_token_ids or [])
    pieces = []
    pending_tokens = []

    def flush_tokens():
        if not pending_tokens:
            return
        try:
            pieces.append(tokenizer.decode(pending_tokens, skip_special_tokens=False))
        except Exception:
            pieces.append(" ".join(str(token_id) for token_id in pending_tokens))
        pending_tokens.clear()

    for token_id, reveal in zip(token_ids, reveal_mask):
        token_id = int(token_id)
        if reveal:
            if token_id not in skip_ids:
                pending_tokens.append(token_id)
        else:
            flush_tokens()
            pieces.append("[Mask]")

    flush_tokens()
    text = " ".join(piece for piece in pieces if piece)
    return escape_carriage_returns(text)


def stream_diffusion_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    input_ids: mx.array,
    pixel_values: Optional[mx.array],
    attention_mask: Optional[mx.array],
    *,
    max_tokens: int,
    skip_special_token_ids,
    temperature: float = DEFAULT_TEMPERATURE,
    max_denoising_steps: Optional[int] = None,
    diffusion_full_canvas: bool = False,
    diffusion_min_canvas_length: Optional[int] = None,
    diffusion_max_canvas_length: Optional[int] = None,
    diffusion_static_cache: bool = False,
    diffusion_sampler: str = "confidence-threshold",
    diffusion_threshold: Optional[float] = None,
    diffusion_compile: bool = False,
    diffusion_show_unmasking: bool = False,
    diffusion_unmasking_interval: int = 1,
    diffusion_unmasking_width: int = DEFAULT_DIFFUSION_UNMASKING_WIDTH,
    mm_token_type_ids: Optional[mx.array] = None,
    prefill_step_size: Optional[int] = None,
    decoder_input_ids: Optional[mx.array] = None,
) -> Generator[GenerationResult, None, None]:
    if input_ids.shape[0] != 1:
        raise ValueError(
            "Diffusion model streaming generation only supports batch size 1."
        )

    generation_config = _diffusion_config_dict(
        getattr(model.config, "generation_config", None)
    )
    config_eos_token_ids = generation_config.get("eos_token_id")
    if config_eos_token_ids is not None and hasattr(tokenizer, "stopping_criteria"):
        tokenizer.stopping_criteria.add_eos_token_ids(config_eos_token_ids)
    text_config = model.config.text_config
    batch_size, prompt_length = input_ids.shape
    prompt_tokens = input_ids.size
    model_canvas_length = int(model.config.canvas_length)
    if diffusion_min_canvas_length is not None and diffusion_min_canvas_length <= 0:
        raise ValueError("diffusion_min_canvas_length must be a positive integer.")
    if diffusion_max_canvas_length is not None and diffusion_max_canvas_length <= 0:
        raise ValueError("diffusion_max_canvas_length must be a positive integer.")
    max_canvas_length = (
        model_canvas_length
        if diffusion_full_canvas
        else min(
            model_canvas_length,
            int(diffusion_max_canvas_length or model_canvas_length),
        )
    )
    min_canvas_length = min(
        max_canvas_length,
        int(diffusion_min_canvas_length or DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH),
    )
    vocab_size = int(text_config.vocab_size)
    decoder_input_ids = _normalize_decoder_input_ids(
        decoder_input_ids,
        batch_size,
        input_ids.dtype,
    )
    max_new_tokens = int(max_tokens or generation_config.get("max_new_tokens", 256))
    if max_denoising_steps is None:
        max_denoising_steps = int(
            generation_config.get("max_denoising_steps")
            or DEFAULT_DIFFUSION_MAX_DENOISING_STEPS
        )
    else:
        max_denoising_steps = int(max_denoising_steps)
    if diffusion_unmasking_interval <= 0:
        raise ValueError("diffusion_unmasking_interval must be a positive integer.")
    if diffusion_unmasking_width < 0:
        raise ValueError("diffusion_unmasking_width must be non-negative.")
    if diffusion_sampler not in ("entropy-bound", "confidence-threshold"):
        raise ValueError(f"Unsupported diffusion sampler: {diffusion_sampler!r}.")
    if diffusion_threshold is None:
        diffusion_threshold = DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD
    if not 0.0 <= diffusion_threshold <= 1.0:
        raise ValueError("diffusion_threshold must be between 0 and 1.")
    if prefill_step_size is not None:
        prefill_step_size = int(prefill_step_size)
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be a positive integer.")

    sampler_config = _diffusion_config_dict(
        generation_config.get("sampler_config", None)
    )
    sampler_name = sampler_config.get("_cls_name", "EntropyBoundSamplerConfig")
    entropy_bound = float(sampler_config.get("entropy_bound", 0.1))
    if sampler_name != "EntropyBoundSamplerConfig":
        raise NotImplementedError(
            f"Diffusion sampler {sampler_name!r} is not supported yet."
        )
    temperature_config = generation_config.get("linear_temperature_schedule_config")
    temperature_config = _diffusion_config_dict(temperature_config)
    if not temperature_config:
        if "t_min" in generation_config or "t_max" in generation_config:
            temperature_config = {
                "t_min": generation_config.get("t_min", 0.4),
                "t_max": generation_config.get("t_max", 0.8),
            }
        else:
            temperature_config = {"t_min": 0.4, "t_max": 0.8}

    diffusion_stopping_config = generation_config.get("diffusion_stopping_config")
    diffusion_stopping_config = _diffusion_config_dict(diffusion_stopping_config)
    if not diffusion_stopping_config:
        diffusion_stopping_config = {
            key: generation_config[key]
            for key in ("confidence_threshold", "stability_threshold")
            if key in generation_config
        }
        if not diffusion_stopping_config:
            diffusion_stopping_config = None

    if attention_mask is None:
        attention_mask = mx.ones((batch_size, prompt_length), dtype=mx.bool_)
    else:
        attention_mask = attention_mask.astype(mx.bool_)

    static_cache_length = _diffusion_static_cache_length(
        prompt_length,
        max_new_tokens,
        max_canvas_length,
    )
    use_static_cache = diffusion_static_cache and static_cache_length > prompt_length
    has_padding = not bool(mx.all(attention_mask).item())
    if use_static_cache:
        decoder_attention_mask = mx.zeros(
            (batch_size, static_cache_length), dtype=mx.bool_
        )
        decoder_attention_mask[:, :prompt_length] = attention_mask
        cached_sequence_length = prompt_length
        kv_cache = model.make_cache(max_size=static_cache_length)
    else:
        decoder_attention_mask = attention_mask if has_padding else None
        cached_sequence_length = prompt_length
        kv_cache = model.make_cache()
    detokenizer = make_streaming_detokenizer(processor)
    prefill_policy_kwargs = {
        "attention_mask": attention_mask,
        "has_padding": has_padding,
        "use_static_cache": use_static_cache,
        "pixel_values": pixel_values,
        "mm_token_type_ids": mm_token_type_ids,
    }
    chunk_prefill = (
        prefill_step_size is not None
        and prompt_length > prefill_step_size
        and _chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            prompt_cache=kv_cache,
            prefill_kwargs=prefill_policy_kwargs,
        )
    )

    generated_tokens = 0
    diffusion_canvas_tokens = 0
    diffusion_denoising_steps = 0
    diffusion_work_tokens = 0
    last_token = None
    prompt_time = 0.0
    generation_tic = time.perf_counter()
    tic = time.perf_counter()
    is_prefill = True
    current_canvas = None
    stopped = False
    stop_reason = "length"

    def make_result(
        text: str,
        *,
        is_draft: bool = False,
        draft_text: str = "",
        diffusion_step: int = 0,
        diffusion_total_steps: int = 0,
        diffusion_canvas_index: int = 0,
        diffusion_block_complete: bool = False,
        finish_reason: Optional[str] = None,
    ) -> GenerationResult:
        generation_time = max(time.perf_counter() - generation_tic, 1e-9)
        return GenerationResult(
            text=text,
            token=last_token,
            logprobs=None,
            prompt_tokens=prompt_tokens,
            generation_tokens=generated_tokens,
            total_tokens=prompt_tokens + generated_tokens,
            prompt_tps=prompt_tokens / max(prompt_time, 1e-9),
            generation_tps=generated_tokens / generation_time,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason,
            diffusion_canvas_tokens=diffusion_canvas_tokens,
            diffusion_denoising_steps=diffusion_denoising_steps,
            diffusion_work_tokens=diffusion_work_tokens,
            diffusion_canvas_tps=diffusion_canvas_tokens / generation_time,
            diffusion_work_tps=diffusion_work_tokens / generation_time,
            is_draft=is_draft,
            draft_text=draft_text,
            diffusion_step=diffusion_step,
            diffusion_total_steps=diffusion_total_steps,
            diffusion_canvas_index=diffusion_canvas_index,
            diffusion_block_complete=diffusion_block_complete,
        )

    with mx.stream(generation_stream):
        self_conditioning_context = model.diffusion_prepare_self_conditioning()
        canvas_index = 0
        while generated_tokens < max_new_tokens:
            canvas_index += 1
            unprocessed_input_ids = input_ids if is_prefill else current_canvas
            if is_prefill:
                kv_cache = model.diffusion_prefill_cache(
                    unprocessed_input_ids,
                    attention_mask=attention_mask if has_padding else None,
                    cache=kv_cache,
                    pixel_values=pixel_values,
                    mm_token_type_ids=mm_token_type_ids,
                    prefill_step_size=prefill_step_size,
                    chunk_prefill=chunk_prefill,
                )
            else:
                kv_cache = model.diffusion_update_cache(
                    unprocessed_input_ids,
                    cache=kv_cache,
                )

            if is_prefill:
                mx.eval([c.state for c in kv_cache])
                prompt_time = time.perf_counter() - tic
                generation_tic = time.perf_counter()
                is_prefill = False

            remaining_tokens = max_new_tokens - generated_tokens
            canvas_length = (
                model_canvas_length
                if diffusion_full_canvas
                else min(max_canvas_length, max(remaining_tokens, min_canvas_length))
            )
            current_decoder_attention_mask = (
                mx.concatenate(
                    [
                        decoder_attention_mask,
                        mx.ones((batch_size, canvas_length), dtype=mx.bool_),
                    ],
                    axis=-1,
                )
                if decoder_attention_mask is not None
                else None
            )
            current_canvas = _diffusion_initial_canvas(
                decoder_input_ids,
                generated_tokens,
                batch_size,
                canvas_length,
                vocab_size,
                input_ids.dtype,
            )
            draft_reveal_mask = mx.zeros(current_canvas.shape, dtype=mx.bool_)
            draft_canvas = current_canvas
            accepted_canvas = current_canvas
            argmax_canvas = current_canvas
            self_conditioning = None
            mask_mapping = model.diffusion_decoder_masks(
                current_canvas,
                kv_cache,
                current_decoder_attention_mask,
            )
            decoder_logits_without_sc, decoder_logits_with_sc = (
                _make_diffusion_decoder_logits_fns(
                    model,
                    kv_cache,
                    mask_mapping,
                    compile_graph=diffusion_compile,
                )
            )
            diffusion_history: List[mx.array] = []
            denoising_steps_this_canvas = 0

            if diffusion_show_unmasking:
                draft_text = _decode_diffusion_masked_draft(
                    tokenizer,
                    [int(token_id) for token_id in draft_canvas[0].tolist()],
                    [False] * canvas_length,
                    skip_special_token_ids,
                    max_chars=diffusion_unmasking_width,
                )
                yield make_result(
                    "",
                    is_draft=True,
                    draft_text=draft_text,
                    diffusion_step=0,
                    diffusion_total_steps=max_denoising_steps,
                    diffusion_canvas_index=canvas_index,
                )

            for cur_step in reversed(range(1, max_denoising_steps + 1)):
                denoising_steps_this_canvas += 1
                try:
                    if self_conditioning is None:
                        processed_logits = decoder_logits_without_sc(current_canvas)
                    else:
                        processed_logits = decoder_logits_with_sc(
                            current_canvas,
                            self_conditioning,
                        )
                except Exception as exc:
                    if not diffusion_compile:
                        raise
                    logger.warning(
                        "Diffusion decoder compilation failed; falling back "
                        "to the eager path: %s",
                        exc,
                    )
                    diffusion_compile = False
                    decoder_logits_without_sc, decoder_logits_with_sc = (
                        _make_diffusion_decoder_logits_fns(
                            model,
                            kv_cache,
                            mask_mapping,
                            compile_graph=False,
                        )
                    )
                    if self_conditioning is None:
                        processed_logits = decoder_logits_without_sc(current_canvas)
                    else:
                        processed_logits = decoder_logits_with_sc(
                            current_canvas,
                            self_conditioning,
                        )
                schedule_temperature = _diffusion_linear_temperature(
                    cur_step,
                    max_denoising_steps,
                    temperature_config,
                )
                if schedule_temperature is not None:
                    processed_logits = processed_logits / schedule_temperature

                argmax_canvas = mx.argmax(processed_logits, axis=-1).astype(
                    input_ids.dtype
                )
                if cur_step == 1 and not diffusion_show_unmasking:
                    break

                denoiser_canvas = (
                    argmax_canvas
                    if temperature <= 0
                    else _diffusion_sample_canvas(
                        processed_logits,
                        input_ids.dtype,
                        temperature,
                    )
                )

                if diffusion_sampler == "entropy-bound":
                    if cur_step > 1:
                        token_entropy = _diffusion_token_entropy(processed_logits)
                        next_self_conditioning = model.diffusion_self_conditioning(
                            processed_logits,
                            self_conditioning_context,
                        )
                    else:
                        token_entropy = _diffusion_token_entropy(processed_logits)
                        next_self_conditioning = None
                    acceptance_mask = _diffusion_entropy_transfer_mask(
                        token_entropy,
                        entropy_bound,
                    )
                    accepted_canvas = mx.where(
                        acceptance_mask,
                        denoiser_canvas,
                        current_canvas,
                    )
                    current_canvas = mx.where(
                        acceptance_mask,
                        accepted_canvas,
                        _diffusion_initialize_canvas(
                            batch_size,
                            canvas_length,
                            vocab_size,
                            input_ids.dtype,
                        ),
                    )
                    draft_reveal_mask = acceptance_mask
                    draft_canvas = argmax_canvas
                else:
                    next_self_conditioning = None
                    unrevealed_mask = ~draft_reveal_mask
                    confidence = _diffusion_token_probability(
                        processed_logits,
                        denoiser_canvas,
                    )
                    acceptance_mask = _diffusion_confidence_transfer_mask(
                        confidence,
                        unrevealed_mask,
                        diffusion_threshold,
                        force_all=cur_step == 1,
                    )
                    accepted_canvas = mx.where(
                        acceptance_mask,
                        denoiser_canvas,
                        draft_canvas,
                    )
                    current_canvas = mx.where(
                        draft_reveal_mask | acceptance_mask,
                        accepted_canvas,
                        _diffusion_initialize_canvas(
                            batch_size,
                            canvas_length,
                            vocab_size,
                            input_ids.dtype,
                        ),
                    )
                    draft_reveal_mask = draft_reveal_mask | acceptance_mask
                    draft_canvas = mx.where(
                        acceptance_mask, accepted_canvas, draft_canvas
                    )

                displayed_step = max_denoising_steps - cur_step + 1
                should_show_unmasking = diffusion_show_unmasking and (
                    displayed_step == 1
                    or cur_step == 1
                    or displayed_step % diffusion_unmasking_interval == 0
                )
                if should_show_unmasking:
                    mx.eval(draft_canvas, draft_reveal_mask)
                    draft_text = _decode_diffusion_masked_draft(
                        tokenizer,
                        [int(token_id) for token_id in draft_canvas[0].tolist()],
                        [bool(v) for v in draft_reveal_mask[0].tolist()],
                        skip_special_token_ids,
                        max_chars=diffusion_unmasking_width,
                    )
                    yield make_result(
                        "",
                        is_draft=True,
                        draft_text=draft_text,
                        diffusion_step=displayed_step,
                        diffusion_total_steps=max_denoising_steps,
                        diffusion_canvas_index=canvas_index,
                    )

                if diffusion_sampler == "confidence-threshold" and bool(
                    mx.all(draft_reveal_mask).item()
                ):
                    accepted_canvas = draft_canvas
                    break

                if _diffusion_stable_and_confident(
                    argmax_canvas,
                    processed_logits,
                    diffusion_history,
                    diffusion_stopping_config,
                ):
                    break

                if cur_step > 1:
                    if next_self_conditioning is None:
                        next_self_conditioning = model.diffusion_self_conditioning(
                            processed_logits,
                            self_conditioning_context,
                        )
                    self_conditioning = next_self_conditioning

            current_canvas = argmax_canvas
            diffusion_canvas_tokens += canvas_length
            diffusion_denoising_steps += denoising_steps_this_canvas
            diffusion_work_tokens += canvas_length * denoising_steps_this_canvas
            mx.eval(current_canvas)

            for token_id in current_canvas[0].tolist():
                last_token = int(token_id)
                generated_tokens += 1

                if tokenizer.stopping_criteria(last_token):
                    stopped = True
                    stop_reason = "stop"
                    break

                detokenizer.add_token(
                    last_token, skip_special_token_ids=skip_special_token_ids
                )
                yield make_result(
                    detokenizer.last_segment,
                    diffusion_canvas_index=canvas_index,
                )

                if generated_tokens >= max_new_tokens:
                    stopped = True
                    stop_reason = "length"
                    break

            # Mark the end of this denoised block so consumers (e.g. the
            # server) can stream block-by-block.
            yield make_result(
                "",
                diffusion_canvas_index=canvas_index,
                diffusion_block_complete=True,
            )

            if stopped:
                break

            if use_static_cache:
                decoder_attention_mask[
                    :, cached_sequence_length : cached_sequence_length + canvas_length
                ] = True
                cached_sequence_length += canvas_length
            elif decoder_attention_mask is not None:
                decoder_attention_mask = mx.concatenate(
                    [
                        decoder_attention_mask,
                        mx.ones((batch_size, canvas_length), dtype=mx.bool_),
                    ],
                    axis=-1,
                )
            mx.clear_cache()

    if prompt_time == 0.0:
        prompt_time = time.perf_counter() - tic
    detokenizer.finalize()
    finish_reason = stop_reason if stopped else "length"
    yield make_result(detokenizer.last_segment, finish_reason=finish_reason)


def _stream_model_diffusion_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    input_ids: mx.array,
    pixel_values: Optional[mx.array],
    attention_mask: Optional[mx.array],
    *,
    max_tokens: int,
    skip_special_token_ids,
    skip_special_tokens: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_denoising_steps: Optional[int] = None,
    block_length: Optional[int] = None,
    verbose: bool = False,
    on_result: Optional[Callable[[GenerationResult], bool]] = None,
    **kwargs,
) -> Generator[GenerationResult, None, None]:
    if input_ids.shape[0] != 1:
        raise ValueError("Diffusion streaming generation only supports batch size 1.")
    if pixel_values is not None and not (
        _has_engine_diffusion_config(getattr(model, "config", None))
        or getattr(getattr(model, "config", None), "vision_config", None) is not None
    ):
        model_type = getattr(getattr(model, "config", None), "model_type", "This model")
        raise ValueError(f"{model_type} is a text-only model.")

    config = getattr(model, "config", None)
    if max_denoising_steps is None:
        max_denoising_steps = kwargs.pop("steps", None)
        if max_denoising_steps is None and not _has_engine_diffusion_config(config):
            max_denoising_steps = getattr(config, "default_diffusion_steps", 32)
    else:
        kwargs.pop("steps", None)

    if block_length is None and not _has_engine_diffusion_config(config):
        block_length = getattr(config, "default_block_length", None) or 32

    tuned_kwargs = {}
    for key, config_attr in (
        ("threshold", "default_diffusion_threshold"),
        ("min_threshold", "default_diffusion_min_threshold"),
        ("editing_threshold", "default_diffusion_editing_threshold"),
        ("num_to_transfer", "default_diffusion_num_to_transfer"),
        ("max_transfer_per_step", "default_diffusion_max_transfer_per_step"),
        ("max_post_steps", "default_diffusion_max_post_steps"),
        ("stability_steps", "default_diffusion_stability_steps"),
    ):
        value = kwargs.pop(key, None)
        if value is None:
            value = getattr(config, config_attr, None)
        if value is not None:
            tuned_kwargs[key] = value

    diffusion_show_unmasking = kwargs.get("diffusion_show_unmasking", False)

    generation_stats = {}
    pending_results: List[GenerationResult] = []

    def emit(result: GenerationResult) -> bool:
        if on_result is not None:
            return bool(on_result(result))
        pending_results.append(result)
        return True

    top_p_arg = None if top_p is None or top_p >= 1.0 else top_p
    top_k_arg = None if top_k is None or top_k <= 0 else top_k
    generated = model.language_model.generate(
        input_ids,
        temperature=temperature,
        block_length=block_length,
        steps=max_denoising_steps,
        gen_length=max_tokens,
        top_p=top_p_arg,
        top_k=top_k_arg,
        eos_early_stop=True,
        visualize=bool(verbose or diffusion_show_unmasking),
        processor=processor,
        tokenizer=tokenizer,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        skip_special_tokens=skip_special_tokens,
        skip_special_token_ids=skip_special_token_ids,
        stats=generation_stats,
        on_result=emit,
        **tuned_kwargs,
        **kwargs,
    )
    mx.eval(generated)

    if generation_stats.get("text_already_printed"):
        for result in pending_results:
            result.text_already_printed = True

    yield from pending_results


def _stream_model_diffusion_from_kwargs(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    input_ids: mx.array,
    pixel_values: Optional[mx.array],
    attention_mask: Optional[mx.array],
    skip_special_token_ids,
    kwargs: Dict[str, Any],
    *,
    skip_special_tokens: bool = False,
    verbose: bool = False,
    on_result: Optional[Callable[[GenerationResult], bool]] = None,
) -> Generator[GenerationResult, None, None]:
    max_denoising_steps = kwargs.pop("max_denoising_steps", None)
    block_length = kwargs.pop("block_length", None)
    yield from _stream_model_diffusion_generate(
        model,
        processor,
        tokenizer,
        input_ids,
        pixel_values,
        attention_mask,
        max_tokens=kwargs.pop("max_tokens", 2048),
        temperature=kwargs.pop("temperature", DEFAULT_TEMPERATURE),
        top_p=kwargs.pop("top_p", None),
        top_k=kwargs.pop("top_k", None),
        skip_special_token_ids=skip_special_token_ids,
        skip_special_tokens=skip_special_tokens,
        max_denoising_steps=max_denoising_steps,
        block_length=block_length,
        verbose=verbose,
        on_result=on_result,
        **kwargs,
    )


def _diffusion_stream_strategy(
    model: nn.Module,
    kwargs: Optional[Dict[str, Any]] = None,
):
    if _uses_model_diffusion_generator(model, kwargs):
        return _stream_model_diffusion_from_kwargs
    return None


def stream_diffusion_generate_from_kwargs(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    input_ids: mx.array,
    pixel_values: Optional[mx.array],
    attention_mask: Optional[mx.array],
    skip_special_token_ids,
    kwargs: Dict[str, Any],
    *,
    skip_special_tokens: bool = False,
    verbose: bool = False,
    on_result: Optional[Callable[[GenerationResult], bool]] = None,
) -> Generator[GenerationResult, None, None]:
    if kwargs.get("logits_processors") is not None:
        raise ValueError(
            "Structured response_format is not supported with diffusion models."
        )
    seed = kwargs.pop("seed", None)
    if seed is not None:
        mx.random.seed(seed)

    stream_strategy = _diffusion_stream_strategy(model, kwargs)
    if stream_strategy is None:
        raise ValueError("Model does not support diffusion generation.")

    with wired_limit(model, [generation_stream]):
        yield from stream_strategy(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            attention_mask,
            skip_special_token_ids=skip_special_token_ids,
            kwargs=kwargs,
            skip_special_tokens=skip_special_tokens,
            verbose=verbose,
            on_result=on_result,
        )
        mx.clear_cache()
