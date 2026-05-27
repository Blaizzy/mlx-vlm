from __future__ import annotations

import logging
import os
import shutil
import time
import unicodedata
from typing import Any, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from ..tokenizer_utils import make_streaming_detokenizer
from .common import GenerationResult, generation_stream, wired_limit

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_TEMPERATURE = 0.0
DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH = 64
DEFAULT_DIFFUSION_UNMASKING_WIDTH = 0
_DIFFUSION_MODEL_TYPE = "diffusion_gemma4"


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def _clip_display_width(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""

    out = []
    width = 0
    for char in text:
        if unicodedata.combining(char):
            char_width = 0
        else:
            char_width = 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
        if width + char_width > max_width:
            break
        out.append(char)
        width += char_width

    return "".join(out)


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
    width = _diffusion_display_limit(requested_width)
    if width is None:
        return response.draft_text
    return _clip_display_width(response.draft_text, width)


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
    preserve_newlines: bool = False,
) -> str:
    width = _diffusion_display_limit(requested_width)
    text = text.replace("\r", "\\r")
    if not preserve_newlines:
        text = text.replace("\n", "\\n")
    if width is None:
        return text
    return _clip_display_width(text, width)


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
        width = _display_width(line)
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


def _is_diffusion_config(config: Any) -> bool:
    return getattr(config, "model_type", None) == _DIFFUSION_MODEL_TYPE


def is_diffusion_model(model: nn.Module) -> bool:
    return _is_diffusion_config(getattr(model, "config", None))


def diffusion_kwargs_from_args(args: Any, config: Any) -> Dict[str, Any]:
    if not _is_diffusion_config(config):
        return {}

    kwargs = {}
    if args.max_denoising_steps is not None:
        kwargs["max_denoising_steps"] = args.max_denoising_steps
    if args.diffusion_full_canvas:
        kwargs["diffusion_full_canvas"] = True
    if args.diffusion_min_canvas_length is not None:
        kwargs["diffusion_min_canvas_length"] = args.diffusion_min_canvas_length
    if args.diffusion_sampler != "auto-regressive-euler":
        kwargs["diffusion_sampler"] = args.diffusion_sampler
        kwargs["diffusion_threshold"] = args.diffusion_threshold
    return kwargs


class DiffusionOutputHandler:
    def __init__(self, model: nn.Module, kwargs: Dict[str, Any], verbose: bool):
        self.verbose = verbose
        self.live_mode = bool(
            kwargs.get("diffusion_show_unmasking", False) and is_diffusion_model(model)
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


def print_diffusion_stats(response: GenerationResult) -> None:
    if response.diffusion_canvas_tokens <= 0:
        return
    print(
        f"Diffusion canvas: {response.diffusion_canvas_tokens} tokens, "
        f"{response.diffusion_canvas_tps:.3f} tokens-per-sec"
    )
    print(
        f"Diffusion work: {response.diffusion_work_tokens} "
        "token-steps, "
        f"{response.diffusion_work_tps:.3f} token-steps-per-sec"
    )


def _diffusion_initialize_canvas(
    batch_size: int,
    canvas_length: int,
    vocab_size: int,
    dtype,
) -> mx.array:
    return mx.random.randint(0, vocab_size, (batch_size, canvas_length)).astype(dtype)


def _diffusion_linear_temperature(
    cur_step: int,
    max_denoising_steps: int,
    schedule_config: Optional[Dict[str, Any]],
) -> Optional[float]:
    if schedule_config is None:
        return None
    t_min = float(schedule_config.get("t_min", 0.4))
    t_max = float(schedule_config.get("t_max", 0.8))
    return t_min + ((t_max - t_min) * ((cur_step - 1) / max_denoising_steps))


def _diffusion_accept_canvas(
    current_canvas: mx.array,
    denoiser_canvas: mx.array,
    cur_step: int,
    canvas_length: int,
    ar_mask_noise_proportion: float,
) -> Tuple[mx.array, mx.array]:
    acceptance_probability = 1.0 / cur_step
    acceptance_mask = (
        mx.random.uniform(shape=current_canvas.shape) < acceptance_probability
    )

    if ar_mask_noise_proportion > 0.0:
        canvas_proportion = mx.linspace(
            0.0,
            1.0 - ar_mask_noise_proportion,
            canvas_length,
        )
        ar_mask = canvas_proportion < acceptance_probability
        acceptance_mask = acceptance_mask | ar_mask[None, :]

    return mx.where(acceptance_mask, denoiser_canvas, current_canvas), acceptance_mask


def _diffusion_renoise_canvas(
    accepted_canvas: mx.array,
    cur_step: int,
    max_denoising_steps: int,
    renoise_ratio_modifier: float,
    vocab_size: int,
    dtype,
) -> mx.array:
    renoise_probability = renoise_ratio_modifier * (
        (cur_step - 1) / max_denoising_steps
    )
    renoise_mask = mx.random.uniform(shape=accepted_canvas.shape) < renoise_probability
    random_canvas = _diffusion_initialize_canvas(
        accepted_canvas.shape[0],
        accepted_canvas.shape[1],
        vocab_size,
        dtype,
    )
    return mx.where(renoise_mask, random_canvas, accepted_canvas)


def _diffusion_sample_canvas(
    processed_logits: mx.array,
    dtype,
    temperature: float,
) -> mx.array:
    if temperature == 0:
        return mx.argmax(processed_logits, axis=-1).astype(dtype)
    return mx.random.categorical(processed_logits.astype(mx.float32)).astype(dtype)


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
    confidence_threshold = float(stopping_config.get("confidence_threshold", 0.05))

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

    log_probs = processed_logits - mx.logsumexp(
        processed_logits, axis=-1, keepdims=True
    )
    probs = mx.exp(log_probs)
    token_entropy = -probs * log_probs
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
        return model(
            cache=kv_cache,
            canvas_ids=current_canvas,
            self_conditioning_logits=None,
            decoder_attention_mask=mask_mapping,
        ).logits

    def with_self_conditioning(current_canvas, self_conditioning_logits):
        return model(
            cache=kv_cache,
            canvas_ids=current_canvas,
            self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=mask_mapping,
        ).logits

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
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text


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
    diffusion_static_cache: bool = False,
    diffusion_sampler: str = "auto-regressive-euler",
    diffusion_threshold: float = 0.9,
    diffusion_compile: bool = False,
    diffusion_show_unmasking: bool = False,
    diffusion_unmasking_interval: int = 1,
    diffusion_unmasking_width: int = DEFAULT_DIFFUSION_UNMASKING_WIDTH,
) -> Generator[GenerationResult, None, None]:
    if pixel_values is not None:
        raise ValueError("Diffusion model vision inputs are not supported yet.")
    if input_ids.shape[0] != 1:
        raise ValueError(
            "Diffusion model streaming generation only supports batch size 1."
        )

    generation_config = _diffusion_config_dict(
        getattr(model.config, "generation_config", None)
    )
    text_config = model.config.text_config
    batch_size, prompt_length = input_ids.shape
    prompt_tokens = input_ids.size
    model_canvas_length = int(model.config.canvas_length)
    if diffusion_min_canvas_length is not None and diffusion_min_canvas_length <= 0:
        raise ValueError("diffusion_min_canvas_length must be a positive integer.")
    min_canvas_length = min(
        model_canvas_length,
        int(diffusion_min_canvas_length or DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH),
    )
    vocab_size = int(text_config.vocab_size)
    max_new_tokens = int(max_tokens or generation_config.get("max_new_tokens", 256))
    max_denoising_steps = int(
        max_denoising_steps or generation_config.get("max_denoising_steps") or 32
    )
    if diffusion_unmasking_interval <= 0:
        raise ValueError("diffusion_unmasking_interval must be a positive integer.")
    if diffusion_unmasking_width < 0:
        raise ValueError("diffusion_unmasking_width must be non-negative.")
    if diffusion_sampler not in ("auto-regressive-euler", "confidence-threshold"):
        raise ValueError(f"Unsupported diffusion sampler: {diffusion_sampler!r}.")
    if not 0.0 <= diffusion_threshold <= 1.0:
        raise ValueError("diffusion_threshold must be between 0 and 1.")

    sampler_config = _diffusion_config_dict(
        generation_config.get("sampler_config", None)
    )
    sampler_name = sampler_config.get("_cls_name", "AutoRegressiveEulerSamplerConfig")
    if sampler_name != "AutoRegressiveEulerSamplerConfig":
        raise NotImplementedError(
            f"Diffusion sampler {sampler_name!r} is not supported yet."
        )
    renoise_ratio_modifier = float(sampler_config.get("renoise_ratio_modifier", 0.8))
    ar_mask_noise_proportion = float(
        sampler_config.get("ar_mask_noise_proportion", 0.0)
    )
    temperature_config = generation_config.get("linear_temperature_schedule_config")
    temperature_config = _diffusion_config_dict(temperature_config)
    if not temperature_config:
        temperature_config = {"t_min": 0.4, "t_max": 0.8}

    diffusion_stopping_config = generation_config.get("diffusion_stopping_config")
    diffusion_stopping_config = _diffusion_config_dict(diffusion_stopping_config)
    if not diffusion_stopping_config:
        diffusion_stopping_config = None

    if attention_mask is None:
        attention_mask = mx.ones((batch_size, prompt_length), dtype=mx.bool_)
    else:
        attention_mask = attention_mask.astype(mx.bool_)

    static_cache_length = _diffusion_static_cache_length(
        prompt_length,
        max_new_tokens,
        model_canvas_length,
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

    def make_result(
        text: str,
        *,
        is_draft: bool = False,
        draft_text: str = "",
        diffusion_step: int = 0,
        diffusion_total_steps: int = 0,
        diffusion_canvas_index: int = 0,
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
        )

    with mx.stream(generation_stream):
        canvas_index = 0
        while generated_tokens < max_new_tokens:
            canvas_index += 1
            unprocessed_input_ids = input_ids if is_prefill else current_canvas
            encoder_attention_mask = attention_mask if is_prefill else None
            _, kv_cache = model.model.encoder(
                unprocessed_input_ids,
                attention_mask=encoder_attention_mask,
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
                else min(model_canvas_length, max(remaining_tokens, min_canvas_length))
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
            current_canvas = _diffusion_initialize_canvas(
                batch_size,
                canvas_length,
                vocab_size,
                input_ids.dtype,
            )
            draft_reveal_mask = mx.zeros(current_canvas.shape, dtype=mx.bool_)
            draft_canvas = current_canvas
            accepted_canvas = current_canvas
            self_conditioning_logits = None
            mask_mapping = model.model.decoder._make_decoder_masks(
                current_canvas[..., None],
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
                    if self_conditioning_logits is None:
                        processed_logits = decoder_logits_without_sc(current_canvas)
                    else:
                        processed_logits = decoder_logits_with_sc(
                            current_canvas,
                            self_conditioning_logits,
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
                    if self_conditioning_logits is None:
                        processed_logits = decoder_logits_without_sc(current_canvas)
                    else:
                        processed_logits = decoder_logits_with_sc(
                            current_canvas,
                            self_conditioning_logits,
                        )
                schedule_temperature = _diffusion_linear_temperature(
                    cur_step,
                    max_denoising_steps,
                    temperature_config,
                )
                if schedule_temperature is not None:
                    processed_logits = processed_logits / schedule_temperature

                denoiser_canvas = _diffusion_sample_canvas(
                    processed_logits, input_ids.dtype, temperature
                )

                if diffusion_sampler == "confidence-threshold":
                    confidence = _diffusion_token_probability(
                        processed_logits,
                        denoiser_canvas,
                    )
                    unrevealed_mask = ~draft_reveal_mask
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
                else:
                    accepted_canvas, acceptance_mask = _diffusion_accept_canvas(
                        current_canvas,
                        denoiser_canvas,
                        cur_step,
                        canvas_length,
                        ar_mask_noise_proportion,
                    )
                    current_canvas = _diffusion_renoise_canvas(
                        accepted_canvas,
                        cur_step,
                        max_denoising_steps,
                        renoise_ratio_modifier,
                        vocab_size,
                        input_ids.dtype,
                    )

                draft_reveal_mask = draft_reveal_mask | acceptance_mask
                draft_canvas = mx.where(acceptance_mask, accepted_canvas, draft_canvas)

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
                    accepted_canvas,
                    processed_logits,
                    diffusion_history,
                    diffusion_stopping_config,
                ):
                    break

                self_conditioning_logits = processed_logits

            current_canvas = accepted_canvas
            diffusion_canvas_tokens += canvas_length
            diffusion_denoising_steps += denoising_steps_this_canvas
            diffusion_work_tokens += canvas_length * denoising_steps_this_canvas
            mx.eval(current_canvas)

            for token_id in current_canvas[0].tolist():
                last_token = int(token_id)
                generated_tokens += 1

                if tokenizer.stopping_criteria(last_token):
                    stopped = True
                    break

                detokenizer.add_token(
                    last_token, skip_special_token_ids=skip_special_token_ids
                )
                yield make_result(detokenizer.last_segment)

                if generated_tokens >= max_new_tokens:
                    stopped = True
                    break

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
    finish_reason = "stop" if stopped else "length"
    yield make_result(detokenizer.last_segment, finish_reason=finish_reason)


def stream_diffusion_generate_from_kwargs(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    input_ids: mx.array,
    pixel_values: Optional[mx.array],
    attention_mask: Optional[mx.array],
    skip_special_token_ids,
    kwargs: Dict[str, Any],
) -> Generator[GenerationResult, None, None]:
    max_denoising_steps = kwargs.pop("max_denoising_steps", None)
    diffusion_full_canvas = kwargs.pop("diffusion_full_canvas", False)
    diffusion_min_canvas_length = kwargs.pop("diffusion_min_canvas_length", None)
    diffusion_static_cache = kwargs.pop("diffusion_static_cache", False)
    diffusion_sampler = kwargs.pop("diffusion_sampler", "auto-regressive-euler")
    diffusion_threshold = kwargs.pop("diffusion_threshold", 0.9)
    diffusion_compile = kwargs.pop("diffusion_compile", False)
    diffusion_show_unmasking = kwargs.pop("diffusion_show_unmasking", False)
    diffusion_unmasking_interval = kwargs.pop("diffusion_unmasking_interval", 1)
    diffusion_unmasking_width = kwargs.pop(
        "diffusion_unmasking_width", DEFAULT_DIFFUSION_UNMASKING_WIDTH
    )
    with wired_limit(model, [generation_stream]):
        yield from stream_diffusion_generate(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            attention_mask,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
            skip_special_token_ids=skip_special_token_ids,
            max_denoising_steps=max_denoising_steps,
            diffusion_full_canvas=diffusion_full_canvas,
            diffusion_min_canvas_length=diffusion_min_canvas_length,
            diffusion_static_cache=diffusion_static_cache,
            diffusion_sampler=diffusion_sampler,
            diffusion_threshold=diffusion_threshold,
            diffusion_compile=diffusion_compile,
            diffusion_show_unmasking=diffusion_show_unmasking,
            diffusion_unmasking_interval=diffusion_unmasking_interval,
            diffusion_unmasking_width=diffusion_unmasking_width,
        )
        mx.clear_cache()
