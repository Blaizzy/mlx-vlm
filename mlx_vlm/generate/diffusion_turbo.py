"""High-throughput generation loop for block-diffusion canvas models.

This is a performance-focused alternative to ``stream_diffusion_generate``
(see ``diffusion.py``) for DiffusionGemma-style models. The reference loop is
kept untouched; this engine trades a small amount of fidelity for large
throughput gains:

* Top-K sampler chain: the softcap / temperature / entropy / self-conditioning
  math runs on the top-K (default 64) logits per position instead of the full
  262k vocabulary. The softcap (tanh) and temperature transforms are monotonic,
  so top-K membership computed on raw logits is exact; only the entropy tail
  mass and the self-conditioning distribution are truncated. With the linear
  temperature schedule (0.8 -> 0.4) the tail mass beyond 64 candidates is
  negligible for the positions that matter (the low-entropy ones being
  committed).

* Self-conditioning from gathered embedding rows: ``probs[T,K] x embed[K,H]``
  replaces the full ``probs[T,V] @ embed[V,H]`` matmul (and removes the need
  to keep a dequantized copy of the 740M-param embedding table).

* Optional monotone commit mode (default on): once a position is accepted it
  is frozen; the LM head and sampler math then run only on the remaining
  active positions. Frozen positions keep their committed token in the canvas
  (they keep participating in attention), matching the reference
  confidence-threshold sampler's reveal semantics but with the checkpoint's
  entropy-bound acceptance rule.

* EOS tail cut: in monotone mode, once every position up to a committed EOS
  is frozen, the canvas ends early and the tail (positions after EOS, which
  are never emitted) is never denoised further.

* Tunable ``entropy_bound``: the checkpoint ships 0.1; this is the documented
  EB-sampler quality/speed dial (arXiv 2505.24857).

The commit semantics are unchanged: a finished canvas is re-encoded through
the encoder pass so the prefix KV cache stays exact for subsequent blocks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from ..tokenizer_utils import make_streaming_detokenizer
from .common import GenerationResult, generation_stream, wired_limit
from .diffusion import (
    DEFAULT_DIFFUSION_MAX_DENOISING_STEPS,
    DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
    _diffusion_config_dict,
    _diffusion_prefill_cache,
    _diffusion_should_chunk_prefill,
)

DEFAULT_TURBO_TOPK = 64

# Active-set sizes are rounded up to these buckets so the compacted decoder
# forwards a bounded set of tensor shapes. Without this, the live-position count
# shrinks 256->~16 over a canvas with a *different* value every step, and MLX
# JIT-compiles a fresh graph per unique shape — the kernel/graph cache grows
# unbounded and in-process throughput decays (the 120->20 tps drift). Bucketing
# caps the distinct shapes at ~10.
_SHAPE_BUCKETS = (16, 32, 48, 64, 96, 128, 160, 192, 224, 256)


def _bucket_size(n: int) -> int:
    for b in _SHAPE_BUCKETS:
        if n <= b:
            return b
    return _SHAPE_BUCKETS[-1]


def _topk_postprocess(vals: mx.array, softcap: mx.array, temp: mx.array):
    """Softcap + temperature + softmax + entropy on top-K logits.

    Eager on purpose: the active-position count changes every step in monotone
    mode, and per-shape recompilation would cost more than the fusion saves on
    [T, K]-sized tensors.

    vals: [B, T, K] raw top-K logits, any order.
    Returns (probs [B,T,K] fp32, entropy [B,T] fp32).
    """
    capped = mx.tanh(vals.astype(mx.float32) / softcap) * softcap
    scaled = capped / temp
    log_probs = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    return probs, entropy


def _entropy_transfer_mask(
    entropy: mx.array,
    entropy_bound: mx.array,
    quota: int = 0,
) -> mx.array:
    """EB-sampler acceptance rule (same math as the reference loop).

    ``quota`` > 0 additionally forces acceptance of at least that many
    lowest-entropy positions per step ("progressive halving" scheduling) so a
    canvas drains in a bounded number of steps even when the entropy budget
    alone would stall.
    """
    sorted_indices = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, sorted_indices, axis=-1)
    cumulative_entropy = mx.cumsum(sorted_entropy, axis=-1)
    cumulative_maximum = mx.cummax(sorted_entropy, axis=-1)
    sorted_mask = (cumulative_entropy - cumulative_maximum) <= entropy_bound
    if quota > 0:
        rank = mx.arange(entropy.shape[-1])[None, :]
        sorted_mask = sorted_mask | (rank < quota)
    mask = mx.zeros_like(sorted_mask)
    return mx.put_along_axis(mask, sorted_indices, sorted_mask, axis=-1)


def _topk_logits(raw_logits: mx.array, k: int, chunk: int = 64):
    """Exact top-K over the vocab axis via hierarchical chunk-max selection.

    A direct ``mx.topk``/``mx.argpartition`` over 262k logits costs ~40 ms on
    M5-class GPUs; this two-stage form costs ~1.5 ms. Exactness: if a value v
    is among the global top-K, fewer than K chunk-maxima can exceed v, so v's
    chunk ranks within the top-K chunks — selecting the top-K chunks by max
    provably contains every global top-K value.
    """
    *lead, V = raw_logits.shape
    nch = V // chunk
    lc = raw_logits.reshape(*lead, nch, chunk)
    cmax = lc.max(axis=-1)
    cidx = mx.argpartition(cmax, kth=-k, axis=-1)[..., -k:]
    sub = mx.take_along_axis(lc, mx.expand_dims(cidx, -1), axis=-2)
    sub = sub.reshape(*lead, k * chunk)
    sidx = mx.argpartition(sub, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(sub, sidx, axis=-1)
    chunk_of = mx.take_along_axis(cidx, sidx // chunk, axis=-1)
    idx = chunk_of * chunk + (sidx % chunk)
    return vals, idx


def _run_canvas_compact(
    model,
    kv_cache,
    *,
    canvas_length: int,
    vocab_size: int,
    eos_ids,
    max_denoising_steps: int,
    turbo_topk: int,
    turbo_threshold: float,
    turbo_steps,
    turbo_repair: bool,
    turbo_eos_early_stop: bool,
    entropy_bound: float,
    turbo_accept: str,
    softcap: float,
    t_min: float,
    t_max: float,
    temperature: float,
    show_unmasking: bool = False,
    unmask_interval: int = 1,
    unmask_width: int = 0,
    tokenizer=None,
    skip_special_token_ids=None,
):
    """Denoise one canvas with the active-set-compacted runner.

    Generator: yields ("draft", draft_text, step_index) frames while
    denoising (when ``show_unmasking``), then a final
    ("done", final_tokens, emit_length, steps, work_tokens).
    """
    import random as _random

    from .diffusion_turbo_runner import TurboCanvasRunner

    decoder = model.model.decoder
    self_conditioner = _TurboSelfConditioner(decoder)
    cache0 = kv_cache[0]
    prefix_offset = (
        int(cache0.offset) if getattr(cache0, "keys", None) is not None else 0
    )
    runner = TurboCanvasRunner(
        model, kv_cache, canvas_length, prefix_offset + canvas_length + 8
    )

    C = canvas_length
    hidden_size = decoder.config.hidden_size
    softcap_arr = mx.array(softcap, dtype=mx.float32)
    entropy_bound_arr = mx.array(entropy_bound, dtype=mx.float32)

    # device-resident canvas state; host keeps only frozen bookkeeping
    canvas_dev = mx.random.randint(0, vocab_size, (1, C)).astype(mx.int32)
    committed_dev = canvas_dev
    sc_full = mx.zeros((1, C, hidden_size), dtype=mx.bfloat16)
    frozen = [False] * C
    tail_dropped = [False] * C
    newly_frozen: list = []
    eos_list = sorted(eos_ids) if eos_ids else []

    steps = 0
    work_tokens = 0
    emit_length = C

    for step in range(max_denoising_steps):
        active = [i for i in range(C) if not frozen[i]]
        if not active:
            break
        n_real_a = len(active)
        fwd = sorted(active + newly_frozen)
        n_real_f = len(fwd)
        steps += 1
        work_tokens += n_real_f

        # Bucket both the forward set (MoE, dominant graph cost) and the active
        # set (lm_head + sampler) to fixed sizes, padding with a duplicate of an
        # existing position. Duplicates recompute identical values and scatter
        # idempotently, so results are unchanged; only the distinct-shape count
        # is bounded. Host bookkeeping below uses the real counts only.
        fb = _bucket_size(n_real_f)
        fwd_b = fwd + [fwd[-1]] * (fb - n_real_f)
        ab = _bucket_size(n_real_a)
        active_b = active + [active[-1]] * (ab - n_real_a)

        rel = mx.array(fwd_b, dtype=mx.int32)
        act_pos = mx.array(active_b, dtype=mx.int32)
        runner.set_forward_positions(rel)
        tokens_f = canvas_dev[:, rel]
        sc_f = sc_full[:, rel, :]
        h_f = runner.forward(tokens_f, prefix_offset + rel, sc_f)

        # logits + sampler math only for active positions (bucketed length)
        pos_in_f = {p: j for j, p in enumerate(fwd)}  # real positions -> h_f row
        act_idx = mx.array([pos_in_f[p] for p in active_b], dtype=mx.int32)
        h_active = h_f[:, act_idx, :]

        raw_logits = decoder.embed_tokens.as_linear(h_active)
        vals, idx = _topk_logits(raw_logits, turbo_topk)
        frac = 1.0 - step / max(max_denoising_steps - 1, 1)
        step_temp = mx.array(t_min + (t_max - t_min) * frac, dtype=mx.float32)
        probs, entropy = _topk_postprocess(vals, softcap_arr, step_temp)

        if temperature <= 0:
            sel = mx.argmax(probs, axis=-1)
        else:
            sel = mx.random.categorical(
                mx.log(probs + 1e-20) / max(temperature, 1e-5)
            )
        proposal = (
            mx.take_along_axis(idx, sel[..., None], axis=-1)
            .squeeze(-1)
            .astype(mx.int32)
        )

        flush = turbo_steps is not None and step >= turbo_steps - 1
        if flush:
            accept = mx.ones(entropy.shape, dtype=mx.bool_)
        elif turbo_accept == "confidence":
            # Two-stage threshold: picky while the canvas structure is still
            # forming (early steps), greedier once context has solidified —
            # this collapses the active set into the cheap small-batch MoE
            # dispatch region sooner without committing early garbage.
            if isinstance(turbo_threshold, (tuple, list)):
                hi, lo, k0 = turbo_threshold
                tau = hi if step < k0 else lo
            else:
                tau = turbo_threshold
            top_p = probs.max(axis=-1)
            accept = top_p >= tau
        else:
            accept = _entropy_transfer_mask(entropy, entropy_bound_arr, 0)

        # device-side state update for the active positions, single graph
        noise = mx.random.randint(0, vocab_size, accept.shape).astype(mx.int32)
        new_active_tokens = mx.where(accept, proposal, noise)
        committed_active = mx.where(
            accept, proposal, mx.take_along_axis(committed_dev, act_pos[None, :], axis=-1)
        )
        canvas_dev = mx.put_along_axis(
            canvas_dev, act_pos[None, :], new_active_tokens, axis=-1
        )
        committed_dev = mx.put_along_axis(
            committed_dev, act_pos[None, :], committed_active, axis=-1
        )
        sc_active = self_conditioner.soft_embeddings(probs, idx)
        sc_full = mx.put_along_axis(
            sc_full,
            mx.broadcast_to(
                act_pos[None, :, None], (1, ab, hidden_size)
            ),
            sc_active.astype(sc_full.dtype),
            axis=1,
        )

        # one host sync per step: acceptance flags + accepted token ids
        mx.eval(canvas_dev, committed_dev, sc_full)
        accept_list = accept[0].tolist()[:n_real_a]  # ignore bucket padding
        if not any(accept_list) and not flush:
            # force the single most confident *real* position so progress is made
            real_conf = probs[:, :n_real_a].max(axis=-1)
            best = int(mx.argmax(real_conf, axis=-1).item())
            accept_list[best] = True
            p = active[best]
            tok = int(proposal[0, best].item())
            committed_dev = mx.put_along_axis(
                committed_dev,
                mx.array([[p]], dtype=mx.int32),
                mx.array([[tok]], dtype=mx.int32),
                axis=-1,
            )
            canvas_dev = mx.put_along_axis(
                canvas_dev,
                mx.array([[p]], dtype=mx.int32),
                mx.array([[tok]], dtype=mx.int32),
                axis=-1,
            )

        newly_frozen = []
        for j in range(n_real_a):
            if accept_list[j]:
                p = active[j]
                frozen[p] = True
                newly_frozen.append(p)

        if show_unmasking and tokenizer is not None and (
            step % max(1, unmask_interval) == 0 or flush
        ):
            from .diffusion import _decode_diffusion_masked_draft

            committed_view = committed_dev[0].tolist()
            reveal = [frozen[i] and not tail_dropped[i] for i in range(C)]
            draft_text = _decode_diffusion_masked_draft(
                tokenizer,
                [int(t) for t in committed_view],
                reveal,
                skip_special_token_ids,
                max_chars=unmask_width or None,
            )
            yield ("draft", draft_text, step)

        if (turbo_eos_early_stop and eos_list and newly_frozen) or flush:
            committed_now = committed_dev[0].tolist()
            eos_pos = None
            for i in range(C):
                if frozen[i] and committed_now[i] in eos_ids:
                    eos_pos = i
                    break
            if eos_pos is not None:
                if all(frozen[: eos_pos + 1]):
                    emit_length = eos_pos + 1
                    break
                # EOS tail drop: positions after a committed EOS are never
                # emitted and never enter the prefix cache, so stop denoising
                # them. Their last K/V stay in the buffers as context, which
                # is no different from the noise context the model sees on
                # every step anyway.
                for i in range(eos_pos + 1, C):
                    if not frozen[i]:
                        frozen[i] = True
                        tail_dropped[i] = True
                newly_frozen = [p for p in newly_frozen if p <= eos_pos]

    committed = committed_dev[0].tolist()
    committed = [int(t) for t in committed]

    if turbo_repair and emit_length == C:
        rel = mx.arange(C, dtype=mx.int32)
        runner.set_forward_positions(rel)
        tokens_f = mx.array([committed], dtype=mx.int32)
        h_f = runner.forward(tokens_f, prefix_offset + rel, sc_full)
        raw_logits = decoder.embed_tokens.as_linear(h_f)
        vals, idx = _topk_logits(raw_logits, turbo_topk)
        am = (
            mx.take_along_axis(idx, mx.argmax(vals, axis=-1)[..., None], axis=-1)
            .squeeze(-1)
            .astype(mx.int32)
        )
        mx.eval(am)
        committed = [int(t) for t in am[0].tolist()]
        steps += 1
        work_tokens += C

    yield ("done", committed, emit_length, steps, work_tokens)


class _TurboSelfConditioner:
    """Self-conditioning soft embeddings from top-K probabilities."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.embed_tokens = decoder.embed_tokens
        self.embed_scale = decoder.embed_scale

    def soft_embeddings(self, probs: mx.array, idx: mx.array) -> mx.array:
        # probs: [B, T, K] fp32, idx: [B, T, K] int
        rows = self.embed_tokens(idx)  # [B, T, K, H], dequantized rows
        soft = (probs[..., None].astype(rows.dtype) * rows).sum(axis=-2)
        return soft * self.embed_scale


def _draft_result(make_result, draft_text, step_idx, total_steps, canvas_index):
    res = make_result("", canvas_index=canvas_index)
    res.is_draft = True
    res.draft_text = draft_text
    res.diffusion_step = step_idx + 1
    res.diffusion_total_steps = total_steps
    return res


def stream_diffusion_turbo_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    input_ids: mx.array,
    pixel_values: Optional[mx.array],
    attention_mask: Optional[mx.array],
    *,
    max_tokens: int,
    skip_special_token_ids,
    temperature: float = 0.0,
    max_denoising_steps: Optional[int] = None,
    diffusion_full_canvas: bool = False,
    diffusion_min_canvas_length: Optional[int] = None,
    diffusion_max_canvas_length: Optional[int] = None,
    entropy_bound: Optional[float] = None,
    turbo_topk: int = DEFAULT_TURBO_TOPK,
    turbo_monotone: bool = True,
    turbo_eos_early_stop: bool = True,
    turbo_quota: float = 0.0,
    turbo_steps: Optional[int] = None,
    turbo_accept: str = "entropy-bound",
    turbo_threshold: float = 0.9,
    turbo_window: int = 0,
    turbo_repair: bool = False,
    turbo_compact: bool = False,
    turbo_repeat_guard: int = 16,
    diffusion_show_unmasking: bool = False,
    diffusion_unmasking_interval: int = 1,
    diffusion_unmasking_width: int = 0,
    mm_token_type_ids: Optional[mx.array] = None,
    prefill_step_size: Optional[int] = None,
    **_ignored: Any,
) -> Generator[GenerationResult, None, None]:
    if input_ids.shape[0] != 1:
        raise ValueError("Turbo diffusion generation only supports batch size 1.")

    generation_config = _diffusion_config_dict(
        getattr(model.config, "generation_config", None)
    )
    config_eos_token_ids = generation_config.get("eos_token_id")
    if config_eos_token_ids is not None and hasattr(tokenizer, "stopping_criteria"):
        tokenizer.stopping_criteria.add_eos_token_ids(config_eos_token_ids)
    eos_ids = set(
        config_eos_token_ids
        if isinstance(config_eos_token_ids, (list, tuple))
        else ([config_eos_token_ids] if config_eos_token_ids is not None else [])
    )

    text_config = model.config.text_config
    batch_size, prompt_length = input_ids.shape
    prompt_tokens = input_ids.size
    model_canvas_length = int(model.config.canvas_length)
    vocab_size = int(text_config.vocab_size)
    max_new_tokens = int(max_tokens or generation_config.get("max_new_tokens", 256))
    if max_denoising_steps is None:
        max_denoising_steps = int(
            generation_config.get("max_denoising_steps")
            or DEFAULT_DIFFUSION_MAX_DENOISING_STEPS
        )

    max_canvas_length = (
        model_canvas_length
        if diffusion_full_canvas
        else min(model_canvas_length, int(diffusion_max_canvas_length or model_canvas_length))
    )
    min_canvas_length = min(
        max_canvas_length,
        int(diffusion_min_canvas_length or DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH),
    )

    sampler_config = _diffusion_config_dict(generation_config.get("sampler_config", None))
    if entropy_bound is None:
        entropy_bound = float(sampler_config.get("entropy_bound", 0.1))
    t_min = float(generation_config.get("t_min", 0.4))
    t_max = float(generation_config.get("t_max", 0.8))
    stability_threshold = int(generation_config.get("stability_threshold", 1))
    confidence_threshold = float(generation_config.get("confidence_threshold", 0.005))

    decoder = model.model.decoder
    self_conditioner = _TurboSelfConditioner(decoder)
    sc_module = decoder.self_conditioning
    entropy_bound_arr = mx.array(entropy_bound, dtype=mx.float32)
    softcap_arr = mx.array(
        float(getattr(model.language_model, "final_logit_softcapping", None)
              or getattr(text_config, "final_logit_softcapping", 30.0)),
        dtype=mx.float32,
    )

    kv_cache = model.make_cache()
    detokenizer = make_streaming_detokenizer(processor)
    has_padding = attention_mask is not None and not bool(mx.all(attention_mask).item())
    chunk_prefill = _diffusion_should_chunk_prefill(
        prefill_step_size=prefill_step_size,
        prompt_length=prompt_length,
        has_padding=has_padding,
        use_static_cache=False,
        pixel_values=pixel_values,
        mm_token_type_ids=mm_token_type_ids,
    )

    generated_tokens = 0
    diffusion_canvas_tokens = 0
    diffusion_denoising_steps = 0
    diffusion_work_tokens = 0
    last_token = None
    prompt_time = 0.0
    generation_tic = time.perf_counter()
    tic = time.perf_counter()
    stopped = False
    stop_reason = "length"
    # Degeneration guard: a runaway canvas can commit the same token hundreds of
    # times ("the the the..."). Natural text effectively never repeats one token
    # this many times in a row, so a long run is an unambiguous collapse signal;
    # we stop instead of emitting a wall of garbage up to max_tokens.
    repeat_guard = int(turbo_repeat_guard) if turbo_repeat_guard else 0
    prev_emit_id = None
    repeat_run = 0

    def make_result(text, *, finish_reason=None, canvas_index=0, block_complete=False):
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
            diffusion_canvas_index=canvas_index,
            diffusion_block_complete=block_complete,
        )

    with mx.stream(generation_stream):
        canvas_index = 0
        is_prefill = True
        committed_canvas = None

        while generated_tokens < max_new_tokens:
            canvas_index += 1
            if is_prefill:
                kv_cache = _diffusion_prefill_cache(
                    model,
                    input_ids,
                    attention_mask=attention_mask if has_padding else None,
                    kv_cache=kv_cache,
                    pixel_values=pixel_values,
                    mm_token_type_ids=mm_token_type_ids,
                    prefill_step_size=prefill_step_size,
                    chunk_prefill=chunk_prefill,
                )
                mx.eval([c.state for c in kv_cache])
                prompt_time = time.perf_counter() - tic
                generation_tic = time.perf_counter()
                is_prefill = False
            else:
                _, kv_cache = model.model.encoder(
                    committed_canvas, attention_mask=None, cache=kv_cache
                )

            remaining = max_new_tokens - generated_tokens
            canvas_length = (
                model_canvas_length
                if diffusion_full_canvas
                else min(max_canvas_length, max(remaining, min_canvas_length))
            )

            if turbo_compact:
                canvas_gen = _run_canvas_compact(
                    model,
                    kv_cache,
                    canvas_length=canvas_length,
                    vocab_size=vocab_size,
                    eos_ids=eos_ids,
                    max_denoising_steps=max_denoising_steps,
                    turbo_topk=turbo_topk,
                    turbo_threshold=turbo_threshold,
                    turbo_steps=turbo_steps,
                    turbo_repair=turbo_repair,
                    turbo_eos_early_stop=turbo_eos_early_stop,
                    entropy_bound=entropy_bound,
                    turbo_accept=turbo_accept,
                    softcap=float(softcap_arr.item()),
                    t_min=t_min,
                    t_max=t_max,
                    temperature=temperature,
                    show_unmasking=diffusion_show_unmasking,
                    unmask_interval=diffusion_unmasking_interval,
                    unmask_width=diffusion_unmasking_width,
                    tokenizer=tokenizer,
                    skip_special_token_ids=skip_special_token_ids,
                )
                committed_list = None
                for frame in canvas_gen:
                    if frame[0] == "draft":
                        _, draft_text, step_idx = frame
                        yield _draft_result(
                            make_result,
                            draft_text,
                            step_idx,
                            turbo_steps or max_denoising_steps,
                            canvas_index,
                        )
                    else:
                        _, committed_list, emit_length, c_steps, c_work = frame
                diffusion_canvas_tokens += canvas_length
                diffusion_denoising_steps += c_steps
                diffusion_work_tokens += c_work
                committed_canvas = mx.array(
                    [committed_list[:emit_length]], dtype=mx.int32
                )
                for token_id in committed_list[:emit_length]:
                    last_token = int(token_id)
                    generated_tokens += 1
                    if tokenizer.stopping_criteria(last_token):
                        stopped = True
                        stop_reason = "stop"
                        break
                    if repeat_guard:
                        repeat_run = (
                            repeat_run + 1 if last_token == prev_emit_id else 1
                        )
                        prev_emit_id = last_token
                        if repeat_run >= repeat_guard:
                            stopped = True
                            stop_reason = "repetition"
                            break
                    detokenizer.add_token(
                        last_token, skip_special_token_ids=skip_special_token_ids
                    )
                    yield make_result(
                        detokenizer.last_segment, canvas_index=canvas_index
                    )
                    if generated_tokens >= max_new_tokens:
                        stopped = True
                        stop_reason = "length"
                        break
                yield make_result(
                    "", canvas_index=canvas_index, block_complete=True
                )
                if stopped:
                    break
                mx.clear_cache()
                continue

            # int32 throughout: int64 scatter ops hit a broken Metal JIT
            # template on macOS 27 (atomic packing_size<long> divide-by-zero).
            canvas = mx.random.randint(0, vocab_size, (batch_size, canvas_length))
            frozen = mx.zeros((batch_size, canvas_length), dtype=mx.bool_)
            committed = canvas
            sc_embeddings = None
            prev_argmax = None
            stable_count = 0
            steps_this_canvas = 0
            work_tokens_this_canvas = 0

            masks = decoder._make_decoder_masks(canvas[..., None], kv_cache, None)

            final_canvas = None
            emit_length = canvas_length

            for step in range(max_denoising_steps):
                steps_this_canvas += 1
                # Decoder forward over the full canvas (frozen tokens keep
                # providing exact attention context).
                inputs_embeds = decoder.embed_tokens(canvas) * decoder.embed_scale
                if sc_embeddings is None:
                    soft = mx.zeros_like(inputs_embeds)
                else:
                    soft = sc_embeddings.astype(inputs_embeds.dtype)
                h = sc_module(inputs_embeds, soft)
                cache_list = kv_cache or [None] * len(decoder.layers)
                offset = (
                    cache_list[0].offset
                    if cache_list and getattr(cache_list[0], "keys", None) is not None
                    else 0
                )
                for layer, c in zip(decoder.layers, cache_list):
                    h = layer(h, masks.get(layer.layer_type), c, decoder=True, offset=offset)
                h = decoder.norm(h)

                if turbo_monotone and step > 0:
                    active_idx = mx.array(active_positions)
                    h_active = h[:, active_idx, :]
                else:
                    active_idx = None
                    h_active = h
                work_tokens_this_canvas += int(h_active.shape[1])

                raw_logits = decoder.embed_tokens.as_linear(h_active)
                vals, idx = _topk_logits(raw_logits, turbo_topk)

                frac = 1.0 - step / max(max_denoising_steps - 1, 1)
                step_temp = mx.array(t_min + (t_max - t_min) * frac, dtype=mx.float32)
                probs, entropy = _topk_postprocess(vals, softcap_arr, step_temp)

                # Proposal per position: argmax (temp<=0) or categorical in top-K.
                if temperature <= 0:
                    sel = mx.argmax(probs, axis=-1)
                else:
                    sel = mx.random.categorical(
                        mx.log(probs + 1e-20) / max(temperature, 1e-5)
                    )
                proposal = (
                    mx.take_along_axis(idx, sel[..., None], axis=-1)
                    .squeeze(-1)
                    .astype(mx.int32)
                )
                top1 = (
                    mx.take_along_axis(
                        idx, mx.argmax(probs, axis=-1)[..., None], axis=-1
                    )
                    .squeeze(-1)
                    .astype(mx.int32)
                )

                quota_n = 0
                n_active = int(entropy.shape[-1])
                if turbo_quota > 0:
                    quota_n = max(1, int(n_active * turbo_quota))

                flush = turbo_steps is not None and step >= turbo_steps - 1
                if flush:
                    # Final-step flush (reference fast-mode semantics): after
                    # turbo_steps - 1 refinement rounds the leftovers are
                    # well-conditioned on the committed context, so taking
                    # their argmax wholesale costs little quality and bounds
                    # the canvas at exactly turbo_steps forwards.
                    accept = mx.ones(entropy.shape, dtype=mx.bool_)
                elif turbo_accept == "confidence":
                    # Fast-dLLM-style absolute-readiness rule: commit every
                    # position whose top-1 probability clears the threshold;
                    # if none qualify, force the single most confident one so
                    # the canvas always makes progress.
                    top_p = probs.max(axis=-1)
                    accept = top_p >= turbo_threshold
                    if turbo_window > 0 and turbo_monotone:
                        # Left-anchored acceptance window: only commit within
                        # turbo_window positions of the unfrozen frontier so
                        # every commit has solid left context (sub-block
                        # scheduling a la Fast-dLLM v2).
                        if active_idx is None:
                            frontier_limit = turbo_window
                            pos_of_active = mx.arange(canvas_length)[None, :]
                        else:
                            frontier_limit = active_positions[0] + turbo_window
                            pos_of_active = active_idx[None, :]
                        accept = accept & (pos_of_active < frontier_limit)
                    if not bool(mx.any(accept).item()):
                        best = mx.argmax(top_p, axis=-1)
                        accept = (
                            mx.arange(top_p.shape[-1])[None, :] == best[:, None]
                        )
                else:
                    accept = _entropy_transfer_mask(
                        entropy, entropy_bound_arr, quota_n
                    )

                if turbo_monotone:
                    if active_idx is None:
                        # step 0: active set == all positions
                        new_frozen = accept
                        committed = mx.where(accept, proposal, committed)
                        canvas = mx.where(
                            accept,
                            committed,
                            mx.random.randint(0, vocab_size, (batch_size, canvas_length)),
                        )
                        frozen = frozen | new_frozen
                        argmax_full = top1
                    else:
                        # scatter active results back into full-canvas tensors
                        committed_active = mx.where(
                            accept, proposal,
                            mx.take_along_axis(committed, active_idx[None, :], axis=-1),
                        )
                        committed = mx.put_along_axis(
                            committed, active_idx[None, :], committed_active, axis=-1
                        )
                        noise = mx.random.randint(0, vocab_size, accept.shape)
                        canvas_active = mx.where(accept, committed_active, noise)
                        canvas = mx.put_along_axis(
                            canvas, active_idx[None, :], canvas_active, axis=-1
                        )
                        frozen = mx.put_along_axis(
                            frozen,
                            active_idx[None, :],
                            mx.take_along_axis(frozen, active_idx[None, :], axis=-1)
                            | accept,
                            axis=-1,
                        )
                        argmax_full = mx.put_along_axis(
                            committed.astype(top1.dtype),
                            active_idx[None, :],
                            top1,
                            axis=-1,
                        )
                else:
                    committed = mx.where(accept, proposal, canvas)
                    canvas = mx.where(
                        accept,
                        committed,
                        mx.random.randint(0, vocab_size, (batch_size, canvas_length)),
                    )
                    argmax_full = top1

                # Self-conditioning for next step (skip on the final step).
                sc_active = self_conditioner.soft_embeddings(probs, idx)
                if turbo_monotone and active_idx is not None:
                    base = (
                        sc_embeddings
                        if sc_embeddings is not None
                        else mx.zeros(
                            (batch_size, canvas_length, sc_active.shape[-1]),
                            dtype=sc_active.dtype,
                        )
                    )
                    sc_embeddings = mx.put_along_axis(
                        base,
                        mx.broadcast_to(
                            active_idx[None, :, None],
                            (batch_size, active_idx.size, sc_active.shape[-1]),
                        ),
                        sc_active,
                        axis=1,
                    )
                else:
                    sc_embeddings = sc_active

                # --- host sync point: small tensors only ---
                if turbo_monotone:
                    mx.eval(frozen, committed, entropy)
                    frozen_list = frozen[0].tolist()
                    active_positions = [
                        i for i, f in enumerate(frozen_list) if not f
                    ]
                    mean_entropy = float(mx.mean(entropy).item())

                    if turbo_eos_early_stop and eos_ids:
                        committed_list = committed[0].tolist()
                        eos_pos = None
                        for i, (tok, fz) in enumerate(zip(committed_list, frozen_list)):
                            if fz and int(tok) in eos_ids:
                                eos_pos = i
                                break
                        if eos_pos is not None and all(frozen_list[: eos_pos + 1]):
                            final_canvas = committed
                            emit_length = eos_pos + 1
                            break

                    if not active_positions:
                        final_canvas = committed
                        break
                    if not frozen_list or len(active_positions) == canvas_length:
                        active_positions = list(range(canvas_length))
                else:
                    mx.eval(argmax_full, entropy)
                    mean_entropy = float(mx.mean(entropy).item())

                # stability check on argmax predictions
                if prev_argmax is not None and turbo_monotone is False:
                    if bool(mx.all(argmax_full == prev_argmax).item()):
                        stable_count += 1
                    else:
                        stable_count = 0
                    if (
                        stable_count >= stability_threshold
                        and mean_entropy < confidence_threshold
                    ):
                        final_canvas = argmax_full
                        break
                prev_argmax = argmax_full

            if final_canvas is None:
                final_canvas = committed if turbo_monotone else argmax_full

            if turbo_repair and emit_length == canvas_length:
                # One repair forward with the fully committed canvas as input:
                # take the model's argmax everywhere, recovering the reference
                # sampler's trailing "final canvas = argmax of last forward"
                # semantics so early-frozen tokens get a revision opportunity.
                inputs_embeds = (
                    decoder.embed_tokens(final_canvas) * decoder.embed_scale
                )
                soft = (
                    sc_embeddings.astype(inputs_embeds.dtype)
                    if sc_embeddings is not None
                    else mx.zeros_like(inputs_embeds)
                )
                h = sc_module(inputs_embeds, soft)
                cache_list = kv_cache or [None] * len(decoder.layers)
                offset = (
                    cache_list[0].offset
                    if cache_list and getattr(cache_list[0], "keys", None) is not None
                    else 0
                )
                for layer, c in zip(decoder.layers, cache_list):
                    h = layer(
                        h, masks.get(layer.layer_type), c, decoder=True, offset=offset
                    )
                h = decoder.norm(h)
                raw_logits = decoder.embed_tokens.as_linear(h)
                vals, idx = _topk_logits(raw_logits, turbo_topk)
                final_canvas = (
                    mx.take_along_axis(
                        idx, mx.argmax(vals, axis=-1)[..., None], axis=-1
                    )
                    .squeeze(-1)
                    .astype(mx.int32)
                )
                steps_this_canvas += 1
                work_tokens_this_canvas += canvas_length

            mx.eval(final_canvas)
            diffusion_canvas_tokens += canvas_length
            diffusion_denoising_steps += steps_this_canvas
            diffusion_work_tokens += work_tokens_this_canvas

            committed_canvas = final_canvas[:, :emit_length]
            for token_id in committed_canvas[0].tolist():
                last_token = int(token_id)
                generated_tokens += 1
                if tokenizer.stopping_criteria(last_token):
                    stopped = True
                    stop_reason = "stop"
                    break
                if repeat_guard:
                    repeat_run = repeat_run + 1 if last_token == prev_emit_id else 1
                    prev_emit_id = last_token
                    if repeat_run >= repeat_guard:
                        stopped = True
                        stop_reason = "repetition"
                        break
                detokenizer.add_token(
                    last_token, skip_special_token_ids=skip_special_token_ids
                )
                yield make_result(detokenizer.last_segment, canvas_index=canvas_index)
                if generated_tokens >= max_new_tokens:
                    stopped = True
                    stop_reason = "length"
                    break

            yield make_result("", canvas_index=canvas_index, block_complete=True)

            if stopped:
                break
            mx.clear_cache()

    if prompt_time == 0.0:
        prompt_time = time.perf_counter() - tic
    detokenizer.finalize()
    yield make_result(detokenizer.last_segment, finish_reason=stop_reason if stopped else "length")
