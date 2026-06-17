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

from typing import Optional

import mlx.core as mx

from .diffusion import (
    DiffusionCanvasDenoiseContext,
    DiffusionCanvasDraft,
    DiffusionCanvasResult,
    _diffusion_config_dict,
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

    from ..models.diffusion_gemma.diffusion_turbo_runner import TurboCanvasRunner

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
        h_f = runner(tokens_f, prefix_offset + rel, sc_f)

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
            sel = mx.random.categorical(mx.log(probs + 1e-20) / max(temperature, 1e-5))
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
            accept,
            proposal,
            mx.take_along_axis(committed_dev, act_pos[None, :], axis=-1),
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
            mx.broadcast_to(act_pos[None, :, None], (1, ab, hidden_size)),
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

        if (
            show_unmasking
            and tokenizer is not None
            and (step % max(1, unmask_interval) == 0 or flush)
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
        h_f = runner(tokens_f, prefix_offset + rel, sc_full)
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


DEFAULT_TURBO_REPEAT_GUARD = 16


class TurboDiffusionDenoiser:
    """High-throughput, opt-in canvas denoiser for DiffusionGemma.

    Plugs into ``stream_diffusion_generate`` via its ``diffusion_canvas_denoiser``
    hook, so it reuses the shared prefill / KV-cache-commit / token-emission /
    stopping / repeat-guard scaffolding and only owns the per-canvas denoise:
    an exact top-K sampler chain, gathered self-conditioning, the compacted
    active-set runner (per-layer K/V buffers + shape bucketing), a two-stage
    confidence schedule, a repair pass, and EOS tail-drop.

    All dials are constructor args (the dispatch layer forwards them from
    ``--gen-kwargs``). ``__call__(ctx)`` implements the denoiser protocol:
    it yields ``DiffusionCanvasDraft`` frames (live unmasking) then exactly one
    ``DiffusionCanvasResult`` with the committed canvas. Turbo always uses the
    compacted runner (``compact`` / ``monotone`` are accepted for API
    compatibility and are effectively always-on).
    """

    def __init__(
        self,
        *,
        topk: int = DEFAULT_TURBO_TOPK,
        monotone: bool = True,
        eos_early_stop: bool = True,
        steps: Optional[int] = None,
        accept: str = "entropy-bound",
        threshold=0.9,
        repair: bool = False,
        compact: bool = True,
        entropy_bound: Optional[float] = None,
    ):
        self.topk = topk
        self.monotone = monotone
        self.eos_early_stop = eos_early_stop
        self.steps = steps
        self.accept = accept
        self.threshold = threshold
        self.repair = repair
        self.compact = compact
        self.entropy_bound = entropy_bound

    def __call__(self, ctx: DiffusionCanvasDenoiseContext):
        model = ctx.model
        gen_cfg = _diffusion_config_dict(
            getattr(model.config, "generation_config", None)
        )
        eos = gen_cfg.get("eos_token_id")
        eos_ids = set(
            eos
            if isinstance(eos, (list, tuple))
            else ([eos] if eos is not None else [])
        )
        sampler_cfg = _diffusion_config_dict(gen_cfg.get("sampler_config", None))
        entropy_bound = (
            self.entropy_bound
            if self.entropy_bound is not None
            else float(sampler_cfg.get("entropy_bound", ctx.entropy_bound))
        )
        t_min = float(ctx.temperature_config.get("t_min", 0.4))
        t_max = float(ctx.temperature_config.get("t_max", 0.8))
        softcap = float(
            getattr(model.language_model, "final_logit_softcapping", None)
            or getattr(model.config.text_config, "final_logit_softcapping", 30.0)
        )
        total_steps = self.steps or ctx.max_denoising_steps

        committed = None
        steps = ctx.max_denoising_steps
        work_tokens = ctx.canvas_length
        for event in _run_canvas_compact(
            model,
            ctx.kv_cache,
            canvas_length=ctx.canvas_length,
            vocab_size=ctx.vocab_size,
            eos_ids=eos_ids,
            max_denoising_steps=ctx.max_denoising_steps,
            turbo_topk=self.topk,
            turbo_threshold=self.threshold,
            turbo_steps=self.steps,
            turbo_repair=self.repair,
            turbo_eos_early_stop=self.eos_early_stop,
            entropy_bound=entropy_bound,
            turbo_accept=self.accept,
            softcap=softcap,
            t_min=t_min,
            t_max=t_max,
            temperature=ctx.temperature,
            show_unmasking=ctx.show_unmasking,
            unmask_interval=ctx.unmasking_interval,
            unmask_width=ctx.unmasking_width,
            tokenizer=ctx.tokenizer,
            skip_special_token_ids=ctx.skip_special_token_ids,
        ):
            if event[0] == "draft":
                yield DiffusionCanvasDraft(event[1], event[2] + 1, total_steps)
            else:
                _, committed, _emit_length, steps, work_tokens = event

        if committed is None:
            raise RuntimeError("Turbo denoiser produced no canvas.")
        yield DiffusionCanvasResult(
            canvas=mx.array([committed], dtype=ctx.input_dtype),
            canvas_tokens=ctx.canvas_length,
            denoising_steps=steps,
            work_tokens=work_tokens,
        )
