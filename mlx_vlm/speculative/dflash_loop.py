"""Vanilla DFlash speculative decoding (single drafted trajectory, no tree).

This is the "pre-tree" loop from Section 3 of the DDTree paper. DDTree tree
construction + tree-attention verification layer on top of this and live in
``tree_verify.py`` / ``ddtree.py``.

Design notes
------------
* State between rounds is (cache offset ``T``, ``target_hidden_buffer`` of
  length ``T``, held bonus token ``b`` at position ``T`` *not yet processed
  through target*). The prefill step seeds this.
* Each round:
    1. Build the drafter's noise block ``[embed(b), embed(mask), …, embed(mask)]``
       (length ``L+1``) and run the drafter to get ``L`` candidate tokens for
       positions ``[T+1, T+L]``.
    2. Verification forward: run the target on ``[b, d₀, …, d_{L-1}]`` (length
       ``L+1``). Output logits ``logit_i`` is the target's prediction at
       position ``T+i+1`` given the prefix processed so far.
    3. Walk: accept ``d_i`` iff ``d_i == argmax(logit_i)``; stop at first
       mismatch with ``k`` accepted. Next bonus = ``argmax(logit_k)`` (or
       ``argmax(logit_L)`` when ``k == L``).
    4. Commit: restore cache snapshot, then replay ``[b, d₀, …, d_{k-1}]``
       through the target *with* ``capture_layer_ids`` to both advance the
       cache and capture the hidden states for the growing
       ``target_hidden_buffer``.

Replay is needed because Qwen3.5's ``Qwen3_5GatedDeltaNet`` recurrent state is
not reversible — snapshot+replay is the only sound way to roll back an
over-shot verification pass for linear-attention layers.
"""

from typing import Any, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models.qwen3_5_dflash import DFlashDraftModel
from .cache_snapshot import restore_caches, snapshot_caches


def _greedy(logits: mx.array) -> mx.array:
    # logits: [..., V]  ->  [..., ]
    return mx.argmax(logits, axis=-1)


def _run_target(
    lm: nn.Module,
    tokens: mx.array,
    cache: List[Any],
    capture_layer_ids: Optional[List[int]] = None,
):
    """Run the target language model on a 2D ``[B, S]`` token chunk, optionally
    capturing intermediate hidden states."""
    out = lm(
        tokens,
        cache=cache,
        capture_layer_ids=capture_layer_ids,
    )
    return out


def _concat_layer_hiddens(hiddens: List[mx.array]) -> mx.array:
    """Stack ``num_target_layers`` hidden states of shape ``[B, S, H]`` along
    the feature dimension → ``[B, S, num_target_layers*H]`` (the layout the
    DFlash drafter expects for ``target_hidden``).
    """
    return mx.concatenate(hiddens, axis=-1)


def dflash_generate(
    target_model: nn.Module,          # outer VLM model (has .language_model)
    drafter: DFlashDraftModel,
    input_ids: mx.array,              # [B, P] prompt token ids
    *,
    max_new_tokens: int = 256,
    block_size: Optional[int] = None,
    target_layer_ids: Optional[List[int]] = None,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pixel_values=None,
    inputs_embeds: Optional[mx.array] = None,
    **lm_kwargs,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(token_id, accepted_in_round)`` pairs.

    ``accepted_in_round`` is ``0`` for the first prefill bonus and for the
    unconditional round-bonus token; for drafted tokens it is the 1-indexed
    position within the accepted prefix. This lets callers measure mean
    acceptance length.
    """
    cfg = drafter.config
    # block_size in the DFlash config is the TOTAL noise block length
    # (1 bonus slot + (block_size - 1) mask slots). So the drafter predicts
    # block_size - 1 new tokens per round — matching the reference
    # z-lab/ddtree dflash.py decode loop.
    block_total = block_size if block_size is not None else cfg.block_size
    L = block_total - 1  # number of drafted (masked) tokens per round
    target_layer_ids = target_layer_ids or cfg.target_layer_ids
    mask_token_id = mask_token_id if mask_token_id is not None else cfg.mask_token_id

    lm = target_model.language_model if hasattr(target_model, "language_model") else target_model
    embed = lm.model.embed_tokens

    # Reset any cached mRoPE state from a previous generation. The
    # Qwen3.5 LanguageModel caches ``_position_ids`` / ``_rope_deltas``
    # on the instance for chunked-prefill fast-path; without this reset
    # a second call with a different prompt length raises a RoPE shape
    # mismatch.
    if hasattr(lm, "_position_ids"):
        lm._position_ids = None
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = None

    # --- Prefill ---------------------------------------------------------
    prompt_cache = lm.make_cache()

    prefill_out = _run_target(
        lm, input_ids, prompt_cache, capture_layer_ids=target_layer_ids
    )
    # Capture committed context features
    target_hidden_buffer = _concat_layer_hiddens(prefill_out.hidden_states)
    mx.eval(target_hidden_buffer)

    # First bonus: target's argmax at last prompt position
    b = int(_greedy(prefill_out.logits[:, -1, :]).item())
    yield b, 0
    emitted = 1
    if eos_token_id is not None and b == eos_token_id:
        return

    # --- Main loop -------------------------------------------------------
    while emitted < max_new_tokens:
        snap = snapshot_caches(prompt_cache)
        T = target_hidden_buffer.shape[1]  # committed context length

        # Build drafter noise: [b, mask*L]  — total length == block_total
        noise_ids = mx.array(
            [[b] + [mask_token_id] * L], dtype=input_ids.dtype
        )  # [1, block_total]
        noise = embed(noise_ids)

        position_ids = mx.arange(T + block_total, dtype=mx.int32)[None, :]
        drafter_hidden = drafter(noise, target_hidden_buffer, position_ids)
        drafter_logits = embed.as_linear(drafter_hidden)  # [1, block_total, V]
        # Drafted tokens occupy slots [1..L] (slot 0 is the known bonus b)
        drafted = _greedy(drafter_logits[:, 1:, :])  # [1, L]
        mx.eval(drafted)

        # Verification input: [b, d0..d_{L-1}], chunk length == block_total.
        # We capture target hidden states here so the drafter sees
        # features from the same chunk size the reference uses (block_total)
        # — critical for accept rate, since the drafter was trained on
        # features collected from block_total-length target forwards.
        verify_tokens = mx.concatenate(
            [mx.array([[b]], dtype=input_ids.dtype), drafted], axis=1
        )  # [1, block_total]
        verify_out = _run_target(
            lm, verify_tokens, prompt_cache, capture_layer_ids=target_layer_ids
        )
        verify_logits = verify_out.logits  # [1, block_total, V]
        target_choices = _greedy(verify_logits)  # [1, block_total]
        mx.eval(target_choices)

        drafted_list = drafted.reshape(-1).tolist()
        target_list = target_choices.reshape(-1).tolist()

        k = 0
        for i in range(L):
            if drafted_list[i] == target_list[i]:
                k += 1
            else:
                break
        next_bonus = target_list[k]

        # --- Update target_hidden_buffer using the VERIFY pass's own
        # captured hidden states, sliced to the accepted prefix
        # [b, d0..d_{k-1}]. Matches reference's
        # extract_context_feature(output.hidden_states)[:, :acc+1, :].
        verify_hidden = _concat_layer_hiddens(verify_out.hidden_states)
        target_hidden_buffer = mx.concatenate(
            [target_hidden_buffer, verify_hidden[:, : k + 1, :]], axis=1
        )

        # --- Cache commit: restore pre-verify snapshot and replay accepted
        # prefix as a single chunk. This is the best correctness/speed
        # trade-off available on Qwen3.5:
        #
        #  * "trim-only" (HF reference style, no-op on linear state) corrupts
        #    output via linear-state pollution from rejected drafted tokens,
        #    because mlx_lm's gated_delta_update does not self-heal the way
        #    HF's fla kernel does.
        #  * Per-token replay is correct but ~1.5× slower than AR since each
        #    token requires a full target forward with kernel-launch overhead.
        #  * Chunked replay trades a small numerical drift in the linear
        #    state (gated_delta_update is non-associative across chunk sizes)
        #    for a ~1.5× throughput win vs per-token. Output diverges from
        #    plain AR by ~1 token per ~50 tokens but remains coherent.
        restore_caches(prompt_cache, snap)
        commit_tokens = mx.array([[b] + drafted_list[:k]], dtype=input_ids.dtype)
        _run_target(lm, commit_tokens, prompt_cache)
        mx.eval(target_hidden_buffer)

        # Emit: b was not yet yielded this round (it was yielded at end of
        # previous round as "next bonus", actually wait — we yielded it there.
        # We only yield drafted and the NEW bonus now.)
        for i in range(k):
            yield drafted_list[i], i + 1
            emitted += 1
            if emitted >= max_new_tokens:
                return
            if eos_token_id is not None and drafted_list[i] == eos_token_id:
                return

        yield next_bonus, 0
        emitted += 1
        if eos_token_id is not None and next_bonus == eos_token_id:
            return
        b = next_bonus
