"""DFlash speculative decoding loop.

This is an MLX port of ``stream_generate`` from z-lab/dflash PR #59
(dflash/model_mlx.py). It runs the DFlash block-diffusion drafter against
any target language model that has full-attention and/or
``Qwen3_5GatedDeltaNet`` layers.

Design notes
------------

Per round:

1. Drafter forward on ``[b, mask, mask, …]`` (length ``block_size``) with a
   stateful ``KVCache``; only the ``acc+1`` newly-committed target features
   from the previous round are passed in. First ``(block_size - 1)`` slot
   predictions are sampled to get ``d_0..d_{L-1}``. The transient noise K/V
   entries are then trimmed out of the drafter cache.
2. Verify: one target forward on ``[b, d_0, …, d_{L-1}]`` (length
   ``block_size``). Hidden states at ``drafter.config.target_layer_ids``
   are captured via ``_LayerHook`` monkey-patches on the selected decoder
   layers — no modification to the model class itself.
3. Walk: accept ``d_i`` iff ``d_i == argmax(verify_logits[i])``; ``k`` is
   the number of accepted drafted tokens.
4. Cache commit:
   * full-attention layers — ``trim_prompt_cache(target_cache, block - k - 1)``
   * gated-delta-net layers — replay the first ``k+1`` positions through
     ``gated_delta_update`` directly using the inputs captured during the
     verify pass (see :class:`_GDNStateCapture`). The captured pre-conv
     ``conv_input`` tensor is also sliced to recover the conv state.

The rollback pattern is much cheaper than re-running the full target model
on the accepted prefix — we touch only the linear-attention layers, skipping
all attention, MLP, and embedding work.

``mx.async_eval`` is used after the drafter and verify forwards so the GPU
queue overlaps with the Python-side walk/commit work.
"""

from threading import RLock
from typing import Iterator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_sampler

from ..models.qwen3_5_dflash import DFlashDraftModel

try:
    import mlx_lm.models.gated_delta  # noqa: F401
    _HAS_GDN = True
except ImportError:  # pragma: no cover
    _HAS_GDN = False


_GDN_PATCH_LOCK = RLock()


# ---------------------------------------------------------------------------
# Hidden-state capture via layer monkey-patch
# ---------------------------------------------------------------------------


class _LayerHook:
    """Wraps a decoder layer so its forward output is stashed into a shared
    list at a known index. The hook forwards all other attribute accesses to
    the wrapped layer so e.g. weight loading still works through it.
    """

    def __init__(self, layer, idx: int, storage: list):
        self._layer = layer
        self._idx = idx
        self._storage = storage

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _patch_model(model, layer_ids):
    """Install layer hooks in-place. Idempotent — safe to call repeatedly."""
    if hasattr(model, "_dflash_hidden_states"):
        return
    model._dflash_hidden_states = [None] * len(layer_ids)
    layers = _get_layers(model)
    for i, lid in enumerate(layer_ids):
        layers[lid] = _LayerHook(layers[lid], i, model._dflash_hidden_states)


# ---------------------------------------------------------------------------
# Gated-delta-net rollback support
# ---------------------------------------------------------------------------


class _GDNStateCapture:
    """Monkey-patches ``Qwen3_5GatedDeltaNet.__call__`` to save the inputs
    to each ``gated_delta_update`` call made during a verify forward. After
    the walk we can replay just the first ``accepted+1`` positions through
    ``gated_delta_update`` to recover the correct linear-attention state
    without running another full target forward pass.
    """

    def __init__(self):
        self.conv_data: list = []
        self._gdn_inputs: list = []
        self._gdn_cls = None
        self._orig_call = None
        self._patched_call = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            _GDN_PATCH_LOCK.release()
            raise

    def _patch(self):
        # Qwen3_5GatedDeltaNet lives in mlx_vlm's qwen3_5/language.py
        from ..models.qwen3_5.language import Qwen3_5GatedDeltaNet

        self._gdn_cls = Qwen3_5GatedDeltaNet
        self._orig_call = Qwen3_5GatedDeltaNet.__call__
        capture = self

        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            mixed_qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(
                B, S, -1, self_layer.head_v_dim
            )
            b = self_layer.in_proj_b(inputs)
            a = self_layer.in_proj_a(inputs)

            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
                if conv_state.shape[0] != B:
                    conv_state = mx.zeros(
                        (B, self_layer.conv_kernel_size - 1, self_layer.conv_dim),
                        dtype=inputs.dtype,
                    )
            else:
                conv_state = mx.zeros(
                    (B, self_layer.conv_kernel_size - 1, self_layer.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                if mask.shape[0] != B:
                    mask = None
                else:
                    mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)

            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            # Remember the conv_input so rollback can slice it to recover the
            # correct conv state for the next round.
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                cache[0] = conv_input[:, -(self_layer.conv_kernel_size - 1):]

            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
                    [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
                    [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
                )
            ]
            state = cache[1] if cache else None
            if state is not None and state.shape[0] != B:
                state = None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            # Save the inputs for rollback. ``state`` here is the pre-verify
            # linear state, which is what we'll replay from.
            capture._gdn_inputs.append(
                (q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask)
            )

            from mlx_lm.models.gated_delta import gated_delta_update as _gdu

            out, new_state = _gdu(
                q, k, v, a, b,
                self_layer.A_log, self_layer.dt_bias,
                state, mask,
                use_kernel=not self_layer.training,
            )
            if cache is not None:
                cache[1] = new_state
            out = self_layer.norm(out, z)
            return self_layer.out_proj(out.reshape(B, S, -1))

        self._patched_call = _capturing_gdn_call
        Qwen3_5GatedDeltaNet.__call__ = _capturing_gdn_call

    def clear(self):
        self.conv_data.clear()
        self._gdn_inputs.clear()

    def close(self):
        if self._closed:
            return
        try:
            if (
                self._gdn_cls is not None
                and self._gdn_cls.__call__ is self._patched_call
            ):
                self._gdn_cls.__call__ = self._orig_call
        finally:
            self._closed = True
            self._gdn_cls = None
            self._orig_call = None
            self._patched_call = None
            _GDN_PATCH_LOCK.release()

    def rollback(self, cache: list, accepted: int, trim: int) -> None:
        """Roll back the target cache to keep only the first ``accepted+1``
        positions of the latest verify chunk.

        * full-attention layers: trim via ``cache.trim`` (fast K/V slice)
        * gated-delta-net layers: replay the pre-conv inputs through
          ``gated_delta_update`` to recover linear state, and slice the
          captured ``conv_input`` for the conv state.
        """
        from mlx_lm.models.gated_delta import gated_delta_update as _gdu

        j = 0
        n = accepted + 1
        for c in cache:
            if c is None:
                continue
            if c.is_trimmable():
                c.trim(trim)
            else:
                q, k, v, a, b, A_log, dt_bias, init_state, mask = self._gdn_inputs[j]
                _, state = _gdu(
                    q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
                    A_log, dt_bias, init_state,
                    None if mask is None else mask[:, :n],
                    use_kernel=True,
                )
                c.cache[1] = state
                conv_input, K = self.conv_data[j]
                c.cache[0] = conv_input[:, accepted + 1 : accepted + K]
                j += 1


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------


def _get_language_model(target_model):
    """Return the inner language-model module regardless of VLM wrapping."""
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    return target_model


def _run_target(target_model, tokens: mx.array, cache: list):
    """Call the target model on ``tokens`` with ``cache`` and return the
    logits. Accepts either an mlx_vlm VLM (where the outer model exposes
    ``__call__(input_ids, cache)``) or a plain mlx_lm LM."""
    lm = _get_language_model(target_model)
    out = lm(tokens, cache=cache)
    if hasattr(out, "logits"):
        return out.logits
    return out


def dflash_generate(
    target_model: nn.Module,
    drafter: DFlashDraftModel,
    input_ids: mx.array,
    *,
    max_new_tokens: int = 256,
    block_size: Optional[int] = None,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    sampler=None,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(token_id, accepted_in_round)`` pairs.

    ``accepted_in_round`` is 0 for the prefill bonus and each round's
    bonus token; it is 1-indexed within the accepted drafted prefix
    otherwise. Output is bit-identical to plain autoregressive greedy.
    """
    cfg = drafter.config
    block_total = block_size if block_size is not None else int(cfg.block_size)
    L = block_total - 1  # drafted slots per round
    target_layer_ids = cfg.target_layer_ids

    if sampler is None:
        sampler = make_sampler(temp=temperature)

    lm = _get_language_model(target_model)
    _patch_model(target_model, target_layer_ids)
    drafter.bind(target_model)

    # Reset Qwen3.5 VLM mRoPE state from any previous generation
    if hasattr(lm, "_position_ids"):
        lm._position_ids = None
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = None

    target_cache = make_prompt_cache(lm)
    draft_cache = drafter.make_cache()

    target_trim_ok = can_trim_prompt_cache(target_cache)
    if not target_trim_ok and not _HAS_GDN:
        raise RuntimeError(
            "DFlash rollback requires either a trimmable target cache or "
            "mlx_lm.models.gated_delta availability."
        )

    capture: Optional[_GDNStateCapture] = (
        _GDNStateCapture() if not target_trim_ok else None
    )

    try:
        with mx.stream(generation_stream):
            logits = _run_target(target_model, input_ids, target_cache)
            hidden = mx.concatenate(target_model._dflash_hidden_states, axis=-1)
        mx.eval(logits, hidden)

        # First bonus token
        b_arr = sampler(logits[:, -1:])
        b = int(b_arr[0, 0].item())
        yield b, 0
        emitted = 1
        if eos_token_id is not None and b == eos_token_id:
            return

        prompt_len = int(input_ids.shape[1])

        while emitted < max_new_tokens:
            bs = min(block_total, max_new_tokens - emitted + 1)
            if bs <= 1:
                # Fall back to one AR step
                one = mx.array([[b]], dtype=input_ids.dtype)
                logits = _run_target(target_model, one, target_cache)
                nb = int(sampler(logits[:, -1:])[0, 0].item())
                yield nb, 0
                emitted += 1
                b = nb
                continue

            # --- Drafter forward --------------------------------------
            with mx.stream(generation_stream):
                block = mx.array(
                    [[b] + [int(cfg.mask_token_id)] * (bs - 1)],
                    dtype=input_ids.dtype,
                )
                draft_logits = drafter(block, hidden, draft_cache)
                trim_n = draft_cache[0].offset - (prompt_len + emitted - 1)
                if trim_n > 0:
                    trim_prompt_cache(draft_cache, trim_n)
                draft_tokens = sampler(draft_logits[:, 1 - bs:])

            # --- Verify forward ---------------------------------------
            if capture is not None:
                capture.clear()
            with mx.stream(generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[b]], dtype=input_ids.dtype), draft_tokens],
                    axis=1,
                )
                verify_logits = _run_target(target_model, verify_input, target_cache)
                hidden = mx.concatenate(
                    target_model._dflash_hidden_states, axis=-1
                )
                target_tokens = sampler(verify_logits)
            mx.async_eval(target_tokens, hidden)

            # --- Walk (exact greedy, matches the reference torch and
            # MLX implementations — first mismatch stops acceptance).
            d_list = draft_tokens[0].tolist()
            t_list = target_tokens[0].tolist()
            accepted = next(
                (i for i in range(len(d_list)) if d_list[i] != t_list[i]),
                len(d_list),
            )
            new_tokens = d_list[:accepted] + [t_list[accepted]]
            new_tokens = new_tokens[: max_new_tokens - emitted]

            # --- Emit -------------------------------------------------
            for i, tok in enumerate(new_tokens):
                # Drafted tokens are 1..accepted, the round bonus is 0
                if i < accepted:
                    yield tok, i + 1
                else:
                    yield tok, 0
                emitted += 1
                if eos_token_id is not None and tok == eos_token_id:
                    return
                if emitted >= max_new_tokens:
                    return

            # --- Cache commit -----------------------------------------
            trim = bs - accepted - 1
            if trim > 0:
                if target_trim_ok:
                    trim_prompt_cache(target_cache, trim)
                elif capture is not None:
                    capture.rollback(target_cache, accepted, trim)

            # Slice the captured hidden states to match the accepted prefix
            hidden = hidden[:, : accepted + 1, :]

            b = new_tokens[-1] if new_tokens else b
    finally:
        if capture is not None:
            capture.close()
