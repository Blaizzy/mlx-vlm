"""Active-set-compacted decoder runner for the turbo diffusion engine.

The stock decoder forwards all 256 canvas positions every denoising step. With
monotone commits most positions freeze early, and below ~48 live tokens the
MoE drops onto MLX's fast qmv dispatch path, so forwarding only the live
positions roughly halves canvas latency.

Design:
  * Per-layer canvas K/V buffers [1, kv_heads, canvas, head_dim]. Every step
    the forward set F (live positions + positions committed on the previous
    step, so their K/V reflect the committed token) is recomputed and its K/V
    scattered into the buffers. Frozen positions keep the K/V from the step
    after they froze (Fast-dLLM-style approximation; the commit encoder pass
    restores exactness for subsequent blocks).
  * Attention for F queries runs over [prefix cache state, full canvas
    buffer] with no mask: every buffer slot is either frozen (stale-but-valid)
    or freshly updated this step, and the 1024 sliding window exceeds
    prefix-window + canvas for all reachable shapes.
  * RoPE for scattered positions is applied from precomputed cos/sin tables
    that replicate ``mx.fast.rope`` (validated numerically against the model's
    own rope modules; see ``rope_tables_match``).

Invariant used in tests: with F = all positions and empty freeze state, one
runner step must match the stock decoder forward.
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx

from ..models.diffusion_gemma.language import _cache_offset, _cache_state


def _rope_tables(dims: int, base: float, positions: int):
    """cos/sin tables [positions, dims//2] replicating mx.fast.rope."""
    # mx.fast.rope uses theta_i = pos * base^(-2i/dims)
    inv_freq = base ** (-(mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    pos = mx.arange(positions, dtype=mx.float32)[:, None]
    angles = pos * inv_freq[None, :]
    return mx.cos(angles), mx.sin(angles)


def _rope_tables_from_freqs(freqs: mx.array, positions: int):
    """cos/sin tables for an explicit freqs vector (theta = pos / freqs)."""
    pos = mx.arange(positions, dtype=mx.float32)[:, None]
    angles = pos / freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def _apply_rope_gathered(x: mx.array, cos: mx.array, sin: mx.array):
    """Non-traditional rope on x [B, H, T, D] with per-T tables [T, D/2]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half].astype(mx.float32)
    x2 = x[..., half:].astype(mx.float32)
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return mx.concatenate([out1, out2], axis=-1).astype(x.dtype)


class TurboCanvasRunner:
    """Drives the decoder layers over a compacted forward set."""

    def __init__(self, model, kv_cache, canvas_length: int, max_position: int):
        self.decoder = model.model.decoder
        self.config = self.decoder.config
        self.layers = self.decoder.layers
        self.canvas_length = canvas_length
        self.kv_cache = kv_cache

        cfg = self.config
        # rope tables per layer type
        sliding_params = cfg.rope_parameters.get("sliding_attention", {})
        full_params = cfg.rope_parameters.get("full_attention", {})
        self.cos_s, self.sin_s = _rope_tables(
            cfg.head_dim, float(sliding_params.get("rope_theta", 10000.0)), max_position
        )
        # full layers: ProportionalRoPE — rotated_dims of the global head dim
        gdim = cfg.global_head_dim or cfg.head_dim
        partial = float(full_params.get("partial_rotary_factor", 1.0))
        rope_angles = int(partial * gdim // 2)
        self.full_rotated = 2 * rope_angles
        factor = float(full_params.get("factor", 1.0))
        base = float(full_params.get("rope_theta", 10000.0))
        exponents = mx.arange(0, self.full_rotated, 2, dtype=mx.float32) / gdim
        freqs = factor * (base**exponents)
        self.cos_f, self.sin_f = _rope_tables_from_freqs(freqs, max_position)
        mx.eval(self.cos_s, self.sin_s, self.cos_f, self.sin_f)

        # prefix state per layer (temporal order), sliding layers trimmed
        self.prefix_k: List[Optional[mx.array]] = []
        self.prefix_v: List[Optional[mx.array]] = []
        window = max(cfg.sliding_window - 1, 0)
        for layer, c in zip(self.layers, kv_cache):
            state = _cache_state(c)
            if state is None:
                self.prefix_k.append(None)
                self.prefix_v.append(None)
                continue
            k, v = state
            offset = _cache_offset(c)
            if layer.layer_type == "sliding_attention":
                enc_len = k.shape[2]
                if window and enc_len > window and offset >= enc_len:
                    k = k[:, :, -window:, :]
                    v = v[:, :, -window:, :]
            self.prefix_k.append(k)
            self.prefix_v.append(v)

        # canvas K/V buffers per layer
        self.buf_k: List[mx.array] = []
        self.buf_v: List[mx.array] = []
        dtype = self.decoder.embed_tokens.scales.dtype if hasattr(
            self.decoder.embed_tokens, "scales"
        ) else mx.bfloat16
        for layer in self.layers:
            attn = layer.self_attn
            self.buf_k.append(
                mx.zeros((1, attn.n_kv_heads, canvas_length, attn.head_dim), dtype=dtype)
            )
            self.buf_v.append(
                mx.zeros((1, attn.n_kv_heads, canvas_length, attn.head_dim), dtype=dtype)
            )

    def forward(self, tokens_f: mx.array, positions_f: mx.array, sc_f: Optional[mx.array]):
        """One decoder pass over the forward set.

        tokens_f: [1, F] int32 canvas tokens at the forward positions
        positions_f: [F] int32 absolute positions (prefix_offset + canvas pos)
        sc_f: optional [1, F, H] self-conditioning embeddings
        Returns hidden states [1, F, H] (post final norm).
        """
        dec = self.decoder
        inputs_embeds = dec.embed_tokens(tokens_f) * dec.embed_scale
        soft = (
            sc_f.astype(inputs_embeds.dtype)
            if sc_f is not None
            else mx.zeros_like(inputs_embeds)
        )
        h = dec.self_conditioning(inputs_embeds, soft)

        B, F, _ = h.shape
        canvas_pos = positions_f  # absolute positions
        cos_s = self.cos_s[canvas_pos]
        sin_s = self.sin_s[canvas_pos]
        cos_f = self.cos_f[canvas_pos]
        sin_f = self.sin_f[canvas_pos]

        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = h
            hn = layer.input_layernorm(h)

            q = attn.q_proj(hn).reshape(B, F, attn.n_heads, attn.head_dim)
            q = attn.q_norm(q).transpose(0, 2, 1, 3)
            k = attn.k_proj(hn).reshape(B, F, attn.n_kv_heads, attn.head_dim)
            v_raw = (
                attn.v_proj(hn).reshape(B, F, attn.n_kv_heads, attn.head_dim)
                if attn.v_proj is not None
                else k
            )
            k = attn.k_norm(k).transpose(0, 2, 1, 3)
            v = attn.v_norm(v_raw).transpose(0, 2, 1, 3)

            if attn.is_sliding:
                q = _apply_rope_gathered(q, cos_s, sin_s)
                k = _apply_rope_gathered(k, cos_s, sin_s)
            else:
                rd = self.full_rotated
                half = attn.head_dim // 2
                rh = rd // 2

                def prope(x):
                    left = x[..., :half]
                    right = x[..., half:]
                    rot = mx.concatenate([left[..., :rh], right[..., :rh]], axis=-1)
                    rot = _apply_rope_gathered(rot, cos_f, sin_f)
                    left = mx.concatenate([rot[..., :rh], left[..., rh:]], axis=-1)
                    right = mx.concatenate([rot[..., rh:], right[..., rh:]], axis=-1)
                    return mx.concatenate([left, right], axis=-1)

                q = prope(q)
                k = prope(k)

            # scatter fresh K/V into the canvas buffers at their positions
            idx = self._scatter_idx
            self.buf_k[li] = mx.put_along_axis(
                self.buf_k[li],
                idx[li][0],
                k.astype(self.buf_k[li].dtype),
                axis=2,
            )
            self.buf_v[li] = mx.put_along_axis(
                self.buf_v[li],
                idx[li][0],
                v.astype(self.buf_v[li].dtype),
                axis=2,
            )

            keys = self.buf_k[li]
            values = self.buf_v[li]
            if self.prefix_k[li] is not None:
                keys = mx.concatenate([self.prefix_k[li], keys], axis=2)
                values = mx.concatenate([self.prefix_v[li], values], axis=2)

            out = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=1.0, mask=None
            )
            out = out.transpose(0, 2, 1, 3).reshape(B, F, -1)
            hn = attn.o_proj(out)

            hn = layer.post_attention_layernorm(hn)
            h = residual + hn

            residual = h
            h1 = layer.pre_feedforward_layernorm(h)
            h1 = layer.mlp(h1)
            h1 = layer.post_feedforward_layernorm_1(h1)

            flat = residual.reshape(-1, residual.shape[-1])
            top_k_indices, top_k_weights = layer.router(flat)
            h2 = layer.pre_feedforward_layernorm_2(flat)
            h2 = layer.experts(h2, top_k_indices, top_k_weights)
            h2 = h2.reshape(residual.shape)
            h2 = layer.post_feedforward_layernorm_2(h2)

            h = layer.post_feedforward_layernorm(h1 + h2)
            h = residual + h
            h = h * layer.layer_scalar

        return dec.norm(h)

    def set_forward_positions(self, canvas_rel_positions: mx.array):
        """Set canvas-relative forward positions before calling forward()."""
        # scatter index per layer: [1, kv_heads, F, head_dim] broadcastable
        self._scatter_idx = []
        for layer in self.layers:
            attn = layer.self_attn
            idx = mx.broadcast_to(
                canvas_rel_positions[None, None, :, None],
                (1, attn.n_kv_heads, canvas_rel_positions.size, attn.head_dim),
            )
            self._scatter_idx.append((idx,))
