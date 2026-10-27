# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn

from ...kernels.lfm3_rope import lfm3_rope
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache, RotatingKVCache
from ..lfm2.language import ShortConv

# MLX's fused attention takes window_size from its sliding-window kernels onward
# (after 0.32.4); with an older MLX the sliding layers keep the band mask.
_SDPA_HAS_WINDOW = "window_size" in (mx.fast.scaled_dot_product_attention.__doc__ or "")


class RotaryEmbedding(nn.Module):
    """Rotate matching coordinates in both head halves, leaving NoPE pairs intact."""

    def __init__(self, dim, parameters):
        super().__init__()
        if parameters.get("rope_type", "default") != "default":
            raise ValueError("LFM3 supports default rotary embeddings only")
        self.dim = dim
        self.base = parameters.get("rope_theta", 1e6)
        self.angles = int(dim * parameters.get("partial_rotary_factor", 1.0)) // 2
        if not 0 < self.angles <= dim // 2:
            raise ValueError("Invalid partial_rotary_factor")

    def padded_inv_freq(self):
        # The inverse frequencies __call__ computes, zero-padded over the NoPE pairs.
        if getattr(self, "_padded_inv_freq", None) is None:
            inv_freq = self.base ** (
                -mx.arange(self.angles, dtype=mx.float32) * 2 / self.dim
            )
            self._padded_inv_freq = mx.pad(inv_freq, [(0, self.dim // 2 - self.angles)])
        return self._padded_inv_freq

    def __call__(self, x, offset=0, position_ids=None):
        if isinstance(offset, mx.array) and offset.ndim > 0:
            offset = offset[:, None]
        positions = (
            mx.arange(x.shape[-2]) + offset if position_ids is None else position_ids
        )
        inv_freq = self.base ** (
            -mx.arange(self.angles, dtype=mx.float32) * 2 / self.dim
        )
        inv_freq = mx.pad(inv_freq, [(0, self.dim // 2 - self.angles)])
        angles = positions[..., None].astype(mx.float32) * inv_freq
        angles = mx.concatenate([angles, angles], axis=-1)
        if angles.ndim == 3:
            angles = angles[:, None]
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate([-b, a], axis=-1) * sin


class Attention(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_parameters[layer_type])

    def __call__(self, x, mask=None, cache=None, position_ids=None):
        batch, length, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        offset = cache.offset if cache is not None else 0
        if self._fused_rope_applies(q, offset, position_ids):
            inv_freq = self.rope.padded_inv_freq()
            q = lfm3_rope(q, inv_freq, offset)
            k = lfm3_rope(k, inv_freq, offset)
        else:
            q = self.rope(q.transpose(0, 2, 1, 3), offset, position_ids)
            k = self.rope(k.transpose(0, 2, 1, 3), offset, position_ids)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if self._fused_window_applies(q, mask, cache):
            # The band mask of a sliding layer is causal attention over the last
            # `window` keys, which the fused attention skips block by block.
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal", window_size=self.window
            )
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
        out = out.transpose(0, 2, 1, 3)
        out = out * mx.sigmoid(self.gate_proj(x))[..., None]
        return self.o_proj(out.reshape(batch, length, -1))

    def _fused_window_applies(self, q, mask, cache):
        return (
            _SDPA_HAS_WINDOW
            and not self.training
            and self.window is not None
            and isinstance(mask, mx.array)
            and mask.dtype == mx.bool_
            and q.shape[0] == 1
            # Short chunks would reach the vector kernel, whose windowed form sums
            # in a different order than with the band mask.
            and q.shape[2] > 32
            and (cache is None or (type(cache) is RotatingKVCache and cache.keep == 0))
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    def _fused_rope_applies(self, q, offset, position_ids):
        # Prefill only: on Max-class GPUs the fused rope at decode, together with the
        # fused short-conv step, leaves too few command buffers per step for MLX to
        # hold the host back, and long-context decode then copies the KV cache.
        return (
            not self.training
            and q.shape[1] > 1
            and position_ids is None
            and (not isinstance(offset, mx.array) or offset.size == 1)
            and q.dtype in (mx.float16, mx.bfloat16)
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        self.is_attention_layer = self.layer_type != "conv"
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.is_attention_layer:
            self.self_attn = Attention(config, self.layer_type)
        else:
            self.conv = ShortConv(config, index)
        self.feed_forward = MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(
                config.hidden_size, eps=config.norm_eps
            )
            self.post_ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, mask=None, cache=None, position_ids=None):
        h = self.operator_norm(x)
        h = (
            self.self_attn(h, mask, cache, position_ids)
            if self.is_attention_layer
            else self.conv(h, mask, cache)
        )
        x = x + (self.post_operator_norm(h) if self.use_sandwich_norm else h)
        h = self.feed_forward(self.ffn_norm(x))
        return x + (self.post_ffn_norm(h) if self.use_sandwich_norm else h)


class Lfm3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self, inputs=None, cache=None, input_embeddings=None, position_ids=None
    ):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        cache = [None] * len(self.layers) if cache is None else cache
        for layer, c in zip(self.layers, cache):
            if layer.is_attention_layer:
                window = (
                    self.config.sliding_window
                    if layer.layer_type == "sliding_attention"
                    else None
                )
                mask = create_attention_mask(h, c, window_size=window)
            else:
                mask = create_ssm_mask(h, c)
            h = layer(h, mask, c, position_ids)
        return self.embedding_norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs=None,
        cache=None,
        inputs_embeds=None,
        input_embeddings=None,
        **kwargs,
    ):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        h = self.model(inputs, cache, inputs_embeds, kwargs.get("position_ids"))
        logits = (
            self.model.embed_tokens.as_linear(h)
            if self.config.tie_word_embeddings
            else self.lm_head(h)
        )
        return LanguageModelOutput(logits=logits, hidden_states=h)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        return [
            (
                ArraysCache(size=1)
                if layer.layer_type == "conv"
                else (
                    RotatingKVCache(max_size=self.config.sliding_window)
                    if layer.layer_type == "sliding_attention"
                    else KVCache()
                )
            )
            for layer in self.layers
        ]
