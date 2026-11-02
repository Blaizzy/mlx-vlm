# Copyright 2026 KRAFTON Inc.
# Adapted from the Raon-1 Mini Transformers reference implementation.
# SPDX-License-Identifier: Apache-2.0

import math

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..gated_delta import gated_delta_kernel, gated_delta_ops
from ..mla import MultiLinear, latent_length, max_absorbed_queries
from ..switch_layers import SwitchGLU


class SiTU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_beta = config.activation_situ_beta
        self.value_beta = config.activation_situ_linear_beta

    def __call__(self, value, gate):
        dtype = value.dtype
        gate, value = gate.astype(mx.float32), value.astype(mx.float32)
        gate = self.gate_beta * mx.tanh(gate / self.gate_beta) * mx.sigmoid(gate)
        value = self.value_beta * mx.tanh(value / self.value_beta)
        return (gate * value).astype(dtype)


class StreamMix(nn.Module):
    def __init__(self, streams, hidden_size):
        super().__init__()
        size = (streams - 1) ** 2
        self.phi_z = mx.zeros((size, streams * hidden_size))
        self.alpha_z = mx.array(1.0)
        self.b_z = mx.zeros((size,))
        self.eta = mx.array(0.0)


class MHC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.streams = n = config.raon_mhc_streams
        self.eps = config.raon_mhc_selector_eps
        self.post_factor = config.raon_mhc_post_factor
        self.phi_pre = mx.zeros((n, n * config.hidden_size))
        self.phi_post = mx.zeros((n, n * config.hidden_size))
        self.alpha_pre = mx.array(1.0)
        self.alpha_post = mx.array(1.0)
        self.b_pre = mx.zeros((n,))
        self.b_post = mx.zeros((n,))
        self.res = StreamMix(n, config.hidden_size)
        # Non-parameter constants for the doubly stochastic residual chart.
        self._basis = mx.array(
            [
                [
                    (
                        1.0
                        if row < col + 1
                        else -float(col + 1) if row == col + 1 else 0.0
                    )
                    / math.sqrt((col + 1) * (col + 2))
                    for col in range(n - 1)
                ]
                for row in range(n)
            ]
        )
        self._identity = mx.eye(n)
        self._chart = (self._identity + mx.ones((n, n)) / n) * 0.5

    def __call__(self, streams):
        n = self.streams
        flat = streams.reshape(-1, n * streams.shape[-1])
        selectors = mx.concatenate([self.phi_pre, self.phi_post, self.res.phi_z])
        # Training applies the RMS factor after the model-dtype projection.
        inv = mx.rsqrt(
            mx.mean(mx.square(flat.astype(mx.float32)), axis=-1, keepdims=True)
            + self.eps
        )
        projected = (flat @ selectors.astype(flat.dtype).T).astype(mx.float32) * inv
        pre = mx.sigmoid(
            self.alpha_pre.astype(mx.float32) * projected[:, :n]
            + self.b_pre.astype(mx.float32)
        )
        post = self.post_factor * mx.sigmoid(
            self.alpha_post.astype(mx.float32) * projected[:, n : 2 * n]
            + self.b_post.astype(mx.float32)
        )
        z = self.res.alpha_z.astype(mx.float32) * projected[
            :, 2 * n :
        ] + self.res.b_z.astype(mx.float32)
        error = self._basis @ z.reshape(-1, n - 1, n - 1) @ self._basis.T
        radius = mx.maximum(
            mx.max(-error / self._chart, axis=(-2, -1), keepdims=True), 0
        )
        chart = self._chart + error / (1 + radius)
        gamma = mx.sigmoid(self.res.eta.astype(mx.float32))
        return pre, post, (1 - gamma) * self._identity + gamma * chart

    @staticmethod
    def aggregate(streams, pre):
        return mx.sum(streams.astype(mx.float32) * pre[..., None], axis=-2).astype(
            streams.dtype
        )

    @staticmethod
    def mix(streams, branch, post, residual):
        return (
            residual @ streams.astype(mx.float32)
            + post[..., None] * branch.astype(mx.float32)[:, None]
        ).astype(streams.dtype)


class RDA(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config.rda_config
        self.num_heads = cfg.get("num_heads", config.num_attention_heads)
        self.head_dim = cfg.get("head_dim", config.head_dim)
        self.width = self.num_heads * self.head_dim
        self.kernel_size = cfg.get("conv_kernel_size", 4)
        self.lower_bound = cfg.get("gate_lower_bound", -5.0)
        self.eps = config.rms_norm_eps
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, self.width, bias=False))
        self.qkv_conv1d = nn.Conv1d(
            3 * self.width,
            3 * self.width,
            self.kernel_size,
            groups=3 * self.width,
            bias=False,
        )
        self.f_a_proj = nn.Linear(config.hidden_size, config.rda_decay_rank, bias=False)
        self.f_b_proj = nn.Linear(config.rda_decay_rank, self.width, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rda_w_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.A_log = mx.zeros((self.num_heads,), mx.float32)
        self.dt_bias = mx.zeros((self.width,), mx.float32)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self.o_proj = nn.Linear(self.width, config.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        qkv = mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        history = None if cache is None else cache[0]
        if history is None:
            history = mx.zeros((batch, self.kernel_size - 1, 3 * self.width), x.dtype)
        signal = mx.concatenate([history, qkv], axis=1)
        if cache is not None:
            cache.update_window(0, signal, self.kernel_size - 1, lengths=cache.lengths)
        # Keep the raw projection window; the write gate acts after convolution.
        conv = mx.conv1d(
            signal.astype(mx.float32),
            self.qkv_conv1d.weight.astype(mx.float32),
            groups=3 * self.width,
        )
        q, k, v = mx.split(nn.silu(conv).astype(x.dtype), 3, axis=-1)
        shape = (batch, length, self.num_heads, self.head_dim)
        q, k, v = (t.reshape(shape) for t in (q, k, v))

        def normalize(t):
            fp = t.astype(mx.float32)
            return (
                fp / mx.maximum(mx.sqrt(mx.sum(fp * fp, axis=-1, keepdims=True)), 1e-6)
            ).astype(x.dtype)

        q, k = normalize(q) * self.head_dim**-0.5, normalize(k)
        beta_raw = self.b_proj(x).astype(mx.float32)
        delta = self.rda_w_proj(x).astype(mx.float32)
        ratio = mx.exp(nn.softplus(-beta_raw) - nn.softplus(-beta_raw - delta))
        v = (v.astype(mx.float32) * ratio[..., None]).astype(x.dtype)
        decay = self.f_b_proj(self.f_a_proj(x)).astype(mx.float32).reshape(shape)
        decay = (
            decay
            + self.dt_bias.astype(mx.float32).reshape(self.num_heads, self.head_dim)
        ) * mx.exp(self.A_log.astype(mx.float32)).reshape(self.num_heads, 1)
        log_decay = (
            -nn.softplus(decay)
            if self.lower_bound is None
            else self.lower_bound * mx.sigmoid(decay)
        )
        g, beta = mx.exp(log_decay), mx.sigmoid(beta_raw).astype(x.dtype)

        def recurrence(state, steps):
            if state is None:
                state = mx.zeros(
                    (batch, self.num_heads, self.head_dim, self.head_dim), mx.float32
                )
            kernel = (
                gated_delta_kernel
                if (
                    not self.training
                    and mx.default_device() == mx.gpu
                    and mx.metal.is_available()
                    and self.head_dim % 32 == 0
                )
                else gated_delta_ops
            )
            return kernel(q, k, v, g, beta, state, mask, steps)

        if cache is None:
            out, _ = recurrence(None, None)
        else:
            out, _ = cache.update_recurrent(1, length, recurrence)
            cache.advance(length)
        normed = mx.fast.rms_norm(
            out.astype(mx.float32), self.o_norm.weight.astype(mx.float32), self.eps
        )
        gated = normed * mx.sigmoid(self.g_proj(x).astype(mx.float32).reshape(shape))
        return self.o_proj(gated.astype(x.dtype).reshape(batch, length, self.width))


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.nope = config.qk_nope_head_dim
        self.extra = config.qk_rope_head_dim
        self.value_dim = config.v_head_dim
        self.rank = config.kv_lora_rank
        self.scale = (self.nope + self.extra) ** -0.5
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(
            config.q_lora_rank, eps=config.mla_latent_norm_eps
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.heads * (self.nope + self.extra), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.rank + self.extra, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.rank, eps=config.mla_latent_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.rank, self.heads)
        self.unembed_out = MultiLinear(self.rank, self.value_dim, self.heads)
        self.o_proj = nn.Linear(
            self.heads * self.value_dim, config.hidden_size, bias=False
        )
        if config.mla_use_output_gate:
            self.g_proj = nn.Linear(
                config.hidden_size, self.heads * self.value_dim, bias=False
            )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(batch, length, self.heads, self.nope + self.extra).transpose(
            0, 2, 1, 3
        )
        q, q_extra = mx.split(q, [self.nope], axis=-1)
        latent, extra = mx.split(self.kv_a_proj_with_mqa(x), [self.rank], axis=-1)
        latent = self.kv_a_layernorm(latent)[:, None]
        extra = extra[:, None]
        if cache is not None:
            latent, extra = cache.update_and_fetch(latent, extra)
        # NoPE retains the extra key channels without applying rotary embeddings.
        scores = (q_extra * self.scale) @ extra.swapaxes(-1, -2)
        if mask is not None:
            scores = mx.where(mask, scores, mx.array(-float("inf"), scores.dtype))
        absorbed = length <= max_absorbed_queries(
            self.rank, self.nope, self.value_dim, latent_length(latent)
        )
        if absorbed:
            q, key, value = self.embed_q(q), latent, latent
        else:
            key, value = self.embed_q(latent, transpose=False), self.unembed_out(latent)
        out = scaled_dot_product_attention(
            q, key, value, cache=cache, scale=self.scale, mask=scores
        )
        if absorbed:
            out = self.unembed_out(out)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if hasattr(self, "g_proj"):
            out = out * mx.sigmoid(self.g_proj(x))
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.activation = SiTU(config)

    def __call__(self, x):
        return self.down_proj(self.activation(self.up_proj(x), self.gate_proj(x)))


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.zeros((config.num_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.num_experts,))
        self.top_k = config.num_experts_per_token
        self.renormalize = config.moe_renormalize
        self.scale = config.routed_scaling_factor

    def __call__(self, x):
        scores = mx.sigmoid(x.astype(mx.float32) @ self.weight.astype(mx.float32).T)
        indices = mx.argpartition(
            -(scores + self.e_score_correction_bias.astype(mx.float32)),
            kth=self.top_k - 1,
            axis=-1,
        )[..., : self.top_k]
        indices = mx.stop_gradient(indices)
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.renormalize:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return indices, weights * self.scale


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Router(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            activation=SiTU(config),
        )
        self.shared_experts = (
            MLP(config, config.moe_intermediate_size * config.num_shared_experts)
            if config.num_shared_experts
            else None
        )

    def __call__(self, x):
        indices, weights = self.gate(x)
        out = (self.switch_mlp(x, indices).astype(mx.float32) * weights[..., None]).sum(
            axis=-2
        )
        if self.shared_experts is not None:
            out = out + self.shared_experts(x).astype(mx.float32)
        return out.astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.is_linear = index + 1 in config.rda_config["rda_layers"]
        self.is_moe = (
            bool(config.num_experts)
            and index >= config.first_k_dense_replace
            and index % config.moe_layer_freq == 0
        )
        self.self_attn = RDA(config) if self.is_linear else MLA(config)
        self.self_attn_raon_mhc = MHC(config)
        self.mlp_raon_mhc = MHC(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.is_moe:
            self.moe = MoE(config)
        else:
            self.mlp = MLP(config, config.intermediate_size)

    def __call__(self, streams, mask=None, cache=None):
        batch, length, n, width = streams.shape
        rows = streams.reshape(-1, n, width)
        pre, post, residual = self.self_attn_raon_mhc(streams)
        x = self.input_layernorm(MHC.aggregate(rows, pre)).reshape(batch, length, width)
        rows = MHC.mix(
            rows, self.self_attn(x, mask, cache).reshape(-1, width), post, residual
        )
        pre, post, residual = self.mlp_raon_mhc(rows.reshape(streams.shape))
        x = self.post_attention_layernorm(MHC.aggregate(rows, pre))
        branch = self.moe(x) if self.is_moe else self.mlp(x)
        return MHC.mix(rows, branch, post, residual).reshape(streams.shape)


class RaonModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.streams = config.raon_mhc_streams

    def __call__(self, inputs, inputs_embeds=None, cache=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        streams = mx.broadcast_to(
            h[:, :, None], (*h.shape[:2], self.streams, h.shape[-1])
        )
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, state in zip(self.layers, cache):
            mask = (
                create_ssm_mask(h, state)
                if layer.is_linear
                else create_attention_mask(h, state, return_array=True)
            )
            streams = layer(streams, mask=mask, cache=state)
        return self.norm(mx.mean(streams.astype(mx.float32), axis=2).astype(h.dtype))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = RaonModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs=None, inputs_embeds=None, cache=None, **kwargs):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = kwargs.get("input_embeddings")
        hidden = self.model(inputs, inputs_embeds, cache)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(2) if layer.is_linear else KVCache() for layer in self.layers
        ]
