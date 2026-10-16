from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..llada2_moe.language import (
    LLaDA2MoeAttention,
    LLaDA2MoeMLP,
    LLaDA2MoeModel,
    LLaDA2MoeSparseMoeBlock,
)


class TextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # The reference rounds the normalized activation before applying gain.
        return mx.fast.rms_norm(x, None, self.eps) * self.weight


class TextAttention(LLaDA2MoeAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if config.use_qk_norm:
            self.query_layernorm = TextRMSNorm(self.head_dim, config.rms_norm_eps)
            self.key_layernorm = TextRMSNorm(self.head_dim, config.rms_norm_eps)

    def _apply_rope(self, q: mx.array, k: mx.array, offset=0, position_ids=None):
        if self.rope_dim <= 0:
            return q, k
        positions = (
            (mx.arange(q.shape[-2], dtype=mx.float32) + offset)[None]
            if position_ids is None
            else position_ids
        )
        frequencies = self.rope_theta ** (
            -mx.arange(0, self.rope_dim, 2, dtype=mx.float32) / self.rope_dim
        )
        angles = positions.astype(mx.float32)[..., None] * frequencies
        angles = mx.concatenate([angles, angles], axis=-1)
        # LLaDA's text backbone rounds cos/sin to BF16 before rotation. The
        # fused RoPE kernel instead keeps them in FP32, which can change routing.
        cos, sin = (
            mx.cos(angles)[:, None].astype(q.dtype),
            mx.sin(angles)[:, None].astype(q.dtype),
        )

        def rotate(x):
            rotary = x[..., : self.rope_dim]
            first, second = mx.split(rotary, 2, axis=-1)
            rotated = rotary * cos + mx.concatenate([-second, first], axis=-1) * sin
            return mx.concatenate([rotated, x[..., self.rope_dim :]], axis=-1)

        return rotate(q), rotate(k)

    def __call__(self, x, mask=None, cache=None, position_ids=None):
        batch, length, _ = x.shape
        qkv = self.query_key_value(x).reshape(
            batch, length, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        q, k, v = mx.split(
            qkv, [self.num_heads, self.num_heads + self.num_key_value_heads], axis=-2
        )
        if self.query_layernorm is not None:
            q, k = self.query_layernorm(q), self.key_layernorm(k)
        q, k, v = [value.transpose(0, 2, 1, 3) for value in (q, k, v)]
        offset = cache.offset if cache is not None else 0
        q, k = self._apply_rope(q, k, offset=offset, position_ids=position_ids)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        return self.dense(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class TextMLP(LLaDA2MoeMLP):
    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextMoE(LLaDA2MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        if config.num_shared_experts:
            self.shared_experts = TextMLP(
                config, config.moe_intermediate_size * config.num_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        indices, _, logits = self.gate(x)
        scores = mx.take_along_axis(mx.sigmoid(logits), indices, axis=-1)
        if self.gate.top_k > 1:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
        scores = scores * self.gate.routed_scaling_factor
        # The checkpoint's fused MoE weights the activation BEFORE the down
        # projection, retaining FP32 routing weights until the multiplication.
        order = mx.argsort(indices.flatten())
        selected = indices.flatten()[order]
        routed = x.reshape(-1, x.shape[-1])[order // self.gate.top_k, None, :]
        gate = self.switch_mlp.gate_proj(routed, selected, sorted_indices=True)
        up = self.switch_mlp.up_proj(routed, selected, sorted_indices=True)
        activated = nn.silu(gate) * up
        activated = (activated * scores.flatten()[order, None, None]).astype(x.dtype)
        output = self.switch_mlp.down_proj(activated, selected, sorted_indices=True)
        output = output[mx.argsort(order)].reshape(*indices.shape, x.shape[-1])
        output = output.astype(mx.float32).sum(axis=-2).astype(x.dtype)
        if self.shared_experts is not None:
            output = output + self.shared_experts(x)
        return output


class LLaDAImageTextEncoder(LLaDA2MoeModel):
    def __init__(self, config):
        super().__init__(config)
        for index, layer in enumerate(self.layers):
            layer.attention = TextAttention(config, index)
            if config.num_experts is not None and index >= config.first_k_dense_replace:
                layer.mlp = TextMoE(config)
            else:
                layer.mlp = TextMLP(config, config.intermediate_size)
            layer.input_layernorm = TextRMSNorm(config.hidden_size, config.rms_norm_eps)
            layer.post_attention_layernorm = TextRMSNorm(
                config.hidden_size, config.rms_norm_eps
            )
        self.norm = TextRMSNorm(config.hidden_size, config.rms_norm_eps)
