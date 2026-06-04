from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from .config import Ideogram4TransformerConfig

LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class Ideogram4MRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, int, int],
    ) -> None:
        super().__init__()
        inv = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        selector = [0] * (head_dim // 2)
        for axis, offset in ((1, 1), (2, 2)):
            for index in range(offset, mrope_section[axis] * 3, 3):
                selector[index] = axis
        self.inv_freq = inv
        self.axis_selector = mx.array(selector, dtype=mx.int32)

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        selected_positions = mx.take(position_ids, self.axis_selector, axis=2)
        freqs = selected_positions.astype(mx.float32) * self.inv_freq.reshape(1, 1, -1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance.astype(x.dtype) + self.eps)
        return x * self.weight


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q = mx.transpose(self.norm_q(q), (0, 2, 1, 3))
        k = mx.transpose(self.norm_k(k), (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        zeros = mx.zeros(same_segment.shape, dtype=x.dtype)
        neg_inf = mx.full(same_segment.shape, -float("inf"), dtype=x.dtype)
        attn_mask = mx.expand_dims(mx.where(same_segment, zeros, neg_inf), axis=1)

        out = scaled_dot_product_attention(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=attn_mask,
        )
        out = out.astype(x.dtype)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.hidden_size
        )
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)
        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
        adaln_input: mx.array,
    ) -> mx.array:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(mod, 4, axis=-1)
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        mlp_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * self.ffn_norm2(mlp_out)


def _sinusoidal_embedding(t: mx.array, dim: int, scale: float = 1e4) -> mx.array:
    t = t.astype(mx.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = mx.exp(mx.arange(half, dtype=mx.float32) * -freq)
    emb = mx.expand_dims(t, axis=-1) * freq
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        scaled = (
            1e4
            * (x.astype(mx.float32) - self.range_min)
            / (self.range_max - self.range_min)
        )
        emb = _sinusoidal_embedding(scaled, self.dim).astype(self.mlp_in.weight.dtype)
        return self.mlp_out(nn.silu(self.mlp_in(emb)))


def _layer_norm_no_affine(x: mx.array, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x.astype(mx.float32), axis=-1, keepdims=True)
    centered = x - mean.astype(x.dtype)
    var = mx.mean(mx.square(centered.astype(mx.float32)), axis=-1, keepdims=True)
    return centered * mx.rsqrt(var.astype(x.dtype) + eps)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaln_modulation(nn.silu(c))
        return self.linear(_layer_norm_no_affine(x) * scale)


class Ideogram4Transformer(nn.Module):
    def __init__(self, config: Ideogram4TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or Ideogram4TransformerConfig()
        head_dim = self.config.emb_dim // self.config.num_heads

        self.input_proj = nn.Linear(
            self.config.in_channels, self.config.emb_dim, bias=True
        )
        self.llm_cond_norm = Ideogram4RMSNorm(self.config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(
            self.config.llm_features_dim, self.config.emb_dim, bias=True
        )
        self.t_embedding = Ideogram4EmbedScalar(
            self.config.emb_dim, input_range=(0.0, 1.0)
        )
        self.adaln_proj = nn.Linear(self.config.emb_dim, self.config.adanln_dim)
        self.embed_image_indicator = nn.Embedding(2, self.config.emb_dim)
        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=self.config.rope_theta,
            mrope_section=self.config.mrope_section,
        )
        self.layers = [
            Ideogram4TransformerBlock(
                hidden_size=self.config.emb_dim,
                intermediate_size=self.config.intermediate_size,
                num_heads=self.config.num_heads,
                norm_eps=self.config.norm_eps,
                adanln_dim=self.config.adanln_dim,
            )
            for _ in range(self.config.num_layers)
        ]
        self.final_layer = Ideogram4FinalLayer(
            hidden_size=self.config.emb_dim,
            out_channels=self.config.in_channels,
            adanln_dim=self.config.adanln_dim,
        )

    def __call__(
        self,
        *,
        llm_features: mx.array | None,
        x: mx.array,
        t: mx.array,
        position_ids: mx.array,
        segment_ids: mx.array,
        indicator: mx.array,
    ) -> mx.array:
        param_dtype = self.input_proj.weight.dtype
        x = x.astype(param_dtype)
        t = t.astype(param_dtype)
        if llm_features is not None:
            llm_features = llm_features.astype(param_dtype)

        llm_token_mask = mx.expand_dims(
            (indicator == LLM_TOKEN_INDICATOR).astype(x.dtype), axis=-1
        )
        output_image_mask = mx.expand_dims(
            (indicator == OUTPUT_IMAGE_INDICATOR).astype(x.dtype), axis=-1
        )

        x = x * output_image_mask
        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.ndim == 1:
            t_cond = mx.expand_dims(t_cond, axis=1)
        adaln_input = nn.silu(self.adaln_proj(t_cond))

        if llm_features is not None:
            llm_features = llm_features * llm_token_mask
            llm_features = self.llm_cond_norm(llm_features)
            llm_features = self.llm_cond_proj(llm_features) * llm_token_mask
            h = x + llm_features
        else:
            h = x
        h = h + self.embed_image_indicator(
            (indicator == OUTPUT_IMAGE_INDICATOR).astype(mx.int32)
        )

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.astype(h.dtype)
        sin = sin.astype(h.dtype)
        for layer in self.layers:
            h = layer(
                h,
                segment_ids=segment_ids,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
            )

        return self.final_layer(h, c=adaln_input).astype(mx.float32)
