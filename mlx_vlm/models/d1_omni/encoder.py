"""The bidirectional LFM2 trunk and the decision head, as in the reference ``encoder.py``.

A media prefix sits in front of the text: prefix queries never see text keys and the
centred convolution never lets the last prefix position read the first text one.
"""

import mlx.core as mx
import mlx.nn as nn

from ..laya.laya import DecisionHead as Layers
from ..lfm2.language import MLP


def _rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, heads, kv_heads = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        self.head_dim = d // heads
        self.q_proj = nn.Linear(d, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(heads * self.head_dim, d, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def __call__(self, x, cos, sin, mask, past=None, keep=None):
        """``past`` is the rotated (k, v) of a media prefix in front of ``x``;
        ``keep`` collects the rotated (k, v) of ``x``."""
        B, L, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).reshape(B, L, -1, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).reshape(B, L, -1, self.head_dim))
        v = self.v_proj(x).reshape(B, L, -1, self.head_dim).transpose(0, 2, 1, 3)
        q, k = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if keep is not None:
            keep.append((k, v))
        if past is not None:
            k, v = (
                mx.concatenate([mx.broadcast_to(a, (B, *a.shape[1:])), b], axis=2)
                for a, b in zip(past, (k, v))
            )
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(B, L, -1))


class ShortConv(nn.Module):
    """``out(C * conv(B * x))`` with a centred 3-tap depthwise convolution."""

    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.conv = nn.Conv1d(d, d, config.conv_L_cache, groups=d, bias=False)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, pad, keep_right, left=None, keep=None):
        """``left`` is the tap input of a media prefix's last position in front of
        ``x``; ``keep`` collects the one at the last position of ``x``."""
        b, c, u = mx.split(self.in_proj(x * pad[..., None]), 3, axis=-1)
        bu = b * u
        if keep is not None:
            keep.append(mx.contiguous(bu[:, -1:]))  # a copy: a view keeps all of bu
        if left is None:
            xp = mx.pad(bu, [(0, 0), (1, 1), (0, 0)])
        else:
            B, _, D = bu.shape
            edge = mx.zeros((B, 1, D), bu.dtype)
            xp = mx.concatenate([mx.broadcast_to(left, (B, 1, D)), bu, edge], axis=1)
        w = self.conv.weight[:, :, 0]
        y = xp[:, :-2] * w[:, 0] + xp[:, 1:-1] * w[:, 1]
        y = y + xp[:, 2:] * keep_right[..., None] * w[:, 2]
        return self.out_proj(c * y)


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(config)
        else:
            self.conv = ShortConv(config)
        self.feed_forward = MLP(
            config.hidden_size,
            config.intermediate_size,
            config.block_multiple_of,
            True,
            config.block_ffn_dim_multiplier,
        )
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class Trunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Layer(config, kind) for kind in config.layer_types]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rope_theta = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads

    def rope(self, length, dtype):
        exponent = mx.arange(0, self.head_dim, 2).astype(mx.float32) / self.head_dim
        inv_freq = 1.0 / (self.rope_theta**exponent)
        freqs = mx.arange(length).astype(mx.float32)[:, None] * inv_freq[None]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    def __call__(self, h, pad, prefix, keep=None):
        """``h`` (B, L, D) embeddings, ``pad`` (B, L) True on real positions,
        ``prefix`` (B,) media lengths (0 for text). ``keep`` collects what text
        after a lone media prefix reads from each layer (see ``prefix_state``)."""
        L = h.shape[1]
        t = mx.arange(L)[None]
        prefix = mx.array(prefix)[:, None]
        mask = pad[:, None, None, :]
        if (prefix > 0).any().item():
            blocked = (t < prefix)[:, :, None] & (t >= prefix)[:, None, :]
            mask = mask & ~blocked[:, None]
        keep_right = (t != prefix - 1).astype(h.dtype)
        padf = pad.astype(h.dtype)
        cos, sin = self.rope(L, h.dtype)
        return self._layers(h, cos, sin, mask, padf, keep_right, keep=keep)

    def prefix_state(self, media):
        """Per layer, what text after the media prefix ``media`` (P, D) reads from
        it: its rotated (k, v), or the tap input at its last position. The prefix
        does not depend on the text, so rows that follow it can share this."""
        P, state = media.shape[0], []
        self(media[None], mx.ones((1, P), mx.bool_), [P], keep=state)
        return P, state

    def after(self, h, pad, prefix):
        """``__call__`` on the text positions of rows ``h`` (B, T, D), ``pad`` (B, T)
        that all follow one media prefix, given its ``prefix_state``."""
        (P, state), (B, T, _) = prefix, h.shape
        cos, sin = self.rope(P + T, h.dtype)
        mask = mx.concatenate([mx.ones((B, P), mx.bool_), pad], axis=1)[:, None, None]
        # Text sees every prefix key; no text position is the last media one.
        padf, keep_right = pad.astype(h.dtype), mx.ones((1, T), h.dtype)
        return self._layers(h, cos[P:], sin[P:], mask, padf, keep_right, past=state)

    def _layers(self, h, cos, sin, mask, pad, keep_right, past=None, keep=None):
        past = past or [None] * len(self.layers)
        for layer, before in zip(self.layers, past):
            x = layer.operator_norm(h)
            if layer.is_attention_layer:
                x = layer.self_attn(x, cos, sin, mask, before, keep)
            else:
                x = layer.conv(x, pad, keep_right, before, keep)
            h = h + x
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return self.embedding_norm(h)


class DecisionHead(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.type_emb = nn.Embedding(3, width)
        self.head = Layers(width, layers)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def __call__(self, h, pad, marker_pos, marker_mask, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        h = self.head(h, pad[:, None, None, :])
        g = h[mx.arange(h.shape[0])[:, None], marker_pos]
        logits = self.scorer(g).squeeze(-1).astype(mx.float32)
        return mx.where(marker_mask, logits, -1e4)
