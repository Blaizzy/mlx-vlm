from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, scaled_dot_product_attention
from .config import ModelConfig

# The reference calls F.rms_norm with eps=None, which resolves to
# finfo(dtype).eps, so the epsilon is dtype dependent and far larger than the
# usual 1e-6. rms_norm_eps_mode is pinned in the config for this reason.
_EPS = {mx.bfloat16: 0.0078125, mx.float16: 0.0009765625, mx.float32: 1.1920929e-07}


def rms_norm(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, None, _EPS.get(x.dtype, 1.1920929e-07))


def fold(scale: mx.array, weight: mx.array) -> mx.array:
    """Scale a projection by a learned scalar, rounded to the weight's dtype.

    The reference multiplies a 0-dim fp32 tensor by a bf16 matrix, which torch
    promotes as a wrapped scalar: the scalar is rounded to bf16 first. Doing the
    product in fp32 instead gives different numbers.
    """
    return scale.astype(weight.dtype) * weight


class MuddMixer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        n, taps, inter = config.num_hidden_layers, config.mudd_taps, config.mudd_inter
        self.dense1 = mx.zeros((inter, config.hidden_size))
        self.dense2 = mx.zeros((n, taps, inter))
        self.bias = mx.zeros((n, taps))
        self.uses_r_way = config.mudd_mlp
        if self.uses_r_way:
            self.dense2_mlp = mx.zeros((n, taps, inter))
            self.bias_mlp = mx.zeros((n, taps))

    def combine(
        self,
        values: List[mx.array],
        x_cur: mx.array,
        layer_idx: int,
        r_way: bool = False,
    ) -> mx.array:
        count = len(values)
        inner = nn.gelu(rms_norm(x_cur) @ self.dense1.astype(x_cur.dtype).T)
        dense2 = self.dense2_mlp if r_way else self.dense2
        bias = self.bias if not r_way else self.bias_mlp
        w = inner @ dense2[layer_idx, :count].astype(inner.dtype).T
        w = w + bias[layer_idx, :count].astype(w.dtype)
        # Ordered left-to-right accumulation in the activation dtype; a stacked
        # contraction is mathematically identical and numerically is not.
        out = w[..., 0:1].astype(values[0].dtype) * values[0]
        for i in range(1, count):
            out = out + w[..., i : i + 1].astype(values[i].dtype) * values[i]
        return out


class Rotary(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        n_pairs, head_dim = config.rope_n_pairs, config.head_dim
        base = config.rope_base_local
        freq = (1.0 / base) ** mx.linspace(0, 1, n_pairs, dtype=mx.float32)
        freq = mx.repeat(freq, 2)
        self._freq = mx.concatenate([freq, mx.zeros(head_dim - 2 * n_pairs)])

    def __call__(self, q: mx.array, k: mx.array, positions: mx.array):
        theta = positions.astype(mx.float32).reshape(-1, 1) * self._freq.reshape(1, -1)
        head_dim = self._freq.size
        cos = mx.cos(theta).astype(mx.bfloat16).reshape(-1, 1, head_dim)
        sin = mx.sin(theta).astype(mx.bfloat16).reshape(-1, 1, head_dim)
        # The odd lanes carry the negated sine, which with the pairwise flip
        # below gives (a, b) -> (a cos + b sin, b cos - a sin).
        lane = mx.where(mx.arange(head_dim) % 2 == 1, -1.0, 1.0).astype(sin.dtype)
        sin = sin * lane.reshape(1, 1, head_dim)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        shape = x.shape
        flipped = mx.flip(x.reshape(*shape[:-1], shape[-1] // 2, 2), axis=-1)
        return cos.astype(x.dtype) * x + sin.astype(x.dtype) * flipped.reshape(shape)


def _causal_window_mask(
    q_len: int, offset: int, window: Optional[int], dtype
) -> mx.array:
    q_idx = mx.arange(offset, offset + q_len).reshape(-1, 1)
    k_idx = mx.arange(offset + q_len).reshape(1, -1)
    keep = k_idx <= q_idx
    if window is not None:
        # k >= q - window, inclusive of the query token: window+1 keys.
        keep = keep & (k_idx >= q_idx - window)
    return mx.where(keep, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary: Rotary):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = self.n_heads // self.n_kv
        self.scale = config.attention_softmax_scale
        self.ve_dim = config.ve_dim
        self.ve_stored_heads = config.ve_stored_heads
        self.ve_gate_scale = config.ve_gate_scale
        self.attn_gate_channels = config.attn_gate_channels
        self.attn_gate_scale = config.attn_gate_scale
        self.xsa_eps = config.xsa_normalize_eps

        self.uses_ve = layer_idx in set(config.ve_layers)
        self.uses_xsa = config.xsa and layer_idx in set(config.xsa_layers)
        self.is_global = layer_idx in set(config.global_layers)
        self.uses_rope = not (config.global_nope and self.is_global)
        window = config.global_window if self.is_global else config.sliding_window
        self.window = None if window < 0 else window
        self.rotary = rotary

        h, kv = config.hidden_size, self.n_kv * self.head_dim
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, kv, bias=False)
        self.v_proj = nn.Linear(h, kv, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.qkv_scale = mx.zeros(())
        self.o_scale = mx.zeros(())
        self.xsa_alpha = mx.zeros((self.n_heads,))
        if self.uses_ve:
            self.ve_gate = mx.zeros((self.ve_stored_heads, config.ve_gate_channels))
        if self.attn_gate_channels:
            self.attn_gate = mx.zeros((self.n_heads, self.attn_gate_channels))

    def _value_embeddings(self, attn_in, v, ve_rows):
        B, L, _ = attn_in.shape
        ve = ve_rows.astype(v.dtype).reshape(B, L, self.ve_stored_heads, self.ve_dim)
        if self.ve_dim < self.head_dim:
            ve = mx.pad(ve, [(0, 0), (0, 0), (0, 0), (0, self.head_dim - self.ve_dim)])
        gate_w = self.ve_gate.astype(attn_in.dtype)
        if self.ve_stored_heads > self.n_kv:
            ve, gate_w = ve[:, :, : self.n_kv], gate_w[: self.n_kv]
        gate = self.ve_gate_scale * mx.sigmoid(
            attn_in[..., : gate_w.shape[-1]] @ gate_w.T
        )
        return v + mx.expand_dims(gate, -1) * ve

    def __call__(self, attn_in, ve_rows, positions, cache=None):
        B, L, _ = attn_in.shape
        q = attn_in @ fold(self.qkv_scale, self.q_proj.weight).T
        k = attn_in @ fold(self.qkv_scale, self.k_proj.weight).T
        v = attn_in @ fold(self.qkv_scale, self.v_proj.weight).T
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        k = k.reshape(B, L, self.n_kv, self.head_dim)
        v = v.reshape(B, L, self.n_kv, self.head_dim)

        if self.uses_ve:
            # Attention and the KV cache both consume the VE-adjusted values.
            v = self._value_embeddings(attn_in, v, ve_rows)

        q, k = rms_norm(q), rms_norm(k)
        if self.uses_rope:
            qf = q.reshape(B * L, self.n_heads, self.head_dim)
            kf = k.reshape(B * L, self.n_kv, self.head_dim)
            qf, kf = self.rotary(qf, kf, positions.reshape(-1))
            q, k = qf.reshape(q.shape), kf.reshape(k.shape)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        mask = _causal_window_mask(L, offset, self.window, q.dtype)
        y = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        y = y.transpose(0, 2, 1, 3)

        if self.uses_xsa:
            vk = self._expand_kv(v.transpose(0, 2, 1, 3)[:, -L:])
            vf = vk.astype(mx.float32)
            vn = vf / mx.maximum(
                mx.linalg.norm(vf, axis=-1, keepdims=True), self.xsa_eps
            )
            proj = (y.astype(mx.float32) * vn).sum(-1, keepdims=True)
            alpha = mx.tanh(self.xsa_alpha.astype(mx.float32)).reshape(1, 1, -1, 1)
            y = y - (alpha * proj * vn).astype(y.dtype)

        if self.attn_gate_channels:
            gate = self.attn_gate_scale * mx.sigmoid(
                attn_in[..., : self.attn_gate_channels]
                @ self.attn_gate.astype(attn_in.dtype).T
            )
            y = y * mx.expand_dims(gate, -1).astype(y.dtype)

        y = y.reshape(B, L, self.n_heads * self.head_dim)
        return y @ fold(self.o_scale, self.o_proj.weight).T

    def _expand_kv(self, x):
        if x.shape[2] < self.n_heads:
            return mx.repeat(x, self.groups, axis=2)
        return x


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.mlp_type = config.mlp_type
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(h, i, bias=False)

    def __call__(self, h):
        up = self.up_proj(h)
        act = (
            nn.silu(self.gate_proj(h)) * up
            if self.mlp_type == "swiglu"
            else nn.relu(up) ** 2
        )
        return self.down_proj(act)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary: Rotary):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, rotary)
        self.mlp = MLP(config)
        self.resid_lambda_attn = mx.zeros(())
        self.post_lambda_attn = mx.zeros(())
        self.resid_lambda_mlp = mx.zeros(())
        self.post_lambda_mlp = mx.zeros(())

    def __call__(self, x, attn_in, residual_base, ve_rows, positions, cache=None):
        attn_out = self.self_attn(attn_in, ve_rows, positions, cache)
        x = (
            self.resid_lambda_attn.astype(x.dtype) * residual_base
            + self.post_lambda_attn.astype(x.dtype) * attn_out
        )
        mlp_out = self.mlp(rms_norm(x))
        return (
            self.resid_lambda_mlp.astype(x.dtype) * x
            + self.post_lambda_mlp.astype(x.dtype) * mlp_out
        )


class LimiteModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.value_embeds = nn.Embedding(
            config.vocab_size, config.ve_stored_heads * config.ve_dim
        )
        rotary = Rotary(config)
        self.layers = [
            DecoderLayer(config, i, rotary) for i in range(config.num_hidden_layers)
        ]
        self.mudd = MuddMixer(config)
        self.tap_idx = config.mudd_tap_idx
        self.retained = {i for taps in self.tap_idx.values() for i in taps}
        self.final_softcap = config.final_softcap

    def __call__(self, inputs, cache=None, positions=None):
        x = rms_norm(self.embed_tokens(inputs))
        ve_rows = self.value_embeds(inputs)
        if positions is None:
            offset = (
                cache[0].offset if cache is not None and cache[0] is not None else 0
            )
            positions = mx.arange(offset, offset + inputs.shape[1])
            positions = mx.broadcast_to(positions, inputs.shape)
        if cache is None:
            cache = [None] * len(self.layers)

        history = {0: x} if self.tap_idx else {}
        for index, layer in enumerate(self.layers):
            if index in self.tap_idx:
                values = [history[i] for i in self.tap_idx[index]]
                attn_in = rms_norm(self.mudd.combine(values, x, index))
                residual_base = (
                    self.mudd.combine(values, x, index, r_way=True)
                    if self.mudd.uses_r_way
                    else x
                )
            else:
                attn_in = rms_norm(x)
                residual_base = x
            x = layer(x, attn_in, residual_base, ve_rows, positions, cache[index])
            if index + 1 in self.retained:
                history[index + 1] = x
        if self.final_softcap > 0:
            cap = self.final_softcap
            x = cap * mx.tanh(x / cap)
        return rms_norm(x)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = LimiteModel(config)
        cap = config.softcap_logits or {}
        self.softcap_a = float(cap.get("a", 0.0))
        self.softcap_b = float(cap.get("b", 0.0))
        self.softcap_c = float(cap.get("c", 1.0))

    def __call__(self, inputs, cache=None, mask=None, **kwargs):
        h = self.model(inputs, cache=cache)
        raw = h @ self.model.embed_tokens.weight.astype(h.dtype).T
        # The head is a sigmoid softcap, not a bare projection, so the values
        # are bounded in (0, a) rather than unbounded logits.
        logits = self.softcap_a * mx.sigmoid(
            (raw.astype(mx.float32) + self.softcap_b) / self.softcap_c
        )
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from ..cache import KVCache

        return [KVCache() for _ in self.model.layers]
