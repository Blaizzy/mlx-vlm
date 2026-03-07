import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchLinear

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from .config import TextConfig


class Tau(nn.Module):
    """Learned position- and data-dependent temperature scaling for Q and V."""

    def __init__(self, n_heads: int, qkv_dim: int):
        super().__init__()
        self.wq = mx.zeros((n_heads, qkv_dim))
        self.wv = mx.zeros((n_heads, qkv_dim))
        self.alpha = mx.zeros((n_heads,))

    def __call__(
        self, qkv_cat: mx.array, positions: mx.array
    ) -> tuple:
        """Compute temperature scaling for Q and V.

        Args:
            qkv_cat: (B, L, qkv_dim) concatenated QKV before split
            positions: (L,) position indices

        Returns:
            tau_q: (B, n_heads, L, 1) query temperature
            tau_v: (B, n_heads, L, 1) value temperature
        """
        # Data-dependent component: project GeLU(QKV) to per-head temperature
        h = nn.gelu(qkv_cat)  # (B, L, qkv_dim)
        # (B, L, qkv_dim) @ (qkv_dim, n_heads) -> (B, L, n_heads)
        tok_q = mx.tanh(h @ self.wq.T)
        tok_v = mx.tanh(h @ self.wv.T)

        # Position-dependent component
        # tau_pos = 1 + (sigmoid(alpha * log(pos+1)) - 0.5)
        log_pos = mx.log(positions.astype(mx.float32) + 1.0)  # (L,)
        # alpha: (n_heads,), log_pos: (L,) -> (n_heads, L)
        alpha_log_pos = self.alpha[:, None] * log_pos[None, :]
        tau_pos = 1.0 + (mx.sigmoid(alpha_log_pos) - 0.5)  # (n_heads, L)

        # Combine: tok component (B, L, n_heads) + pos component (n_heads, L)
        # tau_q/v: (B, n_heads, L)
        dtype = qkv_cat.dtype
        tau_q = tok_q.transpose(0, 2, 1) + tau_pos[None, :, :]  # (B, n_heads, L)
        tau_v = tok_v.transpose(0, 2, 1) + tau_pos[None, :, :]

        return tau_q[..., None].astype(dtype), tau_v[..., None].astype(dtype)


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.rope_dim = config.rope_dim

        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(dim, qkv_dim, bias=config.attention_bias)
        self.proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=config.attention_bias
        )
        self.tau = Tau(self.n_heads, qkv_dim)
        self.rope = nn.RoPE(
            self.rope_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv_out = self.qkv(x)  # (B, L, qkv_dim)

        # Compute tau temperature scaling
        offset = cache.offset if cache is not None else 0
        positions = mx.arange(offset, offset + L)
        tau_q, tau_v = self.tau(qkv_out, positions)  # (B, n_heads, L, 1)

        # Split into Q, K, V
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv_out, [q_dim, q_dim + kv_dim], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply tau scaling to Q and V (before RoPE for Q)
        queries = queries * tau_q
        values = values * tau_v

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(output)


class DenseMLP(nn.Module):
    """Dense MLP for layers 0 to moe_start_layer-1."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class MoEMLP(nn.Module):
    """Mixture of Experts MLP with GeGLU activation for layers >= moe_start_layer."""

    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        inner_dim = config.moe_intermediate_size
        num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.router = nn.Linear(dim, num_experts, bias=True)
        # fc1 outputs 2x inner_dim for GeGLU split
        self.fc1 = SwitchLinear(dim, 2 * inner_dim, num_experts, bias=False)
        self.fc2 = SwitchLinear(inner_dim, dim, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape[:-1]  # batch dims (B, T) or (B,)
        D = x.shape[-1]
        x_2d = x.reshape(-1, D)  # (N, D)
        N = x_2d.shape[0]

        # Route
        gates = self.router(x_2d)  # (N, num_experts)
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne]
        )  # (N, ne)
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(x.dtype)  # (N, ne)

        # Flatten to 1D indices for gather_mm (each token-expert pair)
        flat_inds = inds.reshape(-1)  # (N*ne,)
        x_rep = mx.repeat(x_2d, ne, axis=0)[:, None, :]  # (N*ne, 1, D)

        # fc1: weight stored (E, out, in), swapaxes for gather_mm (E, K=in, N=out)
        fc1_w = self.fc1["weight"].swapaxes(-1, -2)  # (E, D, 2*inner)
        h = mx.gather_mm(x_rep, fc1_w, rhs_indices=flat_inds)  # (N*ne, 1, 2*inner)
        h = h.squeeze(1)  # (N*ne, 2*inner)

        # GeGLU with +1 offset
        h1, g = mx.split(h, 2, axis=-1)
        h = nn.gelu(h1) * (g + 1.0)  # (N*ne, inner)

        # fc2: each expert output through its corresponding expert
        fc2_w = self.fc2["weight"].swapaxes(-1, -2)  # (E, inner, D)
        y = mx.gather_mm(
            h[:, None, :], fc2_w, rhs_indices=flat_inds
        )  # (N*ne, 1, D)
        y = y.squeeze(1).reshape(N, ne, D)  # (N, ne, D)

        # Weighted sum over experts
        y = (y * scores[..., None]).sum(axis=-2)  # (N, D)
        return y.reshape(*orig_shape, D)


class DecoderBlock(nn.Module):
    """Transformer decoder block with parallel attention + MLP residual."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)

        if layer_idx < config.moe_start_layer:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = MoEMLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        # Parallel residual: both attn and mlp operate on same LN'd input
        h = self.ln(x)
        attn_out = self.attn(h, mask, cache)
        mlp_out = self.mlp(h)
        return x + attn_out + mlp_out


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = [
            DecoderBlock(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def layers(self):
        return self.blocks

    @property
    def embed_tokens(self):
        return self.wte

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.wte(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.blocks)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for block, c in zip(self.blocks, cache):
            h = block(h, mask, c)

        return self.post_ln(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        out = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        return LanguageModelOutput(logits=self.lm_head(out))

    @property
    def layers(self):
        return self.model.blocks

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
