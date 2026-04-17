from typing import Optional

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

    def __call__(self, qkv_cat: mx.array, positions: mx.array) -> tuple:
        h = nn.gelu(qkv_cat)
        tok_q = mx.tanh(h @ self.wq.T)
        tok_v = mx.tanh(h @ self.wv.T)

        dtype = qkv_cat.dtype
        log_pos = mx.log(positions + 1.0)
        alpha_log_pos = self.alpha[:, None] * log_pos[None, :]
        tau_pos = (1.0 + (mx.sigmoid(alpha_log_pos) - 0.5)).astype(dtype)

        tau_q = tok_q.transpose(0, 2, 1) + tau_pos[None, :, :]
        tau_v = tok_v.transpose(0, 2, 1) + tau_pos[None, :, :]

        return tau_q[..., None], tau_v[..., None]


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

        qkv_out = self.qkv(x)

        offset = cache.offset if cache is not None else 0
        positions = mx.arange(offset, offset + L)
        tau_q, tau_v = self.tau(qkv_out, positions)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(qkv_out, [q_dim, q_dim + kv_dim], axis=-1)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        queries = queries * tau_q
        values = values * tau_v

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
    def __init__(self, config: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


@mx.compile
def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


@mx.compile
def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class MoEMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        inner_dim = config.moe_intermediate_size
        num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.router = nn.Linear(dim, num_experts, bias=True)
        self.fc1 = SwitchLinear(dim, 2 * inner_dim, num_experts, bias=False)
        self.fc2 = SwitchLinear(inner_dim, dim, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok

        gates = self.router(x)
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne - 1, axis=-1)[..., :ne])
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1),
            axis=-1,
            precise=True,
        )

        x = mx.expand_dims(x, (-2, -3))

        do_sort = inds.size >= 64
        idx = inds
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, inds)

        h = self.fc1(x, idx, sorted_indices=do_sort)
        h1, g = mx.split(h, 2, axis=-1)
        h = nn.gelu(h1) * (g + 1.0)
        y = self.fc2(h, idx, sorted_indices=do_sort)

        if do_sort:
            y = _scatter_unsort(y, inv_order, inds.shape)

        y = y.squeeze(-2)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class DecoderBlock(nn.Module):
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
