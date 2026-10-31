"""Mistral Large 4: latent attention, DSA, LeRoPE and expert parallelism.

Architecture follows Hugging Face's Apache-2.0 Mistral Large 4 reference.
Packed NVFP4 experts retain their original per-block and global scales.
"""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import _BaseCache
from ..switch_layers import QuantizedSwitchLinear, SwitchLinear
from .config import TextConfig


class LatentCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = self.values = self.index_keys = None
        self.offset = 0

    def _append(self, cache, values):
        end = self.offset + values.shape[-2]
        if cache is None or end > cache.shape[-2]:
            capacity = ((values.shape[-2] + self.step - 1) // self.step) * self.step
            padding = mx.zeros(
                (*values.shape[:-2], capacity, values.shape[-1]), dtype=values.dtype
            )
            cache = (
                padding
                if cache is None
                else mx.concatenate([cache[..., : self.offset, :], padding], axis=-2)
            )
        cache[..., self.offset : end, :] = values
        return cache

    def update_latent(self, keys, values):
        end = self.offset + keys.shape[2]
        self.keys = self._append(self.keys, keys)
        self.values = self._append(self.values, values)
        return self.keys[..., :end, :], self.values[..., :end, :]

    def update_index(self, keys):
        self.index_keys = self._append(self.index_keys, keys)
        return self.index_keys[:, : self.offset + keys.shape[1]]

    @property
    def state(self):
        return [
            x[..., : self.offset, :]
            for x in (self.keys, self.values, self.index_keys)
            if x is not None
        ]

    @state.setter
    def state(self, value):
        self._restored_state = value

    @property
    def meta_state(self):
        return (
            str(self.offset),
            str(int(self.keys is not None)),
            str(int(self.index_keys is not None)),
        )

    @meta_state.setter
    def meta_state(self, value):
        offset, latent, index = map(int, value)
        state = iter(self._restored_state)
        self.keys = next(state) if latent else None
        self.values = next(state) if latent else None
        self.index_keys = next(state) if index else None
        self.offset = offset
        del self._restored_state

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return sum(
            x.nbytes for x in (self.keys, self.values, self.index_keys) if x is not None
        )

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


class LeRoPE(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.qk_rope_head_dim
        p = args.rope_parameters
        base = p.get("rope_theta", 10000.0)
        factor = p.get("factor", 1.0)
        original = p.get("original_max_position_embeddings", 512)

        def correction(rotations):
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(p.get("beta_fast", 32.0))), 0)
        high = min(math.ceil(correction(p.get("beta_slow", 1.0))), dim - 1)
        ramp = mx.clip(
            (mx.arange(dim // 2, dtype=mx.float32) - low) / max(high - low, 0.001), 0, 1
        )
        inv = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        self._inv_freq = inv * ((1 - ramp) + ramp / factor)
        self.alpha = mx.zeros((dim // 2,))

    def __call__(self, positions, dtype):
        angles = (
            positions[..., None].astype(mx.float32)
            * self._inv_freq
            * mx.exp(self.alpha.astype(mx.float32))
        )
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def rotate(x, cos, sin):
    """Interleaved input pairs, half-split output (the Mistral convention)."""
    a, b = x[..., ::2], x[..., 1::2]
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


class SoftmaxScaling(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.s = mx.zeros((heads,))
        self.b = mx.ones((heads,))

    def __call__(self, q, positions):
        log = mx.log(positions.astype(mx.float32) + 1).astype(q.dtype)
        scale = (
            self.s.astype(q.dtype)[None, :, None] * log[:, None, :]
            + self.b.astype(q.dtype)[None, :, None]
        )
        return q * scale[..., None]


class Indexer(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.heads = args.index_n_heads
        self.dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.use_rope = args.no_rope_layers[idx]
        self.topk = args.index_topk
        self.wq_b = nn.Linear(args.hidden_size, self.heads * self.dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.dim, bias=False)
        self.q_input_norm = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.heads, bias=False)
        self.ssmax_scaling = SoftmaxScaling(self.heads)

    def __call__(self, x, positions, cos, sin, cache, causal):
        b, t, _ = x.shape
        k = self.k_norm(self.wk(x))
        if self.use_rope:
            k = mx.concatenate(
                [rotate(k[..., : self.rope_dim], cos, sin), k[..., self.rope_dim :]],
                axis=-1,
            )
        if cache is not None:
            k = cache.update_index(k)
        if k.shape[1] <= self.topk:
            return None  # All causal tokens are selected; retain keys for later chunks.
        q = (
            self.wq_b(self.q_input_norm(x))
            .reshape(b, t, self.heads, self.dim)
            .transpose(0, 2, 1, 3)
        )
        q = self.ssmax_scaling(q, positions)
        if self.use_rope:
            q = mx.concatenate(
                [
                    rotate(q[..., : self.rope_dim], cos[:, None], sin[:, None]),
                    q[..., self.rope_dim :],
                ],
                axis=-1,
            )
        scores = mx.maximum(
            q.astype(mx.float32)
            @ k[:, None].swapaxes(-1, -2).astype(mx.float32)
            * self.dim**-0.5,
            0,
        )
        weights = (
            self.weights_proj(x).transpose(0, 2, 1).astype(mx.float32)
            * self.heads**-0.5
        )
        scores = (scores * weights[..., None]).sum(axis=1)
        scores = mx.where(causal[:, 0], scores, -float("inf"))
        return mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]


class Attention(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.idx = idx
        self.source = args.kv_source_layers[idx]
        self.heads = args.num_attention_heads
        self.rope_dim = args.qk_rope_head_dim if args.no_rope_layers[idx] else 0
        self.nope_dim = args.qk_nope_head_dim + (
            0 if self.rope_dim else args.qk_rope_head_dim
        )
        self.head_dim = self.nope_dim + self.rope_dim
        self.latent = args.kv_lora_rank
        self.value_dim = args.v_head_dim
        self.wq = nn.Linear(args.hidden_size, self.heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.heads * self.value_dim, args.hidden_size, bias=False)
        if self.source == idx:
            self.wkv_a_with_mqa = nn.Linear(
                args.hidden_size, self.latent + self.rope_dim, bias=False
            )
            self.kv_a_norm = nn.RMSNorm(self.latent, eps=args.rms_norm_eps)
        self.wkv_b = nn.Linear(
            self.latent, self.heads * (self.nope_dim + self.value_dim), bias=False
        )
        self.ssmax_scaling = SoftmaxScaling(self.heads)
        self.indexer = Indexer(args, idx) if args.indexer_types[idx] == "full" else None

    def __call__(self, x, positions, cos, sin, cache, shared, previous, mask):
        b, t, _ = x.shape
        q = self.wq(x).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.rope_dim:
            q = mx.concatenate(
                [
                    q[..., : self.nope_dim],
                    rotate(q[..., self.nope_dim :], cos[:, None], sin[:, None]),
                ],
                axis=-1,
            )
        if self.source == self.idx:
            kv = self.wkv_a_with_mqa(x)
            latent = self.kv_a_norm(kv[..., : self.latent])[:, None]
            kr = kv[..., self.latent :][:, None]
            if self.rope_dim:
                kr = rotate(kr, cos[:, None], sin[:, None])
            if cache is not None:
                latent, kr = cache.update_latent(latent, kr)
            shared[self.idx] = (latent, kr)
        else:
            latent, kr = shared[self.source]
        q = self.ssmax_scaling(q, positions)
        length = latent.shape[2]
        causal = mx.arange(length)[None, None, None, :] <= positions[:, None, :, None]
        if mask is not None:
            if mask.ndim == 2:
                causal = causal & mask[:, None, None, :length].astype(mx.bool_)
            elif mask.dtype == mx.bool_:
                causal = causal & mask
            else:
                raise ValueError("Mistral Large 4 expects a boolean attention mask")
        topk = (
            self.indexer(x, positions, cos, sin, cache, causal)
            if self.indexer is not None
            else previous
        )
        if topk is not None:
            if t == 1:
                # Expand only selected latents during decode, in sequence order.
                indices = mx.sort(topk, axis=-1)[:, None]
                latent = mx.take_along_axis(latent, indices[..., 0, :, None], axis=2)
                kr = mx.take_along_axis(kr, indices[..., 0, :, None], axis=2)
                causal = mx.take_along_axis(causal, indices, axis=-1)
                length = latent.shape[2]
            else:
                selected = mx.zeros((b, t, length), dtype=mx.bool_)
                selected = mx.put_along_axis(selected, topk, mx.array(True), axis=-1)
                causal = causal & selected[:, None]
        kv = (
            self.wkv_b(latent)
            .reshape(b, length, self.heads, self.nope_dim + self.value_dim)
            .transpose(0, 2, 1, 3)
        )
        kn, values = kv[..., : self.nope_dim], kv[..., self.nope_dim :]
        keys = mx.concatenate(
            [kn, mx.broadcast_to(kr, (b, self.heads, length, self.rope_dim))], axis=-1
        )
        out = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.head_dim**-0.5, mask=causal
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        if cache is not None:
            cache.offset += t
        return self.wo(out), topk


class Experts(nn.Module):
    def __init__(self, args):
        super().__init__()
        n = args.expert_end - args.expert_start
        d, h = args.hidden_size, args.moe_intermediate_size
        linear = (
            partial(
                QuantizedSwitchLinear, bias=False, group_size=16, bits=4, mode="nvfp4"
            )
            if args.nvfp4
            else partial(SwitchLinear, bias=False)
        )
        self.w1 = linear(d, h, n)
        self.w3 = linear(d, h, n)
        self.w2 = linear(h, d, n)

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        return self.w2(
            nn.silu(self.w1(x, indices)) * self.w3(x, indices), indices
        ).squeeze(-2)


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.experts = Experts(args)
        self.shared_experts = (
            MLP(args.hidden_size, args.moe_intermediate_size * args.n_shared_experts)
            if args.n_shared_experts
            else None
        )
        self.group = None
        self.start, self.end = args.expert_start, args.expert_end

    def __call__(self, x):
        a = self.args
        scores = mx.softmax(
            x.astype(mx.float32) @ self.gate.weight.T.astype(mx.float32),
            axis=-1,
            precise=True,
        )
        choice = scores
        if a.n_group > 1:
            grouped = scores.reshape(*scores.shape[:-1], a.n_group, -1)
            group_scores = mx.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
            groups = mx.argpartition(-group_scores, kth=a.topk_group - 1, axis=-1)[
                ..., : a.topk_group
            ]
            keep = mx.any(mx.arange(a.n_group) == groups[..., None], axis=-2)
            choice = mx.where(
                mx.repeat(keep, scores.shape[-1] // a.n_group, axis=-1),
                scores,
                -float("inf"),
            )
        inds = mx.argpartition(-choice, kth=a.num_experts_per_tok - 1, axis=-1)[
            ..., : a.num_experts_per_tok
        ]
        # Ascending expert order matches the reference scatter accumulation.
        inds = mx.stop_gradient(mx.sort(inds, axis=-1))
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = (
            weights
            / (weights.sum(axis=-1, keepdims=True) + 1e-20)
            * a.routed_scaling_factor
        )
        if self.group is None:
            if self.start or self.end != a.n_routed_experts:
                raise ValueError("An expert partition requires a distributed group")
            outputs = self.experts(x, inds)
        else:
            owned = (inds >= self.start) & (inds < self.end)
            local = mx.where(owned, inds - self.start, 0)
            outputs = self.experts(x, local)
            # Preserve each route before reduction, as in MiniMax's EP path.
            outputs = mx.distributed.all_sum(
                mx.where(owned[..., None], outputs, 0), group=self.group
            )
        weighted = (outputs.astype(mx.float32) * weights[..., None]).astype(x.dtype)
        y = weighted[..., 0, :]
        for i in range(1, a.num_experts_per_tok):
            y = y + weighted[..., i, :]
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.attention = Attention(args, idx)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feed_forward = (
            MLP(args.hidden_size, args.intermediate_size)
            if idx < args.first_k_dense_replace
            else MoE(args)
        )

    def __call__(self, x, positions, cos, sin, cache, shared, topk, mask):
        h, topk = self.attention(
            self.attention_norm(x), positions, cos, sin, cache, shared, topk, mask
        )
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x)), topk


class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rotary_emb = LeRoPE(args)

    def __call__(self, input_ids, inputs_embeds=None, cache=None, mask=None):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        offset = cache[0].offset if cache else 0
        positions = mx.broadcast_to(
            mx.arange(offset, offset + x.shape[1])[None], x.shape[:2]
        )
        cos, sin = self.rotary_emb(positions, x.dtype)
        shared, topk = {}, None
        cache = cache or [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x, topk = layer(x, positions, cos, sin, c, shared, topk, mask)
        return self.norm(x)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs=None, inputs_embeds=None, mask=None, cache=None, **kwargs
    ):
        x = self.model(inputs, inputs_embeds, cache, mask)
        logits = self.lm_head(x)
        cap = self.args.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [LatentCache() for _ in self.layers]

    def shard(self, group=None):
        group = group or mx.distributed.init()
        n = self.args.n_routed_experts
        start, end = (
            n * group.rank() // group.size(),
            n * (group.rank() + 1) // group.size(),
        )
        for layer in self.layers:
            moe = layer.feed_forward
            if not isinstance(moe, MoE):
                continue
            if (moe.start, moe.end) != (start, end):
                if (moe.start, moe.end) != (0, n):
                    raise ValueError("Checkpoint partition does not match rank")
                for linear in (moe.experts.w1, moe.experts.w2, moe.experts.w3):
                    for key, value in list(linear.items()):
                        if isinstance(value, mx.array) and value.shape[0] == n:
                            linear[key] = value[start:end]
            moe.start, moe.end, moe.group = start, end, group
