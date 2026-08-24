from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from ..deepseek_v4.hyper_connection import HyperConnection, hc_expand
from ..gated_delta import gated_delta_update
from ..mla import MultiLinear
from ..switch_layers import SwitchGLU
from .config import TextConfig


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate):
        return _limited_swiglu(gate, up, self.limit)


class Glm5NextMLP(nn.Module):
    def __init__(self, config: TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.swiglu_limit = config.swiglu_limit

    def __call__(self, x):
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


@mx.compile
def _expert_select(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> Tuple[mx.array, mx.array]:
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice_scores = scores + correction_bias
    if n_group > 1:
        grouped = mx.unflatten(choice_scores, -1, (n_group, -1))
        group_scores = mx.topk(grouped, 2, axis=-1).sum(axis=-1)
        drop_count = n_group - topk_group
        if drop_count:
            drop = mx.argpartition(group_scores, kth=drop_count - 1, axis=-1)[
                ..., :drop_count
            ]
            group_mask = mx.ones(group_scores.shape, dtype=mx.bool_)
            group_mask = mx.put_along_axis(group_mask, drop, mx.array(False), axis=-1)
            choice_scores = mx.where(
                mx.repeat(group_mask[..., None], grouped.shape[-1], axis=-1).reshape(
                    choice_scores.shape
                ),
                choice_scores,
                -float("inf"),
            )

    indices = mx.argpartition(-choice_scores, kth=top_k - 1, axis=-1)[
        ..., :top_k
    ].astype(mx.int32)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if top_k > 1 and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return indices, weights * routed_scaling_factor


class MoEGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros(
            (config.n_routed_experts, config.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (config.n_routed_experts,), dtype=mx.float32
        )

    def __call__(self, x):
        return _expert_select(
            x.astype(mx.float32) @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm5NextMoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate = MoEGate(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = Glm5NextMLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

    def __call__(self, x):
        indices, weights = self.gate(x)
        routed = self.switch_mlp(x, indices)
        routed = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
        return routed + self.shared_experts(x)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(self, x, state, mask, lengths):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        output = nn.silu(self.conv(conv_input))
        keep = self.kernel_size - 1
        if lengths is None:
            state = mx.contiguous(conv_input[:, -keep:])
        else:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(keep))[..., None]
            state = mx.take_along_axis(conv_input, positions, axis=1)
        return output, state


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self.projection_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.lower_bound = config.linear_lower_bound

        self.q_proj = nn.Linear(config.hidden_size, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.projection_dim, bias=False)
        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.A_log = mx.zeros((self.num_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.projection_dim,), dtype=mx.float32)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, config.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        if cache is None:
            q_state = k_state = v_state = ssm_state = None
            lengths = None
        else:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths

        q, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)
        if cache is not None:
            cache[0], cache[1], cache[2] = q_state, k_state, v_state

        shape = (batch, length, self.num_heads, self.head_dim)
        q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
        eps = 1e-6 / self.head_dim
        q = (self.scale**2) * mx.fast.rms_norm(q, None, eps)
        k = self.scale * mx.fast.rms_norm(k, None, eps)

        a = self.f_b_proj(self.f_a_proj(x)).reshape(shape)
        b = self.b_proj(x).reshape(batch, length, self.num_heads)
        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
            lower_bound=self.lower_bound,
        )
        if cache is not None:
            cache[3] = ssm_state
            cache.advance(length)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(shape)
        out = (self.o_norm(out) * mx.sigmoid(gate)).reshape(batch, length, -1)
        return self.o_proj(out)


def _batch_gather(values: mx.array, indices: mx.array) -> mx.array:
    batch, length = values.shape[:2]
    offsets_shape = (batch,) + (1,) * (indices.ndim - 1)
    offsets = mx.arange(batch).reshape(offsets_shape) * length
    flat_indices = (indices + offsets).reshape(-1)
    return values.reshape(batch * length, *values.shape[2:])[flat_indices].reshape(
        *indices.shape, *values.shape[2:]
    )


class Glm5NextIndexer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        self.always_select_tail = config.index_kpool_always_select_tail
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.index_kpool_compress_ape = mx.zeros(
            (self.index_kpool, self.head_dim), dtype=mx.float32
        )
        self.index_kpool_compress_gate = mx.zeros(
            (self.head_dim, config.hidden_size), dtype=mx.float32
        )

    def _pooled_states(self, packed):
        keys = packed[..., : self.head_dim]
        gates = packed[..., self.head_dim : 2 * self.head_dim]
        valid = packed[..., -1].astype(mx.bool_)
        batch, length = valid.shape
        pool_count = (length + self.index_kpool - 1) // self.index_kpool

        has_key = mx.any(valid, axis=-1)
        first_key = mx.where(
            has_key,
            mx.argmax(valid.astype(mx.int32), axis=-1),
            mx.array(length),
        )
        offsets = mx.arange(pool_count * self.index_kpool).reshape(
            1, pool_count, self.index_kpool
        )
        indices = first_key[:, None, None] + offsets
        safe = mx.clip(indices, 0, max(length - 1, 0))
        grouped_keys = _batch_gather(keys, safe)
        grouped_gates = _batch_gather(gates, safe)
        grouped_valid = _batch_gather(valid[..., None], safe).squeeze(-1)
        grouped_valid = grouped_valid & (indices < length)
        pool_valid = mx.all(grouped_valid, axis=-1)
        indices = mx.where(grouped_valid, indices, -1)

        logits = grouped_gates.astype(mx.float32)
        logits = logits + self.index_kpool_compress_ape[None, None]
        logits = mx.where(grouped_valid[..., None], logits, -float("inf"))
        probs = mx.nan_to_num(mx.softmax(logits, axis=2, precise=True))
        pool_keys = (probs.astype(grouped_keys.dtype) * grouped_keys).sum(axis=2)
        return pool_keys, indices, pool_valid, valid

    def __call__(self, x, q_resid, padding_mask=None, cache=None):
        batch, q_length, _ = x.shape
        q = self.wq_b(q_resid).reshape(batch, q_length, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x))
        gate = x.astype(mx.float32) @ self.index_kpool_compress_gate.T
        if padding_mask is None:
            query_valid = mx.ones((batch, q_length), dtype=mx.bool_)
        else:
            query_valid = padding_mask.astype(mx.bool_)
        packed = mx.concatenate(
            [k, gate.astype(k.dtype), query_valid[..., None].astype(k.dtype)],
            axis=-1,
        )

        if cache is not None:
            packed, _ = cache.update_and_fetch(
                packed[:, None],
                mx.zeros((batch, 1, q_length, 0), dtype=packed.dtype),
            )
            packed = packed[:, 0]
            current_length = cache.offset
        else:
            current_length = q_length

        pool_keys, pool_indices, pool_valid, key_valid = self._pooled_states(packed)
        kv_length = packed.shape[1]
        kv_positions = mx.arange(kv_length)[None, None]
        if isinstance(current_length, mx.array):
            current_length = current_length[:, None, None]
        q_positions = current_length - q_length + mx.arange(q_length)[None, :, None]
        visible = (kv_positions <= q_positions) & key_valid[:, None]

        scores = q.astype(mx.float32) @ pool_keys[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores * self.softmax_scale, 0)
        weights = self.weights_proj(x).astype(mx.float32) * self.n_heads**-0.5
        scores = (scores * weights[..., None]).sum(axis=2)

        pool_end = mx.clip(pool_indices[..., -1], 0, max(kv_length - 1, 0))
        pool_visible = mx.take_along_axis(
            visible,
            mx.broadcast_to(pool_end[:, None], (batch, q_length, pool_end.shape[1])),
            axis=-1,
        )
        candidates = pool_visible & pool_valid[:, None]
        scores = mx.where(candidates, scores, mx.finfo(mx.float32).min)

        select_k = min(self.index_topk // self.index_kpool, pool_indices.shape[1])
        if select_k == pool_indices.shape[1]:
            selected = mx.broadcast_to(
                mx.arange(select_k)[None, None], (batch, q_length, select_k)
            )
        else:
            selected = mx.argpartition(-scores, kth=select_k - 1, axis=-1)[
                ..., :select_k
            ]
        selected_valid = mx.take_along_axis(candidates, selected, axis=-1)
        expanded_pools = mx.broadcast_to(
            pool_indices[:, None],
            (batch, q_length, *pool_indices.shape[1:]),
        )
        selected_indices = mx.take_along_axis(
            expanded_pools,
            selected[..., None],
            axis=2,
        )
        topk = selected_indices.reshape(batch, q_length, -1)
        topk = mx.where(
            mx.repeat(selected_valid[..., None], self.index_kpool, axis=-1).reshape(
                batch, q_length, -1
            ),
            topk,
            -1,
        )

        output_width = self.index_topk
        if self.always_select_tail and self.index_kpool > 1:
            tail_width = self.index_kpool - 1
            visible_count = visible.astype(mx.int32).sum(axis=-1)
            tail_count = visible_count % self.index_kpool
            first_key = mx.where(
                mx.any(key_valid, axis=-1),
                mx.argmax(key_valid.astype(mx.int32), axis=-1),
                kv_length,
            )
            tail_start = first_key[:, None] + visible_count - tail_count
            tail = tail_start[..., None] + mx.arange(tail_width)
            tail_valid = (mx.arange(tail_width)[None, None] < tail_count[..., None]) & (
                tail < kv_length
            )
            safe_tail = mx.clip(tail, 0, max(kv_length - 1, 0))
            tail_valid = tail_valid & mx.take_along_axis(visible, safe_tail, axis=-1)
            topk = mx.concatenate([topk, mx.where(tail_valid, tail, -1)], axis=-1)
            output_width += tail_width

        if topk.shape[-1] < output_width:
            topk = mx.pad(
                topk,
                [(0, 0), (0, 0), (0, output_width - topk.shape[-1])],
                constant_values=-1,
            )
        topk = topk[..., :output_width]
        return mx.where(query_valid[..., None], topk, -1).astype(mx.int32)


def _sparse_prefill_attention(q, k, v, indices, scale, chunk_size=32):
    batch, heads, q_length, dim = q.shape
    kv_length = k.shape[2]
    outputs = []
    for start in range(0, q_length, chunk_size):
        end = min(start + chunk_size, q_length)
        idx = indices[:, start:end]
        valid = (idx >= 0) & (idx < kv_length)
        safe = mx.clip(idx, 0, max(kv_length - 1, 0))
        gather_idx = safe[:, None, :, :, None]
        key_source = mx.broadcast_to(
            k[:, :, None], (batch, heads, end - start, kv_length, dim)
        )
        value_source = mx.broadcast_to(
            v[:, :, None],
            (batch, heads, end - start, kv_length, v.shape[-1]),
        )
        selected_k = mx.take_along_axis(key_source, gather_idx, axis=3)
        selected_v = mx.take_along_axis(value_source, gather_idx, axis=3)
        scores = (
            q[:, :, start:end, None].astype(mx.float32) * selected_k.astype(mx.float32)
        ).sum(axis=-1) * scale
        scores = mx.where(valid[:, None], scores, mx.finfo(mx.float32).min)
        probs = mx.softmax(scores, axis=-1, precise=True).astype(selected_v.dtype)
        outputs.append((probs[..., None] * selected_v).sum(axis=-2))
    return mx.concatenate(outputs, axis=2)


class Glm5NextAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.scale = self.q_head_dim**-0.5
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.indexer = None if self.skip_topk else Glm5NextIndexer(config, layer_idx)

    def __call__(self, x, padding_mask=None, cache=None, prev_topk_indices=None):
        batch, length, _ = x.shape
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = (
            self.q_b_proj(q_resid)
            .reshape(batch, length, self.num_heads, self.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))[:, None]

        if cache is None:
            kv_cache = index_cache = None
        else:
            kv_cache = cache[0]
            index_cache = cache[1] if self.indexer is not None else None
            latent, _ = kv_cache.update_and_fetch(
                latent,
                mx.zeros((batch, 1, length, 0), dtype=latent.dtype),
            )

        if self.indexer is None:
            if prev_topk_indices is None:
                raise ValueError("Shared indexer layer has no previous top-k indices.")
            topk = prev_topk_indices
        else:
            topk = self.indexer(x, q_resid, padding_mask, index_cache)

        if length == 1:
            kv_length = latent.shape[2]
            valid = (topk >= 0) & (topk < kv_length)
            safe = mx.clip(topk, 0, max(kv_length - 1, 0))
            selected = mx.take_along_axis(
                latent,
                safe[:, None, 0, :, None],
                axis=2,
            )
            q = self.embed_q(q)
            out = scaled_dot_product_attention(
                q,
                selected,
                selected,
                cache=kv_cache,
                scale=self.scale,
                mask=valid[:, None],
            )
            out = self.unembed_out(out)
        else:
            keys = self.embed_q(latent, transpose=False)
            values = self.unembed_out(latent)
            out = _sparse_prefill_attention(q, keys, values, topk, self.scale)

        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out), topk


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        self.self_attn = (
            Glm5NextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextAttention(config, layer_idx)
        )
        self.mlp = (
            Glm5NextMoE(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Glm5NextMLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, x, mask=None, cache=None, prev_topk_indices=None):
        residual = x
        collapsed, post, comb = self.attn_hc(x)
        collapsed = self.input_layernorm(collapsed)
        if self.block_type == "linear_attention":
            collapsed = self.self_attn(collapsed, mask, cache)
            topk = prev_topk_indices
        else:
            collapsed, topk = self.self_attn(collapsed, mask, cache, prev_topk_indices)
        x = hc_expand(collapsed, residual, post, comb)

        residual = x
        collapsed, post, comb = self.ffn_hc(x)
        collapsed = self.mlp(self.post_attention_layernorm(collapsed))
        return hc_expand(collapsed, residual, post, comb), topk


class Glm5NextTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array],
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if attention_mask is not None:
            attention_mask = attention_mask.astype(mx.bool_)
            if attention_mask.shape[-1] != h.shape[1]:
                attention_mask = attention_mask[..., -h.shape[1] :]

        h = mx.repeat(h[:, :, None], self.config.hc_mult, axis=2)
        topk = None
        for layer, layer_cache in zip(self.layers, cache):
            layer_mask = attention_mask
            if layer.block_type == "linear_attention" and layer_cache is not None:
                layer_mask = create_ssm_mask(h[:, :, 0], layer_cache)
            h, topk = layer(h, layer_mask, layer_cache, topk)
        return self.norm(h.mean(axis=2))


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = Glm5NextTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        hidden = self.model(inputs, inputs_embeds, cache, attention_mask)
        return LanguageModelOutput(logits=self.lm_head(hidden))

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(path):
            return not (
                path.endswith(("A_log", "dt_bias", "e_score_correction_bias"))
                or ".attn_hc." in path
                or ".ffn_hc." in path
                or "index_kpool_compress" in path
            )

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if any(
                marker in path
                for marker in ("attn_hc", "ffn_hc", "index_kpool_compress")
            ):
                return False
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.block_type == "linear_attention":
                caches.append(ArraysCache(size=4))
            elif layer.self_attn.indexer is None:
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefixes = ("language_model.model.", "model.", "")
        cleaned = {}
        for key, value in weights.items():
            drop = False
            for prefix in prefixes[:-1]:
                layer_prefix = f"{prefix}layers."
                if key.startswith(layer_prefix):
                    try:
                        layer_idx = int(key[len(layer_prefix) :].split(".", 1)[0])
                        drop = layer_idx >= self.args.num_hidden_layers
                    except ValueError:
                        pass
                    break
            if not drop:
                cleaned[key] = value
        weights = cleaned

        remapped = {}
        for key, value in weights.items():
            for site in ("attn", "ffn"):
                for name in ("fn", "base", "scale"):
                    key = key.replace(f".hc_{site}_{name}", f".{site}_hc.{name}")
            for old, new in (
                ("q_conv1d", "q_conv.conv"),
                ("k_conv1d", "k_conv.conv"),
                ("v_conv1d", "v_conv.conv"),
            ):
                if key.endswith(f".{old}.weight"):
                    key = key.replace(f".{old}.weight", f".{new}.weight")
                    if value.ndim == 3:
                        value = value.moveaxis(2, 1)
            remapped[key] = value
        weights = remapped

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"language_model.model.layers.{layer_idx}"
            if isinstance(layer.mlp, Glm5NextMoE):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        key0 = f"{prefix}.mlp.experts.0.{name}.{suffix}"
                        if key0 in weights:
                            values = [
                                weights.pop(
                                    f"{prefix}.mlp.experts.{expert}.{name}.{suffix}"
                                )
                                for expert in range(self.args.n_routed_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{name}.{suffix}"] = (
                                mx.stack(values)
                            )

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if isinstance(layer.self_attn, Glm5NextAttention) and kv_b_key in weights:
                value = weights.pop(kv_b_key)
                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                if quantized:
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases", None)
                    bits = value.shape[-1] * 32 // self.args.kv_lora_rank
                    group_size = self.args.kv_lora_rank // scales.shape[-1]
                    mode = "mxfp8" if biases is None and bits == 8 else "affine"
                    dequantize_kwargs = {
                        "bits": bits,
                        "group_size": group_size,
                        "mode": mode,
                    }
                    if biases is None:
                        value = mx.dequantize(value, scales, **dequantize_kwargs)
                    else:
                        value = mx.dequantize(
                            value, scales, biases, **dequantize_kwargs
                        )
                value = value.reshape(
                    self.args.num_attention_heads,
                    self.args.qk_nope_head_dim + self.args.v_head_dim,
                    self.args.kv_lora_rank,
                )
                wk = mx.contiguous(
                    value[:, : self.args.qk_nope_head_dim].swapaxes(-1, -2)
                )
                wv = mx.contiguous(value[:, self.args.qk_nope_head_dim :])
                if quantized:
                    wk, wk_s, *wk_b = mx.quantize(
                        wk, bits=bits, group_size=group_size, mode=mode
                    )
                    wv, wv_s, *wv_b = mx.quantize(
                        wv, bits=bits, group_size=group_size, mode=mode
                    )
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    if wk_b:
                        weights[f"{attn_prefix}.embed_q.biases"] = wk_b[0]
                    if wv_b:
                        weights[f"{attn_prefix}.unembed_out.biases"] = wv_b[0]
                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv
        return weights
