from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_ssm_mask, scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, KVCache, PoolingCache
from ..deepseek_v4.hyper_connection import HyperConnection, hc_expand
from ..gated_delta import gated_delta_update
from ..mla import MultiLinear
from ..sparse_attention import indexed_sparse_attention
from ..switch_layers import SwitchGLU
from .config import TextConfig
from .speculative_verifier import Glm5NextSpeculativeVerifier

_SPECULATIVE_VERIFIER = Glm5NextSpeculativeVerifier()


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Reference L2 normalization used by the GLM KDA parity oracle."""
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    """Readable recurrent KDA reference used for implementation parity tests."""
    dtype = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query = query * (key_dim**-0.5)
    if state is None:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outputs = []
    for index in range(length):
        q_i = query[:, index]
        k_i = key[:, index]
        v_i = value[:, index]
        g_i = mx.exp(g[:, index])[..., None]
        beta_i = beta[:, index][..., None]
        state = state * g_i
        memory = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - memory) * beta_i
        state = state + k_i[..., None] * delta[..., None, :]
        outputs.append((state * q_i[..., None]).sum(axis=-2))
    return mx.stack(outputs, axis=1).astype(dtype), state


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
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.swiglu_limit = config.swiglu_limit

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(_limited_swiglu(gate, up, self.swiglu_limit))


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

    def __call__(self, x, state, mask, lengths, return_input=False):
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
        if return_input:
            return output, state, conv_input
        return output, state


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self.projection_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.lower_bound = config.linear_lower_bound

        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * self.projection_dim, bias=False
        )
        self.qkv_conv = ShortConv1d(3 * self.projection_dim, self.conv_kernel)

        self.fbg_a_proj = nn.Linear(
            config.hidden_size, 2 * self.head_dim + self.num_heads, bias=False
        )
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.A_log = mx.zeros((self.num_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.projection_dim,), dtype=mx.float32)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        rollback_sink=None,
        linear_fn=None,
        output_linear_fn=None,
        timewise_fn=None,
        output_gate_fn=None,
        scaled_norm_fn=None,
    ):
        linear_fn = linear_fn or (lambda linear, inputs: linear(inputs))
        output_linear_fn = output_linear_fn or linear_fn
        timewise_fn = timewise_fn or (lambda fn, inputs: fn(inputs))
        output_gate_fn = output_gate_fn or (
            lambda norm, output, gate: norm(output) * mx.sigmoid(gate)
        )
        scaled_norm_fn = scaled_norm_fn or (
            lambda inputs, scale, eps: timewise_fn(
                lambda values: scale * mx.fast.rms_norm(values, None, eps),
                inputs,
            )
        )
        batch, length, _ = x.shape
        if mask is not None and mask.dtype == mx.bool_:
            x = mx.where(mask[..., None], x, 0)
        if cache is None:
            qkv_state = ssm_state = None
            lengths = None
        else:
            qkv_state = cache[0]
            if qkv_state is not None and qkv_state.shape[-1] == self.projection_dim:
                states = [cache[index] for index in range(3)]
                qkv_state = (
                    None
                    if any(state is None for state in states)
                    else mx.concatenate(states, axis=-1)
                )
            ssm_state = cache[3]
            lengths = cache.lengths

        qkv_out = self.qkv_conv(
            linear_fn(self.qkv_proj, x),
            qkv_state,
            mask,
            lengths,
            return_input=rollback_sink is not None,
        )
        if rollback_sink is None:
            qkv, qkv_state = qkv_out
            conv_input = None
        else:
            qkv, qkv_state, conv_input = qkv_out
        q, k, v = mx.split(qkv, 3, axis=-1)
        if cache is not None:
            cache[0] = qkv_state
            cache[1] = cache[2] = None

        shape = (batch, length, self.num_heads, self.head_dim)
        q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
        eps = 1e-6 / self.head_dim
        q = scaled_norm_fn(q, self.scale**2, eps)
        k = scaled_norm_fn(k, self.scale, eps)

        f_a, b, g_a = mx.split(
            linear_fn(self.fbg_a_proj, x),
            (self.head_dim, self.head_dim + self.num_heads),
            axis=-1,
        )
        a = linear_fn(self.f_b_proj, f_a).reshape(shape)
        b = b.reshape(batch, length, self.num_heads)
        initial_state = ssm_state
        delta_output = gated_delta_update(
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
            state_steps=(
                length - 1 if rollback_sink is not None and length > 1 else None
            ),
        )
        if rollback_sink is None or length <= 1:
            out, ssm_state = delta_output
            intermediate_states = None
        else:
            out, ssm_state, intermediate_states = delta_output
        if rollback_sink is not None:
            conv_states = (
                None
                if length <= 1
                else mx.stack(
                    [
                        conv_input[:, position + 1 : position + self.conv_kernel]
                        for position in range(length - 1)
                    ],
                    axis=1,
                )
            )
            rollback_sink.append(
                (
                    q,
                    k,
                    v,
                    a,
                    b,
                    self.A_log.reshape(self.num_heads, 1),
                    self.dt_bias.reshape(self.num_heads, self.head_dim),
                    initial_state,
                    mask,
                    conv_input,
                    self.conv_kernel,
                    self.lower_bound,
                    conv_states,
                    intermediate_states,
                )
            )
        if cache is not None:
            cache[3] = ssm_state
            cache.advance(length)

        gate = linear_fn(self.g_b_proj, g_a).reshape(shape)
        out = output_gate_fn(self.o_norm, out, gate).reshape(batch, length, -1)
        return output_linear_fn(self.o_proj, out)


def _batch_gather(values: mx.array, indices: mx.array) -> mx.array:
    batch, length = values.shape[:2]
    offsets_shape = (batch,) + (1,) * (indices.ndim - 1)
    offsets = mx.arange(batch).reshape(offsets_shape) * length
    flat_indices = (indices + offsets).reshape(-1)
    return values.reshape(batch * length, *values.shape[2:])[flat_indices].reshape(
        *indices.shape, *values.shape[2:]
    )


def _sparse_head_gather(values: mx.array, indices: mx.array) -> mx.array:
    """Gather per-query positions directly into fused-SDPA batch layout."""
    batch, heads, kv_length, dim = values.shape
    query_length, topk = indices.shape[1:]
    batch_offsets = (mx.arange(batch) * heads * kv_length)[:, None, None, None]
    head_offsets = (mx.arange(heads) * kv_length)[None, None, :, None]
    flat_indices = batch_offsets + head_offsets + indices[:, :, None]
    return values.reshape(batch * heads * kv_length, dim)[
        flat_indices.reshape(-1)
    ].reshape(batch * query_length, heads, topk, dim)


def _score_index_keys(q, keys, weights, scale):
    scores = q.astype(mx.float32) @ keys[:, None].astype(mx.float32).swapaxes(-1, -2)
    scores = mx.maximum(scores, 0)
    return (scores * (weights * scale)[..., None]).sum(axis=2)


def _exact_pool_select(
    q,
    pool_keys,
    weights,
    pool_ends,
    pool_valid,
    query_positions,
    select_k,
    scale,
    chunk_size=512,
):
    """Select exact top-scoring pools with bounded query-side temporaries."""

    batch = pool_valid.shape[0]
    query_length = query_positions.shape[0]
    pool_count = pool_keys.shape[1]
    selected_chunks = []
    valid_chunks = []
    for start in range(0, query_length, chunk_size):
        end = min(start + chunk_size, query_length)
        length = end - start
        candidates = pool_valid[:, None] & (
            pool_ends[:, None] <= query_positions[None, start:end, None]
        )
        if select_k == pool_count:
            selected = mx.broadcast_to(
                mx.arange(pool_count, dtype=mx.int32)[None, None],
                (batch, length, pool_count),
            )
        else:
            scores = _score_index_keys(
                q[:, start:end], pool_keys, weights[:, start:end], scale
            )
            scores = mx.where(candidates, scores, mx.finfo(mx.float32).min)
            selected = mx.argpartition(-scores, kth=select_k - 1, axis=-1)[
                ..., :select_k
            ].astype(mx.int32)
        selected_chunks.append(selected)
        valid_chunks.append(mx.take_along_axis(candidates, selected, axis=-1))

    if len(selected_chunks) == 1:
        return selected_chunks[0], valid_chunks[0]
    return (
        mx.concatenate(selected_chunks, axis=1),
        mx.concatenate(valid_chunks, axis=1),
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

    def _compress_pools(self, keys, gates):
        batch = keys.shape[0]
        if keys.shape[1] == 0:
            return mx.zeros((batch, 0, self.head_dim), dtype=keys.dtype)

        keys = mx.unflatten(keys, 1, (-1, self.index_kpool))
        gates = mx.unflatten(gates, 1, (-1, self.index_kpool))
        logits = gates.astype(mx.float32)
        logits = logits + self.index_kpool_compress_ape[None, None]
        probs = mx.softmax(logits, axis=2, precise=True).astype(keys.dtype)
        return (probs * keys).sum(axis=2)

    def __call__(
        self,
        x,
        q_resid,
        padding_mask=None,
        cache=None,
        pool_cache=None,
        offset=0,
        cache_update_sink=None,
        linear_fn=None,
        projected=None,
    ):
        linear_fn = linear_fn or (lambda linear, inputs: linear(inputs))
        batch, q_length, _ = x.shape
        if projected is None:
            k = self.k_norm(linear_fn(self.wk, x))
            if cache_update_sink is None:
                gate = x.astype(mx.float32) @ self.index_kpool_compress_gate.T
            else:
                gate = mx.concatenate(
                    [
                        x[:, index : index + 1].astype(mx.float32)
                        @ self.index_kpool_compress_gate.T
                        for index in range(q_length)
                    ],
                    axis=1,
                )
            projected_q = projected_weights = None
        else:
            k, gate, projected_q, projected_weights = projected
        if padding_mask is None:
            query_valid = mx.ones((batch, q_length), dtype=mx.bool_)
        else:
            query_valid = padding_mask.astype(mx.bool_)
        if cache_update_sink is not None:
            cache_update_sink.update(
                indexer=self,
                index_keys=k,
                index_gates=gate.astype(k.dtype),
                query_valid=query_valid,
                cache_offset=offset,
            )
        if cache is not None:
            valid_state, _ = cache.update_and_fetch(
                query_valid[:, None, :, None],
                mx.zeros((batch, 1, q_length, 0), dtype=mx.bool_),
            )
            key_valid = valid_state[:, 0, :, 0].astype(mx.bool_)

            ready_k, ready_gate, _ = pool_cache.accumulate_windows(
                k, gate.astype(k.dtype), offset
            )
            new_pool_keys = self._compress_pools(ready_k, ready_gate)
            pool_keys = pool_cache.update_and_fetch(new_pool_keys)

            pool_count = pool_keys.shape[1]
            first_key = mx.where(
                mx.any(key_valid, axis=-1),
                mx.argmax(key_valid.astype(mx.int32), axis=-1),
                key_valid.shape[1],
            )
            pool_offsets = mx.arange(pool_count * self.index_kpool).reshape(
                1, pool_count, self.index_kpool
            )
            pool_indices = first_key[:, None, None] + pool_offsets
            if isinstance(pool_cache.offset, mx.array):
                pool_valid = mx.arange(pool_count)[None] < pool_cache.offset[:, None]
            else:
                pool_valid = mx.ones((batch, pool_count), dtype=mx.bool_)
        else:
            packed = mx.concatenate(
                [k, gate.astype(k.dtype), query_valid[..., None].astype(k.dtype)],
                axis=-1,
            )
            pool_keys, pool_indices, pool_valid, key_valid = self._pooled_states(packed)

        kv_length = key_valid.shape[1]
        query_positions = kv_length - q_length + mx.arange(q_length)

        pool_count = pool_indices.shape[1]
        select_k = min(self.index_topk // self.index_kpool, pool_count)
        if pool_count == 0:
            topk = mx.zeros((batch, q_length, 0), dtype=mx.int32)
        else:
            pool_end = mx.clip(pool_indices[..., -1], 0, max(kv_length - 1, 0))
            if select_k == pool_count:
                selected, selected_valid = _exact_pool_select(
                    None,
                    pool_keys,
                    None,
                    pool_end,
                    pool_valid,
                    query_positions,
                    select_k,
                    self.softmax_scale,
                )
            else:
                q = projected_q
                if q is None:
                    q = linear_fn(self.wq_b, q_resid).reshape(
                        batch, q_length, self.n_heads, self.head_dim
                    )
                weights = projected_weights
                if weights is None:
                    weights = (
                        linear_fn(self.weights_proj, x).astype(mx.float32)
                        * self.n_heads**-0.5
                    )
                # Keep the score matmul in the model dtype, as serving runtimes
                # do, then aggregate heads in FP32. Query chunking bounds the
                # [B, chunk, H, pools] temporary while evaluating every pool.
                selected, selected_valid = _exact_pool_select(
                    q,
                    pool_keys,
                    weights,
                    pool_end,
                    pool_valid,
                    query_positions,
                    select_k,
                    self.softmax_scale,
                )

            # ``argpartition`` guarantees the selected set, but not its order.
            # Ragged speculative batches can represent the same logical history
            # with different physical padding, which otherwise changes the SDPA
            # reduction order (and eventually close logit decisions).  Restore
            # chronological pool order and keep invalid slots at the end.
            selection_order = mx.argsort(
                mx.where(selected_valid, selected, pool_count), axis=-1
            )
            selected = mx.take_along_axis(selected, selection_order, axis=-1)
            selected_valid = mx.take_along_axis(
                selected_valid, selection_order, axis=-1
            )

            safe_selected = mx.clip(selected, 0, max(pool_count - 1, 0))
            expanded_pools = mx.broadcast_to(
                pool_indices[:, None],
                (batch, q_length, *pool_indices.shape[1:]),
            )
            selected_indices = mx.take_along_axis(
                expanded_pools,
                safe_selected[..., None],
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

        # Keep the complete-pool region at a fixed width before appending the
        # partial tail.  Otherwise the tail's attention slot depends on the
        # largest pool count in another batch row, changing fused-SDPA
        # reduction order for an otherwise identical logical sequence.
        if topk.shape[-1] < self.index_topk:
            topk = mx.pad(
                topk,
                [(0, 0), (0, 0), (0, self.index_topk - topk.shape[-1])],
                constant_values=-1,
            )
        topk = topk[..., : self.index_topk]
        output_width = self.index_topk
        if self.always_select_tail and self.index_kpool > 1:
            tail_width = self.index_kpool - 1
            valid_prefix = mx.cumsum(key_valid.astype(mx.int32), axis=-1)
            safe_query_positions = mx.clip(query_positions, 0, max(kv_length - 1, 0))
            visible_count = mx.take_along_axis(
                valid_prefix,
                mx.broadcast_to(safe_query_positions[None], (batch, q_length)),
                axis=-1,
            )
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
            tail_valid = tail_valid & _batch_gather(
                key_valid[..., None], safe_tail
            ).squeeze(-1)
            topk = mx.concatenate([topk, mx.where(tail_valid, tail, -1)], axis=-1)
            output_width += tail_width

        topk = topk[..., :output_width]
        return mx.where(query_valid[..., None], topk, -1).astype(mx.int32)


def _sparse_prefill_attention(q, k, v, indices, scale, chunk_size=16, use_kernel=True):
    if use_kernel:
        output = indexed_sparse_attention(q, k, v, indices, scale)
        if output is not None:
            return output

    batch, heads, q_length, dim = q.shape
    kv_length = k.shape[2]
    outputs = []
    for start in range(0, q_length, chunk_size):
        end = min(start + chunk_size, q_length)
        idx = indices[:, start:end]
        valid = (idx >= 0) & (idx < kv_length)
        safe = mx.clip(idx, 0, max(kv_length - 1, 0))
        chunk_length = end - start

        # Gather once from [B, H, N, D], then fold each query row into the
        # batch dimension so MLX can use its fused SDPA kernel. The outer loop
        # is only memory tiling; score, mask, softmax, and value reduction are
        # fused rather than materialized as separate FP32 tensors.
        selected_k = _sparse_head_gather(k, safe)
        selected_v = _sparse_head_gather(v, safe)
        chunk_q = (
            q[:, :, start:end]
            .transpose(0, 2, 1, 3)
            .reshape(batch * chunk_length, heads, 1, dim)
        )
        mask = valid.reshape(batch * chunk_length, 1, 1, idx.shape[-1])
        out = mx.fast.scaled_dot_product_attention(
            chunk_q,
            selected_k,
            selected_v,
            scale=scale,
            mask=mask,
        )
        outputs.append(
            out.reshape(batch, chunk_length, heads, v.shape[-1]).transpose(0, 2, 1, 3)
        )
    return mx.concatenate(outputs, axis=2)


# Projected MLA keys/values make prefill much faster, but unlike the latent
# cache they grow by every attention head. Bound the transient optimization so
# million-token contexts retain the compressed-cache memory advantage.
_MAX_PROJECTED_PREFILL_TOKENS = 32768


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
        self.qkv_a_proj = nn.Linear(
            config.hidden_size,
            config.q_lora_rank + config.kv_lora_rank,
            bias=config.attention_bias,
        )
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
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

    def __call__(
        self,
        x,
        padding_mask=None,
        cache=None,
        prev_topk_indices=None,
        last_only: bool = False,
    ):
        batch, length, _ = x.shape
        q_a, kv_a = mx.split(self.qkv_a_proj(x), (self.q_lora_rank,), axis=-1)
        q_resid = self.q_a_layernorm(q_a)
        q = (
            self.q_b_proj(q_resid)
            .reshape(batch, length, self.num_heads, self.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        new_latent = self.kv_a_layernorm(kv_a)[:, None]

        if cache is None:
            kv_cache = index_cache = pool_cache = None
            projected_cache = None
            cache_offset = 0
            latent = new_latent
        else:
            kv_cache = cache[0]
            cache_offset = kv_cache.offset
            if self.indexer is None:
                index_cache = pool_cache = None
                projected_cache = cache[1]
            else:
                index_cache = cache[1]
                pool_cache = cache[2]
                projected_cache = cache[3]
            latent, _ = kv_cache.update_and_fetch(
                new_latent,
                mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype),
            )

        if self.indexer is None:
            if prev_topk_indices is None:
                raise ValueError("Shared indexer layer has no previous top-k indices.")
            topk = prev_topk_indices
        else:
            topk = self.indexer(
                x,
                q_resid,
                padding_mask,
                index_cache,
                pool_cache,
                cache_offset,
            )

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
            previous_length = latent.shape[2] - length
            cache_matches = (
                projected_cache is not None
                and projected_cache.size() == previous_length
                and latent.shape[2] <= _MAX_PROJECTED_PREFILL_TOKENS
            )
            if cache_matches:
                new_keys = self.embed_q(new_latent, transpose=False)
                new_values = self.unembed_out(new_latent)
                keys, values = projected_cache.update_and_fetch(new_keys, new_values)
            else:
                keys = self.embed_q(latent, transpose=False)
                values = self.unembed_out(latent)
            if last_only:
                q = q[:, :, -1:]
                topk = topk[:, -1:]
            out = _sparse_prefill_attention(q, keys, values, topk, self.scale)

        output_length = 1 if last_only and length > 1 else length
        out = out.transpose(0, 2, 1, 3).reshape(batch, output_length, -1)
        return self.o_proj(out), topk


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        self.is_linear = self.block_type == "linear_attention"
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

    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        prev_topk_indices=None,
    ):
        residual = x
        collapsed, post, comb = self.attn_hc(x)
        collapsed = self.input_layernorm(collapsed)
        if self.block_type == "linear_attention":
            collapsed = self.self_attn(collapsed, mask, cache)
            topk = prev_topk_indices
        else:
            collapsed, topk = self.self_attn(
                collapsed,
                mask,
                cache,
                prev_topk_indices,
            )
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
        hidden_sink: Optional[List[mx.array]] = None,
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
            h, topk = layer(
                h,
                layer_mask,
                layer_cache,
                topk,
            )
        h = self.norm(h.mean(axis=2))
        if hidden_sink is not None:
            hidden_sink.append(h)
        return h


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = Glm5NextTextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache
        prefill_kwargs = prefill_kwargs or {}
        if draft_model is None:
            return True
        if draft_kind == "mtp":
            return bool(prefill_kwargs.get("return_hidden", False)) and bool(
                prefill_kwargs.get("return_shared_kv", False)
            )
        return draft_kind is None

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        speculative_verify = kwargs.pop("speculative_verify", False)
        hidden_sink = kwargs.pop("hidden_sink", None)
        if return_hidden and hidden_sink is None:
            hidden_sink = []
        if inputs is None:
            inputs = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if speculative_verify:
            return _SPECULATIVE_VERIFIER(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                cache=cache,
                attention_mask=attention_mask,
                hidden_sink=hidden_sink,
                return_shared_kv=return_shared_kv,
                skip_logits=skip_logits,
            )
        hidden = self.model(
            inputs,
            inputs_embeds,
            cache,
            attention_mask,
            hidden_sink=hidden_sink,
        )
        num_logits_to_keep = kwargs.get("num_logits_to_keep", 0)
        if num_logits_to_keep:
            hidden = hidden[:, -num_logits_to_keep:, :]
        if skip_logits:
            logits = None
        elif self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return _SPECULATIVE_VERIFIER.logits_from_hidden(self, hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return _SPECULATIVE_VERIFIER.argmax_from_hidden(self, hidden)

    def speculative_verify_logits(self, inputs, cache, sampler):
        return _SPECULATIVE_VERIFIER.verify(self, inputs, cache, sampler)

    def speculative_verify_hidden(self, inputs, cache):
        return _SPECULATIVE_VERIFIER.verify(self, inputs, cache)

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        rollback_state,
        accepted,
        block_size: int,
    ) -> int:
        return _SPECULATIVE_VERIFIER.rollback(
            self,
            caches,
            rollback_state,
            accepted,
            block_size,
        )

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
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                indexer = layer.self_attn.indexer
                caches.append(
                    CacheList(
                        KVCache(),
                        KVCache(),
                        PoolingCache(indexer.index_kpool),
                        KVCache(),
                    )
                )
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
            mlp_prefixes = []
            if isinstance(layer.mlp, Glm5NextMLP):
                mlp_prefixes.append(f"{prefix}.mlp")
            elif isinstance(layer.mlp, Glm5NextMoE):
                mlp_prefixes.append(f"{prefix}.mlp.shared_experts")
            for mlp_prefix in mlp_prefixes:
                for suffix in ("weight", "scales", "biases"):
                    source_keys = [
                        f"{mlp_prefix}.{projection}.{suffix}"
                        for projection in ("gate_proj", "up_proj")
                    ]
                    present = [key in weights for key in source_keys]
                    if not any(present):
                        continue
                    if not all(present):
                        continue
                    weights[f"{mlp_prefix}.gate_up_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(key) for key in source_keys], axis=0
                    )

            if isinstance(layer.self_attn, Glm5NextLinearAttention):
                attn_prefix = f"{prefix}.self_attn"
                for destination, sources in (
                    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                    ("fbg_a_proj", ("f_a_proj", "b_proj", "g_a_proj")),
                    (
                        "qkv_conv.conv",
                        ("q_conv.conv", "k_conv.conv", "v_conv.conv"),
                    ),
                ):
                    for suffix in ("weight", "scales", "biases"):
                        source_keys = [
                            f"{attn_prefix}.{source}.{suffix}" for source in sources
                        ]
                        present = [key in weights for key in source_keys]
                        if not any(present):
                            continue
                        if not all(present):
                            continue
                        weights[f"{attn_prefix}.{destination}.{suffix}"] = (
                            mx.concatenate(
                                [weights.pop(key) for key in source_keys], axis=0
                            )
                        )

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
            if isinstance(layer.self_attn, Glm5NextAttention):
                for suffix in ("weight", "scales", "biases", "bias"):
                    source_keys = [
                        f"{attn_prefix}.{projection}.{suffix}"
                        for projection in ("q_a_proj", "kv_a_proj_with_mqa")
                    ]
                    present = [key in weights for key in source_keys]
                    if not any(present):
                        continue
                    if not all(present):
                        continue
                    weights[f"{attn_prefix}.qkv_a_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(key) for key in source_keys], axis=0
                    )

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
