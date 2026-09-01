from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from ..deepseek_v4.hyper_connection import HyperConnection, hc_expand
from ..deepseek_v32.language import DeepseekV32MoE
from ..deepseek_v32.language import Model as DSV32Model
from ..deepseek_v32.language import MoEGate, group_expert_select
from ..gated_delta import gated_delta_update
from ..mla import MultiLinear
from ..mlp import DeepseekMLP
from .config import ModelConfig, TextConfig
from .speculative_verifier import Glm5NextExactSpeculativeVerifier, verify_logits

_SPECULATIVE_VERIFIER = Glm5NextExactSpeculativeVerifier()


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dt = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        var = (x * x).mean(-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        x = self.weight.astype(mx.float32) * x
        x = x * mx.sigmoid(gate.astype(mx.float32))
        return x.astype(dt)


class Glm5NextForgetGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = mx.zeros(self.qkv_dim)
        self.A_log = mx.zeros(self.num_heads)
        self.safe_gate_lower_bound = config.linear_lower_bound

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, S, _ = hidden_states.shape
        fg = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (fg.astype(mx.float32) + self.dt_bias.astype(mx.float32)).reshape(
            B, S, self.num_heads, self.head_dim
        )
        decay = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * g)
        g_softplus = mx.where(g > 20.0, g, mx.log(1.0 + mx.exp(g)))
        return -decay * g_softplus


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    dt = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    query = query * (Dk**-0.5)
    if state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outs = []
    for i in range(S):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = mx.exp(g[:, i])[..., None]
        b_i = beta[:, i][..., None]
        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i[..., None] * delta[..., None, :]
        out_i = (state * q_i[..., None]).sum(axis=-2)
        outs.append(out_i)
    out = mx.stack(outs, axis=1).astype(dt)
    return out, state


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)
        self.fuse_in = True
        self._fused_ready = False

    def _fused_in_proj(self, inputs):
        if not self._fused_ready:
            mods = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.forget_gate.f_a_proj,
                self.g_a_proj,
                self.b_proj,
            ]
            quantized = [hasattr(m, "scales") for m in mods]
            homogeneous = all(quantized) or not any(quantized)
            if homogeneous and all(quantized):
                homogeneous = (
                    len({m.group_size for m in mods}) == 1
                    and len({m.bits for m in mods}) == 1
                )
            if not homogeneous:
                self.fuse_in = False
                self._fused_ready = True
                return None
            pts, acc = [], 0
            for m in mods[:-1]:
                acc += m.weight.shape[0]
                pts.append(acc)
            self._split_pts = pts
            self._fq = hasattr(mods[0], "scales")
            self._fw = mx.concatenate([m.weight for m in mods], axis=0)
            if self._fq:
                self._fs = mx.concatenate([m.scales for m in mods], axis=0)
                self._fb = mx.concatenate([m.biases for m in mods], axis=0)
                self._gs, self._bits = mods[0].group_size, mods[0].bits
            self._fused_ready = True
        if self._fq:
            out = mx.quantized_matmul(
                inputs,
                self._fw,
                self._fs,
                self._fb,
                transpose=True,
                group_size=self._gs,
                bits=self._bits,
            )
        else:
            out = inputs @ self._fw.T
        return mx.split(out, self._split_pts, axis=-1)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        fused = self._fused_in_proj(inputs) if self.fuse_in else None
        if fused is not None:
            q_o, k_o, v_o, fa_o, ga_o, b_o = fused
            mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
        else:
            mixed = mx.concatenate(
                [self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)], axis=-1
            )
            fa_o = self.forget_gate.f_a_proj(inputs)
            ga_o = self.g_a_proj(inputs)
            b_o = self.b_proj(inputs)
        if mask is not None and mask.dtype == mx.bool_:
            mixed = mx.where(mask[..., None], mixed, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.qkv_dim, 2 * self.qkv_dim], axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        fg = self.forget_gate
        a = fg.f_b_proj(fa_o).reshape(B, S, self.num_heads, self.head_dim)
        in_dtype = q.dtype
        q = (_l2norm(q.astype(mx.float32)) * (self.head_dim**-0.5)).astype(in_dtype)
        k = _l2norm(k.astype(mx.float32)).astype(in_dtype)

        state = cache[1] if cache is not None else None
        A_log = fg.A_log.reshape(self.num_heads, 1)
        dt_bias = fg.dt_bias.reshape(self.num_heads, self.head_dim)
        lower_bound = fg.safe_gate_lower_bound
        if gdn_sink is not None:
            gdn_sink.append(
                (
                    q,
                    k,
                    v,
                    a,
                    b_o,
                    A_log,
                    dt_bias,
                    state,
                    conv_input,
                    self.conv_kernel_size,
                    lower_bound,
                )
            )
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b_o,
            A_log,
            dt_bias,
            state=state,
            lower_bound=lower_bound,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        gate = self.g_b_proj(ga_o).reshape(B, S, self.num_heads, self.head_dim)
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return self.o_proj(out)


class Glm5NextIndexer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.index_kpool = args.index_kpool
        self.index_kpool_always_select_tail = args.index_kpool_always_select_tail
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.index_kpool_compress_ape = mx.zeros((self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, self.dim))

    def _pooled_states(self, keys, gate_scores, valid):
        B, S, hd = keys.shape
        kp = self.index_kpool
        P = (S + kp - 1) // kp
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(S)
        )
        pool_offsets = mx.arange(P * kp).reshape(1, P, kp)
        pool_indices = first_key[:, None, None] + pool_offsets
        safe = mx.clip(pool_indices, 0, S - 1)
        flat = safe.reshape(B, P * kp)
        idxC = mx.broadcast_to(flat[..., None], (B, P * kp, hd))
        grouped_keys = mx.take_along_axis(keys, idxC, axis=1).reshape(B, P, kp, hd)
        grouped_gate = mx.take_along_axis(gate_scores, idxC, axis=1).reshape(
            B, P, kp, hd
        )
        grouped_valid = (
            mx.take_along_axis(valid.astype(mx.int32), flat, axis=1).reshape(B, P, kp)
            > 0
        )
        grouped_valid = grouped_valid & (pool_indices < S)
        pool_valid = mx.all(grouped_valid, axis=-1)
        pool_indices = mx.where(grouped_valid, pool_indices, -1)
        logits = grouped_gate + self.index_kpool_compress_ape[None, None]
        logits = mx.where(grouped_valid[..., None], logits, -1e30)
        probs = mx.softmax(logits, axis=2)
        probs = mx.where(mx.isnan(probs), 0.0, probs)
        pool_keys = mx.sum(probs * grouped_keys, axis=2)
        return pool_keys, pool_indices, pool_valid

    def _visible_tail(self, visible, valid):
        B, S, Kv = visible.shape
        kp = self.index_kpool
        mtw = kp - 1
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(Kv)
        )
        visible_count = mx.sum(visible.astype(mx.int32), axis=-1)
        tail_count = visible_count - (visible_count // kp) * kp
        tail_offsets = mx.arange(mtw)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & (
            tail_indices < Kv
        )
        kv_idx = mx.clip(tail_indices, 0, Kv - 1)
        tail_vis = mx.take_along_axis(visible, kv_idx, axis=-1)
        tail_indices = mx.where(tail_valid & tail_vis, tail_indices, -1)
        return tail_indices

    def __call__(self, x, qr, mask, cache=None):
        B, S, _ = x.shape
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).reshape(B, S, self.head_dim)
        gate_scores = x @ self.index_kpool_compress_gate.swapaxes(-1, -2)

        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (B, S):
            valid_cur = mask
        else:
            valid_cur = mx.ones((B, S), dtype=mx.bool_)

        packed = mx.concatenate(
            [k, gate_scores, valid_cur.astype(k.dtype)[..., None]], axis=-1
        )
        if cache is not None:
            keys, _ = cache.update_and_fetch(packed[:, None], mx.zeros((B, 1, S, 0)))
            packed_full = keys[:, 0]
        else:
            packed_full = packed
        T = packed_full.shape[1]
        if getattr(self, "bypass_short", True) and T <= self.index_topk:
            return None
        k_full, gate_full, valid_ch = mx.split(
            packed_full, [self.head_dim, 2 * self.head_dim], axis=-1
        )
        valid = valid_ch[..., 0] > 0

        offset = T - S
        kv_len = T
        kv_pos = mx.arange(T)

        if (
            S == 1
            and cache is not None
            and getattr(cache, "_pool", None) is not None
            and getattr(cache, "_no_pad", False)
            and cache._pool[0].shape[0] == B
        ):
            ck, ci, cv, t_prev = cache._pool
            n_stable = t_prev // self.index_kpool
            s0 = n_stable * self.index_kpool
            pk_s, pi_s, pv_s = self._pooled_states(
                k_full[:, s0:], gate_full[:, s0:], valid[:, s0:]
            )
            pi_s = mx.where(pi_s >= 0, pi_s + s0, -1)
            pool_keys = mx.concatenate([ck[:, :n_stable], pk_s], axis=1)
            pool_indices = mx.concatenate([ci[:, :n_stable], pi_s], axis=1)
            pool_valid = mx.concatenate([cv[:, :n_stable], pv_s], axis=1)
        else:
            pool_keys, pool_indices, pool_valid = self._pooled_states(
                k_full, gate_full, valid
            )
            if cache is not None:
                cache._no_pad = bool(mx.all(valid))
        if cache is not None:
            cache._pool = (pool_keys, pool_indices, pool_valid, T)
        P = pool_keys.shape[1]
        select_k = min(self.index_topk // self.index_kpool, P)
        pool_end = mx.clip(pool_indices[..., -1], 0, kv_len - 1)
        pool_keys_t = pool_keys[:, None].swapaxes(-1, -2)
        tail_on = self.index_kpool_always_select_tail and self.index_kpool > 1
        output_width = self.index_topk + (self.index_kpool - 1 if tail_on else 0)

        chunk = 512 if S > 512 else S
        out = []
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cs = c1 - c0
            q_pos = offset + mx.arange(c0, c1)
            visible = (kv_pos[None, None, :] <= q_pos[None, :, None]) & valid[
                :, None, :
            ]
            scores = q[:, c0:c1] @ pool_keys_t
            scores = mx.maximum(scores * self.softmax_scale, 0.0)
            weights = self.weights_proj(x[:, c0:c1]) * (self.n_heads**-0.5)
            index_scores = mx.sum(weights[..., None] * scores, axis=2)
            pool_visible = mx.take_along_axis(
                visible, mx.broadcast_to(pool_end[:, None, :], (B, cs, P)), axis=-1
            )
            valid_candidates = pool_visible & pool_valid[:, None]
            index_scores = mx.where(valid_candidates, index_scores, -1e30)
            order = mx.argsort(-index_scores, axis=-1)
            selected = order[..., :select_k]
            selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
            pi = mx.broadcast_to(pool_indices[:, None], (B, cs, P, self.index_kpool))
            sel_exp = mx.broadcast_to(
                selected[..., None], (B, cs, select_k, self.index_kpool)
            )
            selected_indices = mx.take_along_axis(pi, sel_exp, axis=2)
            topk = selected_indices.reshape(B, cs, select_k * self.index_kpool)
            sv = mx.broadcast_to(
                selected_valid[..., None], (B, cs, select_k, self.index_kpool)
            ).reshape(B, cs, select_k * self.index_kpool)
            topk = mx.where(sv, topk, -1)
            if tail_on:
                topk = mx.concatenate(
                    [topk, self._visible_tail(visible, valid)], axis=-1
                )
            if topk.shape[-1] < output_width:
                pad = mx.full(
                    (B, cs, output_width - topk.shape[-1]), -1, dtype=topk.dtype
                )
                topk = mx.concatenate([topk, pad], axis=-1)
            topk = topk[..., :output_width]
            topk = mx.where(valid_cur[:, c0:c1][..., None], topk, -1)
            out.append(topk)
        topk = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
        return topk[:, None].astype(mx.int32)


class Glm5NextSparseAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.use_nope = config.mla_use_nope or config.qk_rope_head_dim == 0
        if not self.use_nope:
            raise NotImplementedError(
                "glm5_next implements NoPE MLA only; qk_rope_head_dim>0 with "
                "mla_use_nope=False is not supported."
            )
        self.q_head_dim = config.qk_nope_head_dim
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.indexer = Glm5NextIndexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, _ = cache[0].update_and_fetch(kv_latent, kv_latent)
        else:
            cache = [None] * 2

        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        if 1 < L <= 8:
            if topk_indices is None:
                topk_indices = self._dense_verify_indices(
                    B, L, kv_latent.shape[2], mask
                )
            if topk_indices is not None:
                if (
                    cache is not None
                    and cache[0] is not None
                    and cache[1] is not None
                    and cache[1].keys is not None
                ):
                    cache[0].keys = mx.depends(
                        cache[0].keys, (cache[1].keys, cache[1].values)
                    )
                return self._gathered_verify_attention(q, kv_latent, topk_indices)
        attn_mask = mask
        if topk_indices is not None:
            Kv = kv_latent.shape[2]
            valid_sel = topk_indices >= 0
            if L == 1:
                clamped = mx.clip(topk_indices[:, :, 0, :], 0, Kv - 1)
                idx = clamped[..., None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                sel_mask = valid_sel[:, :, 0, :][:, :, None, :]
                if mask is not None and mask.dtype == mx.bool_:
                    mkeys = mask.reshape(B, -1, Kv)[:, 0, :]
                    gathered = mx.take_along_axis(
                        mx.broadcast_to(mkeys[:, None, :], (B, clamped.shape[1], Kv)),
                        clamped,
                        axis=-1,
                    )
                    sel_mask = sel_mask & gathered[:, :, None, :]
                attn_mask = sel_mask
            else:
                shape = list(topk_indices.shape)
                shape[-1] = Kv + 1
                safe_idx = mx.where(valid_sel, topk_indices, Kv)
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, safe_idx, mx.array(True), axis=-1
                )
                sparse_mask = sparse_mask[..., :Kv]
                if mask is not None and mask.dtype == mx.bool_:
                    sparse_mask = sparse_mask & mask
                attn_mask = sparse_mask

        if (
            cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and cache[1].keys is not None
        ):
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if L == 1:
            q = self.embed_q(q)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=attn_mask
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    @staticmethod
    def _dense_verify_indices(B, L, kv_length, mask):
        positions = kv_length - L + mx.arange(L)
        indices = mx.broadcast_to(mx.arange(kv_length), (B, L, kv_length))
        valid = indices <= positions[None, :, None]
        if mask is not None:
            if mask.dtype != mx.bool_:
                return None
            if mask.ndim == 4:
                mask = mask[:, 0]
            elif mask.ndim == 2:
                if mask.shape[0] == B:
                    mask = mask[:, None]
                else:
                    mask = mask[None]
            if mask.ndim != 3 or mask.shape[-1] != kv_length:
                return None
            valid = valid & mx.broadcast_to(mask, (B, L, kv_length))
        return mx.where(valid, indices, -1)[:, None]

    def _gathered_verify_attention(self, q, kv_latent, topk_indices):
        B, H, L, _ = q.shape
        Kv = kv_latent.shape[2]
        dim = kv_latent.shape[-1]
        sel = topk_indices[:, 0, :, :]
        topk = sel.shape[-1]
        clamped = mx.clip(sel, 0, Kv - 1)
        kv_g = mx.take_along_axis(
            mx.broadcast_to(kv_latent, (B, L, Kv, dim)),
            mx.broadcast_to(clamped[..., None], (B, L, topk, dim)),
            axis=2,
        )
        q_e = self.embed_q(q)
        q_bl = q_e.transpose(0, 2, 1, 3).reshape(B * L, H, 1, dim)
        kv_bl = kv_g.reshape(B * L, 1, topk, dim)
        valid = (sel >= 0).reshape(B * L, 1, 1, topk)
        attn = scaled_dot_product_attention(
            q_bl, kv_bl, kv_bl, cache=None, scale=self.scale, mask=valid
        )
        attn = attn.reshape(B, L, H, dim).transpose(0, 2, 1, 3)
        out = self.unembed_out(attn).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class Glm5NextClampedSwiGLU(nn.Module):
    def __init__(self, limit: Optional[float]):
        super().__init__()
        self.limit = limit

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        if self.limit is not None:
            x_gate = mx.clip(x_gate, a_min=None, a_max=self.limit)
            x_up = mx.clip(x_up, a_min=-self.limit, a_max=self.limit)
        return nn.silu(x_gate) * x_up


class Glm5NextMLP(DeepseekMLP):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__(
            config, hidden_size=hidden_size, intermediate_size=intermediate_size
        )
        self.limit = config.swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.limit is not None:
            gate = mx.clip(gate, a_min=None, a_max=self.limit)
            up = mx.clip(up, a_min=-self.limit, a_max=self.limit)
        return self.down_proj(nn.silu(gate) * up)


class Glm5NextMoEGate(MoEGate):
    def __call__(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm5NextMoE(DeepseekV32MoE):
    def __init__(self, config):
        super().__init__(config)
        self.switch_mlp.activation = Glm5NextClampedSwiGLU(config.swiglu_limit)
        self.gate = Glm5NextMoEGate(config)
        if config.n_shared_experts is not None:
            inter = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm5NextMLP(config, intermediate_size=inter)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.is_linear = layer_type == "linear_attention"
        if self.is_linear:
            self.self_attn = Glm5NextLinearAttention(config)
        else:
            self.self_attn = Glm5NextSparseAttention(config)

        is_sparse = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        self.mlp = Glm5NextMoE(config) if is_sparse else Glm5NextMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.compile_ffn = True
        self._ffn_c = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        gdn_sink: Optional[list] = None,
    ) -> mx.array:
        residual = x
        xc, post, comb = self.attn_hc(x)
        if self.is_linear:
            r = self.self_attn(self.input_layernorm(xc), mask, cache, gdn_sink=gdn_sink)
        else:
            r = self.self_attn(self.input_layernorm(xc), mask, cache)
        x = hc_expand(r, residual, post, comb)
        if self.compile_ffn and x.shape[0] == 1 and x.shape[1] <= 8:
            if self._ffn_c is None:
                self._ffn_c = mx.compile(self._ffn_block)
            return self._ffn_c(x)
        return self._ffn_block(x)

    def _ffn_block(self, x: mx.array) -> mx.array:
        residual = x
        xc, post, comb = self.ffn_hc(x)
        m = self.mlp(self.post_attention_layernorm(xc))
        return hc_expand(m, residual, post, comb)


class Glm5NextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm5NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), 0)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), 0)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        gdn_sink: Optional[list] = None,
        hidden_sink: Optional[list] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        fa_mask = create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c, gdn_sink=gdn_sink)

        h = h.mean(axis=2)
        if hidden_sink is not None:
            hidden_sink.append(h)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        gdn_sink: Optional[list] = [] if capture_layer_ids is not None else None
        hidden_sink: Optional[list] = [] if return_hidden else None

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            gdn_sink=gdn_sink,
            hidden_sink=hidden_sink,
        )
        nlk = kwargs.get("num_logits_to_keep", 0)
        logits = None
        if not skip_logits:
            logits = self._logits(out[:, -nlk:, :] if nlk else out)

        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def _logits(self, normed_hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed_hidden)
        return self.lm_head(normed_hidden)

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return verify_logits(self, self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        out = _SPECULATIVE_VERIFIER(
            self, inputs, cache=cache, capture_layer_ids=[], skip_logits=True
        )
        return out.hidden_states[-1], out.shared_kv_states, out.gdn_states

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        out = _SPECULATIVE_VERIFIER(self, inputs, cache=cache, capture_layer_ids=[])
        return (
            out.hidden_states[-1],
            out.shared_kv_states,
            out.gdn_states,
            sampler(out.logits),
        )

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

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: list,
        accepted,
        block_size: int,
    ) -> int:
        if isinstance(accepted, int):
            accepted_list = [int(accepted)]
        elif isinstance(accepted, mx.array):
            accepted_list = [int(x) for x in accepted.reshape(-1).tolist()]
        else:
            accepted_list = [int(x) for x in accepted]
        max_a = max(accepted_list)
        trim = block_size - (max_a + 1)
        is_batch = len(accepted_list) > 1

        gdn_idx = 0
        for c in caches:
            if c is None:
                continue
            if isinstance(c, ArraysCache):
                q_, k_, v_, a_, b_, A_log_, dt_bias_, init_state, conv_input, K, lb = (
                    gdn_states[gdn_idx]
                )
                gdn_idx += 1
                n = max_a + 1
                _, state_n = gated_delta_update(
                    q_[:, :n],
                    k_[:, :n],
                    v_[:, :n],
                    a_[:, :n],
                    b_[:, :n],
                    A_log_,
                    dt_bias_,
                    state=init_state,
                    lower_bound=lb,
                )
                c[1] = state_n
                c[0] = conv_input[:, n : n + K - 1]
            else:
                if trim > 0 and c.is_trimmable():
                    c.trim(trim)
                indexer_cache = c[1]
                pool = getattr(indexer_cache, "_pool", None)
                if pool is not None:
                    pk, pi, pv, t = pool
                    t2 = t - trim
                    if t2 <= 0:
                        indexer_cache._pool = None
                    else:
                        n_stable = t2 // self.args.index_kpool
                        indexer_cache._pool = (
                            pk[:, :n_stable],
                            pi[:, :n_stable],
                            pv[:, :n_stable],
                            t2,
                        )
        return max_a

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        weights = DSV32Model.sanitize(self, weights)

        remapped = {}
        conv_parts = {}
        fg_parts = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
        for k, v in weights.items():
            nk = k.replace(".hc_attn_", ".attn_hc.").replace(".hc_ffn_", ".ffn_hc.")

            fused = False
            for part in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
                suffix = ".self_attn." + part
                if nk.endswith(suffix):
                    prefix = nk[: -len(part)]
                    conv_parts.setdefault(prefix, {})[part[0]] = v
                    fused = True
                    break
            if fused:
                continue

            for p in fg_parts:
                suffix = ".self_attn." + p
                if nk.endswith(suffix):
                    nk = nk[: -len(p)] + "forget_gate." + p
                    break

            remapped[nk] = v

        for prefix, parts in conv_parts.items():
            if all(c in parts for c in ("q", "k", "v")):
                remapped[prefix + "conv1d.weight"] = mx.concatenate(
                    [parts["q"], parts["k"], parts["v"]], axis=0
                )
            else:
                for c, w in parts.items():
                    remapped[prefix + c + "_conv1d.weight"] = w

        weights = remapped
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        for k, v in list(weights.items()):
            keep = (
                ".attn_hc." in k
                or ".ffn_hc." in k
                or k.endswith("A_log")
                or k.endswith("dt_bias")
            )
            if keep and mx.issubdtype(v.dtype, mx.floating) and v.dtype != mx.float32:
                weights[k] = v.astype(mx.float32)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            if "e_score_correction_bias" in k:
                return False
            if ".attn_hc." in k or ".ffn_hc." in k:
                return False
            if k.endswith("A_log") or k.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if (
                path.endswith("mlp.gate")
                or "e_score_correction_bias" in path
                or ".indexer" in path
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
