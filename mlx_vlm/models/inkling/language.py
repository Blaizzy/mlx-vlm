from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, KVCache
from ..mlp import SwiGLUMLP
from ..switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
from .config import TextConfig as ModelConfig


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_cache_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_cache_tree(v) for k, v in value.items()}
    return value


_CACHE_DICT_STATE = object()


def _subcaches(cache):
    return getattr(cache, "caches", None) or (cache,)


def _snapshot_cache_state(caches):
    """Copy cache state for speculative restore-and-replay."""
    snapshot = []
    for cache in caches:
        if cache is None:
            snapshot.append(None)
            continue
        states = []
        for subcache in _subcaches(cache):
            if (
                isinstance(subcache, ArraysCache)
                or getattr(subcache, "keys", False) is None
            ):
                states.append((_CACHE_DICT_STATE, _clone_cache_tree(vars(subcache))))
            else:
                states.append(_clone_cache_tree(subcache.state))
        snapshot.append(states)
    arrays = [v for _, v in tree_flatten(snapshot) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)
    return snapshot


def _restore_cache_state(caches, snapshot):
    for cache, states in zip(caches, snapshot):
        if cache is None or states is None:
            continue
        for subcache, state in zip(_subcaches(cache), states):
            if isinstance(state, tuple) and state and state[0] is _CACHE_DICT_STATE:
                subcache.__dict__.clear()
                subcache.__dict__.update(_clone_cache_tree(state[1]))
            else:
                subcache.state = _clone_cache_tree(state)


_MASK_SRC = r"""
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if (i >= LQ || j >= S || bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = (int(i) + int(Q_OFF)) - int(j);   // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        uint rbase = ((b * LQ + i) * H + h) * D_REL;
        uint pcol = (uint)dist;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[((b * H + h) * LQ + i) * S + j] = val;
"""
_mask_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask",
    input_names=["rel", "proj"],
    output_names=["out"],
    source=_MASK_SRC,
)


_MASK_V2_SRC = r"""
    // Shape-generic variant of the banded mask: every runtime dimension comes
    // from injected shapes (B/LQ/H from rel, S from the unread shape-carrier
    // input), and the query offset is S - LQ. Only the per-layer constants
    // (dtype, band geometry) are template args, so exactly one pipeline is
    // compiled per layer kind instead of one per (LQ, S) pair.
    const int B  = rel_shape[0];
    const int LQ = rel_shape[1];
    const int H  = rel_shape[2];
    const int S  = kshape_shape[2];
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    if ((int)i >= LQ || (int)j >= S || (int)bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = ((int)i + (S - LQ)) - (int)j;     // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        const size_t rbase = (size_t)b * rel_strides[0]
            + (size_t)i * rel_strides[1] + (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]]
                 * (float)proj[(size_t)d * proj_strides[0]
                               + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[(((size_t)b * H + h) * LQ + i) * S + j] = val;
"""
_mask_v2_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask_v2",
    input_names=["rel", "proj", "kshape"],
    output_names=["out"],
    source=_MASK_V2_SRC,
    ensure_row_contiguous=False,
)


def _rup(a, m):
    return ((a + m - 1) // m) * m


def banded_additive_mask(
    rel, proj, q_offset, S, sliding, rel_extent, left_padding=None, shape_ref=None
):
    """rel: [B, LQ, H, d_rel]; proj: [d_rel, rel_extent] -> additive mask [B, H, LQ, S]."""
    B, LQ, H, d_rel = rel.shape
    dtype = rel.dtype
    q_offset = int(q_offset)
    S = int(S)
    sliding = int(sliding)
    rel_extent = int(rel_extent)
    if (
        shape_ref is not None
        and shape_ref.ndim >= 3
        and shape_ref.shape[2] == S
        and q_offset == S - LQ
        and mx.default_device() == mx.gpu
    ):
        mask = _mask_v2_kernel(
            inputs=[rel, proj, shape_ref],
            template=[
                ("T", dtype),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    elif mx.default_device() == mx.gpu:
        mask = _mask_kernel(
            inputs=[rel, proj],
            template=[
                ("T", dtype),
                ("B", B),
                ("H", H),
                ("LQ", LQ),
                ("S", S),
                ("Q_OFF", q_offset),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    else:
        rl = (rel @ proj).transpose(0, 2, 1, 3)
        qp = mx.arange(LQ) + q_offset
        kp = mx.arange(S)
        dist = qp[:, None] - kp[None, :]
        gidx = mx.broadcast_to(
            mx.clip(dist, 0, rel_extent - 1)[None, None], (B, H, LQ, S)
        )
        pb = mx.take_along_axis(rl, gidx, axis=-1)
        pb = mx.where((dist >= rel_extent)[None, None], mx.array(0.0, dtype), pb)
        masked = dist < 0
        if sliding > 0:
            masked = masked | (dist >= sliding)
        mask = mx.where(masked[None, None], mx.array(-1e30, dtype), pb)

    if left_padding is not None:
        valid = mx.arange(S)[None, :] >= left_padding[:, None]
        mask = mx.where(valid[:, None, None, :], mask, mx.array(-1e30, dtype))
    return mask.astype(dtype)


def _cache_padding_mask(cache, length):
    if cache is None:
        return None
    positions = mx.arange(length)[None, :]
    mask = None
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is not None:
        mask = positions >= left_padding[:, None]
    lengths = getattr(cache, "lengths", None)
    if lengths is not None:
        length_mask = positions < lengths[:, None]
        mask = length_mask if mask is None else mask & length_mask
    return mask


def _next_conv_state(state, inputs, mask):
    state_size = state.shape[1]
    if state_size == 0:
        return state
    if mask is None:
        return mx.concatenate([state, inputs], axis=1)[:, -state_size:]

    valid_count = mx.sum(mask, axis=1).astype(mx.int32)
    valid_start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(state_size)[None, :] + valid_count[:, None] - state_size
    state_indices = state_size + logical
    input_indices = state_size + valid_start[:, None] + mx.maximum(logical, 0)
    indices = mx.where(logical >= 0, input_indices, state_indices)
    combined = mx.concatenate([state, inputs], axis=1)
    indices = mx.broadcast_to(indices[..., None], (*indices.shape, inputs.shape[-1]))
    return mx.take_along_axis(combined, indices, axis=1)


_SCONV_SRC = r"""
    uint c = thread_position_in_grid.x;   // channel
    uint b = thread_position_in_grid.y;   // batch row
    if (c >= C || b >= B) return;
    float w0 = (float)w[c * K + 0];
    float w1 = (float)w[c * K + 1];
    float w2 = (float)w[c * K + 2];
    float w3 = (float)w[c * K + 3];
    // Virtual padded input xp = [state (K-1 rows, fp32); x (L rows, T)].
    for (uint i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int r = (int)(i + k) - (int)(K - 1);  // row into x; negative -> state
            float v = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
            float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
            acc += wk * v;
        }
        // Match the unfused path's rounding: the conv emits bf16 (rounded)
        // before the fp32 residual add; the layer residual is a second bf16
        // add on top (as the decoder layer's x + sconv(r) was).
        float conv_r = (float)((T)acc);
        T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
        if (HAS_RES) {
            out[(b * L + i) * C + c] =
                (T)((float)inner + (float)res[(b * L + i) * C + c]);
        } else {
            out[(b * L + i) * C + c] = inner;
        }
    }
    for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[(b * (K - 1) + s) * C + c] = (r < 0)
            ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
            : (float)x[(b * L + (uint)r) * C + c];
    }
"""
_sconv_kernel = mx.fast.metal_kernel(
    name="inkling_sconv_decode",
    input_names=["x", "state", "w", "res"],
    output_names=["out", "nstate"],
    source=_SCONV_SRC,
)


class InklingShortConvolution(nn.Module):
    """Depthwise causal 1-D conv over the previous ``kernel_size - 1`` states, plus a
    residual add. Kept in fp32 for stability (matches the reference). ``conv_idx`` selects
    this conv's slot in the layer's shared conv cache."""

    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        residual: Optional[mx.array] = None,
    ):
        dt = x.dtype
        K = self.kernel_size
        if (
            cache is not None
            and mask is None
            and K == 4
            and x.shape[1] <= 8
            and mx.default_device() == mx.gpu
        ):
            B, L, C = x.shape
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((B, K - 1, C), dtype=mx.float32)
            out, nstate = _sconv_kernel(
                inputs=[
                    x,
                    state,
                    self.conv.weight.reshape(-1),
                    residual if residual is not None else x,
                ],
                template=[
                    ("T", dt),
                    ("B", B),
                    ("L", L),
                    ("C", C),
                    ("K", K),
                    ("HAS_RES", residual is not None),
                ],
                grid=(_rup(C, 32), B, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, L, C), (B, K - 1, C)],
                output_dtypes=[dt, mx.float32],
            )
            cache[self.conv_idx] = nstate
            return out
        xf = x.astype(mx.float32)
        res = xf
        if mask is not None:
            xf = mx.where(mask[..., None], xf, 0)
        if cache is not None:
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((xf.shape[0], K - 1, xf.shape[-1]), dtype=xf.dtype)
            xp = mx.concatenate([state, xf], axis=1)
            cache[self.conv_idx] = _next_conv_state(state, xf, mask)
        else:
            xp = mx.pad(xf, [(0, 0), (K - 1, 0), (0, 0)])
        out = self.conv(xp.astype(self.conv.weight.dtype)).astype(mx.float32)
        out = (out + res).astype(dt)
        return out if residual is None else residual + out


# Escape hatch: skip the sliding-layer out-of-window K/V slicing.
_SLIDING_KV_SLICE = True


class InklingAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = (
            config.sliding_window_size if self.is_sliding else config.rel_extent
        )
        self.d_rel = config.d_rel
        self.scale = 1.0 / self.head_dim
        self.log_floor = None if self.is_sliding else config.log_scaling_n_floor
        self.log_alpha = config.log_scaling_alpha

        # q/k/v/r share the input row; their weights are stacked at load
        # (see fuse_qkvr) so decode does one matmul instead of four.
        self.qkvr_dims = (
            self.n_heads * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_heads * self.d_rel,
        )
        self.qkvr_proj = nn.Linear(config.hidden_size, sum(self.qkvr_dims), bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.k_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=0
        )
        self.v_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=1
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_proj = mx.zeros((self.d_rel, self.rel_extent))

    def __call__(self, x, cache=None, conv_mask=None):
        B, L, _ = x.shape
        kv = cache[0] if cache is not None else None
        conv = cache[1] if cache is not None else None

        qkvr = self.qkvr_proj(x)
        dq, dk, dv, _ = self.qkvr_dims
        k = self.k_sconv(qkvr[..., dq : dq + dk], cache=conv, mask=conv_mask)
        v = self.v_sconv(qkvr[..., dq + dk : dq + dk + dv], cache=conv, mask=conv_mask)

        k = self.k_norm(k.reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
        S = k.shape[2]

        q = qkvr[..., :dq]
        r = qkvr[..., dq + dk + dv :].reshape(B, L, self.n_heads, self.d_rel)
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        left_padding = getattr(kv, "left_padding", None)
        if _SLIDING_KV_SLICE and self.sliding > 0 and S > L + self.sliding - 1:
            # No query row can reach keys older than its window; slice them
            # off before the mask and SDPA (they were -1e30 masked anyway).
            # The mask keeps the right positions: its offset is S_eff - L,
            # and any left padding shifts with the slice.
            j0 = S - L - (self.sliding - 1)
            k = k[:, :, j0:, :]
            v = v[:, :, j0:, :]
            S = k.shape[2]
            if left_padding is not None:
                left_padding = mx.maximum(left_padding - j0, 0)
        offset = S - L

        mask = banded_additive_mask(
            r,
            self.rel_proj.astype(x.dtype),
            offset,
            S,
            self.sliding,
            self.rel_extent,
            left_padding=left_padding,
            shape_ref=k,
        )
        if self.log_floor is not None:
            qpos = (mx.arange(L) + offset + 1).astype(mx.float32)
            tau = 1.0 + self.log_alpha * mx.log(mx.maximum(qpos / self.log_floor, 1.0))
            tau = tau.reshape(1, 1, L, 1).astype(x.dtype)
            q = q * tau
            mask = mx.where(mask > -1e29, mask * tau, mask)

        out = scaled_dot_product_attention(
            q, k, v, cache=None, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class InklingDenseMLP(SwiGLUMLP):
    """Dense SwiGLU MLP (shared ``SwiGLUMLP``) with a learned output scale."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.hidden_size, config.dense_intermediate_size)
        self.global_scale = mx.ones((1,))

    def __call__(self, x):
        return super().__call__(x) * self.global_scale


class InklingSwitchGLU(SwitchGLU):
    def __init__(self, input_dims, hidden_dims, num_experts, **kwargs):
        super().__init__(input_dims, hidden_dims, num_experts, **kwargs)
        self.gate_scale = mx.ones((num_experts,))  # s13 (fused gate/up scale2)
        self.out_scale = mx.ones((num_experts,))  # s13 * s2
        self._scales_trivial = None

    def _per_expert(self, scale, idx, like):
        s = scale[idx].astype(like.dtype)
        return s.reshape(s.shape + (1,) * (like.ndim - s.ndim))

    def scales_trivial(self):
        # Non-NVFP4 checkpoints carry all-ones expert scales; checked once,
        # after load.
        if self._scales_trivial is None:
            self._scales_trivial = bool(
                (mx.all(self.gate_scale == 1) & mx.all(self.out_scale == 1)).item()
            )
        return self._scales_trivial

    def __call__(self, x, indices) -> mx.array:
        # All-ones expert scales: skip the two gather+mul chains entirely.
        if self.scales_trivial():
            return super().__call__(x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x_gate = x_gate * self._per_expert(self.gate_scale, idx, x_gate)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        x = x * self._per_expert(self.out_scale, idx, x)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


_ROUTE_SRC = r"""
    // One simdgroup (32 lanes) per token; each lane owns ceil(R/32) experts
    // (strided by 32 so logit reads coalesce). Top-K via K rounds of
    // simd_max; ties resolved to the lowest lane (deterministic).
    uint lane = thread_position_in_grid.x;   // 0..31, simd lane
    uint n = thread_position_in_grid.y;      // token
    if (n >= N) return;
    const device T* lg = logits + (size_t)n * (R + SH);
    float ws = wscale[0];
    constexpr int PER = (R + 31) / 32;
    float sc[PER];
    float tl_[PER];
    bool taken[PER];
    for (int t = 0; t < PER; ++t) {
        uint j = lane + (uint)t * 32u;
        taken[t] = false;
        if (j < R) {
            float l = (float)lg[j];
            tl_[t] = l;
            sc[t] = 1.0f / (1.0f + metal::exp(-l)) + (float)corr[j];
        } else {
            sc[t] = -INFINITY;
        }
    }
    uint bidx[K];
    float btl[K];
    for (uint kk = 0; kk < K; ++kk) {
        float lb = -INFINITY; int lt = -1;
        for (int t = 0; t < PER; ++t)
            if (!taken[t] && sc[t] > lb) { lb = sc[t]; lt = t; }
        float gb = simd_max(lb);
        ushort wl = (ushort)simd_min(lb == gb ? lane : 32u);
        uint wj = simd_shuffle(lt >= 0 ? lane + (uint)lt * 32u : 0u, wl);
        float wtl = simd_shuffle(lt >= 0 ? tl_[lt] : 0.0f, wl);
        if (lane == (uint)wl && lt >= 0 && sc[lt] == gb) taken[lt] = true;
        bidx[kk] = wj; btl[kk] = wtl;
    }
    // Routing weights: softmax over logsigmoid of the K routed + SH shared
    // logits, times route_scale * global_scale (folded into ws). Computed
    // redundantly on every lane (cheap; K + SH values).
    float lp[K + SH];
    float m = -INFINITY;
    for (uint t = 0; t < K + SH; ++t) {
        float tv = (t < K) ? btl[t] : (float)lg[R + (t - K)];
        float a = -tv;
        float lad = metal::max(a, 0.0f)
                  + metal::log(1.0f + metal::exp(-metal::fabs(a)));
        lp[t] = -lad;
        m = metal::max(m, lp[t]);
    }
    float se = 0.0f;
    for (uint t = 0; t < K + SH; ++t) se += metal::exp(lp[t] - m);
    float lse = m + metal::log(se);
    if (lane < K) {
        idx[(size_t)n * K + lane] = bidx[lane];
        wk[(size_t)n * K + lane] = (T)(metal::exp(lp[lane] - lse) * ws);
    }
    T gv[SH];
    for (uint s = 0; s < SH; ++s) gv[s] = (T)(metal::exp(lp[K + s] - lse) * ws);
    device T* gp = gamma + (size_t)n * SH * I;
    for (uint t = lane; t < SH * I; t += 32u) gp[t] = gv[t / I];
"""
_route_kernel = mx.fast.metal_kernel(
    name="inkling_moe_route",
    input_names=["logits", "corr", "wscale"],
    output_names=["idx", "wk", "gamma"],
    source=_ROUTE_SRC,
)


# Escape hatch: use gather_qmm for the routed down projection at decode.
_DOWN_COMBINE = True

_DOWN_COMBINE_SRC = r"""
    // Weighted routed-expert down-projection for decode: gather_qmm's
    // vector-per-expert mode runs at ~200 GB/s on the [2048 -> 4096] down
    // shape (vs ~750 broadcast), so this kernel dequantizes q4/g64 rows
    // directly: one threadgroup per output row, one simdgroup per selected
    // expert, one quant group per lane; the top-k weighted sum over experts
    // is folded in, so the [N, K, out] intermediate never exists.
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;   // expert slot (8 sgs, K used)
    uint row  = threadgroup_position_in_grid.y;   // output row [0, OUT)
    uint n    = threadgroup_position_in_grid.z;   // token
    threadgroup float partial[8];
    if (sg < K) {
        uint e = idx[(size_t)n * K + sg];
        const device uint* wr = wq + ((size_t)e * OUT + row) * (IN / 8u);
        const device T* sr = sc + ((size_t)e * OUT + row) * GROUPS;
        const device T* br = bi + ((size_t)e * OUT + row) * GROUPS;
        const device T* xr = xin + ((size_t)n * K + sg) * IN;
        float s = (float)sr[lane];
        float b = (float)br[lane];
        float accq = 0.0f, accx = 0.0f;
        uint base = lane * 8u;      // 8 uint32 = 64 packed q4 values
        uint xbase = lane * 64u;
        for (uint u = 0; u < 8u; ++u) {
            uint w8 = wr[base + u];
            for (uint t = 0; t < 8u; ++t) {
                float xv = (float)xr[xbase + u * 8u + t];
                accq += (float)((w8 >> (4u * t)) & 0xFu) * xv;
                accx += xv;
            }
        }
        float dot = simd_sum(s * accq + b * accx);
        if (lane == 0) {
            // match the unfused chain: down output rounds to T, then the
            // per-expert weight multiply rounds again before the sum
            float dv = (float)((T)dot);
            partial[sg] = (float)((T)(dv * (float)wk[(size_t)n * K + sg]));
        }
    } else if (lane == 0) {
        partial[sg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float tot = 0.0f;
        for (uint t = 0; t < 8u; ++t) tot += partial[t];
        out[(size_t)n * OUT + row] = (T)tot;
    }
"""
_down_combine_kernel = mx.fast.metal_kernel(
    name="inkling_moe_down_combine",
    input_names=["xin", "wq", "sc", "bi", "idx", "wk"],
    output_names=["out"],
    source=_DOWN_COMBINE_SRC,
)


@partial(mx.compile, shapeless=True)
def _swiglu_scaled(gate, up, s):
    return nn.silu(gate) * up * s


class InklingSharedExpertsDense(nn.Module):
    """The ``n_shared`` always-on experts as one dense SwiGLU: expert weights are
    concatenated at load (see ``shared_experts_to_dense``) so the fixed-index
    gather_qmm path becomes three plain matmuls. Per-token expert weights arrive
    pre-broadcast over the expert-major intermediate (``gamma``) and are applied
    before down_proj, which distributes over the concatenated experts exactly."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.up_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.down_proj = nn.Linear(num_experts * hidden_dims, input_dims, bias=False)

    def __call__(self, x, gamma):
        return self.down_proj(_swiglu_scaled(self.gate_proj(x), self.up_proj(x), gamma))


def fuse_qkvr(weights):
    """Stack per-layer q/k/v/r projection tensors (rows, plus scales/biases for
    quantized checkpoints) into the single ``qkvr_proj``. Row-concat of
    quantized matrices is exact: each output row keeps its own groups."""
    out = dict(weights)
    prefixes = {
        k[: -len("q_proj.weight")]
        for k in weights
        if k.endswith(".self_attn.q_proj.weight")
    }
    for p in prefixes:
        for leaf in ("weight", "scales", "biases"):
            parts = [out.pop(f"{p}{n}_proj.{leaf}", None) for n in "qkvr"]
            if all(v is not None for v in parts):
                out[f"{p}qkvr_proj.{leaf}"] = mx.concatenate(parts, axis=0)
            elif any(v is not None for v in parts):
                raise ValueError(f"partial q/k/v/r {leaf} set under {p}")
    return out


def shared_experts_to_dense(weights):
    """Remap SwitchGLU-shaped shared-expert tensors ``[E, out, in]`` (bf16 or
    quantized triplets) to the dense concatenated layout of
    ``InklingSharedExpertsDense``. Expert-major on the intermediate axis, so
    gate/up stack experts along rows and down stacks along input columns."""
    out = {}
    for k, v in weights.items():
        if ".shared_experts." in k and isinstance(v, mx.array) and v.ndim == 3:
            if ".down_proj." in k:
                v = v.transpose(1, 0, 2).reshape(v.shape[1], -1)
            else:
                v = v.reshape(-1, v.shape[2])
        out[k] = v
    return out


class InklingSparseMoE(nn.Module):
    """Sigmoid-gated fine-grained MoE: top-k routed experts (+ correction-bias selection)
    plus always-on shared experts, weighted by a logsigmoid/logsumexp softmax."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.intermediate_size = config.intermediate_size
        self.gate_weight = mx.zeros((self.n_routed + self.n_shared, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed,))
        self.global_scale = mx.ones((1,))
        self.switch_mlp = InklingSwitchGLU(
            config.hidden_size, config.intermediate_size, self.n_routed
        )
        self.shared_experts = InklingSharedExpertsDense(
            config.hidden_size, config.intermediate_size, self.n_shared
        )
        self._wscale = None

    def _route(self, logits):
        """Expert selection + routing weights. On GPU the whole post-matmul
        chain (sigmoid + bias + top-k + logsigmoid softmax + scaling) is one
        kernel; it also emits the shared-expert weights pre-broadcast over the
        expert-major dense intermediate."""
        N = logits.shape[0]
        if self._wscale is None:
            self._wscale = mx.array(
                [self.route_scale], dtype=mx.float32
            ) * self.global_scale.astype(mx.float32)
        if mx.default_device() == mx.gpu:
            return _route_kernel(
                inputs=[logits, self.e_score_correction_bias, self._wscale],
                template=[
                    ("T", logits.dtype),
                    ("N", N),
                    ("R", self.n_routed),
                    ("SH", self.n_shared),
                    ("K", self.top_k),
                    ("I", self.intermediate_size),
                ],
                grid=(32, N, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (N, self.top_k),
                    (N, self.top_k),
                    (N, self.n_shared * self.intermediate_size),
                ],
                output_dtypes=[mx.uint32, logits.dtype, logits.dtype],
            )
        scores = mx.sigmoid(logits.astype(mx.float32))
        sfc = scores[:, : self.n_routed] + self.e_score_correction_bias
        idx = mx.argpartition(-sfc, self.top_k - 1, axis=-1)[:, : self.top_k]
        routed_logits = logits[:, : self.n_routed]
        shared_logits = logits[:, -self.n_shared :]
        tl = mx.concatenate(
            [mx.take_along_axis(routed_logits, idx, axis=-1), shared_logits], axis=-1
        ).astype(mx.float32)
        lp = -mx.logaddexp(mx.zeros_like(tl), -tl)
        w = (
            mx.exp(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
            * self.route_scale
            * self.global_scale
        )
        topk_w = w[:, : self.top_k].astype(logits.dtype)
        gamma = mx.repeat(
            w[:, -self.n_shared :].astype(logits.dtype),
            self.intermediate_size,
            axis=-1,
        )
        return idx.astype(mx.uint32), topk_w, gamma

    def __call__(self, x):
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        gw = self.gate_weight
        if gw.dtype != x.dtype:
            gw = gw.astype(x.dtype)
        logits = xf @ gw.T
        idx, topk_w, gamma = self._route(logits)
        sm = self.switch_mlp
        dp = sm.down_proj
        if (
            _DOWN_COMBINE
            and xf.shape[0] <= 8
            and mx.default_device() == mx.gpu
            and getattr(dp, "bits", None) == 4
            and getattr(dp, "group_size", None) == 64
            and getattr(dp, "mode", "affine") == "affine"
            and getattr(dp, "biases", None) is not None
            and dp.input_dims == 2048  # kernel maps one 64-wide group per lane
            and dp.scales.dtype == x.dtype
            and sm.scales_trivial()
        ):
            # decode: gather_qmm's vector-per-expert mode is ~3.5x off peak on
            # the down shape; dequantize rows directly and fold the weighted
            # expert sum in.
            xe = mx.expand_dims(xf, (-2, -3))
            act = sm.activation(sm.up_proj(xe, idx), sm.gate_proj(xe, idx))
            yr = _down_combine_kernel(
                inputs=[act, dp.weight, dp.scales, dp.biases, idx, topk_w],
                template=[
                    ("T", x.dtype),
                    ("OUT", dp.output_dims),
                    ("IN", dp.input_dims),
                    ("GROUPS", dp.input_dims // 64),
                    ("K", self.top_k),
                ],
                grid=(256, dp.output_dims, xf.shape[0]),
                threadgroup=(256, 1, 1),
                output_shapes=[(xf.shape[0], dp.output_dims)],
                output_dtypes=[x.dtype],
            )[0]
        else:
            yr = (sm(xf, idx) * topk_w[..., None]).sum(axis=-2)
        ys = self.shared_experts(xf, gamma)
        return (yr + ys).reshape(B, L, D).astype(x.dtype)


class InklingDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = InklingAttention(config, layer_idx)
        self.mlp = (
            InklingDenseMLP(config)
            if config.layer_is_dense(layer_idx)
            else InklingSparseMoE(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=2
        )
        self.mlp_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=3
        )

    def __call__(self, x, cache=None, conv_mask=None):
        conv = cache[1] if cache is not None else None
        if conv_mask is None:
            conv_mask = _cache_padding_mask(conv, x.shape[1])
        r = self.self_attn(self.input_layernorm(x), cache=cache, conv_mask=conv_mask)
        h = self.attn_sconv(r, cache=conv, mask=conv_mask, residual=x)
        r = self.mlp(self.post_attention_layernorm(h))
        out = self.mlp_sconv(r, cache=conv, mask=conv_mask, residual=h)
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class InklingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        self.layers = [
            InklingDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids):
        h = self.embed_tokens(input_ids)
        if self.embed_norm is not None:
            h = self.embed_norm(h)
        return h

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        skip_final_norm: bool = False,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c)
        return h if skip_final_norm else self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = InklingModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits_from_norm(self, h):
        h = h / self.config.logits_mup_width_multiplier
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        uv = self.config.unpadded_vocab_size
        if uv is not None and uv < logits.shape[-1]:
            logits = logits[..., :uv]
        return logits

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
        **kwargs,
    ):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        pre_norm = self.model(inputs, cache, inputs_embeds, skip_final_norm=True)
        logits = (
            None if skip_logits else self._logits_from_norm(self.model.norm(pre_norm))
        )
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[pre_norm] if return_hidden else None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._logits_from_norm(self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> Optional[mx.array]:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        snapshot = _snapshot_cache_state(cache)
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, (snapshot, inputs)

    def rollback_speculative_cache(
        self, caches, gdn_states, accepted, block_size
    ) -> int:
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item()) if accepted.size else 0
        elif not isinstance(accepted, int):
            accepted = max(int(a) for a in accepted)
        snapshot, verify_inputs = gdn_states
        _restore_cache_state(caches, snapshot)
        keep = accepted + 1
        if keep > 0:
            self(verify_inputs[:, :keep], cache=caches, skip_logits=True)
        return accepted

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(4)) for _ in self.model.layers]
