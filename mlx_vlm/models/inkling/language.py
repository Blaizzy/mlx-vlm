from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, KVCache
from ..mlp import SwiGLUMLP
from ..switch_layers import SwitchGLU
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


def _snapshot_cache_state(caches):
    """Deep-copy the full state of every cache so a speculative block can be
    rolled back by replay. Inkling's short-conv slots keep only the last K-1
    inputs and cannot be trimmed, so we restore-and-replay instead."""
    snapshot = [None if c is None else _clone_cache_tree(c.state) for c in caches]
    arrays = [v for _, v in tree_flatten(snapshot) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)
    return snapshot


def _restore_cache_state(caches, snapshot):
    for c, s in zip(caches, snapshot):
        if c is not None and s is not None:
            c.state = _clone_cache_tree(s)


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


def _rup(a, m):
    return ((a + m - 1) // m) * m


def banded_additive_mask(rel, proj, q_offset, S, sliding, rel_extent):
    """rel: [B, LQ, H, d_rel]; proj: [d_rel, rel_extent] -> additive mask [B, H, LQ, S]."""
    B, LQ, H, d_rel = rel.shape
    dtype = rel.dtype
    if mx.default_device() == mx.gpu:
        return _mask_kernel(
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
    rl = (rel @ proj).transpose(0, 2, 1, 3)
    qp = mx.arange(LQ) + q_offset
    kp = mx.arange(S)
    dist = qp[:, None] - kp[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, rel_extent - 1)[None, None], (B, H, LQ, S))
    pb = mx.take_along_axis(rl, gidx, axis=-1)
    pb = mx.where((dist >= rel_extent)[None, None], mx.array(0.0, dtype), pb)
    neg = dist < 0
    if sliding > 0:
        neg = neg | (dist >= sliding)
    return mx.where(neg[None, None], mx.array(-1e30, dtype), pb).astype(dtype)


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

    def __call__(self, x: mx.array, cache=None, mask: Optional[mx.array] = None):
        dt = x.dtype
        xf = x.astype(mx.float32)
        res = xf
        if mask is not None:
            xf = mx.where(mask[..., None], xf, 0)
        K = self.kernel_size
        if cache is not None:
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((xf.shape[0], K - 1, xf.shape[-1]), dtype=xf.dtype)
            xp = mx.concatenate([state, xf], axis=1)
            cache[self.conv_idx] = xp[:, -(K - 1) :, :]
        else:
            xp = mx.pad(xf, [(0, 0), (K - 1, 0), (0, 0)])
        out = self.conv(xp.astype(self.conv.weight.dtype)).astype(mx.float32)
        return (out + res).astype(dt)


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

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv * self.head_dim, bias=False
        )
        self.r_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.d_rel, bias=False
        )
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

        q = self.q_proj(x)
        k = self.k_sconv(self.k_proj(x), cache=conv, mask=conv_mask)
        v = self.v_sconv(self.v_proj(x), cache=conv, mask=conv_mask)
        r = self.r_proj(x).reshape(B, L, self.n_heads, self.d_rel)

        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(k.reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        offset = kv.offset if kv is not None else 0
        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
        S = k.shape[2]

        mask = banded_additive_mask(
            r, self.rel_proj.astype(x.dtype), offset, S, self.sliding, self.rel_extent
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
        super().__init__(config.hidden_size, config.intermediate_size)
        self.global_scale = mx.ones((1,))

    def __call__(self, x):
        return super().__call__(x) * self.global_scale


class InklingSparseMoE(nn.Module):
    """Sigmoid-gated fine-grained MoE: top-k routed experts (+ correction-bias selection)
    plus always-on shared experts, weighted by a logsigmoid/logsumexp softmax."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate_weight = mx.zeros((self.n_routed + self.n_shared, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed,))
        self.global_scale = mx.ones((1,))
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, self.n_routed
        )
        self.shared_experts = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, self.n_shared
        )

    def __call__(self, x):
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        logits = xf @ self.gate_weight.astype(x.dtype).T
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
        shared_gammas = w[:, -self.n_shared :]
        topk_w = w[:, : self.top_k]

        yr = (self.switch_mlp(xf, idx) * topk_w[..., None].astype(x.dtype)).sum(axis=-2)
        sh_idx = mx.broadcast_to(
            mx.arange(self.n_shared)[None], (xf.shape[0], self.n_shared)
        )
        ys = (
            self.shared_experts(xf, sh_idx) * shared_gammas[..., None].astype(x.dtype)
        ).sum(axis=-2)
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
        r = self.self_attn(self.input_layernorm(x), cache=cache, conv_mask=conv_mask)
        h = x + self.attn_sconv(r, cache=conv, mask=conv_mask)
        r = self.mlp(self.post_attention_layernorm(h))
        return h + self.mlp_sconv(r, cache=conv, mask=conv_mask)


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
