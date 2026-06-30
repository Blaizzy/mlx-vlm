import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.pipeline import PipelineMixin
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import CacheList, PoolingCache, RotatingKVCache
from .config import ModelConfig
from .hisa_kernel import hisa_select
from .hyper_connection import HyperConnection, HyperHead, hc_expand


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    shared_experts = {k: mxfp8 for k, _ in flat_modules if ".ffn.shared_experts." in k}
    attn = {
        k: mxfp8 for k, _ in flat_modules if ".attn.w" in k or ".attn.indexer.wq" in k
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **shared_experts,
        **attn,
    }


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k].astype(
        mx.int32
    )
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _hash_expert_select(
    input_ids: mx.array,
    logits: mx.array,
    tid2eid: mx.array,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    inds = tid2eid[input_ids].astype(mx.int32)
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache = {}

    def _get_freqs(self, head_dim: int, inverse: bool):
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(
        self,
        x: mx.array,
        offset: Any = 0,
        inverse: bool = False,
    ) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


def _apply_score_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape

    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, N - S))

    full_mask = mx.concatenate([mask, pool_mask], axis=-1)

    return full_mask


def _align_local_mask(mask: Optional[mx.array], local_len: int):
    if mask is None:
        return None

    current_len = mask.shape[-1]
    if current_len == local_len:
        return mask
    if current_len > local_len:
        return mask[..., -local_len:]

    pad_shape = (*mask.shape[:-1], local_len - current_len)
    if mask.dtype == mx.bool_:
        pad = mx.ones(pad_shape, dtype=mask.dtype)
    else:
        pad = mx.zeros(pad_shape, dtype=mask.dtype)
    return mx.concatenate([pad, mask], axis=-1)


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, ape, head_dim):
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


@mx.compile
def _overlap_compress_kv(kv, gate, ape, head_dim):
    B, L, R, D = kv.shape

    gate = gate + ape.astype(gate.dtype)

    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)

    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)

    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    B, H, L, D = q.shape
    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )

    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    pooled_sq = pooled.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = x @ self.weight.T

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds, weights = _hash_expert_select(
                input_ids,
                logits,
                self.tid2eid,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )
        else:
            inds, weights = _expert_select(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )

        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Compressor(nn.Module):
    def __init__(self, config: ModelConfig, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ) -> mx.array:
        B, _, _ = x.shape
        kv = self.wkv(x)
        gate = self.wgate(x)
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            compress_func = (
                _overlap_compress_kv if self.overlap else _simple_compress_kv
            )
            kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))
            new_pooled = compress_func(kv, gate, self.ape, self.head_dim)
            new_pooled = self.norm(new_pooled)
            new_pooled = self.rope(
                new_pooled[:, None],
                offset=pool_base,
            ).squeeze(1)

        if pool_cache is not None:
            new_pooled = pool_cache.update_and_fetch(new_pooled)

        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config: ModelConfig, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5
        self.index_block = getattr(config, "index_block", 0)
        self.index_keep = getattr(config, "index_keep", 0)

    def _hisa_select(self, q: mx.array, pooled: mx.array, x: mx.array, k: int):
        """HISA hierarchical decode selection: coarse block filter -> fine top-k.

        q: (B, n_heads, 1, head_dim); pooled: (B, Np, head_dim). Returns (B, 1, k)
        indices into the prefix, matching the flat path's output. Scans n_blocks
        coarse reps + index_keep*index_block fine candidates instead of all Np.

        paper: https://arxiv.org/abs/2603.28458
        """

        B, Np, hd = pooled.shape
        b = self.index_block
        nb = Np // b
        usable = nb * b
        w = self.weights_proj(x).astype(mx.float32) * (
            self.n_heads**-0.5
        )  # (B,1,n_heads)
        wq = w.swapaxes(-1, -2)[..., None]  # (B,n_heads,1,1)

        # coarse: score block-mean representatives
        rep = pooled[:, :usable].reshape(B, nb, b, hd).mean(axis=2)  # (B,nb,hd)
        cs = mx.maximum(
            q.astype(mx.float32) @ rep[:, None].swapaxes(-1, -2).astype(mx.float32), 0
        )
        cscore = (cs * self.scale * wq).sum(axis=1)  # (B,1,nb)
        Kb = min(self.index_keep, nb)
        top_blk = mx.argpartition(-cscore, kth=Kb - 1, axis=-1)[..., :Kb]  # (B,1,Kb)

        # fine: score only positions inside the retained blocks
        pos = (top_blk[..., None] * b + mx.arange(b)).reshape(B, 1, Kb * b)
        idx = mx.broadcast_to(pos.reshape(B, Kb * b)[..., None], (B, Kb * b, hd))
        cand = mx.take_along_axis(pooled, idx, axis=1)  # (B,Kb*b,hd)
        fs = mx.maximum(
            q.astype(mx.float32) @ cand[:, None].swapaxes(-1, -2).astype(mx.float32), 0
        )
        fscore = (fs * self.scale * wq).sum(axis=1)  # (B,1,Kb*b)
        sel = mx.argpartition(-fscore, kth=k - 1, axis=-1)[..., :k]  # (B,1,k)
        return mx.take_along_axis(pos, sel, axis=-1)

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        position_rope: DeepseekV4RoPE,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None

        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = position_rope(q, offset)

        Np = pooled.shape[1]
        k = min(self.index_topk, Np)
        pmask = pool_cache.make_mask(L, offset) if pool_cache is not None else None

        # HISA hierarchical decode fast-path happends only when there is no mask to honor
        # (single-token decode within the pool window => pmask is None), the
        # prefix is long enough to block, and there are enough fine candidates.

        if (
            L == 1
            and pmask is None
            and pool_cache is not None
            and self.index_block > 0
            and Np >= self.index_block * self.index_keep
            and self.index_keep * self.index_block >= k
        ):
            return self._hisa_select(q, pooled, x, k)

        # HISA batched path for L > 1 (prefill / speculative decode). Honors the
        # causal mask via valid_len = #visible pooled positions per query (the
        # pool mask is contiguous-causal, so the count is the visibility cutoff).
        if (
            L > 1
            and self.index_block > 0
            and Np >= self.index_block * self.index_keep
            and self.index_keep * self.index_block >= k
        ):
            if pmask is None:
                valid_len = mx.full((B, L), Np, dtype=mx.int32)
            else:
                pm = pmask if pmask.ndim == 3 else pmask[None]
                valid_len = mx.broadcast_to(pm, (B, L, pm.shape[-1])).sum(-1)
            weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
            return hisa_select(
                q,
                pooled,
                weights,
                self.scale,
                k,
                self.index_block,
                self.index_keep,
                valid_len,
            )

        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None],
                scores,
                mx.finfo(scores.dtype).min,
            )
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


class LocalAttention(nn.Module):
    """DeepSeek V4 attention with no KV compression."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = 0
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_offset: Optional[Union[int, mx.array]] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = (
            position_offset
            if position_offset is not None
            else (cache.offset if cache is not None else 0)
        )
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        mask = _align_local_mask(mask, kv.shape[2])

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class CompressedAttention(nn.Module):
    """DeepSeek V4 attention with pooled KV compression."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # Compressed layers use Yarn-scaled RoPE
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_offset: Optional[Union[int, mx.array]] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = (
            position_offset
            if position_offset is not None
            else (local_cache.offset if local_cache is not None else 0)
        )
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        mask = _align_local_mask(mask, kv.shape[2])

        # Pool tokens into compressed KV and concatenate with local KV
        pooled = self.compressor(x, pool_cache, offset)
        pooled_mask = None
        if pooled.shape[1] > 0:
            pooled_mask = (
                pool_cache.make_mask(L, offset) if pool_cache is not None else None
            )
            kv = mx.concatenate([kv, pooled[:, None]], axis=2)

        mask = _extend_mask(mask, pooled_mask, kv.shape[2])

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=local_cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class SparseCompressedAttention(nn.Module):
    """DeepSeek V4 attention with sparse indexed pooled KV compression."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        self.indexer = Indexer(config, self.compress_ratio)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_offset: Optional[Union[int, mx.array]] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        comp_cache = cache[1] if cache is not None else None
        idx_cache = cache[2] if cache is not None else None
        offset = (
            position_offset
            if position_offset is not None
            else (local_cache.offset if local_cache is not None else 0)
        )
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        mask = _align_local_mask(mask, kv.shape[2])

        pooled = self.compressor(x, comp_cache, offset)
        pmask = comp_cache.make_mask(L, offset) if comp_cache is not None else None
        topk = self.indexer(x, q_residual, self.rope, idx_cache, offset)
        sinks = self.attn_sink.astype(q.dtype)

        # Local attention
        if pooled.shape[1] == 0:
            out = scaled_dot_product_attention(
                q,
                kv,
                kv,
                cache=local_cache,
                scale=self.scale,
                mask=mask,
                sinks=sinks,
            )

        # Compressed attention
        elif pooled.shape[1] <= self.indexer.index_topk:
            full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
            mask = _extend_mask(mask, pmask, full_kv.shape[2])
            out = scaled_dot_product_attention(
                q,
                full_kv,
                full_kv,
                cache=local_cache,
                scale=self.scale,
                mask=mask,
                sinks=sinks,
            )

        # Sparse compressed attention
        else:
            sparse_mask = None
            if pmask is not None:
                sparse_mask = mx.take_along_axis(
                    pmask[None] if pmask.ndim == 2 else pmask,
                    topk,
                    axis=2,
                )[:, None]
            out = _sparse_pooled_attention(
                q,
                kv,
                pooled,
                topk,
                mask,
                sparse_mask,
                self.scale,
                sinks,
            )

        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


def v4_attention_factory(config: ModelConfig, layer_idx: int) -> nn.Module:
    """Instantiate the appropriate attention module for a given layer."""
    ratio = config.compress_ratios[layer_idx]
    if ratio == 0:
        return LocalAttention(config, layer_idx)
    if ratio == 128:
        return CompressedAttention(config, layer_idx)
    return SparseCompressedAttention(config, layer_idx)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = v4_attention_factory(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
        position_offset: Optional[Union[int, mx.array]] = None,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(
            self.attn_norm(x),
            mask=mask,
            cache=cache,
            position_offset=position_offset,
        )
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        hidden_sink: Optional[list] = None,
        capture_layer_ids: Optional[List[int]] = None,
        skip_final_norm: bool = False,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for local_idx, (layer, layer_cache) in enumerate(
            zip(self.pipeline_layers, cache)
        ):
            h = layer(h, mask, layer_cache, inputs)
            if hidden_sink is not None and (self.start_idx + local_idx) in capture_set:
                # DSpark self-speculation conditions on the per-layer residual
                # reduced over the hc_mult Hyper-Connection copies — the same
                # `h.mean(dim=2)` the reference Transformer.forward captures at
                # each dspark_target_layer_id.
                hidden_sink.append(h.mean(axis=2))

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, CacheList):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # Capture-style speculation (DSpark) collects per-layer hiddens above;
        # the last-layer full-hc hidden is only the MTP "last hidden" path.
        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)

        if skip_final_norm:
            return h

        return self.norm(self.hc_head(h))


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


def _snapshot_cache_state(
    caches: List[Any], incoming_tokens: int = 0
) -> List[Optional[Tuple[Any, Any]]]:
    return [_snapshot_single_cache(cache, incoming_tokens) for cache in caches]


def _needs_replay_snapshot(cache, incoming_tokens: int) -> bool:
    if cache is None:
        return False
    if isinstance(cache, CacheList):
        return any(
            _needs_replay_snapshot(child, incoming_tokens) for child in cache.caches
        )
    if isinstance(cache, PoolingCache):
        return int(cache.remainder) + int(incoming_tokens) >= int(cache.ratio)
    if isinstance(cache, RotatingKVCache):
        return False
    return not (hasattr(cache, "trim") and callable(cache.trim))


def _needs_replay_snapshot_for_cache(
    caches: Optional[List[Any]], incoming_tokens: int = 0
) -> bool:
    if caches is None:
        return False
    return any(_needs_replay_snapshot(cache, incoming_tokens) for cache in caches)


def _snapshot_single_cache(cache, incoming_tokens: int = 0):
    if cache is None:
        return None
    if isinstance(cache, CacheList):
        return (
            "cache_list",
            [_snapshot_single_cache(child, incoming_tokens) for child in cache.caches],
        )
    if isinstance(cache, PoolingCache):
        remainder = int(cache.remainder)
        total = remainder + int(incoming_tokens)
        overwrite_len = remainder
        if incoming_tokens > 0:
            overwrite_len = total % cache.ratio if total >= cache.ratio else 0
        will_overwrite_remainder = (
            remainder > 0 and cache.buf_kv is not None and overwrite_len > 0
        )
        buf_kv = cache.buf_kv[:, :overwrite_len] if will_overwrite_remainder else None
        buf_gate = (
            cache.buf_gate[:, :overwrite_len] if will_overwrite_remainder else None
        )
        pooled_len = None if cache.pooled is None else cache.pooled.shape[1]
        return (
            "pooling",
            remainder,
            _clone_cache_tree(buf_kv),
            _clone_cache_tree(buf_gate),
            pooled_len,
        )
    if isinstance(cache, RotatingKVCache):
        return (
            "rotating",
            _clone_cache_tree(cache.offset),
            int(cache._idx),
            getattr(cache, "start_position", None),
        )
    return (
        "full",
        _clone_cache_tree(getattr(cache, "state", None)),
        _clone_cache_tree(getattr(cache, "meta_state", None)),
    )


def _clear_cache_state(cache) -> None:
    if isinstance(cache, CacheList):
        for child in cache.caches:
            _clear_cache_state(child)
        return
    if hasattr(cache, "keys"):
        cache.keys = None
    if hasattr(cache, "values"):
        cache.values = None
    if hasattr(cache, "offset"):
        cache.offset = 0
    if hasattr(cache, "_idx"):
        cache._idx = 0
    if hasattr(cache, "start_position"):
        cache.start_position = 0
    if hasattr(cache, "buf_kv"):
        cache.buf_kv = None
    if hasattr(cache, "buf_gate"):
        cache.buf_gate = None
    if hasattr(cache, "remainder"):
        cache.remainder = 0
    if hasattr(cache, "pooled"):
        cache.pooled = None


def _restore_single_cache(cache, snapshot) -> None:
    if cache is None or snapshot is None:
        return
    kind = snapshot[0]
    if isinstance(cache, CacheList):
        for child, child_snapshot in zip(cache.caches, snapshot[1]):
            _restore_single_cache(child, child_snapshot)
        return
    if kind == "pooling":
        _, remainder, buf_kv, buf_gate, pooled_len = snapshot
        cache.remainder = int(remainder)
        if buf_kv is not None:
            restore_len = int(buf_kv.shape[1])
            if cache.buf_kv is None or cache.buf_kv.shape[1] < cache.ratio:
                cache.buf_kv = mx.zeros(
                    (buf_kv.shape[0], cache.ratio, buf_kv.shape[2]),
                    dtype=buf_kv.dtype,
                )
            if cache.buf_gate is None or cache.buf_gate.shape[1] < cache.ratio:
                cache.buf_gate = mx.zeros(
                    (buf_gate.shape[0], cache.ratio, buf_gate.shape[2]),
                    dtype=buf_gate.dtype,
                )
            cache.buf_kv[:, :restore_len] = buf_kv
            cache.buf_gate[:, :restore_len] = buf_gate
        if pooled_len is None:
            cache.pooled = None
        elif cache.pooled is not None:
            cache.pooled = cache.pooled[:, :pooled_len]
        return
    if kind == "rotating":
        _, offset, idx, start_position = snapshot
        cache.offset = _clone_cache_tree(offset)
        cache._idx = int(idx)
        if start_position is not None and hasattr(cache, "start_position"):
            cache.start_position = int(start_position)
        return
    _, state, meta_state = snapshot
    if state is None:
        _clear_cache_state(cache)
        return
    if meta_state is not None and hasattr(type(cache), "meta_state"):
        cache.meta_state = _clone_cache_tree(meta_state)
    cache.state = _clone_cache_tree(state)


def _restore_cache_state(
    caches: List[Any], snapshot: List[Optional[Tuple[Any, Any]]]
) -> None:
    for cache, entry in zip(caches, snapshot):
        if cache is None or entry is None:
            continue
        _restore_single_cache(cache, entry)


def _iter_leaf_caches(caches):
    for cache in caches:
        if cache is None:
            continue
        if isinstance(cache, CacheList):
            yield from _iter_leaf_caches(cache.caches)
        else:
            yield cache


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        skip_final_norm = kwargs.pop("skip_final_norm", False)
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        hidden_sink = kwargs.pop("hidden_sink", None)
        if (return_hidden or capture_layer_ids) and hidden_sink is None:
            hidden_sink = []

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            hidden_sink=hidden_sink,
            capture_layer_ids=capture_layer_ids,
            skip_final_norm=skip_final_norm,
        )
        logits = None if skip_logits else self.lm_head(out)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if return_shared_kv else None,
        )

    def _target_hidden(self, hidden: mx.array) -> mx.array:
        if (
            hidden.ndim == 3
            and hidden.shape[-1] == self.args.hc_mult * self.args.hidden_size
        ):
            hidden = hidden.reshape(*hidden.shape[:-1], self.args.hc_mult, -1)
        if hidden.ndim != 4:
            raise ValueError(
                "DeepSeek-V4 speculative hidden must have shape "
                "[batch, tokens, hc_mult, hidden_size]."
            )
        return hidden

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        hidden = self._target_hidden(hidden)
        return self.lm_head(self.model.norm(self.model.hc_head(hidden)))

    def speculative_draft_hidden(self, hidden: mx.array) -> mx.array:
        return self._target_hidden(hidden)

    def _speculative_verify(self, inputs: mx.array, cache, sampler=None):
        incoming_tokens = int(inputs.shape[1])
        cache_snapshot = (
            _snapshot_cache_state(cache, incoming_tokens)
            if _needs_replay_snapshot_for_cache(cache, incoming_tokens)
            else None
        )
        sample_logits = sampler is not None

        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=not sample_logits,
            skip_final_norm=not sample_logits,
        )
        hidden = out.hidden_states[-1]
        rollback_state = (
            (cache_snapshot, inputs) if cache_snapshot is not None else None
        )
        if not sample_logits:
            return hidden, {}, rollback_state

        return hidden, {}, rollback_state, sampler(out.logits)

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        # Greedy MTP verification is faster with one batched LM-head projection
        # than with per-position deferred projections on Metal.
        return self._speculative_verify(inputs, cache, sampler)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        return self._speculative_verify(inputs, cache)

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted,
        block_size: int,
    ) -> int:
        if isinstance(accepted, int):
            accepted = mx.array([accepted])

        max_a = int(accepted.max().item())
        if gdn_states:
            cache_snapshot, verify_inputs = gdn_states
            accepted_list = [int(a) for a in accepted.tolist()]
            if len(set(accepted_list)) != 1:
                raise ValueError(
                    "DeepSeek-V4 speculative rollback requires uniform acceptance."
                )
            _restore_cache_state(caches, cache_snapshot)
            keep = max_a + 1
            if keep > 0:
                self(verify_inputs[:, :keep], cache=caches, skip_logits=True)
            return max_a

        n = max_a + 1
        trim = block_size - n
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for cache in _iter_leaf_caches(caches):
            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)
            if is_batch and hasattr(cache, "_idx") and max_a > 0:
                keys = getattr(cache, "keys", None)
                values = getattr(cache, "values", None)
                if keys is None or values is None:
                    continue
                kv_len = cache._idx
                verify_start = kv_len - n
                for bi, valid_end in enumerate(valid_ends.tolist()):
                    start = verify_start + int(valid_end)
                    if start < kv_len:
                        zero_row_tail = getattr(cache, "zero_row_tail", None)
                        if callable(zero_row_tail):
                            zero_row_tail(bi, start, kv_len)
                        else:
                            cache.keys[bi, :, start:kv_len, :] = 0
                            cache.values[bi, :, start:kv_len, :] = 0

        return max_a

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    @property
    def quant_predicate(self):
        quantization_config = make_quantization_config(self)

        def predicate(path, _):
            path = path.removeprefix("language_model.")
            return quantization_config.get(path, True)

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                        PoolingCache(ratio),
                    )
                )
            else:
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.attn.sharding_group = group
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b,
                "all-to-sharded",
                segments=self.args.o_groups,
                group=group,
            )
            shard_inplace(layer.attn.wo_a, "sharded-to-all", group=group)
            layer.attn.attn_sink = mx.split(layer.attn.attn_sink, N)[rank]
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
