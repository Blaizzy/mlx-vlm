import math
from functools import lru_cache
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..deepseek_v4.hyper_connection import _hc_split_sinkhorn_ops, hc_expand
from ..deepseek_v4.language import (
    DeepseekV4MLP,
    DeepseekV4RoPE,
    LimitedSwiGLU,
    _sparse_pooled_attention,
)
from ..mla import MultiLinear
from ..switch_layers import SwitchGLU
from .config import ModelConfig
from .engram import Engram, EngramLayout, NgramHashState
from .fakequant import fake_quant_fp4_e4m3, fake_quant_fp4_ue8m0, fake_quant_fp8_ue8m0


class SharedIndexState:
    """Per-forward handoff between index layers, replacing the reference global.

    Layers run in order and every source writes before its consumers read. The
    candidate source must run on every step before deeper indexers consume its mask.
    """

    def __init__(self):
        self.index_k = None
        self.candidates = None
        self.compress_kv = None
        self.topk_idxs = None


@lru_cache(64)
def _index_cos_sin(length: int, rope_dim: int, theta: float, yarn: tuple):
    """RoPE tables with DeepSeek-YaRN interpolation, one row per position."""
    dims = rope_dim
    inv_freq = 1.0 / (theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    factor, beta_fast, beta_slow, orig_len = yarn

    def correction_dim(num_rotations):
        return (
            dims
            * math.log(orig_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(theta))
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
    if low == high:
        high += 0.001
    ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
    smooth = 1 - mx.clip(ramp, 0, 1)
    inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

    freqs = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq[None, :]
    return mx.cos(freqs), mx.sin(freqs)


def _yarn_params(config: ModelConfig):
    scaling = config.rope_scaling or {}
    return (
        scaling.get("factor", 1),
        scaling.get("beta_fast", 32),
        scaling.get("beta_slow", 1),
        scaling.get("original_max_position_embeddings", 65536),
    )


def _apply_index_rotary(
    x: mx.array, cos: mx.array, sin: mx.array, rope_dim: int
) -> mx.array:
    """Rotate the trailing `rope_dim` entries of each head, halves style."""
    dtype = x.dtype
    x = x.astype(mx.float32)
    passive, rot = x[..., :-rope_dim], x[..., -rope_dim:]
    x1, x2 = mx.split(rot, 2, axis=-1)
    out = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return mx.concatenate([passive, out], axis=-1).astype(dtype)


def select_candidate_blocks(
    logits: mx.array, compress_lens, topk_blocks: int, block_size: int
) -> mx.array:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with unreachable positions already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain
    int during decode, or broadcasts against logits' leading dims during prefill. Returns a
    bool mask shaped like `logits`, so consuming layers just mask and never think about
    blocks again.
    """
    width = logits.shape[-1]
    pad_len = (-width) % block_size
    if pad_len:
        logits = mx.concatenate(
            [
                logits,
                mx.full(logits.shape[:-1] + (pad_len,), -mx.inf),
            ],
            axis=-1,
        )
    scores = logits.reshape(*logits.shape[:-1], -1, block_size).max(axis=-1)
    num_blocks = scores.shape[-1]

    last = (compress_lens - 1) // block_size
    if isinstance(compress_lens, mx.array):
        flat = mx.reshape(last, (-1,))
        match = mx.reshape(mx.arange(num_blocks), (1, -1)) == flat[:, None]
        match = mx.reshape(match, compress_lens.shape[:-1] + (num_blocks,))
    else:
        match = mx.arange(num_blocks) == last
    pinned = mx.where(match, mx.array(mx.inf), scores)
    order = mx.argsort(-pinned, axis=-1)
    rank = mx.argsort(order, axis=-1)
    keep = (rank < min(topk_blocks, num_blocks)) & (scores > -mx.inf)
    return mx.repeat(keep, block_size, axis=-1)[..., :width]


class Indexer(nn.Module):
    """Keeps the `index_topk` best compressed positions per query.

    A small side attention: query heads against one shared key per compressed position,
    scores rectified then combined by `weights_proj`. With a candidate source this is the
    second of two levels; `select_candidate_blocks` is the first. Only compression-source
    layers publish index keys; every other indexer reads them from shared state. Queries
    and keys pass through the QAT fake-quant before scoring.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.owns_k = layer_idx in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.rope_theta = config.compress_rope_theta
        self.yarn = _yarn_params(config)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.scale = self.index_head_dim**-0.5
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self._k_cache = None

    def _publish_keys(self, latent: mx.array, start_pos: int, batch: int):
        ratio = self.compress_ratio
        n_latent = latent.shape[1]
        base = start_pos // ratio
        positions = (base + mx.arange(n_latent)) * ratio
        cos, sin = _index_cos_sin(
            max((base + n_latent - 1) * ratio + 1, start_pos + n_latent + 1),
            self.rope_head_dim,
            self.rope_theta,
            self.yarn,
        )
        k = self.k_norm(self.wk(latent)).astype(mx.float32)
        k = _apply_index_rotary(
            k,
            cos[positions][None, :, :],
            sin[positions][None, :, :],
            self.rope_head_dim,
        )
        k = fake_quant_fp4_ue8m0(k)
        cache = self._k_cache
        if cache is None:
            cache = mx.zeros(
                (batch, base + n_latent, self.index_head_dim), dtype=mx.float32
            )
        else:
            if cache.shape[0] < batch:
                cache = mx.concatenate(
                    [
                        cache,
                        mx.zeros(
                            (
                                batch - cache.shape[0],
                                cache.shape[1],
                                self.index_head_dim,
                            ),
                            dtype=mx.float32,
                        ),
                    ],
                    axis=0,
                )
            if cache.shape[1] < base + n_latent:
                cache = mx.concatenate(
                    [
                        cache,
                        mx.zeros(
                            (
                                cache.shape[0],
                                base + n_latent - cache.shape[1],
                                self.index_head_dim,
                            ),
                            dtype=mx.float32,
                        ),
                    ],
                    axis=1,
                )
        parts = []
        if base > 0:
            parts.append(cache[:batch, :base])
        parts.append(k)
        if cache.shape[1] > base + n_latent:
            parts.append(cache[:batch, base + n_latent :])
        head = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if cache.shape[0] > batch:
            cache = mx.concatenate([head, cache[batch:]], axis=0)
        else:
            cache = head
        self._k_cache = cache
        return cache

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        latent,
        start_pos: int,
        offset: int,
        shared: SharedIndexState,
    ) -> mx.array:
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it
        into index keys here, which has to happen before attention overwrites that same
        storage with the rotated values."""
        batch, seqlen = x.shape[0], x.shape[1]
        ratio, end_pos = self.compress_ratio, start_pos + seqlen

        if self.owns_k and latent is not None:
            shared.index_k = self._publish_keys(latent, start_pos, batch)

        cos, sin = _index_cos_sin(
            end_pos, self.rope_head_dim, self.rope_theta, self.yarn
        )
        positions = mx.arange(start_pos, end_pos)
        q = self.wq_b(qr).reshape(batch, seqlen, self.n_heads, self.index_head_dim)
        q = _apply_index_rotary(
            q.astype(mx.float32),
            cos[positions][None, :, None, :],
            sin[positions][None, :, None, :],
            self.rope_head_dim,
        )
        q = fake_quant_fp4_ue8m0(q)

        index_k = shared.index_k[:batch, : end_pos // ratio].astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32) * (
            self.scale * self.n_heads**-0.5
        )
        scores = mx.einsum("bsnd,btd->bsnt", q, index_k)
        scores = mx.maximum(scores, 0) * weights[..., None]
        scores = scores.sum(axis=2)

        compress_lens = (mx.arange(start_pos + 1, end_pos + 1) // ratio)[:, None]
        visible = mx.arange(index_k.shape[1])[None, :] < compress_lens
        scores = mx.where(visible, scores, -mx.inf)

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                scores,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            scores = mx.where(shared.candidates, scores, -mx.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = mx.sort(mx.argsort(-scores, axis=-1)[..., :topk], axis=-1).astype(
            mx.int32
        )
        return mx.where(idxs < compress_lens, idxs + offset, -1).astype(mx.int32)


def sanitize_moe_weights(weights: dict, ffn_prefix: str, n_routed: int) -> dict:
    """Rename shared experts and stack per-expert tensors for SwitchGLU.

    Shared ``w1/w2/w3`` become ``gate/down/up_proj``; routed
    ``experts.{e}.w1/w2/w3`` stack over experts into
    ``switch_mlp.{gate,down,up}_proj``. Shared by the backbone and the
    DSpark stages, which use identical MoE layouts.

    Checkpoints converted by the reference PipeNetwork port already store the
    routed experts stacked under ``experts.{gate,down,up}_proj``; those only
    need the move onto ``switch_mlp``.
    """
    w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    remapped = {}
    for k, v in weights.items():
        if k.startswith(ffn_prefix + ".shared_experts."):
            for old, new in w_remap.items():
                k = k.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
        remapped[k] = v
    weights = remapped
    prefix = f"{ffn_prefix}.experts"
    for dst in ("gate_proj", "down_proj", "up_proj"):
        for suffix in ("weight", "scales", "biases"):
            stacked_key = f"{prefix}.{dst}.{suffix}"
            if stacked_key in weights:
                weights[f"{ffn_prefix}.switch_mlp.{dst}.{suffix}"] = weights.pop(
                    stacked_key
                )
    for src, dst in (
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
    ):
        for suffix in ("weight", "scales", "biases"):
            key0 = f"{prefix}.0.{src}.{suffix}"
            if key0 not in weights:
                continue
            absent = [
                e
                for e in range(n_routed)
                if f"{prefix}.{e}.{src}.{suffix}" not in weights
            ]
            if absent:
                shown = ", ".join(str(e) for e in absent[:8])
                more = f" (+{len(absent) - 8} more)" if len(absent) > 8 else ""
                raise ValueError(
                    f"{prefix}.{{expert}}.{src}.{suffix}: checkpoint provides "
                    f"{n_routed - len(absent)} of {n_routed} experts; missing {shown}{more}"
                )
            stacked = [
                weights.pop(f"{prefix}.{e}.{src}.{suffix}") for e in range(n_routed)
            ]
            weights[f"{ffn_prefix}.switch_mlp.{dst}.{suffix}"] = mx.stack(stacked)
    return weights


class DeepseekV41MoEGate(nn.Module):
    """MoE gating with a separate correction bias for image-span tokens.

    The bias steers expert selection only; routing weights come from the unbiased
    scores. Selection matches the shared noaux_tc top-k; only the bias source is
    V4.1-specific, everything else reuses the proven mechanism.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"Unsupported DeepSeek-V4.1 scoring function: {config.scoring_func}"
            )
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = mx.zeros((self.num_experts, config.hidden_size))
        self.bias = mx.zeros((self.num_experts,), dtype=mx.float32)
        self.bias_vl = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(
        self, x: mx.array, image_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        scores = mx.sqrt(nn.softplus(x.astype(mx.float32) @ self.weight.T))
        bias = self.bias
        if image_mask is not None:
            bias = mx.where(image_mask[..., None], self.bias_vl, self.bias)
        inds = mx.argpartition(-(scores + bias), kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ].astype(mx.int32)
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, weights * self.routed_scaling_factor


class DeepseekV41MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(
        self,
        config: ModelConfig,
        moe_intermediate_size: Optional[int] = None,
        n_routed_experts: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
    ):
        super().__init__()
        self.gate = DeepseekV41MoEGate(config)
        if n_routed_experts is not None:
            self.gate.num_experts = n_routed_experts
            self.gate.top_k = num_experts_per_tok or self.gate.top_k
            self.gate.weight = mx.zeros((n_routed_experts, config.hidden_size))
            self.gate.bias = mx.zeros((n_routed_experts,), dtype=mx.float32)
            self.gate.bias_vl = mx.zeros((n_routed_experts,), dtype=mx.float32)
        inter = moe_intermediate_size or config.moe_intermediate_size
        routed = n_routed_experts or config.n_routed_experts
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            inter,
            routed,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=inter,
            swiglu_limit=config.swiglu_limit,
        )

    def __call__(self, x: mx.array, image_mask: Optional[mx.array] = None) -> mx.array:
        inds, scores = self.gate(x, image_mask)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        return y + self.shared_experts(x)


def _apply_rope_at_positions(
    x: mx.array, positions: mx.array, rope_dim: int, theta: float, yarn: tuple
) -> mx.array:
    """Rotate strided positions (compressor latents) with explicit tables."""
    table_len = int(mx.max(positions).item()) + 1 if positions.size else 1
    cos, sin = _index_cos_sin(table_len, rope_dim, theta, yarn)
    shape = (1,) * (x.ndim - positions.ndim - 1) + positions.shape + (cos.shape[-1],)
    rows = cos[positions].reshape(shape)
    return _apply_index_rotary(x, rows, sin[positions].reshape(shape), rope_dim)


class DeepseekV41Attention(nn.Module):
    """Latent attention over window KV plus top-k compressed positions.

    Modes derive from layer role: kv sources run Full (own compressor and indexer),
    index-source non-owners run Reindex (shared KV and K, own rescoring), the rest
    run Reuse (shared KV and shared Top-K, no indexer), and ratio-0 layers run
    window-only. Every mode computes its own queries and sliding-window KV.
    Deliberate deviations from the reference: an ordered shift ring replaces the
    indexed ring buffer (same visible sets, mask-addressed); window and compressed
    KV stay in separate gathers so no concatenation offset is needed; the extra
    query rms-norm in the V4 port is omitted (the reference has none).
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        if self.compress_ratio == 0:
            self.mode = "local"
        elif self.is_kv_source:
            self.mode = "full"
        elif self.is_index_source:
            self.mode = "reindex"
        else:
            self.mode = "reuse"
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.scale = self.head_dim**-0.5
        self.window_size = config.sliding_window

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

        if self.compress_ratio:
            self.rope = DeepseekV4RoPE(
                config.qk_rope_head_dim,
                config.compress_rope_theta,
                config.rope_scaling,
                config.max_position_embeddings,
            )
        else:
            self.rope = DeepseekV4RoPE(
                config.qk_rope_head_dim,
                config.rope_theta,
                None,
                config.max_position_embeddings,
            )
        self.compressor = Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = Indexer(config, layer_idx) if self.is_index_source else None
        self._latent_theta = config.compress_rope_theta
        scaling = config.rope_scaling or {}
        self._latent_yarn = (
            scaling.get("factor", 1),
            scaling.get("beta_fast", 32),
            scaling.get("beta_slow", 1),
            scaling.get("original_max_position_embeddings", 65536),
        )
        self._window_cache = None
        self._compress_cache = None
        self._cache_len = 0

    def _grow_cache(self, cache, batch: int, length: int, dim: int):
        if cache is None:
            return mx.zeros((batch, length, dim), dtype=mx.float32)
        if cache.shape[0] < batch:
            cache = mx.concatenate(
                [
                    cache,
                    mx.zeros(
                        (batch - cache.shape[0], cache.shape[1], dim),
                        dtype=mx.float32,
                    ),
                ],
                axis=0,
            )
        if cache.shape[1] < length:
            cache = mx.concatenate(
                [
                    cache,
                    mx.zeros(
                        (cache.shape[0], length - cache.shape[1], dim),
                        dtype=mx.float32,
                    ),
                ],
                axis=1,
            )
        return cache

    def _window_part(self, x: mx.array, start_pos: int):
        """Window KV slice for this step plus its validity mask.

        Prefill attends over the current chunk; later steps attend the last
        `window_size` tokens. The buffer keeps every token in order (not a
        fixed ring) so speculative rollback can truncate it.
        """
        batch, seqlen = x.shape[0], x.shape[1]
        win = self.window_size
        kv = self.kv_norm(self.wkv(x)).reshape(batch, 1, seqlen, self.head_dim)
        kv = self.rope(kv, start_pos).reshape(batch, seqlen, self.head_dim)
        kv = fake_quant_fp8_ue8m0(kv.astype(mx.float32)).astype(kv.dtype)
        need_len = start_pos + seqlen
        cache = self._window_cache
        if cache is None:
            cache = mx.zeros((batch, 0, self.head_dim), dtype=mx.float32)
        if cache.shape[0] < batch:
            cache = mx.concatenate(
                [
                    cache,
                    mx.zeros(
                        (batch - cache.shape[0], cache.shape[1], self.head_dim),
                        dtype=mx.float32,
                    ),
                ],
                axis=0,
            )
        if cache.shape[1] < need_len:
            cache = mx.concatenate(
                [
                    cache,
                    mx.zeros(
                        (cache.shape[0], need_len - cache.shape[1], self.head_dim),
                        dtype=mx.float32,
                    ),
                ],
                axis=1,
            )
        parts = []
        if start_pos > 0:
            parts.append(cache[:batch, :start_pos])
        parts.append(kv)
        if cache.shape[1] > need_len:
            parts.append(cache[:batch, need_len:])
        head = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if cache.shape[0] > batch:
            cache = mx.concatenate([head, cache[batch:]], axis=0)
        else:
            cache = head
        self._window_cache = cache
        self._cache_len = need_len
        if start_pos == 0:
            part, base = cache[:batch, :need_len], 0
        else:
            base = max(0, start_pos - win + 1)
            part = cache[:batch, base:need_len]
        positions = mx.arange(base, base + part.shape[1])
        queries = mx.arange(start_pos, start_pos + seqlen)
        valid = (
            (positions[None, :] <= queries[:, None])
            & (positions[None, :] > queries[:, None] - win)
            & (positions[None, :] >= 0)
        )
        return part, valid[None, None]

    def _compress_part(self, x: mx.array, qr: mx.array, start_pos: int, shared):
        """Shared compressed KV slice plus this layer's Top-K indices."""
        batch, seqlen = x.shape[0], x.shape[1]
        ratio = self.compress_ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            if latent is not None:
                n_latent = latent.shape[1]
                positions = (start_pos // ratio + mx.arange(n_latent)) * ratio
                latent = _apply_rope_at_positions(
                    latent.astype(mx.float32),
                    positions,
                    self.config.qk_rope_head_dim,
                    self._latent_theta,
                    self._latent_yarn,
                )
                latent = fake_quant_fp4_e4m3(latent)
                base = start_pos // ratio
                cache = self._grow_cache(
                    self._compress_cache, batch, base + n_latent, self.head_dim
                )
                parts = []
                if base > 0:
                    parts.append(cache[:batch, :base])
                parts.append(latent)
                if cache.shape[1] > base + n_latent:
                    parts.append(cache[:batch, base + n_latent :])
                head = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
                if cache.shape[0] > batch:
                    cache = mx.concatenate([head, cache[batch:]], axis=0)
                else:
                    cache = head
                self._compress_cache = cache
                shared.compress_kv = cache
        compress_len = (start_pos + seqlen) // ratio
        if compress_len == 0:
            pool = mx.zeros((batch, 0, self.head_dim), dtype=x.dtype)
            return pool, mx.zeros((batch, seqlen, 0), dtype=mx.int32)
        pool = shared.compress_kv[:batch, :compress_len]
        if self.is_index_source:
            idxs = self.indexer(x, qr, latent, start_pos, 0, shared)
            shared.topk_idxs = idxs
        else:
            idxs = shared.topk_idxs
        return pool, idxs

    def __call__(self, x: mx.array, start_pos: int, shared: SharedIndexState):
        batch, seqlen = x.shape[0], x.shape[1]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(batch, seqlen, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, start_pos)

        window_kv, window_mask = self._window_part(x, start_pos)
        out = None
        if self.compress_ratio:
            pool, idxs = self._compress_part(x, qr, start_pos, shared)
            if pool.shape[1]:
                out = _sparse_pooled_attention(
                    q,
                    window_kv[:, None],
                    pool,
                    idxs,
                    window_mask,
                    (idxs != -1)[:, None],
                    self.scale,
                    self.attn_sink.astype(q.dtype),
                )
        if out is None:
            mask = window_mask
            kv = window_kv[:, None]
            out = mx.fast.scaled_dot_product_attention(
                q,
                kv,
                kv,
                scale=self.scale,
                mask=mask,
                sinks=self.attn_sink.astype(q.dtype),
            )
        out = self.rope(out, start_pos, inverse=True)

        out = out.reshape(batch, self.o_groups, -1, seqlen, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent before RoPE, or None while a group is still filling up -- so during
    decode it only yields every `compress_ratio` steps, holding the partial group in state.
    Pre-RoPE is deliberate: the indexer needs the unrotated form, so attention rotates
    afterwards. Ratio 1 is a plain projection with no gate and no fp32 promotion.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        compress_ratio = config.compress_ratios[layer_idx]
        assert compress_ratio >= 1
        self.compress_ratio = compress_ratio
        self.head_dim = config.head_dim
        self.norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        if compress_ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self._kv_state = None
        self._score_state = None
        self._undo = []

    def _push_undo(self, batch: int, slot: int):
        """Record a slot's prior contents so a rejected position can be undone.

        Slots are position-addressed (``pos % ratio``), so a speculative block
        that crosses a compression boundary overwrites slots belonging to
        already-committed positions. Those are not recoverable by replay.
        """
        kv = None if self._kv_state is None else self._kv_state[:batch, slot : slot + 1]
        sc = (
            None
            if self._score_state is None
            else self._score_state[:batch, slot : slot + 1]
        )
        self._undo.append((slot, batch, kv, sc))
        if len(self._undo) > 64:
            del self._undo[:-64]

    def undo(self, n: int):
        """Restore slot state for the last ``n`` written positions."""
        for _ in range(min(int(n), len(self._undo))):
            slot, batch, kv, sc = self._undo.pop()
            if kv is not None:
                self._kv_state = self._write_slot(self._kv_state, batch, slot, kv)
            if sc is not None:
                self._score_state = self._write_slot(self._score_state, batch, slot, sc)

    def _grow_state(self, batch: int):
        if self._kv_state is None:
            self._kv_state = mx.zeros(
                (batch, self.compress_ratio, self.head_dim), dtype=mx.float32
            )
            self._score_state = mx.full(
                (batch, self.compress_ratio, self.head_dim), -mx.inf
            )
        elif self._kv_state.shape[0] < batch:
            extra = batch - self._kv_state.shape[0]
            self._kv_state = mx.concatenate(
                [
                    self._kv_state,
                    mx.zeros(
                        (extra, self.compress_ratio, self.head_dim), dtype=mx.float32
                    ),
                ],
                axis=0,
            )
            self._score_state = mx.concatenate(
                [
                    self._score_state,
                    mx.full((extra, self.compress_ratio, self.head_dim), -mx.inf),
                ],
                axis=0,
            )

    def _stow_remainder(self, state: mx.array, batch: int, vals: mx.array):
        remainder = vals.shape[1]
        head = mx.concatenate([vals, state[:batch, remainder:]], axis=1)
        if state.shape[0] > batch:
            return mx.concatenate([head, state[batch:]], axis=0)
        return head

    def _write_slot(self, state: mx.array, batch: int, slot: int, vals: mx.array):
        parts = []
        if slot > 0:
            parts.append(state[:batch, :slot])
        parts.append(vals)
        if slot + 1 < self.compress_ratio:
            parts.append(state[:batch, slot + 1 :])
        head = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if state.shape[0] > batch:
            return mx.concatenate([head, state[batch:]], axis=0)
        return head

    def __call__(self, x: mx.array, start_pos: int):
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(self.wkv(x))

        batch, seqlen = x.shape[0], x.shape[1]
        dtype = x.dtype
        xf = x.astype(mx.float32)
        kv, score = self.wkv(xf), self.wgate(xf)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            self._undo.clear()
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                self._grow_state(batch)
                self._kv_state = self._stow_remainder(
                    self._kv_state, batch, kv[:, cutoff:]
                )
                self._score_state = self._stow_remainder(
                    self._score_state, batch, score[:, cutoff:]
                )
            kv = kv[:, :cutoff].reshape(batch, -1, ratio, self.head_dim)
            score = score[:, :cutoff].reshape(batch, -1, ratio, self.head_dim)
            kv = (kv * mx.softmax(score, axis=2)).sum(axis=2)
        else:
            self._grow_state(batch)
            pooled = []
            for i in range(seqlen):
                pos = start_pos + i
                slot = pos % ratio
                self._push_undo(batch, slot)
                self._kv_state = self._write_slot(
                    self._kv_state, batch, slot, kv[:, i : i + 1, :]
                )
                self._score_state = self._write_slot(
                    self._score_state, batch, slot, score[:, i : i + 1, :]
                )
                if (pos + 1) % ratio == 0:
                    pooled.append(
                        (
                            self._kv_state[:batch]
                            * mx.softmax(self._score_state[:batch], axis=1)
                        ).sum(axis=1, keepdims=True)
                    )
            should_compress = bool(pooled)
            if should_compress:
                kv = mx.concatenate(pooled, axis=1) if len(pooled) > 1 else pooled[0]
        if not should_compress:
            return None
        return self.norm(kv.astype(dtype))


def make_identity_pre_mix(batch: int, seqlen: int, hc_mult: int) -> mx.array:
    """Initial one-hot mix: the first copy passes through untouched."""
    pre_mix = mx.zeros((batch, seqlen, hc_mult), dtype=mx.float32)
    return mx.concatenate(
        [mx.ones((batch, seqlen, 1), dtype=mx.float32), pre_mix[:, :, 1:]], axis=-1
    )


def hc_mix_coeffs(
    x: mx.array,
    hc_fn: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    norm_eps: float,
):
    """Collapse coefficients for the next sublayer: pre / post / comb."""
    mixes = mx.fast.rms_norm(x.flatten(-2).astype(mx.float32), None, norm_eps) @ hc_fn.T
    return _hc_split_sinkhorn_ops(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
    )


class DeepseekV41Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer
    input) and `hc_post` (expand back out, mixing the residual in through `comb`).
    The coefficients a sublayer computes are used by the *next* one (single-pass mHC):
    attention consumes the incoming mix and produces `attn_pre` for the FFN, which
    produces `ffn_pre` for the next block. Coefficient math reuses the proven
    sinkhorn path; only the flat V4.1 key layout and the shifted consumption are new.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, engram_layout=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.attn = DeepseekV41Attention(config, layer_idx)
        self.ffn = DeepseekV41MoE(config)
        self.engram = None
        if engram_layout is not None and layer_idx in engram_layout.layer_ids:
            self.engram = Engram(config, layer_idx, engram_layout)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=self.norm_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.ones((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.ones((3,), dtype=mx.float32)

    def hc_mixes(
        self, x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array
    ):
        """Collapse coefficients for the next sublayer: pre / post / comb."""
        return hc_mix_coeffs(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
        )

    @staticmethod
    def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
        """Collapse the hc copies into one, weighted by pre_mix."""
        return (
            (pre_mix[..., None].astype(mx.float32) * x.astype(mx.float32))
            .sum(axis=2)
            .astype(x.dtype)
        )

    def __call__(
        self,
        h: mx.array,
        start_pos: int,
        pre_mix: mx.array,
        image_mask: Optional[mx.array],
        shared: SharedIndexState,
    ):
        residual = h
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.hc_pre(h, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, shared)
        x = hc_expand(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        x = hc_expand(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class ParallelHead(nn.Module):
    """Vocabulary projection kept in fp32 so logits come out fp32 directly."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = mx.zeros(
            (config.vocab_size, config.hidden_size), dtype=mx.float32
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) @ self.weight.T


class DeepseekV41Cache:
    """Generation state: shared cross-layer handoff plus the token offset.

    Layer caches live inside the modules (growing underscore attributes, like the
    reference buffers); this object only carries what must cross call boundaries.
    A zero offset marks a fresh generation and resets module state on entry.
    """

    def __init__(self):
        self.shared = SharedIndexState()
        self.offset = 0
        self._model = None

    @property
    def state(self):
        """Arrays the generator materializes between prefill chunks.

        Layer state lives inside the modules, so surface it here; evaluating it
        per chunk is what keeps a long prefill from becoming one command buffer.
        """
        model = self._model
        if model is None:
            return []
        arrays = []
        for layer in model.layers:
            attn = layer.attn
            for array in (attn._window_cache, attn._compress_cache):
                if array is not None:
                    arrays.append(array)
            if attn.indexer is not None and attn.indexer._k_cache is not None:
                arrays.append(attn.indexer._k_cache)
            if attn.compressor is not None:
                for array in (
                    attn.compressor._kv_state,
                    attn.compressor._score_state,
                ):
                    if array is not None:
                        arrays.append(array)
        engram = model.engram_hash
        if engram is not None and engram._cache is not None:
            arrays.append(engram._cache)
        return arrays

    def trim(self, n: int) -> int:
        """Drop the last `n` appended tokens so a speculative block rolls back.

        The framework calls this with the rejected-token count on commit and
        the whole-block advance on abort. Prefix-addressed buffers truncate;
        the compressor's slot state is position-addressed and rewrites itself
        on resume, so it needs no trim.
        """
        n = min(self.offset, int(n))
        if n <= 0:
            return 0
        self.offset -= n
        if self._model is not None:
            self._model._trim_caches(self.offset, n)
        return n


class LanguageModel(nn.Module):
    """Embed, expand to hc copies, run the blocks, collapse, project to logits."""

    requires_uniform_batch_acceptance = True

    def __init__(self, config: ModelConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.layout = EngramLayout.from_config(config)
        self.engram_hash = None
        self._engram_source = None
        if self.layout is not None and tokenizer is not None:
            self.engram_hash = NgramHashState(config, self.layout, tokenizer)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV41Block(config, i, self.layout)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelHead(config)
        self.target_layer_ids = list(config.dspark_target_layer_ids)

    def _ensure_engram_hash(self):
        """Build the n-gram hash state on first use from the checkpoint directory.

        The loader only learns the model path after construction, so this cannot
        happen in ``__init__``.
        """
        if self.engram_hash is not None or self.layout is None:
            return
        source = self._engram_source
        if source is None:
            return
        import json
        import os

        token_map = None
        cached = os.path.join(str(source), "engram_token_map.json")
        if os.path.exists(cached):
            with open(cached) as handle:
                token_map = json.load(handle)
        tokenizer = None
        if token_map is None:
            tok_file = os.path.join(str(source), "tokenizer.json")
            if not os.path.exists(tok_file):
                self._engram_source = None
                return
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
        self.engram_hash = NgramHashState(
            self.config, self.layout, tokenizer=tokenizer, token_map=token_map
        )

    def make_cache(self):
        cache = DeepseekV41Cache()
        cache._model = self
        return [cache]

    def _trim_caches(self, offset: int, n: int = 0):
        for layer in self.layers:
            attn = layer.attn
            attn._cache_len = min(attn._cache_len, offset)
            if n and attn.compressor is not None:
                attn.compressor.undo(n)
            if attn._window_cache is not None:
                attn._window_cache = attn._window_cache[:, :offset]
            if attn.compress_ratio and attn._compress_cache is not None:
                attn._compress_cache = attn._compress_cache[
                    :, : offset // attn.compress_ratio
                ]
            indexer = attn.indexer
            if indexer is not None and indexer._k_cache is not None:
                indexer._k_cache = indexer._k_cache[
                    :, : offset // indexer.compress_ratio
                ]
        if self.engram_hash is not None and self.engram_hash._cache is not None:
            self.engram_hash._cache = self.engram_hash._cache[:, :offset]

    def _reset_caches(self):
        for layer in self.layers:
            layer.attn._window_cache = None
            layer.attn._compress_cache = None
            layer.attn._cache_len = 0
            if layer.attn.indexer is not None:
                layer.attn.indexer._k_cache = None
            if layer.attn.compressor is not None:
                layer.attn.compressor._kv_state = None
                layer.attn.compressor._score_state = None
                layer.attn.compressor._undo.clear()

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache=None,
        image_mask: Optional[mx.array] = None,
        engram_hashes: Optional[mx.array] = None,
        inputs: Optional[mx.array] = None,
        n_to_process: Optional[int] = None,
        capture_layer_ids: Optional[List[int]] = None,
    ) -> LanguageModelOutput:
        """Extra `inputs`/`n_to_process` are generate-protocol passengers.

        Chunked prefill slices `inputs_embeds` per call and advances the cache
        offset; the raw `inputs` ids carry nothing this architecture needs
        (no hash routing), so they are accepted and ignored.
        """
        entry = cache[0] if cache else DeepseekV41Cache()
        start_pos = entry.offset
        if start_pos == 0:
            self._reset_caches()
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds
        batch, seqlen = h.shape[0], h.shape[1]
        h = mx.broadcast_to(
            h[..., None, :],
            (batch, seqlen, self.config.hc_mult, self.config.hidden_size),
        )
        if engram_hashes is None and input_ids is not None:
            self._ensure_engram_hash()
            if self.engram_hash is not None:
                engram_hashes = self.engram_hash(input_ids, start_pos)
        main_hiddens = []
        capture_ids = (
            self.target_layer_ids if capture_layer_ids is None else capture_layer_ids
        )
        pre_mix = make_identity_pre_mix(batch, seqlen, self.config.hc_mult)
        shared = entry.shared
        for i, layer in enumerate(self.layers):
            if layer.engram is not None and engram_hashes is not None:
                h = layer.engram(
                    h,
                    engram_hashes[:, :, layer.engram.layer_hash_index, :],
                    None if image_mask is None else ~image_mask,
                )
            if i in capture_ids:
                main_hiddens.append(h.mean(axis=2))
            h, pre_mix = layer(h, start_pos, pre_mix, image_mask, shared)
        h = DeepseekV41Block.hc_pre(h, pre_mix)
        logits = self.head(self.norm(h))
        entry.offset = start_pos + seqlen
        hidden = [mx.concatenate(main_hiddens, axis=-1)] if main_hiddens else None
        return LanguageModelOutput(logits=logits, hidden_states=hidden)
