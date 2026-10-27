"""Unlimited-OCR-2 decoder attention: page2d RoPE + DASH block-sparse routing.

Unlimited-OCR-2 (``sparse_attention_config.training_stage == "dash"``) replaces
the R-SWA ring cache of Unlimited-OCR with Sparse Reference Sliding Window
Attention. Every query attends

* a token-level local window of ``sparse_local_block * sparse_block_size``
  positions (128), unbiased, and
* ``dash_topk_blocks`` remote 32-token blocks (12) picked by a small per-layer
  router, each with a learned log-prior bias.

The router scores blocks with a single-head index projection: ``head_cls``
summarizes each block's index keys into one key, a query scores the summaries,
1.5-entmax over the top ``DASH_INFER_TOPM`` (16) scores gives the prior, and
the mean-centred log prior of the first ``K`` blocks becomes their bias. Remote
candidates are restricted to the image blocks of the query's own page
(``<PAGE>`` markers number the pages).

Positions use a 2-axis RoPE over (running offset inside the token's page,
page parity) instead of the absolute position.

This mirrors the upstream inference path (``configure_dash_inference`` in the
checkpoint's ``modeling_deepseekv2.py``, parity with its vLLM plugin). MLX
deviations, all value-preserving:

* decode gathers the 512 selected keys instead of masking the full cache, so
  per-token cost no longer grows with the document;
* block summaries are computed once per completed block and cached (only
  completed blocks can ever be selected), instead of being recomputed from all
  index keys on every step;
* page/RoPE state advances in-graph, so decode never syncs the sampled token
  to the host.
"""

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from ..base import scaled_dot_product_attention
from ..cache import KVCache, QuantizedKVCache

_SCORE_NEG = -1.0e9
_ACT_CLAMP = 1.0e4
_ENTMAX_NEG = -1.0e4
_DEFAULT_TOPM = 16


def is_dash_config(sparse_config: Optional[Dict]) -> bool:
    return (
        isinstance(sparse_config, dict)
        and sparse_config.get("training_stage") == "dash"
        and bool(sparse_config.get("use_sparse_attention", False))
    )


@dataclass
class DashParams:
    """Inference knobs read from ``sparse_attention_config``."""

    block_size: int = 32
    local_blocks: int = 4
    index_dim: int = 128
    topk: int = 12
    topm: int = _DEFAULT_TOPM
    sigma: float = 1.0
    image_only: bool = True
    image_token_id: int = 128815
    page_token_id: Optional[int] = 128827
    per_page_image_tokens: int = 273
    page2d: bool = True
    rope_tok_theta: float = 10000.0
    rope_page_theta: float = 256.0
    rope_page_pairs: int = 1
    rope_tok_freq_mode: str = "resample"
    rope_page_mod: int = 2
    idx_scale: list = field(default_factory=list)
    floor: list = field(default_factory=list)

    @property
    def local_window(self) -> int:
        return self.local_blocks * self.block_size

    @classmethod
    def from_config(cls, sp: Dict, num_layers: int, head_dim: int):
        index_dim = int(sp.get("sparse_index_dim", head_dim))
        freq = sp.get("sparse_attention_freq") or [1] * num_layers
        scale_default = float(sp.get("dash_scaling_factor", 2.0))
        floor_default = float(sp.get("dash_prior_floor", 1e-2))
        scales, floors = [], []
        for layer in range(num_layers):
            # Per-layer arrays are indexed by the sparse-layer ordinal.
            slot = sum(1 for f in freq[:layer] if f)
            scale = _per_layer(sp.get("dash_scale_per_layer"), slot, scale_default)
            floors.append(
                _per_layer(sp.get("dash_floor_per_layer"), slot, floor_default)
            )
            scales.append(scale / math.sqrt(index_dim))
        main_mode = str(sp.get("sparse_main_rope_mode", "1d")).lower()
        stage0 = str(sp.get("dash_stage0", "indexer")).lower()
        if stage0 != "head_cls":
            raise ValueError(
                f"Unlimited-OCR-2: dash_stage0={stage0!r} is not supported "
                "(only 'head_cls' checkpoints are)."
            )
        return cls(
            block_size=int(sp.get("sparse_block_size", 128)),
            local_blocks=max(1, int(sp.get("sparse_local_block", 1))),
            index_dim=index_dim,
            topk=int(sp.get("dash_topk_blocks") or 12),
            sigma=float(sp.get("dash_sigma", 1.0)),
            image_only=bool(sp.get("dash_image_only", True)),
            image_token_id=int(sp.get("image_token_id", 128815)),
            page_token_id=sp.get("page_token_id"),
            per_page_image_tokens=int(sp.get("per_page_image_tokens", 273)),
            page2d=main_mode == "page2d",
            rope_tok_theta=float(sp.get("sparse_rope_tok_theta", 10000.0)),
            rope_page_theta=float(sp.get("sparse_rope_page_theta", 256.0)),
            rope_page_pairs=int(sp.get("sparse_rope_page_pairs", 8)),
            rope_tok_freq_mode=str(
                sp.get("sparse_main_rope_tok_freq_mode", "truncate")
            ).lower(),
            rope_page_mod=int(sp.get("sparse_rope_page_mod", 0) or 0),
            idx_scale=scales,
            floor=floors,
        )


def _per_layer(values, slot, default):
    if isinstance(values, (list, tuple)) and 0 <= slot < len(values):
        return float(values[slot])
    return float(default)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def page2d_inv_freq(p: DashParams, dim: int):
    half = dim // 2
    n_tok = half - p.rope_page_pairs
    ar = np.arange(n_tok, dtype=np.float32)
    if p.rope_tok_freq_mode == "truncate":
        expo = ar * np.float32(2.0 / dim)
    else:
        expo = ar / np.float32(n_tok)
    inv_tok = np.float32(1.0) / np.power(np.float32(p.rope_tok_theta), expo)
    pr = np.arange(p.rope_page_pairs, dtype=np.float32) / np.float32(p.rope_page_pairs)
    inv_page = np.float32(1.0) / np.power(np.float32(p.rope_page_theta), pr)
    return mx.array(inv_tok.astype(np.float32)), mx.array(inv_page.astype(np.float32))


def yarn_inv_freq(dim: int, base: float, rope_scaling: Optional[Dict], max_pos: int):
    """HF ``LlamaRotaryEmbedding`` inverse frequencies (default or yarn)."""
    pos_freqs = np.power(
        np.float32(base), np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    )
    inv_extra = np.float32(1.0) / pos_freqs
    rope_type = None
    if rope_scaling:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type != "yarn":
        return mx.array(inv_extra), 1.0
    factor = float(rope_scaling["factor"])
    original = rope_scaling.get("original_max_position_embeddings") or max_pos
    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0

    def correction_dim(rot):
        return (dim * math.log(original / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1
    ).astype(np.float32)
    extra_factor = np.float32(1.0) - ramp
    inv_inter = np.float32(1.0) / (np.float32(factor) * pos_freqs)
    inv = inv_inter * (1 - extra_factor) + inv_extra * extra_factor
    return mx.array(inv.astype(np.float32)), float(attention_factor)


def rope_cos_sin(angles: mx.array, dtype, scaling: float = 1.0):
    """``angles`` [L, half] f32 -> cos/sin [1, 1, L, 2*half] in ``dtype``."""
    emb = mx.concatenate([angles, angles], axis=-1)
    cos = mx.cos(emb)
    sin = mx.sin(emb)
    if scaling != 1.0:
        cos = cos * scaling
        sin = sin * scaling
    return cos.astype(dtype)[None, None], sin.astype(dtype)[None, None]


def _rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


_apply_rope_decode = mx.compile(apply_rope)


# ---------------------------------------------------------------------------
# Prompt page metadata (host side, once per prompt)
# ---------------------------------------------------------------------------


class PromptPages:
    """Page bookkeeping derived from the prompt token ids.

    Matches ``DeepseekV2Indexer._build_page_aware_ids`` (prefill query/image
    page ids with PAGE_SCOPE_PREFILL on), ``_build_page_rope_pos`` (page2d
    coordinates) and ``_uocr_dash_decode_allow`` (decode candidate state).
    """

    def __init__(self, ids: np.ndarray, p: DashParams):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        self.length = S = int(ids.shape[0])
        bs = p.block_size
        img = ids == p.image_token_id
        page_id = -1 if p.page_token_id is None else int(p.page_token_id)
        mk = ids == page_id
        cum = np.cumsum(mk.astype(np.int64))
        ar = np.arange(S)
        last_img = int(np.max(np.where(img, ar, -1))) if S else -1
        prompt_pages = int(cum[last_img]) if last_img >= 0 else 0
        marker_mode = prompt_pages > 0
        per_page = max(1, int(p.per_page_image_tokens))
        running = np.cumsum(img.astype(np.int64)) - 1
        in_prefill = ar <= last_img

        # Prefill candidate pages (query -> same-page image blocks).
        ipi = np.where(marker_mode, cum - 1, running // per_page)
        ipi = np.where(img, ipi, -1)
        raw_page = cum - 1
        resp_page = raw_page - (prompt_pages if marker_mode else 0)
        qpi = np.where(in_prefill, raw_page, resp_page)
        qpi = np.where(qpi < 0, -1, qpi)
        self.page_aware = bool(img.any()) and bool((qpi >= 0).any())
        nblk = -(-S // bs)
        pad = nblk * bs - S
        blk_pages = np.pad(ipi, (0, pad), constant_values=-1).reshape(nblk, bs)
        self.blk_max = blk_pages.max(axis=-1)
        big = np.where(blk_pages >= 0, blk_pages, 1 << 30).min(axis=-1)
        self.blk_min = np.where(self.blk_max >= 0, big, -1)
        self.query_pages = qpi

        # Decode candidate state (pages of the response restart at 0).
        dec_page = np.where(cum > 0, cum - 1, running // per_page)
        dec_page = np.where(img, dec_page, -1)
        dec_blk = np.pad(dec_page, (0, pad), constant_values=-1).reshape(nblk, bs)
        dec_max = dec_blk.max(axis=-1)
        dec_min = np.where(dec_blk >= 0, dec_blk, 1 << 30).min(axis=-1)
        dec_min = np.where(dec_max >= 0, dec_min, -1)
        self.dec_blk_min = mx.array(dec_min.astype(np.int32))
        self.dec_blk_max = mx.array(dec_max.astype(np.int32))
        self.dec_prompt_pages = prompt_pages if last_img >= 0 else 0
        self.marker_count = int(cum[-1]) if S else 0

        # page2d coordinates.
        self.coords = None
        if p.page2d and img.any() and mk.any() and marker_mode:
            page = np.where(in_prefill, cum, cum - prompt_pages)
            page = np.maximum(page, 0)
            n_pg = int(page.max()) + 1
            tok = np.zeros(S, dtype=np.int64)
            counts = np.zeros(n_pg, dtype=np.int64)
            for i, pg in enumerate(page.tolist()):
                tok[i] = counts[pg]
                counts[pg] += 1
            pf_cnt = np.bincount(page[in_prefill], minlength=n_pg)
            self.coords = (tok, page)
            self.n_pages = n_pg
            self.pf_cnt = mx.array(pf_cnt.astype(np.int32))

    def prefill_candidates(self, positions: np.ndarray, nblk: int):
        """[L, nblk] bool same-page image blocks per query (None = unconstrained)."""
        if not self.page_aware:
            return None
        qp = self.query_pages[positions][:, None]
        n0 = min(nblk, self.blk_max.shape[0])
        bmin = self.blk_min[None, :n0]
        bmax = self.blk_max[None, :n0]
        same = (qp >= 0) & (bmin >= 0) & (bmin <= qp) & (bmax >= qp)
        if n0 < nblk:
            same = np.pad(same, ((0, 0), (0, nblk - n0)))
        same = same | ~same.any(axis=-1, keepdims=True)
        return mx.array(same)


# ---------------------------------------------------------------------------
# Generation state + cache
# ---------------------------------------------------------------------------


class DashState:
    """Prompt metadata and in-graph decode state shared by every layer cache."""

    def __init__(self, params: DashParams):
        self.params = params
        self.prompt: Optional[PromptPages] = None
        self.page = None
        self.tok = None
        self.markers = None

    def start(self, prompt_ids):
        self.prompt = PromptPages(prompt_ids, self.params)
        coords = self.prompt.coords
        if coords is not None:
            self.tok = mx.array(int(coords[0][-1]), dtype=mx.int32)
            self.page = mx.array(int(coords[1][-1]), dtype=mx.int32)
        self.markers = mx.array(self.prompt.marker_count, dtype=mx.int32)


class _DashCache:
    """Router state shared by floating-point and quantized DASH caches.

    Only completed blocks can ever be routed (a query's remote blocks end
    ``local_blocks`` blocks before its own), so the cache keeps one summary key
    per completed block and the index keys of the block still being filled.
    """

    # Generic Hybrid/TurboQuant conversion cannot retain the router state.
    # Uniform quantization uses DashQuantizedKVCache below.
    preserve_auxiliary_kv_state = True

    def __init__(self, state: DashState, **kwargs):
        super().__init__(**kwargs)
        self.dash_state = state
        self.pending = None
        self.summaries = None
        self.num_summaries = 0

    def is_trimmable(self):
        return False

    def add_index_keys(self, idx_k: mx.array, head_cls: mx.array):
        """Append index keys [L, d] f32 and summarize newly completed blocks."""
        bs = self.dash_state.params.block_size
        k = idx_k if self.pending is None else mx.concatenate([self.pending, idx_k])
        n_full = k.shape[0] // bs
        if n_full:
            blocks = k[: n_full * bs].reshape(n_full, bs, -1)
            self._append_summaries(_summarize_blocks(blocks, head_cls))
        rest = k.shape[0] - n_full * bs
        self.pending = k[n_full * bs :] if rest else None

    def _append_summaries(self, s: mx.array):
        n = s.shape[0]
        need = self.num_summaries + n
        if self.summaries is None or need > self.summaries.shape[0]:
            cap = block_capacity(need)
            grown = mx.zeros((cap, s.shape[-1]), dtype=s.dtype)
            if self.summaries is not None and self.num_summaries:
                grown[: self.num_summaries] = self.summaries[: self.num_summaries]
            self.summaries = grown
        self.summaries[self.num_summaries : need] = s
        self.num_summaries = need

    def summary_buffer(self, capacity: int, dim: int):
        """Summaries padded with zeros to ``capacity`` rows (decode routing)."""
        if self.summaries is None or self.summaries.shape[0] < capacity:
            grown = mx.zeros((capacity, dim), dtype=mx.float32)
            if self.num_summaries:
                grown[: self.num_summaries] = self.summaries[: self.num_summaries]
            self.summaries = grown
        return self.summaries[:capacity]

    def block_summaries(self):
        if not self.num_summaries:
            return None
        return self.summaries[: self.num_summaries]


class DashKVCache(_DashCache, KVCache):
    """Full KV cache plus the router's block summaries."""

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        base = super().to_quantized(group_size=group_size, bits=bits)
        cache = DashQuantizedKVCache(self.dash_state, group_size=group_size, bits=bits)
        cache.keys, cache.values, cache.offset = base.keys, base.values, base.offset
        cache.pending = self.pending
        cache.summaries = self.summaries
        cache.num_summaries = self.num_summaries
        return cache


class DashQuantizedKVCache(_DashCache, QuantizedKVCache):
    """Packed K/V tensors with unquantized DASH routing and page state."""


def _summarize_blocks(blocks: mx.array, head_cls: mx.array):
    """head_cls attention inside each block -> one summary key per block."""
    d = blocks.shape[-1]
    aw = (blocks * head_cls.astype(mx.float32)).sum(-1) * (1.0 / math.sqrt(d))
    aw = mx.clip(aw, -_ACT_CLAMP, _ACT_CLAMP)
    w = mx.softmax(aw, axis=-1, precise=True)
    return (w[..., None] * blocks).sum(axis=1)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class RouterRMSNorm(nn.Module):
    """Gemma-style RMSNorm (scale = 1 + weight) with the router's fp32 clamps."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return _router_norm(x, self.weight, self.eps)


class DashRouter(nn.Module):
    def __init__(self, hidden_size: int, index_dim: int, eps: float):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.q_norm = RouterRMSNorm(index_dim, eps)
        self.k_norm = RouterRMSNorm(index_dim, eps)
        self.head_cls = mx.zeros((index_dim,))

    def index_qk(self, x, cos, sin, decode: bool = False):
        """Index query/key [L, d] f32 for hidden states ``x`` [1, L, hidden]."""
        if not isinstance(self.q_proj, nn.Linear) or "bias" in self.q_proj:
            return self._index_qk(x, cos, sin)
        fn = _router_qk_decode if decode else _router_qk
        return fn(
            x,
            self.q_proj.weight,
            self.k_proj.weight,
            self.q_norm.weight,
            self.k_norm.weight,
            cos,
            sin,
            self.q_norm.eps,
        )

    def _index_qk(self, x, cos, sin):
        hs = mx.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        q = self.q_norm(self.q_proj(hs))[:, None]
        k = self.k_norm(self.k_proj(hs))[:, None]
        return _router_rope(q, k, cos, sin)


def _router_rope(q, k, cos, sin):
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    q = mx.nan_to_num(q[0, 0].astype(mx.float32), nan=0.0, posinf=0.0, neginf=0.0)
    k = mx.nan_to_num(k[0, 0].astype(mx.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return q, k


def _router_norm(x, weight, eps):
    x32 = mx.nan_to_num(x.astype(mx.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x32 = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    x32 = mx.clip(x32, -_ACT_CLAMP, _ACT_CLAMP)
    out = mx.clip(x32 * (1.0 + weight.astype(mx.float32)), -_ACT_CLAMP, _ACT_CLAMP)
    return out.astype(x.dtype)


def _router_qk(x, q_w, k_w, q_norm_w, k_norm_w, cos, sin, eps):
    hs = mx.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    q = _router_norm(hs @ q_w.T, q_norm_w, eps)[:, None]
    k = _router_norm(hs @ k_w.T, k_norm_w, eps)[:, None]
    return _router_rope(q, k, cos, sin)


# Decode shapes are fixed (one token), so the ~40 router ops trace once.
_router_qk_decode = mx.compile(_router_qk)


def entmax15(x: mx.array):
    """1.5-entmax over the last axis of rows already sorted in descending order."""
    x = (x - x[..., :1]) / 2
    m = x.shape[-1]
    rho = mx.arange(1, m + 1, dtype=x.dtype)
    mean = mx.cumsum(x, axis=-1) / rho
    mean_sq = mx.cumsum(x * x, axis=-1) / rho
    ss = rho * (mean_sq - mean * mean)
    delta = (1 - ss) / rho
    tau = mean - mx.sqrt(mx.maximum(delta, 0))
    support = mx.maximum((tau <= x).sum(axis=-1, keepdims=True), 1)
    tau_star = mx.take_along_axis(tau, support - 1, axis=-1)
    return mx.square(mx.maximum(x - tau_star, 0))


def block_scores(idx_q, summaries, positions, nblk: int, p: DashParams):
    """[L, nblk] f32 remote block scores (``_SCORE_NEG`` = not a remote block)."""
    L = idx_q.shape[0]
    if summaries is None:
        return mx.full((L, nblk), _SCORE_NEG, dtype=mx.float32)
    n_sum = summaries.shape[0]
    z = (idx_q @ summaries.T) * (1.0 / math.sqrt(idx_q.shape[-1]))
    z = mx.clip(z, -_ACT_CLAMP, _ACT_CLAMP)
    qblk = positions[:, None] // p.block_size
    keep = mx.arange(n_sum)[None, :] <= (qblk - p.local_blocks)
    z = mx.where(keep, z, _SCORE_NEG)
    if n_sum < nblk:
        z = mx.pad(z, [(0, 0), (0, nblk - n_sum)], constant_values=_SCORE_NEG)
    return z


def route_select(scores, cand, idx_scale: float, floor: float, p: DashParams):
    """vLLM ``DashRouterMath.route_and_select`` -> (topi, bias, valid), each [L, K]."""
    nblk = scores.shape[-1]
    remote = scores > (_SCORE_NEG * 0.5)
    if cand is None:
        cand = remote
    else:
        cand = remote & cand
        cand = mx.where(cand.any(axis=-1, keepdims=True), cand, remote)
    x = mx.where(cand, scores * idx_scale, _ENTMAX_NEG)
    m = max(1, min(p.topm, nblk))
    order = mx.argsort(-x, axis=-1)[..., :m]
    xv = mx.take_along_axis(x, order, axis=-1)
    prior = entmax15(xv)
    k = max(1, min(p.topk, m))
    topi = order[..., :k]
    valid = mx.take_along_axis(cand, topi, axis=-1)
    logp = mx.log(mx.maximum(prior[..., :k], floor))
    cnt = mx.maximum(valid.sum(axis=-1, keepdims=True), 1)
    mu = mx.where(valid, logp, 0).sum(axis=-1, keepdims=True) / cnt
    bias = mx.where(valid, (logp - mu) * (1.0 / p.sigma), 0)
    return topi, bias, valid


def _decode_select(idx_q, summaries, cand, pos, idx_scale, floor, p: DashParams):
    """Routing + gather plan for one decode query.

    ``summaries`` / ``cand`` are padded to a block capacity (a multiple of
    ``_BLOCK_CAPACITY``) so the shapes, and the compiled trace, only change
    every ``_BLOCK_CAPACITY`` blocks. Padded blocks are never remote: block
    ``b`` is routable only if ``b <= pos // block_size - local_blocks``, which
    also implies it is complete. Returns the key indices [K*bs + local] and
    their additive mask, equal to ``route_select`` + the dense decode mask.
    """
    bs = p.block_size
    lw = p.local_window
    cap = summaries.shape[0]
    z = (idx_q @ summaries.T) * (1.0 / math.sqrt(idx_q.shape[-1]))
    z = mx.clip(z, -_ACT_CLAMP, _ACT_CLAMP)
    remote = mx.arange(cap)[None] <= (pos // bs - p.local_blocks)
    sel = remote & cand
    sel = mx.where(sel.any(axis=-1, keepdims=True), sel, remote)
    x = mx.where(sel, z * idx_scale, _ENTMAX_NEG)
    m = max(1, min(p.topm, cap))
    order = mx.argsort(-x, axis=-1)[..., :m]
    prior = entmax15(mx.take_along_axis(x, order, axis=-1))
    k = max(1, min(p.topk, m))
    topi = order[0, :k]
    valid = mx.take_along_axis(sel, order[..., :k], axis=-1)[0]
    logp = mx.log(mx.maximum(prior[0, :k], floor))
    mu = mx.where(valid, logp, 0).sum() / mx.maximum(valid.sum(), 1)
    bias = mx.where(valid, (logp - mu) * (1.0 / p.sigma), 0)

    blk_idx = topi[:, None] * bs + mx.arange(bs)[None, :]
    # A block token inside the local window is attended there, unbiased.
    blk_ok = valid[:, None] & (blk_idx <= pos - lw)
    blk_mask = mx.where(blk_ok, bias[:, None], -mx.inf).reshape(-1)
    loc_idx = pos - lw + 1 + mx.arange(lw)
    loc_mask = mx.where(loc_idx >= 0, 0.0, -mx.inf)
    # Masked slots of a partial or invalid block still need an in-range index.
    idx = mx.concatenate([mx.minimum(blk_idx.reshape(-1), pos), mx.maximum(loc_idx, 0)])
    return idx, mx.concatenate([blk_mask, loc_mask])


_BLOCK_CAPACITY = 64
_SELECT_FNS: Dict[int, Any] = {}


def _compiled_select(p: DashParams):
    """One compiled ``_decode_select`` shared by every layer of a model."""
    fn = _SELECT_FNS.get(id(p))
    if fn is None:
        fn = _SELECT_FNS[id(p)] = mx.compile(partial(_decode_select, p=p))
    return fn


def block_capacity(nblk: int) -> int:
    return -(-nblk // _BLOCK_CAPACITY) * _BLOCK_CAPACITY


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


@dataclass
class DashContext:
    """Per-forward inputs shared by every layer."""

    cos: mx.array
    sin: mx.array
    offset: int
    length: int
    decode: bool
    candidates: Optional[mx.array]  # prefill [L, nblk] / decode [1, capacity]
    pos: Optional[mx.array] = None  # decode query position (int32 scalar)


class DashAttention(nn.Module):
    def __init__(self, config, layer_idx: int, params: DashParams):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5
        bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.dash_router = DashRouter(dim, params.index_dim, config.rms_norm_eps)
        self.params = params
        self.idx_scale = params.idx_scale[layer_idx]
        self.floor = params.floor[layer_idx]
        # Array scalars keep one compiled decode trace for every layer.
        self._decode_consts = (
            mx.array(self.idx_scale, dtype=mx.float32),
            mx.array(self.floor, dtype=mx.float32),
        )
        self._select = _compiled_select(params)

    def __call__(self, x: mx.array, ctx: DashContext, cache: DashKVCache):
        B, L, _ = x.shape
        if B != 1:
            raise ValueError("Unlimited-OCR-2 DASH attention supports batch size 1.")
        p = self.params
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        rope = _apply_rope_decode if ctx.decode else apply_rope
        q = rope(q, ctx.cos, ctx.sin)
        k = rope(k, ctx.cos, ctx.sin)
        keys, values = cache.update_and_fetch(k, v)

        idx_q, idx_k = self.dash_router.index_qk(x, ctx.cos, ctx.sin, ctx.decode)
        cache.add_index_keys(idx_k, self.dash_router.head_cls)
        if ctx.decode:
            summaries = cache.summary_buffer(ctx.candidates.shape[-1], idx_k.shape[-1])
            idx, mask = self._select(
                idx_q, summaries, ctx.candidates, ctx.pos, *self._decode_consts
            )
            out = self._decode(q, keys, values, idx, mask, cache=cache)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
        total = ctx.offset + L
        nblk = -(-total // p.block_size)
        positions = mx.arange(ctx.offset, total)
        scores = block_scores(idx_q, cache.block_summaries(), positions, nblk, p)
        cand = ctx.candidates
        if cand is not None and cand.shape[-1] != nblk:
            cand = cand[..., :nblk]
        topi, bias, valid = route_select(scores, cand, self.idx_scale, self.floor, p)

        out = self._prefill(
            q, keys, values, topi, bias, valid, ctx.offset, nblk, cache=cache
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    def _prefill(
        self, q, keys, values, topi, bias, valid, offset, nblk, chunk=2048, cache=None
    ):
        """Selected blocks (+bias) union the local window, as an additive mask."""
        p = self.params
        L = q.shape[2]
        S = offset + L
        kpos = mx.arange(S)
        kblk = kpos // p.block_size
        blk = mx.full((L, nblk), -mx.inf, dtype=mx.float32)
        blk = mx.put_along_axis(blk, topi, mx.where(valid, bias, -mx.inf), axis=-1)
        outs = []
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            qpos = mx.arange(offset + c0, offset + c1)[:, None]
            base = blk[c0:c1][:, kblk]
            causal = kpos[None, :] <= qpos
            local = causal & (kpos[None, :] > (qpos - p.local_window))
            mask = mx.where(local, 0.0, mx.where(causal, base, -mx.inf))
            outs.append(
                scaled_dot_product_attention(
                    q[:, :, c0:c1],
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=mask.astype(q.dtype)[None, None],
                )
            )
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

    def _decode(self, q, keys, values, idx, mask, cache=None):
        """Gather the selected blocks and the local window: K*bs + local keys."""
        k, v = tree_map(lambda x: mx.take(x, idx, axis=2), (keys, values))
        return scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask.astype(q.dtype)[None, None, None],
        )


# ---------------------------------------------------------------------------
# Per-forward context
# ---------------------------------------------------------------------------


class DashRope:
    def __init__(self, params: DashParams, head_dim: int, config):
        self.params = params
        self.inv_tok, self.inv_page = (
            page2d_inv_freq(params, head_dim) if params.page2d else (None, None)
        )
        self.inv_1d, self.scaling_1d = yarn_inv_freq(
            head_dim,
            float(config.rope_theta),
            config.rope_scaling,
            int(config.max_position_embeddings),
        )

    def page2d(self, tok: mx.array, page: mx.array, dtype):
        if self.params.rope_page_mod > 0:
            page = page % self.params.rope_page_mod
        ang = mx.concatenate(
            [
                tok.astype(mx.float32)[:, None] * self.inv_tok[None],
                page.astype(mx.float32)[:, None] * self.inv_page[None],
            ],
            axis=-1,
        )
        return rope_cos_sin(ang, dtype)

    def linear(self, positions: mx.array, dtype):
        ang = positions.astype(mx.float32)[:, None] * self.inv_1d[None]
        return rope_cos_sin(ang, dtype, self.scaling_1d)


def build_context(
    state: DashState,
    rope: DashRope,
    inputs: Optional[mx.array],
    offset: int,
    length: int,
    dtype,
) -> DashContext:
    p = state.params
    prompt = state.prompt
    total = offset + length
    nblk = -(-total // p.block_size)
    if offset + length <= prompt.length:
        positions = np.arange(offset, total)
        if prompt.coords is not None:
            tok, page = prompt.coords
            cos, sin = rope.page2d(
                mx.array(tok[positions]), mx.array(page[positions]), dtype
            )
        else:
            cos, sin = rope.linear(mx.array(positions), dtype)
        cand = prompt.prefill_candidates(positions, nblk)
        return DashContext(cos, sin, offset, length, False, cand)

    if length != 1 or offset < prompt.length:
        raise ValueError(
            "Unlimited-OCR-2 decodes one token at a time after the prompt."
        )
    token = inputs.reshape(-1)[-1].astype(mx.int32)
    is_page = token == (-1 if p.page_token_id is None else int(p.page_token_id))
    state.markers = state.markers + is_page.astype(mx.int32)
    if prompt.coords is not None:
        page = mx.where(
            is_page, mx.minimum(state.page + 1, prompt.n_pages - 1), state.page
        )
        tok = mx.where(is_page, prompt.pf_cnt[page], state.tok + 1)
        state.page, state.tok = page, tok
        cos, sin = rope.page2d(tok.reshape(1), page.reshape(1), dtype)
    else:
        cos, sin = rope.linear(mx.array([offset]), dtype)

    cap = block_capacity(nblk)
    if p.image_only:
        qp = state.markers - 1 - prompt.dec_prompt_pages
        qp = mx.where(qp >= 0, qp, -1)
        bmin, bmax = prompt.dec_blk_min, prompt.dec_blk_max
        same = (bmin >= 0) & (bmin <= qp) & (bmax >= qp) & (qp >= 0)
        open_row = (qp < 0) | ~same.any()
        n0 = min(cap, same.shape[0])
        allow = same[:n0]
        if n0 < cap:
            allow = mx.concatenate([allow, mx.zeros((cap - n0,), dtype=mx.bool_)])
        cand = (allow | open_row)[None]
    else:
        cand = mx.ones((1, cap), dtype=mx.bool_)
    pos = mx.array(offset, dtype=mx.int32)
    return DashContext(cos, sin, offset, 1, True, cand, pos)


def make_dash_cache(params: DashParams, num_layers: int) -> list:
    state = DashState(params)
    return [DashKVCache(state) for _ in range(num_layers)]


__all__ = [
    "DashAttention",
    "DashContext",
    "DashKVCache",
    "DashParams",
    "DashQuantizedKVCache",
    "DashRope",
    "DashState",
    "build_context",
    "is_dash_config",
    "make_dash_cache",
]
