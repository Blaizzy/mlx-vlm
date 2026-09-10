import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class SharedIndexState:
    """Per-forward handoff between index layers, replacing the reference global.

    Layers run in order and every source writes before its consumers read. The
    candidate source must run on every step before deeper indexers consume its mask.
    """

    def __init__(self):
        self.index_k = None
        self.candidates = None


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
    trimmed = width - width % block_size
    scores = (
        logits[..., :trimmed].reshape(*logits.shape[:-1], -1, block_size).max(axis=-1)
    )
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
    layers publish index keys; every other indexer reads them from shared state. The fp4
    QAT quantization of queries and keys lands with the quantization work.
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
        if start_pos == 0:
            positions = mx.arange(n_latent) * ratio
        else:
            positions = mx.array([start_pos + 1 - ratio])
        cos, sin = _index_cos_sin(
            start_pos + latent.shape[1] + 1 if start_pos == 0 else start_pos + 2,
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
        base = start_pos // ratio
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

        index_k = shared.index_k[:batch, : end_pos // ratio].astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32) * (
            self.scale * self.n_heads**-0.5
        )
        scores = mx.einsum("bsnd,btd->bsnt", q, index_k)
        scores = mx.maximum(scores, 0) * weights[..., None]
        scores = scores.sum(axis=2)

        if start_pos == 0:
            compress_lens = (mx.arange(1, seqlen + 1) // ratio)[:, None]
            visible = mx.arange(index_k.shape[1])[None, :] < compress_lens
            scores = mx.where(visible, scores, -mx.inf)
        else:
            compress_lens = end_pos // ratio

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
