"""DSpark draft attention for a DeepSeek-V4 target.

A faithful port of the reference ``inference/model.py::DSparkAttention`` (the
``compress_ratio == 0`` windowed MLA variant): low-rank Q (``wq_a -> q_norm -> wq_b``)
with a per-head RMS, a single shared KV head, a sliding-window KV ring buffer, top-k
sparse attention with a denominator sink, and a grouped low-rank output projection.

The reference's TileLang ``sparse_attn`` kernel and interleaved-pair RoPE are reimplemented
in plain MLX here (the draft block is tiny — K≈5 query positions — so a dense formulation
is ample). There is no KV compressor/indexer; those live on the base target. The window is
passed in (not module-state) so the speculative round loop can keep one per sequence.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import DeepseekV4DSparkConfig


class RMSNorm(nn.Module):
    """fp32 RMSNorm matching the reference (compute in fp32, cast back)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xf = x.astype(mx.float32)
        xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (self.weight * xf).astype(dtype)


def precompute_rope(
    dim: int, seqlen: int, base: float = 10000.0
) -> Tuple[mx.array, mx.array]:
    """No-YaRN RoPE tables. Returns ``(cos, sin)`` each ``[seqlen, dim/2]``."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(seqlen).astype(mx.float32)
    theta = mx.outer(t, freqs)
    return mx.cos(theta), mx.sin(theta)


def apply_rotary_emb(
    x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False
) -> mx.array:
    """Interleaved-pair RoPE (reference ``apply_rotary_emb``): consecutive dims are
    (real, imag). ``x`` is ``[b, s, rd]`` or ``[b, s, h, rd]`` with position axis 1."""
    rd = x.shape[-1]
    half = rd // 2
    seqlen = x.shape[1]
    xf = x.astype(mx.float32).reshape(*x.shape[:-1], half, 2)
    xr, xi = xf[..., 0], xf[..., 1]
    if x.ndim == 3:
        c = cos.reshape(1, seqlen, half)
        s = sin.reshape(1, seqlen, half)
    else:
        c = cos.reshape(1, seqlen, 1, half)
        s = sin.reshape(1, seqlen, 1, half)
    if inverse:
        s = -s
    out = mx.stack([xr * c - xi * s, xr * s + xi * c], axis=-1).reshape(x.shape)
    return out.astype(x.dtype)


def _rope_last(
    x: mx.array, cos: mx.array, sin: mx.array, rd: int, inverse: bool = False
) -> mx.array:
    """Rotate only the last ``rd`` dims (the rope slice); pass the nope dims through."""
    return mx.concatenate(
        [x[..., :-rd], apply_rotary_emb(x[..., -rd:], cos, sin, inverse)], axis=-1
    )


def sparse_attn(
    q: mx.array,
    kv: mx.array,
    attn_sink: mx.array,
    topk_idxs: mx.array,
    softmax_scale: float,
) -> mx.array:
    """Top-k gathered single-KV-head attention with a denominator-only sink.

    q: ``[b, m, h, d]``; kv: ``[b, n, d]``; attn_sink: ``[h]``; topk_idxs: ``[b, m, t]``
    (``-1`` masks a slot). Returns ``[b, m, h, d]``.
    """
    b, m, h, d = q.shape
    t = topk_idxs.shape[-1]

    mask = topk_idxs != -1
    safe = mx.where(mask, topk_idxs, 0).astype(mx.int32)
    flat = safe.reshape(b, m * t)
    idx_exp = mx.broadcast_to(flat[:, :, None], (b, m * t, d)).astype(mx.int32)
    kv_g = mx.take_along_axis(kv.astype(mx.float32), idx_exp, axis=1).reshape(
        b, m, t, d
    )

    qf = q.astype(mx.float32)
    scores = mx.matmul(qf, mx.swapaxes(kv_g, -1, -2)) * softmax_scale  # [b, m, h, t]
    scores = mx.where(mask[:, :, None, :], scores, -mx.array(float("inf")))

    smax = mx.max(scores, axis=-1, keepdims=True)
    ex = mx.exp(scores - smax)
    num = mx.matmul(ex, kv_g)
    sink = mx.exp(attn_sink[None, None, :, None] - smax)
    denom = mx.sum(ex, axis=-1, keepdims=True) + sink
    return (num / denom).astype(q.dtype)


def _dspark_topk_idxs(
    window_size: int, bsz: int, block_size: int, start_pos: int
) -> mx.array:
    win = mx.arange(min(window_size, start_pos + 1))
    block = window_size + mx.arange(block_size)
    row = mx.concatenate([win, block]).astype(mx.int32)
    return mx.broadcast_to(row[None, None, :], (bsz, block_size, row.shape[0]))


class DSparkKVCache:
    """Sliding-window ring buffer for the drafter's shared single-head KV.

    Plain object (not an ``nn.Module``) so it never lands in a parameter tree. Slot
    ``p % window_size`` holds the KV for absolute position ``p``; attention is order-
    agnostic over the window, so the ring ordering is irrelevant to correctness.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window: Optional[mx.array] = None  # [b, win, head_dim]

    def prefill(self, main_kv: mx.array) -> None:
        b, seqlen, d = main_kv.shape
        win = self.window_size
        if seqlen <= win:
            pad = mx.zeros((b, win - seqlen, d), dtype=main_kv.dtype)
            self.window = mx.concatenate([main_kv, pad], axis=1)
        else:
            cutoff = seqlen % win
            last = main_kv[:, seqlen - win :]
            self.window = mx.concatenate(
                [last[:, win - cutoff :], last[:, : win - cutoff]], axis=1
            )

    def update(self, position: int, v: mx.array) -> None:
        win = self.window_size
        sel = mx.arange(win) == (position % win)
        self.window = mx.where(sel[None, :, None], v[:, None, :], self.window)

    def read(self) -> mx.array:
        return self.window


class DSparkAttention(nn.Module):
    def __init__(self, config: DeepseekV4DSparkConfig, max_seq_len: int = 8192):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.softmax_scale = config.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            self.n_groups * self.o_lora_rank, config.hidden_size, bias=False
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        cos, sin = precompute_rope(self.rope_head_dim, max_seq_len, config.rope_theta)
        self._cos, self._sin = cos, sin

    def _main_kv(self, main_x: mx.array, start_pos: int) -> mx.array:
        rd = self.rope_head_dim
        seqlen = main_x.shape[1]
        main_kv = self.kv_norm(self.wkv(main_x))
        return _rope_last(
            main_kv,
            self._cos[start_pos : start_pos + seqlen],
            self._sin[start_pos : start_pos + seqlen],
            rd,
        )

    def __call__(
        self,
        x: mx.array,
        start_pos: int,
        main_x: mx.array,
        window: DSparkKVCache,
    ) -> mx.array:
        rd = self.rope_head_dim
        b, seqlen, _ = main_x.shape

        main_kv = self._main_kv(main_x, start_pos)
        if start_pos == 0:
            window.prefill(main_kv)
            return x

        _, block_size, _ = x.shape
        bcos = self._cos[start_pos + seqlen : start_pos + seqlen + block_size]
        bsin = self._sin[start_pos + seqlen : start_pos + seqlen + block_size]

        q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(
            b, block_size, self.n_heads, self.head_dim
        )
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + self.eps)
        q = _rope_last(q, bcos, bsin, rd)
        kv = _rope_last(self.kv_norm(self.wkv(x)), bcos, bsin, rd)

        window.update(start_pos, main_kv[:, 0])
        kv_full = mx.concatenate([window.read(), kv], axis=1)
        topk = _dspark_topk_idxs(self.window_size, b, block_size, start_pos)
        o = sparse_attn(q, kv_full, self.attn_sink, topk, self.softmax_scale)
        o = _rope_last(o, bcos, bsin, rd, inverse=True)

        o = o.reshape(b, block_size, self.n_groups, -1)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        # out[..,g,r] = sum_d o[..,g,d] * wo_a[g,r,d]
        o = mx.sum(o[..., None, :] * wo_a[None, None, :, :, :], axis=-1)
        return self.wo_b(o.reshape(b, block_size, self.n_groups * self.o_lora_rank))

    def advance_window(
        self, main_x: mx.array, position: int, window: DSparkKVCache
    ) -> None:
        """Append one committed token's main KV to the window without drafting."""
        main_kv = self._main_kv(main_x[:, None, :], position)
        window.update(position, main_kv[:, 0])

    def seed_window(self, main_x: mx.array, window: DSparkKVCache) -> None:
        """Vectorized initial window fill from the full committed context (positions 0..S-1)."""
        window.prefill(self._main_kv(main_x, 0))

    def draft(
        self, x: mx.array, block_start: int, win_valid: int, window: DSparkKVCache
    ) -> mx.array:
        """Draft a block attending to the already-seeded window (eager round-loop path).

        Unlike :meth:`__call__` decode, no anchor main KV is added — the committed context
        is already in ``window`` (the round loop feeds it via :meth:`advance_window` /
        :meth:`seed_window`). The block's ``block_size`` query slots sit at rope positions
        ``block_start .. block_start + block_size - 1``; ``win_valid`` is the number of valid
        window positions (``min(window_size, committed)``).
        """
        rd = self.rope_head_dim
        b, block_size, _ = x.shape
        bcos = self._cos[block_start : block_start + block_size]
        bsin = self._sin[block_start : block_start + block_size]

        q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(
            b, block_size, self.n_heads, self.head_dim
        )
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + self.eps)
        q = _rope_last(q, bcos, bsin, rd)
        kv = _rope_last(self.kv_norm(self.wkv(x)), bcos, bsin, rd)

        kv_full = mx.concatenate([window.read(), kv], axis=1)
        win = mx.arange(min(self.window_size, win_valid))
        block = self.window_size + mx.arange(block_size)
        row = mx.concatenate([win, block]).astype(mx.int32)
        topk = mx.broadcast_to(row[None, None, :], (b, block_size, row.shape[0]))
        o = sparse_attn(q, kv_full, self.attn_sink, topk, self.softmax_scale)
        o = _rope_last(o, bcos, bsin, rd, inverse=True)

        o = o.reshape(b, block_size, self.n_groups, -1)
        wo_a = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        o = mx.sum(o[..., None, :] * wo_a[None, None, :, :, :], axis=-1)
        return self.wo_b(o.reshape(b, block_size, self.n_groups * self.o_lora_rank))
