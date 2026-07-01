from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

# DSpark draft checkpoints store plain RMSNorm weights (no Gemma "1 + weight" shift),
# so reuse Gemma 4's no-shift norm + GeGLU + softcap primitives directly.
from ....models.gemma4.language import (
    RMSNormNoScale,
    RMSNormZeroShift,
    geglu,
    logit_softcap,
)
from .config import Gemma4DSparkConfig
from .heads import DSparkConfidenceHead, DSparkMarkovHead


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [b, seq, heads, hd]; cos/sin: [seq, hd] (NeoX half-split convention)."""
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
    return x * cos + _rotate_half(x) * sin


def rope_tables(
    position_ids: mx.array, head_dim: int, theta: float, partial: float
) -> Tuple[mx.array, mx.array]:
    """Proportional (partial) RoPE: the first ``partial*head_dim`` dims rotate, the rest
    pass through (zero frequency → cos 1 / sin 0). Returns cos/sin of shape [seq, head_dim].
    """
    rope_angles = int(partial * head_dim // 2)
    inv_rot = 1.0 / (
        theta ** (mx.arange(0, 2 * rope_angles, 2).astype(mx.float32) / head_dim)
    )
    nope = head_dim // 2 - rope_angles
    inv_freq = (
        mx.concatenate([inv_rot, mx.zeros((nope,), dtype=mx.float32)])
        if nope > 0
        else inv_rot
    )
    freqs = position_ids.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


class Gemma4DSparkAttention(nn.Module):
    """Gemma 4 global attention for the DSpark draft: K=V sharing, attention scale 1.0,
    partial RoPE, q/k RMSNorm, weightless v RMSNorm. The committed context K/V live in the
    persistent draft ``KVCache``; the proposed block's K/V are concatenated transiently so
    only context is retained across rounds (the block is re-drafted each round)."""

    def __init__(self, config: Gemma4DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.nh = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.hd = config.head_dim
        self.k_eq_v = config.attention_k_eq_v
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(h, self.nkv * self.hd, bias=False)
        self.v_proj = (
            None if self.k_eq_v else nn.Linear(h, self.nkv * self.hd, bias=False)
        )
        self.o_proj = nn.Linear(self.nh * self.hd, h, bias=False)
        self.q_norm = RMSNormZeroShift(self.hd, config.rms_norm_eps)
        self.k_norm = RMSNormZeroShift(self.hd, config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.hd, config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        x_ctx: mx.array,
        cache: KVCache,
        cos_ctx: mx.array,
        sin_ctx: mx.array,
        cos_blk: mx.array,
        sin_blk: mx.array,
    ) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]

        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.nh, self.hd))
        ck = self.k_proj(x_ctx)
        cv = ck if self.k_eq_v else self.v_proj(x_ctx)
        pk = self.k_proj(x)
        pv = pk if self.k_eq_v else self.v_proj(x)
        ctx_keys = self.k_norm(ck.reshape(B, S, self.nkv, self.hd))
        ctx_values = self.v_norm(cv.reshape(B, S, self.nkv, self.hd))
        prop_keys = self.k_norm(pk.reshape(B, L, self.nkv, self.hd))
        prop_values = self.v_norm(pv.reshape(B, L, self.nkv, self.hd))

        queries = _apply_rope(queries, cos_blk, sin_blk).transpose(0, 2, 1, 3)
        ctx_keys = _apply_rope(ctx_keys, cos_ctx, sin_ctx).transpose(0, 2, 1, 3)
        prop_keys = _apply_rope(prop_keys, cos_blk, sin_blk).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.transpose(0, 2, 1, 3)  # value is not rotated
        prop_values = prop_values.transpose(0, 2, 1, 3)

        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        # Whole block denoised at once → block self-attention is non-causal.
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0, mask=None
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4DSparkConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Gemma4DSparkLayer(nn.Module):
    """Gemma 4 decoder layer: four sandwich norms + per-layer ``layer_scalar``, GeGLU MLP."""

    def __init__(self, config: Gemma4DSparkConfig):
        super().__init__()
        h, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = Gemma4DSparkAttention(config)
        self.mlp = Gemma4MLP(config)
        self.input_layernorm = RMSNormZeroShift(h, eps)
        self.post_attention_layernorm = RMSNormZeroShift(h, eps)
        self.pre_feedforward_layernorm = RMSNormZeroShift(h, eps)
        self.post_feedforward_layernorm = RMSNormZeroShift(h, eps)
        self.layer_scalar = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x, x_ctx, cache, cos_ctx, sin_ctx, cos_blk, sin_blk) -> mx.array:
        h = self.post_attention_layernorm(
            self.self_attn(
                self.input_layernorm(x),
                x_ctx,
                cache,
                cos_ctx,
                sin_ctx,
                cos_blk,
                sin_blk,
            )
        )
        x = x + h
        h = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        x = x + h
        return x * self.layer_scalar


class Gemma4DSparkDraftModel(nn.Module):
    """DSpark self-speculative drafter for a Gemma 4 target.

    Conforms to the mlx-vlm speculative round-loop contract (``reset``/``make_cache``/
    ``draft_block`` + ``config.target_layer_ids`` + ``accept_lens``/``draft_lens``). The
    drafter is standalone — it ships its own ``embed_tokens``/``lm_head`` and does not bind
    to the target. ``draft_block`` proposes ``block_size`` tokens via Gemma decoder layers
    cross-attending to the projected target hidden, with a low-rank Markov logit bias chained
    across the block; ``_last_confidence`` exposes the (advisory) acceptance scores.
    """

    def __init__(self, config: Gemma4DSparkConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.fc = nn.Linear(config.fc_in, config.hidden_size, bias=False)
        self.hidden_norm = RMSNormZeroShift(config.hidden_size, config.rms_norm_eps)
        self.layers = [
            Gemma4DSparkLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNormZeroShift(config.hidden_size, config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(config.vocab_size, config.markov_rank)
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + config.markov_rank,
            bias=True,
        )
        self.embed_scale = float(config.hidden_size) ** 0.5
        self._last_confidence: Optional[mx.array] = None
        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def reset(self, target_model: Any = None) -> List[KVCache]:
        # DSpark is standalone (own embed/head) — nothing to bind to the target.
        self.accept_lens = []
        self.draft_lens = []
        self._last_confidence = None
        return self.make_cache()

    def _embed(self, ids: mx.array) -> mx.array:
        # Gemma scales token embeddings by sqrt(hidden) in the embedding's own dtype.
        e = self.embed_tokens(ids)
        return e * mx.array(self.embed_scale, dtype=e.dtype)

    def _rope(self, start: int, length: int) -> Tuple[mx.array, mx.array]:
        pos = mx.arange(start, start + length)
        return rope_tables(
            pos,
            self.config.head_dim,
            self.config.rope_theta,
            self.config.partial_rotary_factor,
        )

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden.astype(mx.float32))
        softcap = self.config.final_logit_softcapping
        if softcap:
            logits = logit_softcap(softcap, logits)
        return logits

    def _block_hidden(
        self, block_ids: mx.array, target_hidden: mx.array, cache: List[KVCache]
    ) -> mx.array:
        h = self._embed(block_ids)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        offset = cache[0].offset
        S, L = h_ctx.shape[1], h.shape[1]
        cos_ctx, sin_ctx = self._rope(offset, S)
        cos_blk, sin_blk = self._rope(offset + S, L)
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, c, cos_ctx, sin_ctx, cos_blk, sin_blk)
        return self.norm(h)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List[KVCache],
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        if isinstance(last_bonus, int):
            anchor = mx.array([[last_bonus]], dtype=token_dtype)
        else:
            anchor = last_bonus.reshape(-1, 1).astype(token_dtype)
        B = anchor.shape[0]
        masks = mx.full(
            (B, block_size - 1), int(self.config.mask_token_id), dtype=token_dtype
        )
        block_ids = mx.concatenate([anchor, masks], axis=1)

        block_hidden = self._block_hidden(block_ids, hidden, cache)
        base_logits = self._logits(block_hidden)  # [B, block_size, V]

        prev = anchor[:, 0]
        drafts, markov_embeds = [], []
        for i in range(block_size):
            bias, embed = self.markov_head(prev)
            prev = sampler(base_logits[:, i] + bias).astype(token_dtype)
            drafts.append(prev)
            markov_embeds.append(embed)
        draft_tokens = mx.stack(drafts, axis=1)  # [B, block_size]
        markov_embed = mx.stack(markov_embeds, axis=1)  # [B, block_size, rank]
        self._last_confidence = self.confidence_head(
            block_hidden, markov_embed
        )  # [B, block_size]
        return draft_tokens

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            out[k] = v
        return out
