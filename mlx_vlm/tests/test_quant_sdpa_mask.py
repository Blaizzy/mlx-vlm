"""TDD: quantized SDPA mask broadcast under GQA multi-row batching (#1567)."""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.base import (
    align_attention_mask_to_scores,
    quantized_scaled_dot_product_attention,
)
from mlx_vlm.models.cache import create_causal_mask

GROUP = 64
BITS = 8


def _quant_kv(B, n_kv, L, D, dtype=mx.float16):
    keys = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    values = mx.random.normal((B, n_kv, L, D)).astype(dtype)
    return (
        mx.quantize(keys, group_size=GROUP, bits=BITS),
        mx.quantize(values, group_size=GROUP, bits=BITS),
    )


class TestAlignAttentionMaskToScores:
    def test_none_passthrough(self):
        scores = mx.zeros((2, 8, 2, 4, 4))
        assert align_attention_mask_to_scores(None, scores) is None

    def test_str_passthrough(self):
        scores = mx.zeros((2, 8, 2, 4, 4))
        assert align_attention_mask_to_scores("causal", scores) == "causal"

    def test_4d_batch_mask_to_5d_gqa_scores(self):
        """Server crash geometry: mask (B,1,L,K) vs scores (B,H_kv,G,L,K)."""
        B, H_kv, G, L, K = 2, 8, 2, 18, 18
        scores = mx.zeros((B, H_kv, G, L, K))
        mask = create_causal_mask(L, offset=0, left_padding=mx.array([1, 0]))
        assert mask.shape == (B, 1, L, K)

        aligned = align_attention_mask_to_scores(mask, scores)
        assert aligned is not None
        # Must broadcast onto scores without aliasing B onto H_kv
        out = mx.where(aligned, scores, mx.array(-1.0))
        mx.eval(out)
        assert out.shape == scores.shape

    def test_live_server_shapes(self):
        """Exact shapes from #1567 concurrent curl repro."""
        scores = mx.zeros((2, 8, 2, 18, 18))
        mask = mx.ones((2, 1, 18, 18), dtype=mx.bool_)
        aligned = align_attention_mask_to_scores(mask, scores)
        mx.eval(mx.where(aligned, scores, mx.array(0.0)))

    def test_2d_causal_still_broadcasts(self):
        scores = mx.zeros((2, 8, 2, 4, 4))
        mask = create_causal_mask(4, offset=0)  # (4, 4), no batch pad
        aligned = align_attention_mask_to_scores(mask, scores)
        mx.eval(mx.where(aligned, scores, mx.array(0.0)))


class TestQuantizedSdpaMultiRowGqa:
    def test_batch2_gqa_with_left_padding_mask(self):
        """End-to-end quant SDPA: B=2, GQA, batch left-pad mask (the #1567 path)."""
        B, n_q, n_kv, L, D = 2, 16, 8, 4, GROUP  # n_repeats=2
        queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
        q_keys, q_values = _quant_kv(B, n_kv, L, D)
        mask = create_causal_mask(L, offset=0, left_padding=mx.array([1, 0]))

        out = quantized_scaled_dot_product_attention(
            queries,
            q_keys,
            q_values,
            scale=D**-0.5,
            mask=mask,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(out)
        assert out.shape == (B, n_q, L, D)

    def test_batch2_gqa_matches_server_head_layout(self):
        """Qwen3-0.6B-like: 16 Q / 8 KV in smoke used (2,8,2,L,K) scores."""
        B, n_q, n_kv, L, D = 2, 16, 8, 18, GROUP
        queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
        q_keys, q_values = _quant_kv(B, n_kv, L, D)
        mask = create_causal_mask(L, offset=0, left_padding=mx.array([0, 0]))

        out = quantized_scaled_dot_product_attention(
            queries,
            q_keys,
            q_values,
            scale=D**-0.5,
            mask=mask,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(out)
        assert out.shape == (B, n_q, L, D)

    def test_batch1_gqa_still_works(self):
        B, n_q, n_kv, L, D = 1, 16, 8, 4, GROUP
        queries = mx.random.normal((B, n_q, L, D)).astype(mx.float16)
        q_keys, q_values = _quant_kv(B, n_kv, L, D)
        mask = create_causal_mask(L, offset=0, left_padding=mx.array([0]))

        out = quantized_scaled_dot_product_attention(
            queries,
            q_keys,
            q_values,
            scale=D**-0.5,
            mask=mask,
            group_size=GROUP,
            bits=BITS,
        )
        mx.eval(out)
        assert out.shape == (B, n_q, L, D)
