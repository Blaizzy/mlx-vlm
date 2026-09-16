"""TDD: quantized SDPA mask broadcast under GQA multi-row batching (#1567)."""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.models.base import align_attention_mask_to_scores

GROUP = 64
BITS = 8


class TestAlignAttentionMaskToScores:
    def test_str_passthrough(self):
        scores = mx.zeros((2, 8, 2, 4, 4))
        assert align_attention_mask_to_scores("causal", scores) == "causal"
