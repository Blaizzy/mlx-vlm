"""Regression tests for the Gemma 4 blockwise bidirectional vision mask overlay.

Continuous batching can hand the language model a ``(B, 1, q, kv)`` attention
mask with ``q != kv`` (a chunked-prefill or coalesced-decode step). The
bidirectional "same image block" overlay is built over the full key sequence
-> ``(B, 1, kv, kv)``, so ``base_mask | overlay`` used to broadcast-fail and
500 the whole batch::

    ValueError: [broadcast_shapes] Shapes (2,1,16,583) and (1,1,583,583) ...

The overlay is only meaningful on a square (``q == kv``) full-sequence prefill
mask; on a non-square step it must be skipped (fall back to the causal base
mask) rather than crash.
"""

import mlx.core as mx

from mlx_vlm.models.gemma4.language import Gemma4TextModel


def _new_model():
    # The overlay and _block_sequence_ids_for_mask read no weights or config, so
    # an uninitialised instance drives the exact production code path cheaply.
    return Gemma4TextModel.__new__(Gemma4TextModel)


def _causal(q, kv):
    # Boolean causal mask for the last `q` query positions of a length-`kv`
    # sequence: query row i may attend key j when j <= (kv - q) + i.
    qi = mx.arange(kv - q, kv)[:, None]
    kj = mx.arange(kv)[None, :]
    return qi >= kj


def _image_block_token_types(batch, length, lo, hi):
    mm = mx.zeros((batch, length), dtype=mx.int32)
    mm[:, lo:hi] = 1  # token_type 1 == vision
    return mm


def test_nonsquare_batched_mask_does_not_crash():
    # Miniature of the production shapes: batch=2, q=4, kv=12, one image block.
    model = _new_model()
    base_mask = mx.broadcast_to(_causal(4, 12), (2, 1, 4, 12))
    mm = _image_block_token_types(2, 12, 3, 9)

    out = model._apply_blockwise_bidirectional_overlay(base_mask, mm)
    mx.eval(out)

    assert out.shape == (2, 1, 4, 12)
    # q != kv -> the overlay cannot apply; the causal base mask is returned as-is.
    assert mx.array_equal(out, base_mask).item()


def test_square_prefill_mask_still_applies_overlay():
    # Square (q == kv) prefill: the overlay MUST still open intra-image-block
    # bidirectional attention that the causal base mask forbade.
    model = _new_model()
    size = 12
    base_mask = mx.broadcast_to(_causal(size, size), (1, 1, size, size))
    mm = _image_block_token_types(1, size, 3, 9)

    out = model._apply_blockwise_bidirectional_overlay(base_mask, mm)
    mx.eval(out)

    assert out.shape == (1, 1, size, size)
    # Query 4 attending later key 7 (same image block): causal=False -> overlay True.
    assert bool(out[0, 0, 4, 7].item())
    assert not bool(base_mask[0, 0, 4, 7].item())
    # A non-vision query keeps causal masking (query 1 cannot see future key 5).
    assert not bool(out[0, 0, 1, 5].item())


def test_make_masks_cached_chunk_with_vision_tokens_does_not_crash():
    # End-to-end through the real _make_masks: a cached-chunk prefill step
    # (KV offset) that carries vision tokens reproduces the production crash —
    # base_mask is (B, 1, q, kv) with q != kv while the overlay spans the full kv.
    from mlx_vlm.models import cache, gemma4_unified

    config = gemma4_unified.TextConfig(
        hidden_size=8,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        vocab_size=32,
        hidden_size_per_layer_input=0,
        sliding_window=8,
        sliding_window_pattern=1,
        layer_types=["full_attention"],
        use_bidirectional_attention="vision",
    )
    model = Gemma4TextModel(config)
    hidden_states = mx.zeros((1, 4, config.hidden_size))  # current chunk: q = 4
    kv_cache = cache.KVCache()
    kv_cache.offset = 8  # 8 already cached -> kv = 12
    # Full-sequence token types (cache + current chunk) with a vision block.
    mm_token_type_ids = mx.array([[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

    mask = model._make_masks(hidden_states, [kv_cache], mm_token_type_ids)[0]
    mx.eval(mask)

    # The non-square step falls back to the causal base mask (q=4, kv=12) — no crash.
    assert not isinstance(mask, str)
    assert mask.shape[-2] == 4
    assert mask.shape[-1] == 12
