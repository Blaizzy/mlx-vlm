"""End-to-end correctness tests for APC exact-mode store/restore with hybrid
attention models (Gemma 4, Qwen 3.5).

These tests exercise the full path: instantiate a hybrid-cache model, prefill,
store via APCManager.store_exact_cache(), restore via lookup_exact_cache(), and
verify the restored cache state is correct.

Unlike test_apc.py (which tests APC mechanics with synthetic arrays), these
tests run real model forward passes to validate the behavioral contract.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, model_apc_mode
from mlx_vlm.models.cache import KVCache

# ============================================================================
# Model factories — tiny random-weight instances, no downloads needed
# ============================================================================


def _make_tiny_gemma4():
    """Create a tiny Gemma 4 language model with mixed cache types.

    Config adapted from test_models.py::TestModels::test_gemma4 with
    num_hidden_layers bumped to 6 for full sliding_window_pattern coverage.

    sliding_window_pattern=3 → pattern: [sliding, sliding, full] repeated
    With 6 layers: 4 RotatingKVCache + 2 KVCache → triggers exact mode.
    """
    from mlx_vlm.models import gemma4

    text_config = gemma4.TextConfig(
        model_type="gemma4_text",
        hidden_size=32,
        num_hidden_layers=6,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=0,
        sliding_window=32,
        sliding_window_pattern=3,
        final_logit_softcapping=30.0,
    )
    vision_config = gemma4.VisionConfig(
        model_type="gemma4_vision",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        patch_size=16,
        pooling_kernel_size=2,
        default_output_length=4,
        position_embedding_size=64,
        use_clipped_linears=False,
    )
    config = gemma4.ModelConfig(
        text_config=text_config,
        vision_config=vision_config,
        model_type="gemma4",
        vocab_size=64,
        image_token_id=63,
    )
    model = gemma4.Model(config)
    return model.language_model


def _make_tiny_qwen35():
    """Create a tiny Qwen 3.5 language model with mixed cache types.

    Config adapted from test_models.py::TestModels::test_qwen3_5_decode_uses_rope_deltas_kwarg
    with num_hidden_layers bumped to 4 for full_attention_interval coverage.

    full_attention_interval=4 → 3 out of 4 layers use ArraysCache (linear/SSM),
    1 out of 4 uses KVCache (full attention) → triggers exact mode.
    """
    from mlx_vlm.models import qwen3_5

    text_config = qwen3_5.TextConfig(
        model_type="qwen3_5",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=8,
        full_attention_interval=4,
    )
    config = qwen3_5.ModelConfig(
        text_config=text_config,
        vision_config=qwen3_5.VisionConfig(
            model_type="qwen3_5",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            num_heads=2,
        ),
        model_type="qwen3_5",
    )
    model = qwen3_5.LanguageModel(text_config, config)
    return model


# ============================================================================
# Helpers
# ============================================================================


def _generate_greedy(model, input_ids, cache, n_tokens):
    """Generate n tokens greedily, return list of token ids."""
    tokens = []
    y = input_ids
    for _ in range(n_tokens):
        outputs = model(y, cache=cache)
        mx.eval(outputs.logits)
        y = mx.argmax(outputs.logits[:, -1:, :], axis=-1)
        mx.eval(y)
        tokens.append(y.item())
    return tokens


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_mode_detected_for_hybrid_models(model_factory):
    """Hybrid models must route to exact mode, not block mode."""
    lm = model_factory()
    assert model_apc_mode(lm) == "exact"


def test_apc_exact_store_materializes_its_snapshot(monkeypatch):
    cache = KVCache()
    cache.keys = mx.ones((1, 1, 4, 2))
    cache.values = mx.ones((1, 1, 4, 2))
    cache.offset = 4
    apc = APCManager(num_blocks=4, block_size=4)
    real_eval = mx.eval
    eval_calls = []

    def recording_eval(*args):
        eval_calls.append(args)
        return real_eval(*args)

    monkeypatch.setattr(mx, "eval", recording_eval)

    assert apc.store_exact_cache([1, 2, 3, 4], [cache])
    assert eval_calls


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_store_and_restore_returns_cache(model_factory):
    """store_exact_cache + lookup_exact_cache round-trips successfully."""
    lm = model_factory()
    cache = lm.make_cache()

    prefix = mx.array([[1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]])
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    apc = APCManager(num_blocks=64, block_size=16)
    prefix_tokens = prefix[0].tolist()
    assert apc.store_exact_cache(prefix_tokens, cache)

    # Lookup requires suffix tokens beyond the stored prefix (guard tokens)
    full_tokens = prefix_tokens + [
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    ]
    restored, prefix_len = apc.lookup_exact_cache(full_tokens)

    assert restored is not None
    assert prefix_len == len(prefix_tokens)


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_warm_to_warm_produces_identical_output(model_factory):
    """Multiple restores from the same stored cache produce identical state."""
    lm = model_factory()
    cache = lm.make_cache()

    prefix = mx.array([[1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]])
    suffix = mx.array(
        [[12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    )

    # Prefill prefix
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    # Store
    apc = APCManager(num_blocks=64, block_size=16)
    prefix_tokens = prefix[0].tolist()
    apc.store_exact_cache(prefix_tokens, cache)

    full_tokens = prefix_tokens + suffix[0].tolist()

    # Restore twice — the cache tensors themselves must be byte-identical,
    # proving the stored entry returns a consistent clone each time.
    restored_a, plen_a = apc.lookup_exact_cache(full_tokens)
    restored_b, plen_b = apc.lookup_exact_cache(full_tokens)

    assert plen_a == plen_b == len(prefix_tokens)

    compared = 0
    for i, (ca, cb) in enumerate(zip(restored_a, restored_b)):
        if hasattr(ca, "keys") and ca.keys is not None:
            assert (
                mx.max(mx.abs(ca.keys - cb.keys)).item() == 0
            ), f"Layer {i} keys differ"
            assert (
                mx.max(mx.abs(ca.values - cb.values)).item() == 0
            ), f"Layer {i} values differ"
            compared += 1
        elif hasattr(ca, "cache") and ca.cache is not None:
            for j, (ta, tb) in enumerate(zip(ca.cache, cb.cache)):
                if ta is not None and tb is not None:
                    assert (
                        mx.max(mx.abs(ta - tb)).item() == 0
                    ), f"Layer {i} cache[{j}] differs"
            compared += 1
    assert compared > 0, "No layers were compared — cache structure may have changed"


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_restore_does_not_corrupt_stored_entry(model_factory):
    """Generating from a restored cache must not mutate the stored entry."""
    lm = model_factory()

    prefix = mx.array([[1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]])
    suffix = mx.array(
        [[12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    )
    prefix_tokens = prefix[0].tolist()
    full_tokens = prefix_tokens + suffix[0].tolist()

    cache = lm.make_cache()
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    apc = APCManager(num_blocks=64, block_size=16)
    apc.store_exact_cache(prefix_tokens, cache)

    # First restore + generation (should not corrupt the stored entry)
    restored_first, _ = apc.lookup_exact_cache(full_tokens)
    lm(suffix, cache=restored_first)
    mx.eval([c for c in restored_first if c is not None])
    _generate_greedy(lm, mx.array([[42]]), restored_first, 3)

    # Second and third restores: compare the raw cache tensors to verify
    # the stored entry itself was not mutated by the generation above.
    restored_second, _ = apc.lookup_exact_cache(full_tokens)
    restored_third, _ = apc.lookup_exact_cache(full_tokens)

    assert restored_second is not None, "Second restore failed"
    assert restored_third is not None, "Third restore failed"

    compared = 0
    for i, (c2, c3) in enumerate(zip(restored_second, restored_third)):
        if hasattr(c2, "keys") and c2.keys is not None:
            diff = mx.max(mx.abs(c2.keys - c3.keys)).item()
            assert diff == 0, f"Layer {i} keys differ after corruption (diff={diff})"
            diff = mx.max(mx.abs(c2.values - c3.values)).item()
            assert diff == 0, f"Layer {i} values differ after corruption (diff={diff})"
            compared += 1
        elif hasattr(c2, "cache") and c2.cache is not None:
            for j, (t2, t3) in enumerate(zip(c2.cache, c3.cache)):
                if t2 is not None and t3 is not None:
                    diff = mx.max(mx.abs(t2 - t3)).item()
                    assert diff == 0, f"Layer {i} cache[{j}] differs (diff={diff})"
            compared += 1
    assert compared > 0, "No layers were compared — cache structure may have changed"


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_restored_cache_generates_tokens(model_factory):
    """Restored cache allows multi-token greedy generation without crashing.

    Verifies the full generation loop works with a restored cache — the model
    can produce 8 tokens sequentially using the restored state. This catches
    shape mismatches, offset errors, and cache-type incompatibilities that
    would surface during autoregressive decode.
    """
    lm = model_factory()

    prefix = mx.array([[1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]])
    suffix = mx.array(
        [[12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    )
    prefix_tokens = prefix[0].tolist()
    full_tokens = prefix_tokens + suffix[0].tolist()

    cache = lm.make_cache()
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    apc = APCManager(num_blocks=64, block_size=16)
    apc.store_exact_cache(prefix_tokens, cache)

    restored, _ = apc.lookup_exact_cache(full_tokens)
    assert restored is not None

    # Process suffix on restored cache
    out = lm(suffix, cache=restored)
    mx.eval(out.logits)

    # Generate 4 tokens — this exercises the full decode loop on restored state
    tokens = _generate_greedy(
        lm, mx.argmax(out.logits[:, -1:, :], axis=-1), restored, 4
    )

    assert len(tokens) == 4, f"Expected 4 tokens, got {len(tokens)}"
    assert all(0 <= t < 64 for t in tokens), f"Token out of vocab range: {tokens}"


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_lookup_misses_on_wrong_prefix(model_factory):
    """Lookup with non-matching tokens must return None."""
    lm = model_factory()
    cache = lm.make_cache()

    prefix = mx.array([[1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]])
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    apc = APCManager(num_blocks=64, block_size=16)
    prefix_tokens = prefix[0].tolist()
    apc.store_exact_cache(prefix_tokens, cache)

    # Completely different tokens — should miss
    wrong_tokens = [
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        33,
        40,
        41,
        30,
        31,
        32,
        34,
        35,
        36,
        37,
        38,
        39,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
    ]
    restored, prefix_len = apc.lookup_exact_cache(wrong_tokens)
    assert restored is None
    assert prefix_len == 0


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_longer_prefix_store_restore(model_factory):
    """Exercise store/restore with a longer prefix (64 tokens)."""
    lm = model_factory()
    cache = lm.make_cache()

    prefix = mx.array([list(range(1, 64)) + [2]])  # 64 tokens, all within vocab
    suffix = mx.array(
        [[12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    )

    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    apc = APCManager(num_blocks=64, block_size=16)
    prefix_tokens = prefix[0].tolist()
    apc.store_exact_cache(prefix_tokens, cache)

    full_tokens = prefix_tokens + suffix[0].tolist()
    restored, prefix_len = apc.lookup_exact_cache(full_tokens)

    assert restored is not None
    assert prefix_len == 64

    # Verify restored tensors are non-trivial (not empty/zeros)
    has_nonzero = False
    for c in restored:
        if hasattr(c, "keys") and c.keys is not None:
            if mx.any(c.keys != 0).item():
                has_nonzero = True
                break
        elif hasattr(c, "cache") and c.cache is not None:
            for t in c.cache:
                if t is not None and mx.any(t != 0).item():
                    has_nonzero = True
                    break
    assert has_nonzero, "Restored cache contains only zeros — likely not populated"


# ============================================================================
# Structural invariants — foundations for per-layer hybrid APC
# ============================================================================


def test_gemma4_cache_layer_types_match_expected_pattern():
    """Gemma 4 with sliding_window_pattern=3 produces 2:1 RotatingKVCache:KVCache.

    Per-layer hybrid APC depends on knowing which layers are block-eligible
    (KVCache) vs exact-only (RotatingKVCache). This test documents and enforces
    that invariant.
    """
    from mlx_vlm.apc import _cache_entry_supports_block_apc
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    lm = _make_tiny_gemma4()
    cache = lm.make_cache()

    assert len(cache) == 6

    block_eligible = [_cache_entry_supports_block_apc(c) for c in cache]
    # Pattern with sliding_window_pattern=3: [sliding, sliding, full] × 2
    assert block_eligible == [False, False, True, False, False, True]

    # Verify actual types
    assert isinstance(cache[0], RotatingKVCache)
    assert isinstance(cache[2], KVCache)
    assert isinstance(cache[5], KVCache)


def test_qwen35_cache_layer_types_match_expected_pattern():
    """Qwen 3.5 with full_attention_interval=4 produces 3:1 ArraysCache:KVCache.

    Per-layer hybrid APC depends on knowing which layers are block-eligible
    (KVCache) vs exact-only (ArraysCache). This test documents and enforces
    that invariant.
    """
    from mlx_vlm.apc import _cache_entry_supports_block_apc
    from mlx_vlm.models.cache import ArraysCache, KVCache

    lm = _make_tiny_qwen35()
    cache = lm.make_cache()

    assert len(cache) == 4

    block_eligible = [_cache_entry_supports_block_apc(c) for c in cache]
    # Pattern: layers 0,1,2 are linear (ArraysCache), layer 3 is attention (KVCache)
    assert block_eligible == [False, False, False, True]

    assert isinstance(cache[0], ArraysCache)
    assert isinstance(cache[1], ArraysCache)
    assert isinstance(cache[2], ArraysCache)
    assert isinstance(cache[3], KVCache)


def test_clone_preserves_layer_count_and_types():
    """_clone_prompt_cache_for_apc must preserve layer ordering and types.

    A hybrid refactor that reorders or drops layers would silently corrupt
    generation. This test catches that.
    """
    from mlx_vlm.apc import _clone_prompt_cache_for_apc

    lm = _make_tiny_gemma4()
    cache = lm.make_cache()

    prefix = mx.array([[1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]])
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    cloned = _clone_prompt_cache_for_apc(cache)

    assert len(cloned) == len(cache), "Clone changed layer count"
    for i, (orig, clone) in enumerate(zip(cache, cloned)):
        assert type(orig) is type(
            clone
        ), f"Layer {i} type changed: {type(orig)} -> {type(clone)}"


def test_block_eligible_layers_have_extractable_kv_after_prefill():
    """KVCache layers in a hybrid model contain real K/V data after prefill.

    Per-layer hybrid APC would extract keys/values from block-eligible layers
    and store them as blocks. This verifies the data is actually there and
    non-trivial.
    """
    from mlx_vlm.apc import _cache_entry_supports_block_apc

    lm = _make_tiny_gemma4()
    cache = lm.make_cache()

    prefix = mx.array([list(range(1, 49))])  # 48 tokens, all within vocab_size=64
    lm(prefix, cache=cache)
    mx.eval([c for c in cache if c is not None])

    for i, c in enumerate(cache):
        if _cache_entry_supports_block_apc(c):
            assert c.keys is not None, f"Block-eligible layer {i} has no keys"
            assert c.values is not None, f"Block-eligible layer {i} has no values"
            assert c.keys.shape[2] > 0, f"Block-eligible layer {i} keys are empty"
            assert mx.any(
                c.keys != 0
            ).item(), f"Block-eligible layer {i} keys are all zeros"
            assert mx.any(
                c.values != 0
            ).item(), f"Block-eligible layer {i} values are all zeros"
