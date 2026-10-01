"""Compiled decode must preserve positions, cache transitions, and weight updates."""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm.models.deepseek_v41.engram import NgramHashState
from mlx_vlm.models.deepseek_v41.language import (
    BatchDeepseekV41Cache,
    DeepseekV41Cache,
    LanguageModel,
    _apply_rope_at_positions,
)
from mlx_vlm.tests.test_deepseek_v41_sharding import quantize_experts, small_config


@pytest.mark.parametrize("quantized", [False, True])
def test_compiled_decode_cache_and_weight_updates(quantized):
    mx.random.seed(137)
    config = small_config()
    config.sliding_window = 4
    reference = LanguageModel(config)
    reference.head.weight = mx.random.normal(reference.head.weight.shape) * 0.05
    reference.update(tree_map(lambda p: p.astype(mx.bfloat16), reference.parameters()))
    reference.head.weight = reference.head.weight.astype(mx.float32)
    optimized = LanguageModel(config)
    if quantized:
        quantize_experts(reference)
        quantize_experts(optimized)
    optimized.load_weights(tree_flatten(reference.parameters()))
    for model in (reference, optimized):
        model.engram_hash = NgramHashState(
            config, model.layout, token_map=[i % 7 for i in range(config.vocab_size)]
        )
    optimized.enable_decode_compilation()
    mx.eval(reference.parameters(), optimized.parameters())
    caches = [
        BatchDeepseekV41Cache([DeepseekV41Cache(len(model.layers)) for _ in range(4)])
        for model in (reference, optimized)
    ]

    def compare(tokens):
        tokens = tokens % config.vocab_size
        expected = reference(tokens, cache=[caches[0]]).logits
        actual = optimized(tokens, cache=[caches[1]]).logits
        mx.eval(actual, expected, caches[0].state, caches[1].state)
        assert mx.allclose(actual, expected, atol=0.02, rtol=0.02).item()
        assert caches[0].offset.tolist() == caches[1].offset.tolist()

    compare(mx.array([[1 + row, 7, 9] for row in range(4)]))
    for step in range(8):
        compare(mx.array([[13 + row + step] for row in range(4)]))
    # Replacing a captured parameter must affect the next call to a compiled graph.
    replacement = reference.layers[0].attn.wq_a.weight * 0.75
    for model in (reference, optimized):
        model.layers[0].attn.wq_a.weight = replacement
    compare(mx.array([[41 + row] for row in range(4)]))
    for cache in caches:
        cache.filter([2, 0])
    compare(mx.array([[43], [47]]))
    # Transition back to prefill, then to compiled decode at a new position.
    compare(mx.array([[51, 53, 55], [57, 59, 61]]))
    compare(mx.array([[63], [65]]))


@pytest.mark.parametrize("length", [1, 255, 256, 257, 1025])
def test_known_rope_extent_matches_position_readback(length):
    positions = mx.arange(0, length, 2)
    x = mx.random.normal((2, positions.size, 64)).astype(mx.bfloat16)
    args = (x, positions, 32, 10000.0, (1, 32, 1, 65536))
    expected = _apply_rope_at_positions(*args)
    actual = _apply_rope_at_positions(*args, table_len=length)
    assert mx.array_equal(actual, expected).item()
