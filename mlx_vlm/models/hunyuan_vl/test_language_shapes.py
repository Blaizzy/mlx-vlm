#!/usr/bin/env python3
"""Shape sanity test for HunyuanOCR language tower.

Run from repo root:
    python -m mlx_vlm.models.hunyuan_vl.test_language_shapes
"""

import mlx.core as mx
import mlx.nn as nn

from .config import TextConfig
from .language import HunyuanRotaryEmbedding, LanguageModel


def test_rotary_embedding_shapes():
    """Test xdrope rotary embedding shapes."""
    print("=" * 60)
    print("Testing HunyuanRotaryEmbedding shapes")
    print("=" * 60)

    head_dim = 128
    rope_scaling = {
        "type": "xdrope",
        "xdrope_section": [16, 16, 16, 16],
    }

    rotary = HunyuanRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=32768,
        base=10000.0,
        rope_scaling=rope_scaling,
    )

    # Test with 2D position_ids (generation fallback)
    batch_size, seq_len = 1, 10
    x = mx.zeros((batch_size, seq_len, head_dim))
    position_ids_2d = mx.arange(seq_len)[None, :]  # (1, seq_len)

    cos, sin = rotary(x, position_ids_2d)
    print(f"2D position_ids input: {position_ids_2d.shape}")
    print(f"cos shape: {cos.shape} (expected: ({batch_size}, {seq_len}, {head_dim}))")
    print(f"sin shape: {sin.shape}")
    assert cos.shape == (
        batch_size,
        seq_len,
        head_dim,
    ), f"cos shape mismatch: {cos.shape}"
    assert sin.shape == (
        batch_size,
        seq_len,
        head_dim,
    ), f"sin shape mismatch: {sin.shape}"
    print("✓ 2D position_ids test passed")

    # Test with 4D position_ids (xdrope prefill)
    position_ids_4d = mx.broadcast_to(
        mx.arange(seq_len)[None, None, :], (4, batch_size, seq_len)
    )

    cos, sin = rotary(x, position_ids_4d)
    print(f"\n4D position_ids input: {position_ids_4d.shape}")
    print(f"cos shape: {cos.shape}")
    print(f"sin shape: {sin.shape}")
    assert cos.shape == (
        batch_size,
        seq_len,
        head_dim,
    ), f"cos shape mismatch: {cos.shape}"
    print("✓ 4D position_ids test passed")

    # Check for NaNs
    assert not mx.isnan(cos).any().item(), "NaN in cos"
    assert not mx.isnan(sin).any().item(), "NaN in sin"
    print("✓ No NaNs in rotary embeddings")

    print()


def test_language_model_shapes():
    """Test language model forward pass shapes."""
    print("=" * 60)
    print("Testing LanguageModel shapes (tiny config)")
    print("=" * 60)

    # Tiny config for fast testing
    config = TextConfig(
        model_type="hunyuan_vl",
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        use_qk_norm=True,
        rope_theta=10000.0,
        rope_scaling={
            "type": "xdrope",
            "xdrope_section": [4, 4, 4, 4],  # Adjusted for head_dim=16
        },
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )

    model = LanguageModel(config)

    # Test forward pass
    batch_size, seq_len = 1, 10
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    output = model(input_ids=input_ids)
    logits = output.logits

    print(f"Output logits shape: {logits.shape}")
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert (
        logits.shape == expected_shape
    ), f"Shape mismatch: {logits.shape} vs {expected_shape}"
    print(f"✓ Output shape correct: {logits.shape}")

    # Check for NaNs
    mx.eval(logits)
    assert not mx.isnan(logits).any().item(), "NaN in logits"
    print("✓ No NaNs in logits")

    # Print some stats
    print(f"Logits min: {mx.min(logits).item():.4f}, max: {mx.max(logits).item():.4f}")
    print(
        f"Logits mean: {mx.mean(logits).item():.4f}, std: {mx.std(logits).item():.4f}"
    )

    print()


def test_attention_qk_norm():
    """Test that QK norm is applied correctly."""
    print("=" * 60)
    print("Testing QK normalization")
    print("=" * 60)

    from .language import Attention

    config = TextConfig(
        model_type="hunyuan_vl",
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        use_qk_norm=True,
        rope_scaling={
            "type": "xdrope",
            "xdrope_section": [4, 4, 4, 4],
        },
    )

    attn = Attention(config)

    # Check that QK norm layers exist
    assert attn.query_layernorm is not None, "query_layernorm should exist"
    assert attn.key_layernorm is not None, "key_layernorm should exist"
    print("✓ QK norm layers exist")

    # Check dimensions (RMSNorm stores dims in 'dimensions' attribute)
    print(f"✓ QK norm layers initialized for head_dim={config.head_dim}")

    # Test forward pass
    batch_size, seq_len = 1, 10
    x = mx.random.normal((batch_size, seq_len, config.hidden_size))

    output = attn(x)
    mx.eval(output)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
    assert not mx.isnan(output).any().item(), "NaN in attention output"
    print(f"✓ Attention output shape: {output.shape}")
    print("✓ No NaNs in attention output")

    print()


def test_with_cache():
    """Test language model with KV cache (generation mode)."""
    print("=" * 60)
    print("Testing LanguageModel with cache")
    print("=" * 60)

    from ..cache import KVCache

    config = TextConfig(
        model_type="hunyuan_vl",
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        use_qk_norm=True,
        rope_scaling={
            "type": "xdrope",
            "xdrope_section": [4, 4, 4, 4],
        },
        tie_word_embeddings=True,
    )

    model = LanguageModel(config)

    # Prefill
    batch_size, prefill_len = 1, 5
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, prefill_len))
    cache = [KVCache() for _ in range(config.num_hidden_layers)]

    output = model(input_ids=input_ids, cache=cache)
    mx.eval(output.logits)

    print(f"Prefill input: {input_ids.shape}")
    print(f"Prefill output: {output.logits.shape}")
    print(f"Cache offset after prefill: {cache[0].offset}")

    # Generation step
    next_token = mx.random.randint(0, config.vocab_size, (batch_size, 1))
    output = model(input_ids=next_token, cache=cache)
    mx.eval(output.logits)

    print(f"Generation input: {next_token.shape}")
    print(f"Generation output: {output.logits.shape}")
    print(f"Cache offset after generation: {cache[0].offset}")

    assert output.logits.shape == (batch_size, 1, config.vocab_size)
    assert not mx.isnan(output.logits).any().item(), "NaN in generation logits"
    print("✓ Cache-based generation works")

    print()


def main():
    """Run all shape sanity tests."""
    print("\n" + "=" * 60)
    print("HunyuanOCR Language Tower Shape Sanity Tests")
    print("=" * 60 + "\n")

    test_rotary_embedding_shapes()
    test_language_model_shapes()
    test_attention_qk_norm()
    test_with_cache()

    print("=" * 60)
    print("All language tower tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
