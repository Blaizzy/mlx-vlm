#!/usr/bin/env python3
"""Shape sanity test for HunyuanOCR vision tower.

Run from repo root:
    python -m mlx_vlm.models.hunyuan_vl.test_vision_shapes
"""

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig
from .vision import VisionModel, VisionPatchMerger


def test_token_count_formula():
    """Test that token count matches formula: (H/merge) * (W/merge + 1) + 2"""
    print("=" * 60)
    print("Testing token count formula")
    print("=" * 60)

    config = VisionConfig(
        model_type="hunyuan_vl",
        hidden_size=64,
        out_hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        patch_size=16,
        num_channels=3,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        max_image_size=512,
    )

    model = VisionModel(config)

    # Test various grid sizes
    test_cases = [
        (4, 4),  # 64x64 image -> 4x4 grid -> 2x2 merged -> 2*(2+1)+2 = 8 tokens
        (8, 8),  # 128x128 image -> 8x8 grid -> 4x4 merged -> 4*(4+1)+2 = 22 tokens
        (16, 16),  # 256x256 image -> 16x16 grid -> 8x8 merged -> 8*(8+1)+2 = 74 tokens
        (8, 16),  # 128x256 image -> 8x16 grid -> 4x8 merged -> 4*(8+1)+2 = 38 tokens
    ]

    for grid_h, grid_w in test_cases:
        expected = model.get_num_tokens(grid_h, grid_w)

        # Calculate manually
        merge = config.spatial_merge_size
        merged_h = grid_h // merge
        merged_w = grid_w // merge
        manual = merged_h * (merged_w + 1) + 2

        print(f"Grid ({grid_h}x{grid_w}) -> Merged ({merged_h}x{merged_w})")
        print(f"  Expected tokens: {expected}")
        print(f"  Manual calc: {manual}")
        assert expected == manual, f"Token count mismatch: {expected} vs {manual}"
        print(f"  ✓ Match!")

    print()


def test_vision_model_shapes():
    """Test vision model forward pass shapes."""
    print("=" * 60)
    print("Testing VisionModel shapes (tiny config)")
    print("=" * 60)

    config = VisionConfig(
        model_type="hunyuan_vl",
        hidden_size=64,
        out_hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        patch_size=16,
        num_channels=3,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        max_image_size=512,
    )

    model = VisionModel(config)

    # Test with 128x128 image (8x8 grid -> 4x4 merged)
    batch_size = 1
    H, W = 128, 128
    pixel_values = mx.random.normal((batch_size, H, W, config.num_channels))

    print(f"Input shape: {pixel_values.shape}")

    output = model(pixel_values)
    mx.eval(output)

    grid_h = H // config.patch_size
    grid_w = W // config.patch_size
    expected_tokens = model.get_num_tokens(grid_h, grid_w)

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {expected_tokens}, {config.out_hidden_size})")

    assert output.shape == (
        batch_size,
        expected_tokens,
        config.out_hidden_size,
    ), f"Shape mismatch: {output.shape}"
    print("✓ Output shape correct")

    # Check for NaNs
    assert not mx.isnan(output).any().item(), "NaN in output"
    print("✓ No NaNs in output")

    print()


def test_patch_merger_tokens():
    """Test patch merger output token structure."""
    print("=" * 60)
    print("Testing VisionPatchMerger token structure")
    print("=" * 60)

    config = VisionConfig(
        model_type="hunyuan_vl",
        hidden_size=64,
        out_hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        patch_size=16,
        num_channels=3,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        max_image_size=512,
    )

    merger = VisionPatchMerger(config)

    # Simulate ViT output: 8x8 grid = 64 patches
    batch_size = 1
    grid_h, grid_w = 8, 8
    num_patches = grid_h * grid_w
    hidden_states = mx.random.normal((batch_size, num_patches, config.hidden_size))

    print(f"Input: {hidden_states.shape} (grid {grid_h}x{grid_w})")

    output = merger(hidden_states, grid_h, grid_w)
    mx.eval(output)

    # Expected: merged to 4x4, with newlines and begin/end
    merged_h = grid_h // config.spatial_merge_size
    merged_w = grid_w // config.spatial_merge_size
    expected_tokens = merged_h * (merged_w + 1) + 2

    print(f"Output: {output.shape}")
    print(f"Expected tokens: {expected_tokens}")
    print(f"  - Merged patches: {merged_h * merged_w}")
    print(f"  - Newline tokens: {merged_h}")
    print(f"  - Begin/End tokens: 2")

    assert output.shape == (
        batch_size,
        expected_tokens,
        config.out_hidden_size,
    ), f"Shape mismatch: {output.shape}"
    print("✓ Token count correct")

    assert not mx.isnan(output).any().item(), "NaN in output"
    print("✓ No NaNs in output")

    print()


def test_nchw_input():
    """Test that NCHW input format works."""
    print("=" * 60)
    print("Testing NCHW input format")
    print("=" * 60)

    config = VisionConfig(
        model_type="hunyuan_vl",
        hidden_size=64,
        out_hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        patch_size=16,
        num_channels=3,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        max_image_size=512,
    )

    model = VisionModel(config)

    # NCHW format
    batch_size = 1
    H, W = 128, 128
    pixel_values_nchw = mx.random.normal((batch_size, config.num_channels, H, W))

    print(f"Input shape (NCHW): {pixel_values_nchw.shape}")

    output = model(pixel_values_nchw)
    mx.eval(output)

    grid_h = H // config.patch_size
    grid_w = W // config.patch_size
    expected_tokens = model.get_num_tokens(grid_h, grid_w)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, expected_tokens, config.out_hidden_size)
    print("✓ NCHW input works correctly")

    print()


def main():
    """Run all vision tower shape tests."""
    print("\n" + "=" * 60)
    print("HunyuanOCR Vision Tower Shape Sanity Tests")
    print("=" * 60 + "\n")

    test_token_count_formula()
    test_vision_model_shapes()
    test_patch_merger_tokens()
    test_nchw_input()

    print("=" * 60)
    print("All vision tower tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
