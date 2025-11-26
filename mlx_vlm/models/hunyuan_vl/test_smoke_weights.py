#!/usr/bin/env python3
"""Smoke test for HunyuanOCR with real weights.

Run from repo root:
    python -m mlx_vlm.models.hunyuan_vl.test_smoke_weights
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .hunyuan_vl import Model


def load_weights(model_path: str):
    """Load weights from safetensors files."""
    from mlx.utils import tree_flatten, tree_unflatten

    model_path = Path(model_path)

    # Load index to find weight files
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_files = set(index["weight_map"].values())
    else:
        # Single file
        weight_files = ["model.safetensors"]

    # Load all weights
    weights = {}
    for wf in weight_files:
        wf_path = model_path / wf
        if wf_path.exists():
            print(f"Loading {wf}...")
            w = mx.load(str(wf_path))
            weights.update(w)

    return weights


def test_load_and_forward():
    """Test loading weights and running a forward pass."""
    print("=" * 60)
    print("HunyuanOCR Smoke Test with Real Weights")
    print("=" * 60)

    # Path to local weights (relative to repo root)
    model_path = Path(__file__).parent.parent.parent.parent / "hunyuanocr"

    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        print("Skipping smoke test - no weights available")
        return False

    print(f"Model path: {model_path}")

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    print("Loading config...")
    config = ModelConfig.from_dict(config_dict)
    print(f"  Model type: {config.model_type}")
    print(f"  Text hidden size: {config.text_config.hidden_size}")
    print(f"  Vision hidden size: {config.vision_config.hidden_size}")
    print(f"  Num text layers: {config.text_config.num_hidden_layers}")
    print(f"  Num vision layers: {config.vision_config.num_hidden_layers}")

    # Create model
    print("\nCreating model...")
    model = Model(config)

    # Load weights
    print("\nLoading weights...")
    weights = load_weights(model_path)
    print(f"  Loaded {len(weights)} weight tensors")

    # Sanitize weights
    print("\nSanitizing weights...")
    sanitized = model.sanitize(weights)
    print(f"  Sanitized {len(sanitized)} weight tensors")

    # Check for key mismatches
    model_params = dict(model.parameters())
    model_keys = set(model_params.keys())
    weight_keys = set(sanitized.keys())

    missing = model_keys - weight_keys
    extra = weight_keys - model_keys

    if missing:
        print(f"\n⚠ Missing keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if extra:
        print(f"\n⚠ Extra keys ({len(extra)}):")
        for k in sorted(extra)[:10]:
            print(f"    {k}")
        if len(extra) > 10:
            print(f"    ... and {len(extra) - 10} more")

    if not missing and not extra:
        print("✓ All keys match!")

    # Load weights into model
    print("\nLoading weights into model...")
    try:
        model.load_weights(list(sanitized.items()))
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        return False

    # Test forward pass (text only, no image)
    print("\nTesting text-only forward pass...")
    batch_size, seq_len = 1, 10
    input_ids = mx.array([[config.bos_token_id] + [100] * (seq_len - 1)])

    try:
        output = model(input_ids=input_ids)
        mx.eval(output.logits)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output.logits.shape}")
        print(
            f"  Logits range: [{mx.min(output.logits).item():.4f}, {mx.max(output.logits).item():.4f}]"
        )

        if mx.isnan(output.logits).any().item():
            print("✗ NaN detected in logits!")
            return False
        print("✓ Text-only forward pass successful")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test with image placeholder
    print("\nTesting forward pass with image...")
    H, W = 128, 128  # Small test image
    grid_h = H // config.vision_config.patch_size
    grid_w = W // config.vision_config.patch_size
    num_tokens = model.vision_tower.get_num_tokens(grid_h, grid_w)

    # Create input with image placeholders
    image_token_id = config.image_token_id
    input_ids_with_image = mx.array(
        [[config.bos_token_id] + [image_token_id] * num_tokens + [100] * 5]
    )

    # Flatten patches manually: (1, H, W, 3) -> (num_patches, C*P*P)
    P = config.vision_config.patch_size
    C = 3
    num_patches = grid_h * grid_w
    pixel_values = mx.random.normal((num_patches, C * P * P))

    image_grid_thw = mx.array([[1, grid_h, grid_w]])

    try:
        output = model(
            input_ids=input_ids_with_image,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        mx.eval(output.logits)
        print(f"  Input shape: {input_ids_with_image.shape}")
        print(f"  Image shape: {pixel_values.shape}")
        print(f"  Image tokens: {num_tokens}")
        print(f"  Output shape: {output.logits.shape}")
        print(
            f"  Logits range: [{mx.min(output.logits).item():.4f}, {mx.max(output.logits).item():.4f}]"
        )

        if mx.isnan(output.logits).any().item():
            print("✗ NaN detected in logits!")
            return False
        print("✓ Image forward pass successful")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("Smoke test passed! ✓")
    print("=" * 60)
    return True


def main():
    success = test_load_and_forward()
    if not success:
        print("\nSmoke test failed or skipped.")
        exit(1)


if __name__ == "__main__":
    main()
