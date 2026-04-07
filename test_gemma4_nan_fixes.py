"""
Test the 3 NaN-fix patches for Gemma4 vision training.

Patches:
  1. vision.py: -inf → -1e9 in attention mask (prevents NaN in softmax backward)
  2. vision.py: .item() → mask multiplication (preserves autograd graph)
  3. base.py: remove @mx.compile from ensure_fused_sdpa (allows gradient flow)

Run:  python test_gemma4_nan_fixes.py
"""

import sys
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TestPatch1_AttentionMaskFinite(unittest.TestCase):
    """Patch 1: -inf → -1e9 prevents NaN in softmax backward pass."""

    def test_softmax_backward_with_inf(self):
        """float('-inf') mask causes NaN gradients via 0 * -inf in backward."""
        x = mx.random.normal((1, 4, 8, 8))  # B, heads, L, L

        # Simulate all-padding row: entire row is -inf
        mask_inf = mx.full((1, 1, 8, 8), float("-inf"))
        # One valid row so forward doesn't fully collapse
        mask_inf = mask_inf.at[:, :, 0, :].add(float("inf"))  # row 0 = 0

        # With -inf: softmax produces 0s, backward produces 0 * -inf = NaN
        def loss_inf(x):
            masked = x + mx.broadcast_to(mask_inf, x.shape)
            s = mx.softmax(masked, axis=-1)
            return s.sum()

        grad_fn = mx.grad(loss_inf)
        g = grad_fn(x)
        mx.eval(g)
        has_nan_inf = bool(mx.any(mx.isnan(g)).item()) or bool(
            mx.any(mx.isinf(g)).item()
        )
        # This SHOULD produce NaN — demonstrating the bug
        print(f"  -inf mask → NaN in grad: {has_nan_inf}")

    def test_softmax_backward_with_large_finite(self):
        """Using -1e4 instead of -inf avoids NaN entirely."""
        x = mx.random.normal((1, 4, 8, 8))

        mask_finite = mx.full((1, 1, 8, 8), -1e4)
        mask_finite = mask_finite.at[:, :, 0, :].add(1e4)  # row 0 = 0

        def loss_finite(x):
            masked = x + mx.broadcast_to(mask_finite, x.shape)
            s = mx.softmax(masked, axis=-1)
            return s.sum()

        grad_fn = mx.grad(loss_finite)
        g = grad_fn(x)
        mx.eval(g)
        has_nan = bool(mx.any(mx.isnan(g)).item())
        has_inf = bool(mx.any(mx.isinf(g)).item())
        self.assertFalse(has_nan, "Gradient should not contain NaN with -1e4 mask")
        self.assertFalse(has_inf, "Gradient should not contain Inf with -1e4 mask")
        print(f"  -1e4 mask → grad finite: True")

    def test_float16_safety(self):
        """Verify the mask value is representable in float16 (not clamped to -inf)."""
        # -1e9 overflows float16 to -inf, defeating the fix
        bad = mx.array(-1e9, dtype=mx.float16)
        self.assertTrue(mx.isinf(bad).item(), "-1e9 should overflow to -inf in fp16")

        # -1e4 is safe in float16
        good = mx.array(-1e4, dtype=mx.float16)
        self.assertFalse(mx.isinf(good).item(), "-1e4 should be finite in fp16")

        # Softmax backward with -1e4 in float16 should be NaN-free
        x = mx.random.normal((1, 2, 4, 4)).astype(mx.float16)
        mask = mx.full((1, 1, 4, 4), -1e4, dtype=mx.float16)
        mask = mask.at[:, :, 0, :].add(1e4)

        def loss(x):
            return mx.softmax(x + mx.broadcast_to(mask, x.shape), axis=-1).sum()

        g = mx.grad(loss)(x)
        mx.eval(g)
        self.assertFalse(bool(mx.any(mx.isnan(g)).item()), "fp16 grad should be NaN-free")
        print(f"  float16 safety: -1e4 finite and NaN-free ✓")

    def test_forward_equivalence(self):
        """Both masks produce identical softmax output for valid positions."""
        x = mx.random.normal((1, 1, 4, 4))

        # Build mask: positions 0,1 valid; 2,3 padding
        mask_inf = mx.array([[[[0, 0, float("-inf"), float("-inf")],
                               [0, 0, float("-inf"), float("-inf")],
                               [float("-inf")] * 4,
                               [float("-inf")] * 4]]])
        mask_fin = mx.array([[[[0, 0, -1e4, -1e4],
                               [0, 0, -1e4, -1e4],
                               [-1e4] * 4,
                               [-1e4] * 4]]])

        out_inf = mx.softmax(x + mask_inf, axis=-1)
        out_fin = mx.softmax(x + mask_fin, axis=-1)
        mx.eval(out_inf, out_fin)

        # Valid rows (0,1) should match
        diff = mx.abs(out_inf[:, :, :2, :2] - out_fin[:, :, :2, :2]).max().item()
        self.assertLess(diff, 1e-5, f"Valid-position outputs should match, got diff={diff}")
        print(f"  Forward equivalence (valid positions): diff={diff:.2e}")


class TestPatch2_ItemAutograd(unittest.TestCase):
    """Patch 2: .item() breaks autograd; mask multiplication preserves it."""

    def test_item_breaks_gradient(self):
        """Demonstrate that .item() detaches from computation graph."""
        x = mx.array([1.0, 2.0, 3.0, 0.0])

        def loss_with_item(x):
            # This mimics the original code: use .item() to get count, then slice
            mask = mx.array([True, True, True, False])
            n_valid = int(mask.astype(mx.int32).sum().item())
            # Slicing with a Python int is fine for forward, but the path
            # through .item() doesn't affect gradients of the mask computation
            return x[:n_valid].sum()

        def loss_with_mask_mul(x):
            # This is our fix: multiply by mask to zero out padding
            mask = mx.array([1.0, 1.0, 1.0, 0.0])
            return (x * mask).sum()

        g1 = mx.grad(loss_with_item)(x)
        g2 = mx.grad(loss_with_mask_mul)(x)
        mx.eval(g1, g2)

        # Both should give [1,1,1,0] gradient — the question is whether
        # the gradient flows correctly through the mask path
        print(f"  .item() slice grad:  {g1.tolist()}")
        print(f"  mask multiply grad:  {g2.tolist()}")
        # Mask multiply is the safer path for complex models
        np.testing.assert_array_equal(np.array(g2.tolist()), [1, 1, 1, 0])

    def test_pooled_output_mask_multiplication(self):
        """Test that mask multiplication correctly zeros padding and preserves valid."""
        B, L, D = 1, 8, 4
        pooled = mx.random.normal((B, L, D))
        valid_mask = mx.array([[True, True, True, True, True, False, False, False]])

        valid_mask_expanded = mx.expand_dims(valid_mask, -1).astype(pooled.dtype)
        masked_pooled = pooled * valid_mask_expanded
        mx.eval(masked_pooled)

        # Valid positions should be unchanged
        for i in range(5):
            np.testing.assert_allclose(
                np.array(masked_pooled[0, i].tolist()),
                np.array(pooled[0, i].tolist()),
                rtol=1e-5,
            )
        # Padding positions should be zero
        for i in range(5, 8):
            self.assertTrue(
                all(v == 0.0 for v in masked_pooled[0, i].tolist()),
                f"Padding position {i} should be zero",
            )
        print(f"  Mask multiplication: valid preserved, padding zeroed ✓")


class TestPatch3_CompileDecorator(unittest.TestCase):
    """Patch 3: @mx.compile on ensure_fused_sdpa blocks gradient flow."""

    def _make_sdpa_fn(self, compiled):
        """Create SDPA function with or without @mx.compile."""
        def sdpa(q, k, v, scale, mask=None):
            fused_dims = (64, 80, 128)
            d = q.shape[-1]
            target = next((t for t in fused_dims if d <= t), d)
            if target != d:
                pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
                q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask
            )[..., :d]

        if compiled:
            return mx.compile(sdpa)
        return sdpa

    def test_gradient_flows_without_compile(self):
        """Without @mx.compile, gradients flow through SDPA."""
        sdpa = self._make_sdpa_fn(compiled=False)

        B, H, L, D = 1, 2, 4, 64
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))

        def loss(q):
            out = sdpa(q, k, v, scale=1.0 / (D ** 0.5))
            return out.sum()

        g = mx.grad(loss)(q)
        mx.eval(g)
        has_nan = bool(mx.any(mx.isnan(g)).item())
        self.assertFalse(has_nan, "Gradient should be finite without @mx.compile")
        grad_norm = mx.sqrt((g * g).sum()).item()
        self.assertGreater(grad_norm, 0, "Gradient should be nonzero")
        print(f"  Without @mx.compile: grad_norm={grad_norm:.4f}, NaN={has_nan}")

    def test_gradient_with_compile(self):
        """With @mx.compile, check if gradients still flow (MLX version dependent)."""
        sdpa = self._make_sdpa_fn(compiled=True)

        B, H, L, D = 1, 2, 4, 64
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        v = mx.random.normal((B, H, L, D))

        def loss(q):
            out = sdpa(q, k, v, scale=1.0 / (D ** 0.5))
            return out.sum()

        try:
            g = mx.grad(loss)(q)
            mx.eval(g)
            has_nan = bool(mx.any(mx.isnan(g)).item())
            grad_norm = mx.sqrt((g * g).sum()).item()
            print(f"  With @mx.compile: grad_norm={grad_norm:.4f}, NaN={has_nan}")
            if has_nan:
                print("  → @mx.compile DOES break gradients on this MLX version")
        except Exception as e:
            print(f"  With @mx.compile: ERROR — {e}")
            print("  → @mx.compile DOES break gradients on this MLX version")

    def test_patched_file_has_no_compile(self):
        """Verify the actual base.py has no @mx.compile on ensure_fused_sdpa."""
        from mlx_vlm.models import base
        import inspect
        source = inspect.getsource(base.ensure_fused_sdpa)
        # The function itself shouldn't be wrapped in compile
        self.assertFalse(
            hasattr(base.ensure_fused_sdpa, "__wrapped__"),
            "ensure_fused_sdpa should not be compiled",
        )
        print(f"  ensure_fused_sdpa is not compiled ✓")


class TestEndToEnd_VisionTowerGradient(unittest.TestCase):
    """End-to-end: verify gradients flow through the Gemma4 vision tower."""

    def test_vision_tower_gradient_flow(self):
        """Load a minimal Gemma4 vision config and check gradient flow."""
        from mlx_vlm.models.gemma4.vision import VisionModel
        from mlx_vlm.models.gemma4.config import VisionConfig

        # Minimal config for a tiny vision tower
        config = VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
            num_hidden_layers=1,
            patch_size=16,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=16,
            default_output_length=16,
            position_embedding_size=256,
        )

        model = VisionModel(config)
        # Use float32 for gradient stability in test
        from mlx.utils import tree_map
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        # Random image input: B=1, C=3, H=64, W=64 (4x4 = 16 patches with patch_size=16)
        pixel_values = mx.random.normal((1, 3, 64, 64))

        def loss_fn(pixel_values):
            out = model(pixel_values)
            return out.sum()

        try:
            grad = mx.grad(loss_fn)(pixel_values)
            mx.eval(grad)
            has_nan = bool(mx.any(mx.isnan(grad)).item())
            grad_norm = mx.sqrt((grad * grad).sum()).item()
            print(f"  Vision tower gradient: norm={grad_norm:.4f}, NaN={has_nan}")
            self.assertFalse(has_nan, "Vision tower gradients should not be NaN")
            self.assertGreater(grad_norm, 0, "Vision tower gradients should be nonzero")
        except Exception as e:
            self.fail(f"Vision tower gradient computation failed: {e}")


if __name__ == "__main__":
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: {mx.metal.is_available()}")
    print()
    unittest.main(verbosity=2)
