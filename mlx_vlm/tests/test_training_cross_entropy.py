"""Numerical checks for shared token losses."""

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.losses.cross_entropy import cross_entropy, make_loss_mask


class CrossEntropyTest(unittest.TestCase):
    def setUp(self):
        self.logits = mx.array([[[1.0, 2.0, 0.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]])
        self.targets = mx.array([[1, 2, 0]])

    def test_unmasked_loss(self):
        loss, metrics = cross_entropy(self.logits, self.targets)
        expected = nn.losses.cross_entropy(self.logits, self.targets).mean()
        self.assertTrue(mx.allclose(loss, expected).item())
        self.assertEqual(metrics["num_tokens"].item(), 3)

    def test_inclusive_shifted_boundaries(self):
        mask = make_loss_mask(self.targets, mx.array([[2, 3]]))
        self.assertEqual(mask.tolist(), [[False, True, True]])
        loss, metrics = cross_entropy(self.logits, self.targets, mask)
        expected = nn.losses.cross_entropy(
            self.logits[:, 1:], self.targets[:, 1:]
        ).mean()
        self.assertTrue(mx.allclose(loss, expected).item())
        self.assertEqual(metrics["num_tokens"].item(), 2)

    def test_ignored_and_masked_sentinels(self):
        labels = mx.array([[-100, 2, -999]])
        mask = mx.array([[True, True, False]])
        loss, metrics = cross_entropy(self.logits, labels, mask)
        expected = nn.losses.cross_entropy(
            self.logits[:, 1:2], self.targets[:, 1:2]
        ).mean()
        self.assertTrue(mx.allclose(loss, expected).item())
        self.assertEqual(metrics["num_tokens"].item(), 1)
        grad = mx.grad(lambda x: cross_entropy(x, labels, mask)[0])(self.logits)
        self.assertTrue(mx.all(grad[:, 0] == 0).item())
        self.assertTrue(mx.all(grad[:, 2] == 0).item())

    def test_empty_supervision(self):
        labels = mx.full((1, 3), -100, dtype=mx.int32)
        loss, metrics = cross_entropy(self.logits, labels)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["num_tokens"].item(), 0)
        grad = mx.grad(lambda x: cross_entropy(x, labels)[0])(self.logits)
        self.assertTrue(mx.all(grad == 0).item())

    def test_vision_counts_are_independent_of_loss_mask_and_preserve_gradients(self):
        mask = mx.array([[False, False, True]])
        vision = mx.array([[True, True, False]])
        expected_loss, expected_metrics = cross_entropy(self.logits, self.targets, mask)
        loss, metrics = cross_entropy(
            self.logits, self.targets, mask, vision_mask=vision
        )
        self.assertEqual(metrics["num_vision_tokens"].item(), 2)
        self.assertEqual(metrics["num_tokens"].item(), 1)
        self.assertEqual(metrics["weight"].item(), expected_metrics["weight"].item())
        self.assertTrue(mx.array_equal(loss, expected_loss).item())
        expected_grad = mx.grad(lambda x: cross_entropy(x, self.targets, mask)[0])(
            self.logits
        )
        actual_grad = mx.grad(
            lambda x: cross_entropy(x, self.targets, mask, vision_mask=vision)[0]
        )(self.logits)
        self.assertTrue(mx.array_equal(actual_grad, expected_grad).item())
        _, empty_metrics = cross_entropy(
            self.logits,
            self.targets,
            mx.zeros(mask.shape, dtype=mx.bool_),
            vision_mask=vision,
        )
        self.assertEqual(empty_metrics["num_tokens"].item(), 0)
        self.assertEqual(empty_metrics["num_vision_tokens"].item(), 2)
        self.assertEqual(expected_metrics["num_vision_tokens"].item(), 0)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            cross_entropy(self.logits, self.targets[:, :2])
        with self.assertRaises(ValueError):
            cross_entropy(self.logits, self.targets, mx.ones((1, 1)))
        with self.assertRaises(ValueError):
            make_loss_mask(self.targets, mx.array([[1]]))
        with self.assertRaisesRegex(ValueError, "vision_mask"):
            cross_entropy(self.logits, self.targets, vision_mask=mx.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
