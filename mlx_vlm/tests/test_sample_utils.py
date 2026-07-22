import unittest

import mlx.core as mx

from mlx_vlm.sample_utils import apply_min_p, apply_top_k, make_sampler, top_p_sampling


class TestTopPSampling(unittest.TestCase):
    def test_unbatched_shape(self):
        """Unbatched [V] input should return a scalar array."""
        for top_p in [0.9, 1.0]:
            with self.subTest(top_p=top_p):
                token = top_p_sampling(mx.zeros([100]), top_p=top_p, temperature=1.0)
                self.assertEqual(token.ndim, 0)

    def test_batched_shape(self):
        """Batched [B, V] input should return a [B] array."""
        for top_p in [0.9, 1.0]:
            with self.subTest(top_p=top_p):
                tokens = top_p_sampling(
                    mx.zeros([4, 100]), top_p=top_p, temperature=1.0
                )
                self.assertEqual(tokens.shape, (4,))

    def test_per_row_sampling(self):
        """Each row must sample from its own nucleus, not a shared index space.

        The bug applied row-0's sort order to all rows, causing rows to
        sample from the wrong token distribution.
        """
        V = 8
        # Logits are so peaked that the outcome is deterministic regardless of seed.
        logits = mx.array(
            [
                [100.0] + [-100.0] * (V - 1),  # row 0: token 0
                [-100.0] + [100.0] + [-100.0] * (V - 2),  # row 1: token 1
            ]
        )
        tokens = top_p_sampling(logits, top_p=0.9, temperature=1.0)
        mx.eval(tokens)
        self.assertEqual(tokens[0].item(), 0)
        self.assertEqual(tokens[1].item(), 1)

    def test_three_dim_input_shape(self):
        """3D [B, T, V] logits (e.g. MTP speculative verify output) should
        return a [B, T] array.

        Regression test: top_p_sampling previously hard-coded
        ``sampled_pos[:, None]`` which only works for 2D inputs. With 3D
        logits, take_along_axis produced a [B, T, T] tensor and the trailing
        ``.squeeze(-1)`` raised ``Cannot squeeze axis -1 with size T which is
        not equal to 1``. Reproduced by speculative generation in server.py.
        """
        B, T, V = 2, 4, 100
        for top_p in [0.9, 1.0]:
            with self.subTest(top_p=top_p):
                tokens = top_p_sampling(
                    mx.zeros([B, T, V]), top_p=top_p, temperature=1.0
                )
                self.assertEqual(tokens.shape, (B, T))

    def test_three_dim_per_position_sampling(self):
        """Each (batch, time) position must sample from its own row."""
        V = 8
        # Peaked logits make the outcome deterministic regardless of seed.
        row_a = [100.0] + [-100.0] * (V - 1)  # token 0
        row_b = [-100.0] + [100.0] + [-100.0] * (V - 2)  # token 1
        row_c = [-100.0] * (V - 1) + [100.0]  # token V-1
        row_d = [-100.0, -100.0] + [100.0] + [-100.0] * (V - 3)  # token 2
        logits = mx.array([[row_a, row_b, row_c, row_d]])  # [1, 4, V]
        tokens = top_p_sampling(logits, top_p=0.9, temperature=1.0)
        mx.eval(tokens)
        self.assertEqual(tokens.shape, (1, 4))
        self.assertEqual(tokens[0, 0].item(), 0)
        self.assertEqual(tokens[0, 1].item(), 1)
        self.assertEqual(tokens[0, 2].item(), V - 1)
        self.assertEqual(tokens[0, 3].item(), 2)

    def test_bfloat16_input(self):
        """bfloat16 path should return correct shapes for both input ranks."""
        token = top_p_sampling(
            mx.zeros([50], dtype=mx.bfloat16), top_p=0.9, temperature=1.0
        )
        self.assertEqual(token.ndim, 0)

        tokens = top_p_sampling(
            mx.zeros([3, 50], dtype=mx.bfloat16), top_p=0.9, temperature=1.0
        )
        self.assertEqual(tokens.shape, (3,))


class TestValidationDoesNotCorruptCompile(unittest.TestCase):
    """Regression for #1654: a parameter ValueError must be raised from outside
    the @mx.compile kernel, so it cannot corrupt MLX's trace state for
    subsequent compiled sampler calls in the same process."""

    def test_min_p_error_then_sampler_works(self):
        x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        with self.assertRaises(ValueError):
            mx.eval(apply_min_p(x, -1.0))
        toks = make_sampler(temp=1.0, min_p=0.1)(x)
        mx.eval(toks)
        self.assertEqual(toks.shape, (1,))

    def test_top_k_error_then_sampler_works(self):
        x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        with self.assertRaises(ValueError):
            mx.eval(apply_top_k(x, -5))
        toks = make_sampler(temp=1.0, top_k=3)(x)
        mx.eval(toks)
        self.assertEqual(toks.shape, (1,))


if __name__ == "__main__":
    unittest.main()
