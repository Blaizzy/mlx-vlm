import unittest

import mlx.core as mx

from mlx_vlm.sample_utils import top_p_sampling


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


if __name__ == "__main__":
    unittest.main()
