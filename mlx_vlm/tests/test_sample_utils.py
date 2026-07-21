import unittest

import mlx.core as mx

from mlx_vlm.sample_utils import apply_top_n_sigma, make_sampler, top_p_sampling


def _kept(logits, n_sigma):
    """Indices that survive apply_top_n_sigma (i.e. not masked to -inf)."""
    out = apply_top_n_sigma(logits, n_sigma)
    mx.eval(out)
    return [i for i, v in enumerate(out.tolist()) if v != -float("inf")]


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


class TestTopNSigma(unittest.TestCase):
    LOGITS = mx.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_threshold_masks_below(self):
        """n_sigma=1 -> threshold 4-1.414=2.586, keep only logits 3 and 4."""
        self.assertEqual(_kept(self.LOGITS, 1.0), [3, 4])

    def test_wider_threshold_keeps_more(self):
        """n_sigma=2 -> threshold 4-2.828=1.172, keep logits 2,3,4."""
        self.assertEqual(_kept(self.LOGITS, 2.0), [2, 3, 4])

    def test_zero_keeps_only_max(self):
        """n_sigma=0 -> threshold == max, only the top logit survives (greedy)."""
        self.assertEqual(_kept(self.LOGITS, 0.0), [4])

    def test_large_n_keeps_all(self):
        """A large n_sigma masks nothing."""
        self.assertEqual(_kept(self.LOGITS, 100.0), [0, 1, 2, 3, 4])

    def test_batched_rows_independent(self):
        """Threshold is computed per row."""
        logits = mx.array([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0, 0.0]])
        out = apply_top_n_sigma(logits, 1.0)
        mx.eval(out)
        keep = [
            [i for i, v in enumerate(r) if v != -float("inf")] for r in out.tolist()
        ]
        self.assertEqual(keep, [[3, 4], [0, 1]])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            mx.eval(apply_top_n_sigma(self.LOGITS, -1.0))

    def test_make_sampler_selects_survivor(self):
        """make_sampler(top_n_sigma=...) only ever samples a surviving token.

        Peaked logits make the survivor set {3, 4} deterministic; over many
        draws the sampler must never emit a masked token.
        """
        sampler = make_sampler(temp=1.0, top_n_sigma=1.0)
        logits = mx.array([[0.0, 1.0, 2.0, 3.0, 4.0]] * 64)
        toks = sampler(logits)
        mx.eval(toks)
        self.assertTrue(all(t in (3, 4) for t in toks.tolist()))

    def test_disabled_by_default(self):
        """top_n_sigma=0 (default) leaves the sampler unfiltered."""
        sampler = make_sampler(temp=1.0)
        toks = sampler(mx.array([[0.0, 1.0, 2.0, 3.0, 4.0]] * 16))
        mx.eval(toks)
        self.assertEqual(toks.shape, (16,))


if __name__ == "__main__":
    unittest.main()
