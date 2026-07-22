import unittest

import mlx.core as mx
import numpy as np

from mlx_vlm.sample_utils import (
    apply_min_p,
    apply_p_less,
    apply_top_k,
    apply_top_n_sigma,
    make_dry,
    make_logits_processors,
    make_sampler,
    top_p_sampling,
)


def _dry_out(seq, vocab, **kwargs):
    proc = make_dry(**kwargs)
    out = proc(mx.array(seq, dtype=mx.int32), mx.zeros((1, vocab)))
    mx.eval(out)
    return np.asarray(out.tolist(), dtype=np.float64)[0]


def _np_p_less_keep(logits, temp):
    z = np.asarray(logits, dtype=np.float64) / temp
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    L = float((p * p).sum())
    return p >= L, p, L


def _p_less_kept(logits, temp):
    out = apply_p_less(logits, temp)
    mx.eval(out)
    return [i for i, v in enumerate(out.tolist()) if v != -float("inf")]


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

    def test_float16_large_vocab(self):
        """std must be computed in float32: summing a large float16 vocab
        overflows to inf, which would otherwise disable filtering entirely."""
        mx.random.seed(0)
        V = 152000
        logits = (mx.random.normal((1, V)) * 3).astype(mx.float16)
        out = apply_top_n_sigma(logits, 1.0)
        mx.eval(out)
        kept = int((out[0] != -float("inf")).sum().item())
        self.assertLess(kept, V)
        self.assertGreater(kept, 0)


class TestPLess(unittest.TestCase):
    def test_matches_numpy_reference(self):
        """Mask equals (p >= sum p^2) from an independent numpy implementation,
        across random logits, vocab sizes and temperatures."""
        rng = np.random.default_rng(0)
        for _ in range(40):
            V = int(rng.integers(8, 200))
            temp = float(rng.choice([0.5, 0.7, 1.0, 1.3, 2.0]))
            logits = rng.normal(0, 3, size=V).astype(np.float32)
            keep_ref, p, L = _np_p_less_keep(logits, temp)
            out = np.asarray(
                apply_p_less(mx.array(logits), temp).tolist(), dtype=np.float64
            )
            keep_mlx = np.isfinite(out)
            far = np.abs(p - L) > 1e-5
            self.assertTrue(np.array_equal(keep_ref[far], keep_mlx[far]))

    def test_argmax_always_survives(self):
        """L = sum p^2 <= max p, so the most likely token is never masked."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            V = int(rng.integers(2, 128))
            temp = float(rng.choice([0.5, 1.0, 2.0]))
            logits = rng.normal(0, 3, size=V).astype(np.float32)
            out = apply_p_less(mx.array(logits), temp)
            mx.eval(out)
            self.assertNotEqual(out.tolist()[int(np.argmax(logits))], -float("inf"))

    def test_peaked_keeps_top_only(self):
        """A sharply peaked distribution collapses to the argmax."""
        self.assertEqual(_p_less_kept(mx.array([10.0, 0.0, 0.0, 0.0, 0.0]), 1.0), [0])

    def test_higher_temp_keeps_more(self):
        """Flattening the distribution (higher temp) never shrinks the set."""
        logits = mx.array([4.0, 2.0, 1.0, 0.5, 0.0, -1.0])
        counts = [len(_p_less_kept(logits, t)) for t in (0.5, 1.0, 2.0, 5.0)]
        self.assertEqual(counts, sorted(counts))

    def test_batched_rows_independent(self):
        """Threshold is computed per row."""
        logits = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.9, 0.8, 0.7, 0.6]])
        out = apply_p_less(logits, 1.0)
        mx.eval(out)
        rows = [
            [i for i, v in enumerate(r) if v != -float("inf")] for r in out.tolist()
        ]
        self.assertEqual(rows[0], [0])
        self.assertGreater(len(rows[1]), 1)

    def test_make_sampler_survivor_only(self):
        """make_sampler(p_less=True) only ever samples a surviving token."""
        sampler = make_sampler(temp=1.0, p_less=True)
        toks = sampler(mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]] * 128))
        mx.eval(toks)
        self.assertTrue(all(t == 0 for t in toks.tolist()))

    def test_disabled_by_default(self):
        """p_less=False (default) leaves the sampler unfiltered."""
        sampler = make_sampler(temp=1.0)
        toks = sampler(mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 16))
        mx.eval(toks)
        self.assertEqual(toks.shape, (16,))


class TestDRY(unittest.TestCase):
    def test_no_penalty_without_repeat(self):
        """Distinct tokens (no repeated suffix) are left unchanged."""
        out = _dry_out([1, 2, 3, 4, 5], 8, multiplier=2.0, base=3.0)
        self.assertTrue(np.all(out == 0.0))

    def test_penalizes_repeated_continuation(self):
        """[1,2,3,4,1,2,3] -> token 4 continued a length-3 match; penalty is
        multiplier * base ** (3 - allowed_length)."""
        out = _dry_out(
            [1, 2, 3, 4, 1, 2, 3], 8, multiplier=2.0, base=3.0, allowed_length=2
        )
        self.assertAlmostEqual(out[4], -6.0, places=5)
        self.assertTrue(np.all(np.delete(out, 4) == 0.0))

    def test_allowed_length_gating(self):
        """A length-1 match is not penalized when allowed_length=2."""
        out = _dry_out([5, 9, 5], 10, multiplier=1.0, base=2.0, allowed_length=2)
        self.assertTrue(np.all(out == 0.0))
        out = _dry_out([5, 9, 5], 10, multiplier=1.0, base=2.0, allowed_length=1)
        self.assertAlmostEqual(out[9], -1.0, places=5)

    def test_penalty_grows_with_match_length(self):
        short = _dry_out(
            [1, 2, 8, 1, 2], 10, multiplier=1.0, base=2.0, allowed_length=1
        )
        long = _dry_out(
            [1, 2, 3, 8, 1, 2, 3], 10, multiplier=1.0, base=2.0, allowed_length=1
        )
        self.assertLess(long[8], short[8])

    def test_sequence_breaker_resets_match(self):
        """A breaker token stops backward matching, shrinking the match."""
        seq = [1, 2, 3, 1, 2, 3]
        without = _dry_out(seq, 8, multiplier=1.0, base=2.0, allowed_length=2)
        with_break = _dry_out(
            seq, 8, multiplier=1.0, base=2.0, allowed_length=2, sequence_breakers=[2]
        )
        self.assertLess(without[1], 0.0)
        self.assertTrue(np.all(with_break == 0.0))

    def test_context_size_limits_search(self):
        seq = [3, 3, 3, 3]
        full = _dry_out(seq, 8, multiplier=1.0, base=2.0, allowed_length=2)
        capped = _dry_out(
            seq, 8, multiplier=1.0, base=2.0, allowed_length=2, context_size=2
        )
        self.assertLess(full[3], 0.0)
        self.assertTrue(np.all(capped == 0.0))

    def test_make_logits_processors_integration(self):
        self.assertEqual(len(make_logits_processors(dry_multiplier=0.0)), 0)
        procs = make_logits_processors(
            dry_multiplier=2.0, dry_base=3.0, dry_allowed_length=2
        )
        self.assertEqual(len(procs), 1)
        out = procs[0](
            mx.array([1, 2, 3, 4, 1, 2, 3], dtype=mx.int32), mx.zeros((1, 8))
        )
        mx.eval(out)
        self.assertAlmostEqual(out.tolist()[0][4], -6.0, places=5)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            make_dry(-1.0)
        with self.assertRaises(ValueError):
            make_dry(1.0, base=0.0)
        with self.assertRaises(ValueError):
            make_dry(1.0, allowed_length=0)


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
