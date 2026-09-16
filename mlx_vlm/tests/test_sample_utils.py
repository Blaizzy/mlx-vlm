import unittest

import mlx.core as mx
import numpy as np

from mlx_vlm.sample_utils import (
    apply_min_p,
    apply_p_less,
    apply_top_k,
    apply_top_n_sigma,
    apply_typical_p,
    make_sampler,
    top_p_sampling,
)


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


class TestTopPSampling(unittest.TestCase):
    def test_unbatched_shape(self):
        """Unbatched [V] input should return a scalar array."""
        for top_p in [0.9, 1.0]:
            with self.subTest(top_p=top_p):
                token = top_p_sampling(mx.zeros([100]), top_p=top_p, temperature=1.0)
                self.assertEqual(token.ndim, 0)

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

    def test_peaked_keeps_top_only(self):
        """A sharply peaked distribution collapses to the argmax."""
        self.assertEqual(_p_less_kept(mx.array([10.0, 0.0, 0.0, 0.0, 0.0]), 1.0), [0])

    def test_make_sampler_survivor_only(self):
        """make_sampler(p_less=True) only ever samples a surviving token."""
        sampler = make_sampler(temp=1.0, p_less=True)
        toks = sampler(mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]] * 128))
        mx.eval(toks)
        self.assertTrue(all(t == 0 for t in toks.tolist()))


def _np_typical_keep(logits, typical_p):
    logp = np.asarray(logits, dtype=np.float64)
    logp = logp - logp.max()
    logp = logp - np.log(np.exp(logp).sum())
    p = np.exp(logp)
    ent = -(p * logp).sum()
    shifted = np.abs(-logp - ent)
    order = np.argsort(shifted, kind="stable")
    cum_before_sorted = np.cumsum(p[order]) - p[order]
    keep = np.zeros(len(p), dtype=bool)
    cum_before = np.zeros(len(p))
    keep[order] = cum_before_sorted < typical_p
    cum_before[order] = cum_before_sorted
    return keep, cum_before


class TestTypicalP(unittest.TestCase):
    def _apply(self, logits, typical_p):
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out = apply_typical_p(lp, typical_p)
        mx.eval(out)
        return np.asarray(out.tolist(), dtype=np.float64)

    def test_matches_numpy_reference(self):
        """Mask equals the locally-typical set from an independent numpy
        implementation, away from the single boundary token."""
        rng = np.random.default_rng(0)
        for _ in range(40):
            V = int(rng.integers(8, 200))
            typical_p = float(rng.choice([0.2, 0.5, 0.9, 0.95]))
            logits = rng.normal(0, 3, size=V).astype(np.float32)
            keep_ref, cum_before = _np_typical_keep(logits, typical_p)
            keep_mlx = np.isfinite(self._apply(mx.array(logits), typical_p))
            far = np.abs(cum_before - typical_p) > 1e-4
            self.assertTrue(np.array_equal(keep_ref[far], keep_mlx[far]))

    def test_make_sampler_selects_survivor(self):
        """A sharply peaked distribution collapses to the argmax."""
        sampler = make_sampler(temp=1.0, typical_p=0.3)
        toks = sampler(mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]] * 64))
        mx.eval(toks)
        self.assertTrue(all(t == 0 for t in toks.tolist()))

    def test_invalid_raises(self):
        x = mx.array([0.0, 1.0, 2.0])
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                mx.eval(apply_typical_p(x, bad))


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
