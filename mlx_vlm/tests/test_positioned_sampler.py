"""Regression tests for _PositionedTargetSampler top-k / min-p / seed support.

The server's continuous-batching decode (GenerationBatch._step ->
_sample_with_positions -> sampler.sample_target) historically honored only
temperature and top_p: per-request top_k and min_p were silently dropped, so
a request with top_k=1 at temperature=2.0 produced the same gibberish as an
unconstrained request. These tests pin the contract that top-k truncation
and min-p filtering (mirroring mlx_lm.sample_utils) apply to every row's
normalized logprobs before the categorical/top-p draw, in both the
positioned (sample_target) and plain (__call__) paths, and that per-request
seeds keep working alongside the filters.
"""

import unittest

import mlx.core as mx

from mlx_vlm.generate import ar as generate_ar
from mlx_vlm.server import generation as server_generation

# The class is duplicated between the generate and server layers; both must
# honor the full parameter set, so every behavioral test runs against both.
SAMPLER_VARIANTS = [
    ("generate_ar", generate_ar._PositionedTargetSampler),
    ("server_generation", server_generation._PositionedTargetSampler),
]


def _normalized_logprobs(probs):
    """Rows of probabilities -> normalized logprobs, as GenerationBatch._step
    feeds the sampler (logits - logsumexp)."""
    logits = mx.log(mx.array(probs))
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _tiled_draws(sampler, row_probs, n_draws):
    """Sample the same distribution at n distinct positions in one batched
    sample_target call (distinct positions -> distinct stateless keys)."""
    logprobs = mx.broadcast_to(
        _normalized_logprobs([row_probs]), (n_draws, len(row_probs))
    )
    tokens = sampler.sample_target(
        logprobs, row_ids=[0] * n_draws, positions=list(range(n_draws))
    )
    mx.eval(tokens)
    return tokens.tolist()


class TestTopK(unittest.TestCase):
    def test_top_k_one_is_greedy_even_at_high_temperature(self):
        """The acceptance criterion for the server bug: top_k=1 must pin every
        row to its argmax no matter the temperature. Rows peak at different
        tokens to prove truncation is per-row, not batch-global."""
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=5.0, top_p=1.0, seed=0, top_k=1)
                logprobs = _normalized_logprobs(
                    [
                        [0.80, 0.05, 0.05, 0.05, 0.05],
                        [0.05, 0.05, 0.05, 0.05, 0.80],
                    ]
                )
                tokens = sampler.sample_target(
                    logprobs, row_ids=[0, 0], positions=[3, 7]
                )
                mx.eval(tokens)
                self.assertEqual(tokens.tolist(), [0, 4])

    def test_top_k_restricts_candidates_but_still_samples(self):
        """With top_k=3 at high temperature, draws across many positions must
        stay inside the top-3 token set yet hit more than one of them —
        truncation without collapsing to greedy."""
        row = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=5.0, top_p=1.0, seed=0, top_k=3)
                draws = _tiled_draws(sampler, row, 200)
                self.assertTrue(set(draws).issubset({0, 1, 2}), draws)
                self.assertGreaterEqual(len(set(draws)), 2, draws)

    def test_top_k_at_or_above_vocab_is_a_no_op(self):
        """top_k >= vocab must not raise and must draw exactly what the
        unconstrained sampler draws for the same seed and positions."""
        row = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                unconstrained = cls(temperature=2.0, top_p=1.0, seed=9)
                oversized = cls(temperature=2.0, top_p=1.0, seed=9, top_k=99)
                self.assertEqual(
                    _tiled_draws(oversized, row, 64),
                    _tiled_draws(unconstrained, row, 64),
                )

    def test_top_k_applies_in_plain_call_path(self):
        """__call__ (the non-positioned fallback used when row ids/positions
        are unavailable) must apply the same truncation."""
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=5.0, top_p=1.0, seed=0, top_k=1)
                logprobs = _normalized_logprobs(
                    [
                        [0.80, 0.05, 0.05, 0.05, 0.05],
                        [0.05, 0.05, 0.05, 0.05, 0.80],
                    ]
                )
                tokens = sampler(logprobs)
                mx.eval(tokens)
                self.assertEqual(tokens.tolist(), [0, 4])

    def test_top_k_composes_with_top_p_draw(self):
        """top_k truncation must happen before the top-p draw path
        (0 < top_p < 1 routes through _sample_top_p_one)."""
        row = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=5.0, top_p=0.999, seed=0, top_k=2)
                draws = _tiled_draws(sampler, row, 200)
                self.assertTrue(set(draws).issubset({0, 1}), draws)


class TestMinP(unittest.TestCase):
    def test_min_p_filters_below_scaled_threshold(self):
        """min_p=0.5 with top probability 0.5 keeps only tokens with
        p >= 0.25 (mlx_lm semantics: threshold scales with the top token's
        probability on the unscaled distribution). Survivors {0, 1} must both
        appear across draws — filtering, not greedy collapse."""
        row = [0.50, 0.30, 0.10, 0.05, 0.05]
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=3.0, top_p=1.0, seed=0, min_p=0.5)
                draws = _tiled_draws(sampler, row, 200)
                self.assertTrue(set(draws).issubset({0, 1}), draws)
                self.assertEqual(set(draws), {0, 1}, draws)

    def test_min_p_one_keeps_only_the_top_token(self):
        """min_p=1.0 leaves just the argmax (threshold == top probability)."""
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=4.0, top_p=1.0, seed=0, min_p=1.0)
                draws = _tiled_draws(sampler, [0.50, 0.30, 0.10, 0.05, 0.05], 50)
                self.assertEqual(set(draws), {0})

    def test_min_p_applies_in_plain_call_path(self):
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=4.0, top_p=1.0, seed=0, min_p=1.0)
                logprobs = _normalized_logprobs([[0.50, 0.30, 0.10, 0.05, 0.05]])
                for _ in range(20):
                    tokens = sampler(logprobs)
                    mx.eval(tokens)
                    self.assertEqual(tokens.tolist(), [0])

    def test_min_p_is_independent_per_row(self):
        """Each row's threshold must come from that row's own top probability
        (vectorized across the batch, not computed batch-globally)."""
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=3.0, top_p=1.0, seed=0, min_p=0.5)
                logprobs = _normalized_logprobs(
                    [
                        # top p=0.9 -> threshold 0.45 -> survivor {0} only
                        [0.90, 0.04, 0.03, 0.02, 0.01],
                        # top p=0.4 -> threshold 0.2 -> survivors {3, 4}
                        [0.10, 0.05, 0.05, 0.40, 0.40],
                    ]
                )
                for pos in range(50):
                    tokens = sampler.sample_target(
                        logprobs, row_ids=[0, 0], positions=[pos, pos]
                    )
                    mx.eval(tokens)
                    row0, row1 = tokens.tolist()
                    self.assertEqual(row0, 0)
                    self.assertIn(row1, (3, 4))

    def test_min_p_handles_bfloat16_logprobs(self):
        """The decode loop can hand the sampler bfloat16 logprobs; filtering
        must preserve dtype handling and still constrain the candidate set."""
        row = [0.50, 0.30, 0.10, 0.05, 0.05]
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=3.0, top_p=1.0, seed=0, min_p=0.5, top_k=0)
                logprobs = mx.broadcast_to(
                    _normalized_logprobs([row]).astype(mx.bfloat16), (100, len(row))
                )
                tokens = sampler.sample_target(
                    logprobs, row_ids=[0] * 100, positions=list(range(100))
                )
                mx.eval(tokens)
                self.assertTrue(set(tokens.tolist()).issubset({0, 1}))


class TestSeed(unittest.TestCase):
    def test_same_seed_reproduces_the_stream(self):
        """Pin of existing behavior: positioned draws are a pure function of
        (seed, row_id, position), so a fresh sampler with the same seed must
        replay the same tokens."""
        row = [1.0 / 256] * 256
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                a = _tiled_draws(cls(temperature=2.0, top_p=1.0, seed=11), row, 64)
                b = _tiled_draws(cls(temperature=2.0, top_p=1.0, seed=11), row, 64)
                self.assertEqual(a, b)

    def test_different_seeds_diverge(self):
        """Per-request seed must actually change the sampled stream (the
        companion server bug: every request collapsed onto DEFAULT_SEED)."""
        row = [1.0 / 256] * 256
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                a = _tiled_draws(cls(temperature=2.0, top_p=1.0, seed=1), row, 64)
                b = _tiled_draws(cls(temperature=2.0, top_p=1.0, seed=2), row, 64)
                self.assertNotEqual(a, b)

    def test_seed_reproducible_with_filters_active(self):
        """Seed and top_k/min_p must compose: same seed + same filters replay
        identically, different seeds still diverge inside the filtered set."""
        row = [0.30, 0.20, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04]
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                kwargs = dict(temperature=3.0, top_p=1.0, top_k=4, min_p=0.2)
                a = _tiled_draws(cls(seed=7, **kwargs), row, 64)
                b = _tiled_draws(cls(seed=7, **kwargs), row, 64)
                c = _tiled_draws(cls(seed=8, **kwargs), row, 64)
                self.assertEqual(a, b)
                self.assertNotEqual(a, c)
                self.assertTrue(set(a + c).issubset({0, 1, 2, 3}))


class TestDefaultPathUnchanged(unittest.TestCase):
    """With filters disabled (top_k=0, min_p=0.0) the positioned draw must
    stay byte-identical to the pre-top-k/min-p implementation.

    Expected tokens were captured by running these exact scenarios against
    v0.6.3 (commit 5a4222a, mlx 0.31.2) before the filters existed; regenerate
    them the same way if the underlying mlx RNG ever changes.
    """

    def _fixture_logprobs(self):
        logits = mx.array(
            [[((r * 5 + v * 3) % 7) * 0.5 for v in range(8)] for r in range(4)],
            dtype=mx.float32,
        )
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def test_categorical_path_pinned(self):
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=0.7, top_p=1.0, seed=42)
                tokens = sampler.sample_target(
                    self._fixture_logprobs(),
                    row_ids=[0, 0, 0, 0],
                    positions=[2, 9, 17, 33],
                )
                mx.eval(tokens)
                self.assertEqual(tokens.tolist(), [6, 2, 5, 7])

    def test_top_p_path_pinned(self):
        for name, cls in SAMPLER_VARIANTS:
            with self.subTest(sampler=name):
                sampler = cls(temperature=0.9, top_p=0.85, seed=7)
                tokens = sampler.sample_target(
                    self._fixture_logprobs(),
                    row_ids=[0, 0, 0, 0],
                    positions=[0, 1, 2, 3],
                )
                mx.eval(tokens)
                self.assertEqual(tokens.tolist(), [4, 5, 1, 3])


class TestDefaultSamplerSelection(unittest.TestCase):
    """generate_step's default-sampler choice, extracted as
    generate_ar._default_sampler. Historically a seeded call that also set
    top_k/min_p silently dropped the seed (fell back to mlx_lm's
    make_sampler); now the positioned sampler carries all four knobs."""

    def test_seeded_call_with_filters_keeps_positioned_sampler(self):
        sampler = generate_ar._default_sampler(
            temperature=1.0, top_p=0.9, min_p=0.2, top_k=4, seed=11
        )
        self.assertIsInstance(sampler, generate_ar._PositionedTargetSampler)
        self.assertEqual(sampler.temperature, 1.0)
        self.assertEqual(sampler.top_p, 0.9)
        self.assertEqual(sampler.top_k, 4)
        self.assertEqual(sampler.min_p, 0.2)
        self.assertEqual(sampler.seed, 11)

    def test_unseeded_call_falls_back_to_make_sampler(self):
        """Seedless library calls keep mlx_lm's global-RNG sampler so their
        run-to-run nondeterminism is unchanged."""
        sampler = generate_ar._default_sampler(
            temperature=1.0, top_p=0.9, min_p=0.0, top_k=0, seed=None
        )
        self.assertNotIsInstance(sampler, generate_ar._PositionedTargetSampler)
        self.assertTrue(callable(sampler))

    def test_temperature_zero_is_greedy_even_with_seed(self):
        sampler = generate_ar._default_sampler(
            temperature=0.0, top_p=1.0, min_p=0.0, top_k=0, seed=3
        )
        logprobs = _normalized_logprobs([[0.05, 0.80, 0.05, 0.05, 0.05]])
        tokens = sampler(logprobs)
        mx.eval(tokens)
        self.assertEqual(tokens.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
