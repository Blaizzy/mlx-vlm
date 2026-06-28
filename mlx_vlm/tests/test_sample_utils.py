import math
import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.sample_utils import (
    SlidingWindowNoRepeatNGramProcessor,
    make_sliding_window_no_repeat_ngram_processor,
    top_p_sampling,
)


class TestSlidingWindowNoRepeatNGramProcessor(unittest.TestCase):
    def test_blocks_repeated_ngram_completion(self):
        processor = SlidingWindowNoRepeatNGramProcessor(3, window_size=20)
        logits = processor(
            mx.array([10, 11, 12, 7, 10, 11], dtype=mx.int32),
            mx.zeros([20]),
        )
        mx.eval(logits)

        self.assertTrue(math.isinf(logits[12].item()))
        self.assertEqual(logits[13].item(), 0)

    def test_respects_sliding_window(self):
        processor = SlidingWindowNoRepeatNGramProcessor(3, window_size=3)
        logits = processor(
            mx.array([10, 11, 12, 7, 10, 11], dtype=mx.int32),
            mx.zeros([20]),
        )
        mx.eval(logits)

        self.assertEqual(logits[12].item(), 0)

    def test_respects_whitelist(self):
        processor = SlidingWindowNoRepeatNGramProcessor(
            3, window_size=20, whitelist_token_ids=[12]
        )
        logits = processor(
            mx.array([10, 11, 12, 7, 10, 11], dtype=mx.int32),
            mx.zeros([20]),
        )
        mx.eval(logits)

        self.assertEqual(logits[12].item(), 0)

    def test_batched_sequences_get_independent_bans(self):
        processor = SlidingWindowNoRepeatNGramProcessor(3, window_size=20)
        logits = processor(
            mx.array(
                [
                    [10, 11, 12, 10, 11],
                    [20, 21, 22, 20, 21],
                ],
                dtype=mx.int32,
            ),
            mx.zeros([2, 30]),
        )
        mx.eval(logits)

        self.assertTrue(math.isinf(logits[0, 12].item()))
        self.assertEqual(logits[0, 22].item(), 0)
        self.assertTrue(math.isinf(logits[1, 22].item()))
        self.assertEqual(logits[1, 12].item(), 0)


class TestMakeSlidingWindowNoRepeatNGramProcessor(unittest.TestCase):
    def test_unlimited_ocr_defaults_match_reference_inference(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="unlimited-ocr"))

        single = make_sliding_window_no_repeat_ngram_processor(model, media=["page"])
        multi = make_sliding_window_no_repeat_ngram_processor(
            model, media=["page1", "page2"]
        )

        self.assertEqual(single.ngram_size, 35)
        self.assertEqual(single.window_size, 128)
        self.assertEqual(multi.ngram_size, 35)
        self.assertEqual(multi.window_size, 1024)

    def test_non_unlimited_model_requires_explicit_size(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="other"))

        self.assertIsNone(make_sliding_window_no_repeat_ngram_processor(model))

        processor = make_sliding_window_no_repeat_ngram_processor(
            model, no_repeat_ngram_size=4, ngram_window=16
        )
        self.assertEqual(processor.ngram_size, 4)
        self.assertEqual(processor.window_size, 16)


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


if __name__ == "__main__":
    unittest.main()
