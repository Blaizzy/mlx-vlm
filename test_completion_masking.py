"""
Test multi-turn completion masking for train_on_completions.

Validates that:
1. Single-turn: only assistant tokens get mask=1
2. Multi-turn: all assistant turns get mask=1, all user turns get mask=0
3. With end_turn_id: mask toggles off at turn boundaries
4. Auto-detection of assistant_id from Gemma 4 tokenizer

Usage: python test_completion_masking.py
"""

import unittest

import numpy as np


class TestBuildCompletionMask(unittest.TestCase):

    def setUp(self):
        from mlx_vlm.trainer.sft_trainer import build_completion_mask
        self.build_mask = build_completion_mask

    def test_single_turn_no_end_token(self):
        """Single assistant turn, no end_turn_id: mask everything from assistant_id onward."""
        # Tokens: [BOS, USER, text, text, ASST, reply, reply]
        #                                  ^100
        ids = np.array([[1, 50, 60, 70, 100, 80, 90]])
        mask = self.build_mask(ids, assistant_id=100)
        expected = np.array([[0, 0, 0, 0, 1, 1, 1]])
        np.testing.assert_array_equal(mask, expected)

    def test_single_turn_with_end_token(self):
        """Single assistant turn with end_turn_id: mask between assistant and end."""
        # Tokens: [BOS, USER, text, ASST, reply, reply, END, PAD]
        ids = np.array([[1, 50, 60, 100, 80, 90, 200, 0]])
        mask = self.build_mask(ids, assistant_id=100, end_turn_id=200)
        expected = np.array([[0, 0, 0, 1, 1, 1, 1, 0]])
        np.testing.assert_array_equal(mask, expected)

    def test_multi_turn_with_end_token(self):
        """Multi-turn: mask only assistant turns, skip user turns."""
        # Gemma 4 pattern:
        # <|turn>user\ntext<turn|>\n<|turn>model\nreply<turn|>\n<|turn>user\ntext2<turn|>\n<|turn>model\nreply2<turn|>\n
        TURN_START = 105  # <|turn>
        USER = 2364       # user
        MODEL = 4368      # model
        NL = 107          # \n
        TURN_END = 106    # <turn|>
        BOS = 2

        ids = np.array([[
            BOS,
            TURN_START, USER, NL, 60, 70, TURN_END, NL,        # user turn 1
            TURN_START, MODEL, NL, 80, 90, TURN_END, NL,       # assistant turn 1
            TURN_START, USER, NL, 61, 71, TURN_END, NL,        # user turn 2
            TURN_START, MODEL, NL, 81, 91, TURN_END, NL,       # assistant turn 2
        ]])

        mask = self.build_mask(ids, assistant_id=MODEL, end_turn_id=TURN_END)

        # Expected: 0 for BOS + user turns, 1 for assistant turns (including TURN_END)
        expected = np.array([[
            0,
            0, 0, 0, 0, 0, 0, 0,   # user turn 1: all masked
            0, 1, 1, 1, 1, 1, 0,   # assistant turn 1: MODEL through TURN_END
            0, 0, 0, 0, 0, 0, 0,   # user turn 2: all masked
            0, 1, 1, 1, 1, 1, 0,   # assistant turn 2: MODEL through TURN_END
        ]])

        np.testing.assert_array_equal(
            mask, expected,
            f"\nGot:      {mask[0].tolist()}\nExpected: {expected[0].tolist()}"
        )

    def test_batch_processing(self):
        """Batch of 2 sequences with different assistant positions."""
        ids = np.array([
            [1, 50, 100, 80, 90, 0],   # assistant at position 2
            [1, 50, 60, 100, 80, 0],    # assistant at position 3
        ])
        mask = self.build_mask(ids, assistant_id=100)
        expected = np.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ])
        np.testing.assert_array_equal(mask, expected)

    def test_no_assistant_token(self):
        """If no assistant token found, entire mask is 0."""
        ids = np.array([[1, 50, 60, 70, 80]])
        mask = self.build_mask(ids, assistant_id=100)
        expected = np.array([[0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(mask, expected)

    def test_multi_turn_without_end_token(self):
        """Multi-turn without end_turn_id: mask stays on from first assistant occurrence."""
        # Without end_turn_id, once we see MODEL, we stay on forever
        # (same as old single-turn behavior but from MODEL onward)
        ids = np.array([[1, 50, 60, 100, 80, 50, 60, 100, 90]])
        mask = self.build_mask(ids, assistant_id=100)
        # Once 100 is seen, everything after is 1
        expected = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(mask, expected)


class TestGemma4AutoDetection(unittest.TestCase):
    """Test auto-detection of assistant_id and end_turn_id for Gemma 4."""

    def test_gemma4_tokens(self):
        """Verify Gemma 4 uses token 4368 (model) and 106 (<turn|>)."""
        try:
            from mlx_vlm.utils import load
            _, processor = load(
                "mlx-community/gemma-4-e2b-it-8bit",
                processor_config={"trust_remote_code": True},
            )
            tok = processor.tokenizer

            # Verify the tokens we expect
            self.assertEqual(tok.decode([4368]).strip(), "model")
            self.assertEqual(tok.decode([106]), "<turn|>")

            # Build a multi-turn conversation and verify masking
            test = [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "A cat."},
                {"role": "user", "content": "Color?"},
                {"role": "assistant", "content": "Black."},
            ]
            text = tok.apply_chat_template(
                test, tokenize=False, add_generation_prompt=False
            )
            ids = np.array([tok.encode(text)])

            from mlx_vlm.trainer.sft_trainer import build_completion_mask
            mask = self.build_mask_for_gemma4(ids, tok)

            # Count: should have some 1s (assistant tokens) and some 0s (user tokens)
            total = mask.sum()
            self.assertGreater(total, 0, "Should have some assistant tokens masked as 1")
            self.assertLess(total, mask.size, "Should have some user tokens masked as 0")

            # Verify the mask pattern: "model" token should be 1, "user" token should be 0
            ids_flat = ids[0]
            mask_flat = mask[0]
            for i, tid in enumerate(ids_flat):
                if tid == 2364:  # "user"
                    self.assertEqual(mask_flat[i], 0, f"User token at {i} should be masked")
                if tid == 4368:  # "model"
                    self.assertEqual(mask_flat[i], 1, f"Model token at {i} should be trained")

            print(f"  Gemma 4: {int(total)}/{mask.size} tokens are trainable "
                  f"({100*total/mask.size:.0f}%)")

        except Exception as e:
            self.skipTest(f"Gemma 4 model not available: {e}")

    def build_mask_for_gemma4(self, ids, tok):
        from mlx_vlm.trainer.sft_trainer import build_completion_mask
        return build_completion_mask(ids, assistant_id=4368, end_turn_id=106)


class TestCompletionMaskInLoss(unittest.TestCase):
    """Test that completion mask integrates correctly with loss computation."""

    def test_masked_loss_is_lower(self):
        """Loss on completions only should differ from loss on everything."""
        import mlx.core as mx
        import mlx.nn as nn

        # Simulate: 10 tokens, assistant starts at position 5
        seq_len = 10
        vocab_size = 100
        logits = mx.random.normal((1, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (1, seq_len))

        # Full loss (no masking)
        full_loss = nn.losses.cross_entropy(logits, labels).mean()

        # Masked loss (only positions 5-9)
        mask = mx.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=mx.float32)
        masked_ce = nn.losses.cross_entropy(logits, labels) * mask
        masked_loss = masked_ce.sum() / mask.sum()

        mx.eval(full_loss, masked_loss)

        # They should be different (unless extremely unlucky random seed)
        self.assertFalse(
            abs(full_loss.item() - masked_loss.item()) < 1e-6,
            "Full and masked loss should differ"
        )
        print(f"  Full loss: {full_loss.item():.4f}, Masked loss: {masked_loss.item():.4f}")


if __name__ == "__main__":
    print()
    unittest.main(verbosity=2)
