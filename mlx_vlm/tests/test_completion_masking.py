import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_vlm.trainer.datasets import (
    PreferenceVisionDataset,
    VisionDataset,
    build_completion_mask,
    resolve_completion_token_ids,
)
from mlx_vlm.trainer.orpo_trainer import get_logps, iterate_batches as iterate_orpo
from mlx_vlm.trainer.sft_trainer import (
    iterate_batches,
    vision_language_loss_fn,
)


class TestBuildCompletionMask(unittest.TestCase):
    def test_single_turn_no_end_token(self):
        ids = np.array([[1, 50, 60, 70, 100, 80, 90]])
        mask = build_completion_mask(ids, assistant_id=100)
        expected = mx.array([[0, 0, 0, 0, 1, 1, 1]])
        self.assertTrue(mx.array_equal(mask, expected))

    def test_single_turn_with_end_token(self):
        ids = np.array([[1, 50, 60, 100, 80, 90, 200, 0]])
        mask = build_completion_mask(ids, assistant_id=100, end_turn_id=200)
        expected = mx.array([[0, 0, 0, 1, 1, 1, 1, 0]])
        self.assertTrue(mx.array_equal(mask, expected))

    def test_multi_turn_with_end_token(self):
        turn_start = 105
        user = 2364
        model = 4368
        nl = 107
        turn_end = 106
        bos = 2

        ids = np.array(
            [
                [
                    bos,
                    turn_start,
                    user,
                    nl,
                    60,
                    70,
                    turn_end,
                    nl,
                    turn_start,
                    model,
                    nl,
                    80,
                    90,
                    turn_end,
                    nl,
                    turn_start,
                    user,
                    nl,
                    61,
                    71,
                    turn_end,
                    nl,
                    turn_start,
                    model,
                    nl,
                    81,
                    91,
                    turn_end,
                    nl,
                ]
            ]
        )

        mask = build_completion_mask(ids, assistant_id=model, end_turn_id=turn_end)
        expected = mx.array(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ]
            ]
        )
        self.assertTrue(mx.array_equal(mask, expected))

    def test_multi_turn_without_end_token_with_user_id(self):
        ids = np.array([[1, 50, 60, 100, 80, 50, 60, 100, 90]])
        mask = build_completion_mask(ids, assistant_id=100, user_id=50)
        expected = mx.array([[0, 0, 0, 1, 1, 0, 0, 1, 1]])
        self.assertTrue(mx.array_equal(mask, expected))

    def test_no_assistant_token(self):
        ids = np.array([[1, 50, 60, 70, 80]])
        mask = build_completion_mask(ids, assistant_id=100)
        expected = mx.array([[0, 0, 0, 0, 0]])
        self.assertTrue(mx.array_equal(mask, expected))


class FakeTokenizer:
    token_map = {
        1: "<|turn|>",
        2: "<|im_end|>",
        10: "user",
        11: "assistant",
        20: "U",
        21: "A",
    }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "unused"

    def encode(self, text):
        return [1, 10, 20, 2, 1, 11, 21, 2]

    def decode(self, ids):
        return self.token_map[ids[0]]


class FakeGemmaTokenizer:
    token_map = {
        1: "<start_of_turn>",
        2: "<end_of_turn>",
        10: "user",
        11: "model",
        20: "U",
        21: "A",
    }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "unused"

    def encode(self, text):
        return [1, 10, 20, 2, 1, 11, 21, 2]

    def decode(self, ids):
        return self.token_map[ids[0]]


class TestResolveCompletionTokenIds(unittest.TestCase):
    def test_detects_role_and_end_turn_ids(self):
        assistant_id, end_turn_id, user_id = resolve_completion_token_ids(
            FakeTokenizer()
        )
        self.assertEqual(assistant_id, 11)
        self.assertEqual(end_turn_id, 2)
        self.assertEqual(user_id, 10)

    def test_detects_gemma_role_and_end_turn_ids(self):
        assistant_id, end_turn_id, user_id = resolve_completion_token_ids(
            FakeGemmaTokenizer()
        )
        self.assertEqual(assistant_id, 11)
        self.assertEqual(end_turn_id, 2)
        self.assertEqual(user_id, 10)


class TestCompletionMaskDatasets(unittest.TestCase):
    @patch("mlx_vlm.trainer.datasets.apply_chat_template")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_vision_dataset_returns_completion_mask(
        self, mock_prepare_inputs, mock_apply_chat_template
    ):
        mock_apply_chat_template.return_value = ""
        mock_prepare_inputs.return_value = {
            "input_ids": mx.array([1, 50, 100, 80, 200]),
            "attention_mask": mx.array([1, 1, 1, 1, 1]),
            "pixel_values": None,
        }

        dataset = VisionDataset(
            [{"messages": [{"role": "user", "content": "Hello"}]}],
            {"model_type": "test_model", "image_token_index": 1},
            object(),
            train_on_completions=True,
            assistant_id=100,
            end_turn_id=200,
        )

        item = dataset[0]
        expected = mx.array([0, 0, 1, 1, 1])
        self.assertTrue(mx.array_equal(item["completion_mask"], expected))

    @patch("mlx_vlm.utils.prepare_inputs")
    def test_preference_dataset_returns_prefixed_completion_masks(
        self, mock_prepare_inputs
    ):
        mock_prepare_inputs.side_effect = [
            {
                "input_ids": mx.array([1, 100, 80, 200]),
                "attention_mask": mx.array([1, 1, 1, 1]),
            },
            {
                "input_ids": mx.array([1, 100, 90, 200]),
                "attention_mask": mx.array([1, 1, 1, 1]),
            },
        ]

        dataset = PreferenceVisionDataset(
            [{"chosen": "chosen", "rejected": "rejected"}],
            {"model_type": "test_model", "image_token_index": 1},
            object(),
            train_on_completions=True,
            assistant_id=100,
            end_turn_id=200,
        )

        item = dataset[0]
        expected = mx.array([0, 1, 1, 1])
        self.assertTrue(mx.array_equal(item["chosen_completion_mask"], expected))
        self.assertTrue(mx.array_equal(item["rejected_completion_mask"], expected))


class TestCompletionMaskCollation(unittest.TestCase):
    def test_sft_iterate_batches_pads_completion_mask(self):
        dataset = [
            {
                "input_ids": mx.array([1, 100, 80]),
                "attention_mask": mx.array([1, 1, 1]),
                "completion_mask": mx.array([0, 1, 1]),
                "pixel_values": None,
            },
            {
                "input_ids": mx.array([1, 50]),
                "attention_mask": mx.array([1, 1]),
                "completion_mask": mx.array([0, 0]),
                "pixel_values": None,
            },
        ]

        batch = next(iterate_batches(dataset, batch_size=2, max_seq_length=32))

        self.assertEqual(batch["completion_mask"].shape, (2, 32))
        self.assertTrue(
            mx.array_equal(
                batch["completion_mask"][:, :3],
                mx.array([[0, 1, 1], [0, 0, 0]]),
            )
        )

    def test_orpo_iterate_batches_pads_completion_masks(self):
        dataset = [
            {
                "chosen_input_ids": mx.array([1, 100, 80]),
                "chosen_attention_mask": mx.array([1, 1, 1]),
                "chosen_completion_mask": mx.array([0, 1, 1]),
                "rejected_input_ids": mx.array([1, 100]),
                "rejected_attention_mask": mx.array([1, 1]),
                "rejected_completion_mask": mx.array([0, 1]),
            }
        ]

        batch = next(iterate_orpo(dataset, batch_size=1, max_seq_length=32))

        self.assertEqual(batch["chosen"]["completion_mask"].shape, (1, 32))
        self.assertEqual(batch["rejected"]["completion_mask"].shape, (1, 32))
        self.assertTrue(
            mx.array_equal(batch["chosen"]["completion_mask"][:, :3], mx.array([[0, 1, 1]]))
        )
        self.assertTrue(
            mx.array_equal(batch["rejected"]["completion_mask"][:, :2], mx.array([[0, 1]]))
        )


class TestCompletionMaskLossGuards(unittest.TestCase):
    def test_sft_loss_requires_completion_mask_when_enabled(self):
        batch = {
            "input_ids": mx.array([[1, 2, 3]]),
            "attention_mask": mx.array([[1, 1, 1]]),
            "pixel_values": None,
        }
        with self.assertRaisesRegex(ValueError, "completion_mask"):
            vision_language_loss_fn(None, batch, train_on_completions=True)

    def test_orpo_get_logps_requires_completion_mask_when_enabled(self):
        batch = {
            "input_ids": mx.array([[1, 2, 3]]),
            "attention_mask": mx.array([[1, 1, 1]]),
            "pixel_values": None,
        }
        with self.assertRaisesRegex(ValueError, "completion_mask"):
            get_logps(None, batch, train_on_completions=True)


if __name__ == "__main__":
    unittest.main()
