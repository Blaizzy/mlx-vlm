"""VLM batching preserves supervision and processor media alignment."""

import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_vlm.trainer.vlm.dpo.trainer import make_task
from mlx_vlm.trainer.vlm.sft.runtime import collate_vlm_batch, forward_vlm_logits


class VLMBatchingTest(unittest.TestCase):
    def test_padding_preserves_attention_and_completion_positions(self):
        batch = collate_vlm_batch(
            [
                {
                    "input_ids": np.array([[1, 2, 3]]),
                    "completion_mask": mx.array([0, 1, 1]),
                },
                {"input_ids": mx.array([4, 5]), "attention_mask": np.array([1, 0])},
            ],
            max_seq_length=8,
            pad_to_multiple=4,
            pad_token_id=7,
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 7], [4, 5, 7, 7]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 0], [1, 0, 0, 0]])
        self.assertEqual(
            batch["completion_mask"].tolist(), [[0, 1, 1, 0], [0, 0, 0, 0]]
        )

    def test_variable_image_features_keep_media_order(self):
        batch = collate_vlm_batch(
            [
                {
                    "input_ids": mx.array([1, 2]),
                    "pixel_values": mx.ones((2, 4)),
                    "image_grid_thw": mx.array([[1, 2, 3], [1, 3, 4]]),
                },
                {
                    "input_ids": mx.array([3, 4]),
                    "pixel_values": mx.full((3, 4), 2),
                    "image_grid_thw": np.array([[1, 4, 5]]),
                },
            ],
            max_seq_length=8,
        )
        self.assertEqual(batch["pixel_values"].shape, (5, 4))
        self.assertEqual(batch["pixel_values"][:, 0].tolist(), [1, 1, 2, 2, 2])
        self.assertEqual(
            batch["image_grid_thw"].tolist(), [[1, 2, 3], [1, 3, 4], [1, 4, 5]]
        )

    def test_truncation_requires_preserved_media_placeholders(self):
        row = {"input_ids": mx.array([1, 9, 2, 3]), "pixel_values": mx.ones((2, 4))}
        with self.assertRaisesRegex(ValueError, "no image token ID"):
            collate_vlm_batch([row], max_seq_length=3)
        batch = collate_vlm_batch([row], max_seq_length=3, image_token_id=9)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 9, 2]])
        row["input_ids"] = mx.array([1, 2, 3, 9])
        with self.assertRaisesRegex(ValueError, "remove image placeholders"):
            collate_vlm_batch([row], max_seq_length=3, image_token_id=9)
        for field in ("audio_features", "video_values"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "audio or video"),
            ):
                collate_vlm_batch(
                    [{**row, field: mx.ones((2, 4))}],
                    max_seq_length=3,
                    image_token_id=9,
                )

    def test_forward_keeps_media_kwargs_and_gemma_mask_handling(self):
        class Model:
            config = {"model_type": "gemma4_unified"}

            def __call__(self, ids, pixels, mask, **kwargs):
                self.inputs = ids, pixels, mask, kwargs
                return {"logits": mx.ones((*ids.shape, 5), dtype=mx.float16)}

        model = Model()
        batch = {
            "input_ids": mx.array([[1, 2]]),
            "attention_mask": mx.ones((1, 2)),
            "completion_mask": mx.ones((1, 2)),
            "labels": mx.array([[1, 2]]),
            "pixel_values": mx.ones((2, 4)),
            "image_grid_thw": mx.array([[1, 2, 3]]),
        }
        logits = forward_vlm_logits(
            model, batch["input_ids"], batch["attention_mask"], batch
        )
        self.assertEqual(logits.dtype, mx.float32)
        self.assertIs(model.inputs[1], batch["pixel_values"])
        self.assertIsNone(model.inputs[2])
        self.assertEqual(set(model.inputs[3]), {"image_grid_thw"})


class DPOReferenceTest(unittest.TestCase):
    def test_task_rejects_different_reference_configs_before_training(self):
        for policy, reference, message in (
            ({"model_type": "a"}, {"model_type": "b"}, "model_type"),
            ({"image_token_id": 9}, {"image_token_id": 8}, "image_token_id"),
            (
                {"text_config": {"vocab_size": 8}},
                SimpleNamespace(text_config=SimpleNamespace(vocab_size=9)),
                "vocabulary",
            ),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                make_task(
                    SimpleNamespace(config=policy),
                    None,
                    reference_model=SimpleNamespace(config=reference),
                )
