import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.lora import LoRaLayer, replace_lora_with_linear
from mlx_vlm.trainer.sft_trainer import (
    TrainingArgs,
    iterate_batches,
    train,
    vision_language_loss_fn,
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.mock_hf_dataset = MagicMock()
        self.mock_config = {"model_type": "test_model", "image_token_index": 1}
        self.mock_processor = MagicMock()
        self.mock_image_processor = MagicMock()

    @patch("mlx_vlm.trainer.datasets.get_prompt")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_dataset_getitem(self, mock_prepare_inputs, mock_get_prompt):
        dataset = VisionDataset(
            self.mock_hf_dataset,
            self.mock_config,
            self.mock_processor,
        )

        mock_get_prompt.return_value = ""

        mock_item = {
            "images": ["image1.jpg"],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.mock_hf_dataset.__getitem__.return_value = mock_item

        mock_prepare_inputs.return_value = {
            "input_ids": mx.array([1, 2, 3]),
            "pixel_values": mx.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            ),
            "attention_mask": mx.array([1, 1, 1]),
            "image_grid_thw": (1, 1, 1),
            "image_sizes": [224, 224],
        }

        result = dataset[0]

        mock_prepare_inputs.assert_called_once()
        self.assertIn("pixel_values", result)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("image_grid_thw", result)
        self.assertIn("image_sizes", result)

        self.assertTrue(mx.array_equal(result["input_ids"], mx.array([1, 2, 3])))
        self.assertTrue(
            mx.array_equal(
                result["pixel_values"],
                mx.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            )
        )
        self.assertTrue(mx.array_equal(result["attention_mask"], mx.array([1, 1, 1])))
        self.assertEqual(result["image_grid_thw"], (1, 1, 1))
        self.assertEqual(result["image_sizes"], [224, 224])

    def test_dataset_initialization(self):
        dataset = VisionDataset(
            self.mock_hf_dataset,
            self.mock_config,
            self.mock_processor,
        )

        self.assertEqual(len(dataset), len(self.mock_hf_dataset))
        self.assertEqual(dataset.config, self.mock_config)
        self.assertEqual(dataset.processor, self.mock_processor)

    @patch("mlx_vlm.trainer.datasets.get_prompt")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_dataset_getitem_falls_back_to_image_token_id(
        self, mock_prepare_inputs, mock_get_prompt
    ):
        """Test that image_token_id is used when image_token_index is missing."""
        config_with_token_id = {"model_type": "test_model", "image_token_id": 151655}

        dataset = VisionDataset(
            self.mock_hf_dataset,
            config_with_token_id,
            self.mock_processor,
        )

        mock_item = {
            "images": ["image1.jpg"],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.mock_hf_dataset.__getitem__.return_value = mock_item
        mock_get_prompt.return_value = "Mocked prompt"
        mock_prepare_inputs.return_value = {
            "input_ids": mx.array([1, 2, 3]),
            "pixel_values": mx.array([[0.1, 0.2, 0.3]]),
            "attention_mask": mx.array([1, 1, 1]),
        }

        dataset[0]
        call_kwargs = mock_prepare_inputs.call_args[1]
        self.assertEqual(call_kwargs["image_token_index"], 151655)

    def test_dataset_getitem_raises_when_image_token_keys_missing(self):
        """Test that a clear ValueError is raised when neither key exists."""
        config_missing_token = {"model_type": "test_model"}

        dataset = VisionDataset(
            self.mock_hf_dataset,
            config_missing_token,
            self.mock_processor,
        )

        mock_item = {
            "images": ["image1.jpg"],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.mock_hf_dataset.__getitem__.return_value = mock_item

        with self.assertRaises(ValueError) as context:
            dataset[0]

        self.assertIn("image_token_index", str(context.exception))


class TestTrainer(unittest.TestCase):
    def setUp(self):
        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = mx.zeros((1,))

            def __call__(self, *args, **kwargs):
                return DummyOutput(logits=mx.zeros((4, 3, 10)))

        self.mock_model = DummyModel()
        self.mock_optimizer = MagicMock()
        self.mock_optimizer.learning_rate = 1e-4

    @patch("mlx_vlm.trainer.sft_trainer.iterate_batches")
    @patch("mlx_vlm.trainer.sft_trainer.mx.save_safetensors")
    def test_trainer_initialization(self, mock_save_safetensors, mock_iterate_batches):
        mock_batch = {
            "input_ids": mx.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "attention_mask": mx.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            "pixel_values": mx.array(
                [[[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]]]
            ),
            "labels": mx.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        }
        mock_iterate_batches.return_value = iter([mock_batch])

        result = train(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            train_dataset=MagicMock(__len__=lambda self: 4),
            val_dataset=None,
            args=TrainingArgs(iters=1, batch_size=4),
        )

        self.assertIsNone(result)
        self.mock_optimizer.update.assert_called()
        mock_save_safetensors.assert_called()


class TestLoRA(unittest.TestCase):
    def test_vision_language_loss_forwards_full_sequence(self):
        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits

        class RecordingModel:
            def __init__(self):
                self.seen_input_shape = None
                self.seen_attention_shape = None

            def __call__(self, input_ids, pixel_values, attention_mask, **kwargs):
                self.seen_input_shape = input_ids.shape
                self.seen_attention_shape = attention_mask.shape
                # Return full-length logits; loss function should shift after forward.
                return DummyOutput(logits=mx.zeros((1, 4, 8), dtype=mx.float32))

        model = RecordingModel()
        batch = {
            "input_ids": mx.array([[1, 2, 3, 4]], dtype=mx.int32),
            "attention_mask": mx.array([[1, 1, 1, 1]], dtype=mx.int32),
            "pixel_values": None,
            "image_grid_thw": mx.array([[1, 1, 1]], dtype=mx.int32),
        }

        loss = vision_language_loss_fn(model, batch)

        self.assertEqual(model.seen_input_shape, (1, 4))
        self.assertEqual(model.seen_attention_shape, (1, 4))
        self.assertEqual(loss.shape, ())

    @patch("mlx_vlm.trainer.sft_trainer.mx.distributed.init")
    def test_iterate_batches_uses_flattened_input_length(self, mock_dist_init):
        mock_world = MagicMock()
        mock_world.rank.return_value = 0
        mock_world.size.return_value = 1
        mock_dist_init.return_value = mock_world

        dataset = [
            {"input_ids": mx.array([list(range(40))], dtype=mx.int32)},
            {"input_ids": mx.array([9, 8, 7], dtype=mx.int32)},
        ]

        batch = next(
            iterate_batches(
                dataset=dataset,
                batch_size=2,
                max_seq_length=128,
                train=False,
            )
        )

        # The flattened 40-token example should determine max_len; current padding logic adds
        # one extra slot before rounding up to a multiple of 32, so padded_len becomes 65.
        self.assertEqual(batch["input_ids"].shape, (2, 65))
        first_len = int(mx.sum(batch["attention_mask"][0]).item())
        self.assertEqual(first_len, 40)

    def test_lora_layer_uses_alpha_over_rank_scaling(self):
        linear = nn.Linear(2, 2, bias=False)
        linear.weight = mx.zeros((2, 2), dtype=mx.float32)

        lora = LoRaLayer(linear=linear, rank=2, alpha=4.0, dropout=0.0)
        lora.A = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        lora.B = mx.array([[3.0, 5.0], [7.0, 11.0]], dtype=mx.float32)

        x = mx.array([[2.0, 3.0]], dtype=mx.float32)
        y = lora(x)

        raw_update = (x @ lora.A) @ lora.B
        expected = (lora.alpha / lora.rank) * raw_update

        self.assertTrue(mx.allclose(y, expected))

    def test_replace_lora_with_linear_uses_alpha_over_rank_scaling(self):
        linear = nn.Linear(2, 2, bias=True)
        linear.weight = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        linear.bias = mx.zeros((2,), dtype=mx.float32)

        lora = LoRaLayer(linear=linear, rank=2, alpha=4.0, dropout=0.0)
        lora.A = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        lora.B = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)

        class DummyModel:
            def __init__(self, layer):
                self.layers = [layer]

        model = DummyModel(lora)
        expected_update = (lora.alpha / lora.rank) * (lora.A @ lora.B)
        expected_weight = lora.original_layer.weight + expected_update
        replace_lora_with_linear(model)

        self.assertTrue(mx.allclose(model.layers[0].weight, expected_weight))


if __name__ == "__main__":
    unittest.main()
