import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.trainer import TrainingArgs, train


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
            self.mock_image_processor,
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
            self.mock_image_processor,
        )

        self.assertEqual(len(dataset), len(self.mock_hf_dataset))
        self.assertEqual(dataset.config, self.mock_config)
        self.assertEqual(dataset.processor, self.mock_processor)
        self.assertEqual(dataset.image_processor, self.mock_image_processor)


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

    @patch("mlx_vlm.trainer.trainer.iterate_batches")
    @patch("mlx_vlm.trainer.trainer.mx.save_safetensors")
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

    @patch("mlx_vlm.trainer.trainer.iterate_batches")
    @patch("mlx_vlm.trainer.trainer.mx.save_safetensors")
    def test_train_smoke(self, mock_save_safetensors, mock_iterate_batches):
        mock_batch = {
            "input_ids": mx.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "attention_mask": mx.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            "pixel_values": mx.array(
                [[[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]]]
            ),
            "labels": mx.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        }
        mock_iterate_batches.return_value = iter([mock_batch])

        train(
            model=self.mock_model,
            optimizer=self.mock_optimizer,
            train_dataset=MagicMock(__len__=lambda self: 4),
            val_dataset=None,
            args=TrainingArgs(iters=1, batch_size=4),
        )

        self.mock_optimizer.update.assert_called()
        mock_save_safetensors.assert_called()


if __name__ == "__main__":
    unittest.main()

    @patch("mlx_vlm.trainer.trainer.get_prompt")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_dataset_getitem_falls_back_to_image_token_id(
        self, mock_prepare_inputs, mock_get_prompt
    ):
        """Test that image_token_id is used when image_token_index is missing."""
        config_with_token_id = {"model_type": "test_model", "image_token_id": 151655}

        dataset = Dataset(
            self.mock_hf_dataset,
            config_with_token_id,
            self.mock_processor,
            self.mock_image_processor,
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

        result = dataset[0]
        call_kwargs = mock_prepare_inputs.call_args[1]
        self.assertEqual(call_kwargs["image_token_index"], 151655)

    def test_dataset_getitem_raises_when_image_token_keys_missing(self):
        """Test that a clear KeyError is raised when neither key exists."""
        config_missing_token = {"model_type": "test_model"}

        dataset = Dataset(
            self.mock_hf_dataset,
            config_missing_token,
            self.mock_processor,
            self.mock_image_processor,
        )

        mock_item = {
            "images": ["image1.jpg"],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.mock_hf_dataset.__getitem__.return_value = mock_item

        with self.assertRaises(KeyError) as context:
            dataset[0]

        self.assertIn("image_token_index", str(context.exception))
