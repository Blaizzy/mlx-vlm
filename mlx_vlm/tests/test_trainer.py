import unittest
from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.trainer import Dataset, Trainer


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.mock_hf_dataset = MagicMock()
        self.mock_config = {"model_type": "test_model", "image_token_index": 1}
        self.mock_processor = MagicMock()
        self.mock_image_processor = MagicMock()

    def test_dataset_initialization(self):
        dataset = Dataset(
            self.mock_hf_dataset,
            self.mock_config,
            self.mock_processor,
            self.mock_image_processor,
            take=10,
            split="train",
        )

        self.assertEqual(len(dataset), len(self.mock_hf_dataset["train"].take(10)))
        self.assertEqual(dataset.config, self.mock_config)
        self.assertEqual(dataset.processor, self.mock_processor)
        self.assertEqual(dataset.image_processor, self.mock_image_processor)

    @patch("mlx_vlm.trainer.trainer.get_prompt")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_dataset_getitem(self, mock_prepare_inputs, mock_get_prompt):
        dataset = Dataset(
            self.mock_hf_dataset,
            self.mock_config,
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
            "input_ids": mx.array([1, 2, 3]),  # input_ids
            "pixel_values": mx.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            ),  # pixel_values
            "attention_mask": mx.array([1, 1, 1]),  # mask
            "image_grid_thw": (1, 1, 1),  # image_grid_thw
            "image_sizes": [224, 224],  # image_sizes
        }

        result = dataset[0]

        mock_prepare_inputs.assert_called_once()
        self.assertIn("pixel_values", result)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("image_grid_thw", result)
        self.assertIn("image_sizes", result)

        # Check if the returned values match the mocked input
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


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock(spec=nn.Module)
        self.mock_optimizer = MagicMock()
        self.trainer = Trainer(self.mock_model, self.mock_optimizer)

    def test_trainer_initialization(self):
        self.assertEqual(self.trainer.model, self.mock_model)
        self.assertEqual(self.trainer.optimizer, self.mock_optimizer)
        self.assertFalse(self.trainer.train_on_completions)
        self.assertEqual(self.trainer.assistant_id, 77091)

    def test_loss_fn(self):
        batch = {
            "pixel_values": mx.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "input_ids": mx.array([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": mx.array([[1, 1, 1], [1, 1, 0]]),
            "image_grid_thw": (1, 1, 1),
            "image_sizes": [224, 224],
            "aspect_ratio_ids": mx.array([[1, 2], [3, 4]]),
            "aspect_ratio_mask": mx.array([[1, 1], [1, 0]]),
            "cross_attention_mask": mx.array([[1, 1], [1, 0]]),
        }

        mock_logits = mx.array([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]])
        # Create a mock LanguageModelOutput with the logits
        mock_output = Mock()
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output

        loss = self.trainer.loss_fn(self.mock_model, batch)

        self.assertIsInstance(loss, mx.array)
        self.assertEqual(loss.shape, ())  # Scalar value

    @patch.object(Trainer, "loss_fn")
    @patch("mlx.nn.value_and_grad")
    def test_train_step(self, mock_value_and_grad, mock_loss_fn):
        mock_batch = MagicMock()
        mock_loss = mx.array(0.5)
        mock_grads = {"param1": mx.array([0.1, 0.2]), "param2": mx.array([0.3, 0.4])}

        mock_value_and_grad.return_value = lambda *args, **kwargs: (
            mock_loss,
            mock_grads,
        )

        loss = self.trainer.train_step(mock_batch)

        self.mock_optimizer.update.assert_called_once_with(self.mock_model, mock_grads)
        self.assertEqual(loss, mock_loss)


if __name__ == "__main__":
    unittest.main()
