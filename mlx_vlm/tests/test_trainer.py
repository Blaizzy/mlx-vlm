"""Training workflows and trainer utilities."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.lora import LoRaLayer
from mlx_vlm.trainer.lora_layers import LoRALinear
from mlx_vlm.trainer.sft_trainer import (
    TrainingArgs,
    iterate_batches,
    train,
    vision_language_loss_fn,
)
from mlx_vlm.trainer.utils import (
    apply_lora_layers,
    find_all_linear_names,
    get_peft_model,
)

# Training workflows


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.mock_hf_dataset = MagicMock()
        self.mock_config = {"model_type": "test_model", "image_token_index": 1}
        self.mock_processor = MagicMock()
        self.mock_image_processor = MagicMock()

    @patch("mlx_vlm.trainer.datasets.apply_chat_template")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_dataset_getitem(self, mock_prepare_inputs, mock_apply_chat_template):
        dataset = VisionDataset(
            self.mock_hf_dataset, self.mock_config, self.mock_processor
        )

        mock_apply_chat_template.return_value = ""

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
        self.assertTrue(mx.array_equal(result["image_sizes"], mx.array([224, 224])))

    @patch("mlx_vlm.trainer.datasets.apply_chat_template")
    def test_dataset_getitem_raises_when_image_token_keys_missing(
        self, mock_apply_chat_template
    ):
        """Test that a clear ValueError is raised when neither key exists."""
        config_missing_token = {"model_type": "test_model"}
        mock_apply_chat_template.return_value = ""

        dataset = VisionDataset(
            self.mock_hf_dataset, config_missing_token, self.mock_processor
        )

        mock_item = {
            "images": ["image1.jpg"],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.mock_hf_dataset.__getitem__.return_value = mock_item

        with self.assertRaises(ValueError) as context:
            dataset[0]

        self.assertIn("image_token_index", str(context.exception))

    def test_dataset_initialization(self):
        dataset = VisionDataset(
            self.mock_hf_dataset, self.mock_config, self.mock_processor
        )

        self.assertEqual(len(dataset), len(self.mock_hf_dataset))
        self.assertEqual(dataset.config, self.mock_config)
        self.assertEqual(dataset.processor, self.mock_processor)

    @patch("mlx_vlm.trainer.datasets.apply_chat_template")
    @patch("mlx_vlm.utils.prepare_inputs")
    def test_dataset_adds_completion_mask_from_chat_template(
        self, mock_prepare_inputs, mock_apply_chat_template
    ):
        dataset = VisionDataset(
            self.mock_hf_dataset,
            self.mock_config,
            self.mock_processor,
            train_on_completions=True,
        )

        messages = [
            {"role": "user", "content": "Use the tool."},
            {"role": "assistant", "content": "tool call"},
        ]
        self.mock_hf_dataset.__getitem__.return_value = {"messages": messages}

        def fake_apply_chat_template(
            _processor, _config, conversation, add_generation_prompt, **_kwargs
        ):
            if add_generation_prompt:
                self.assertEqual(conversation, messages[:-1])
                return "prefix"
            self.assertEqual(conversation, messages)
            return "full"

        def fake_prepare_inputs(**kwargs):
            prompts = kwargs["prompts"]
            if prompts == ["prefix"]:
                return {"input_ids": mx.array([[1, 2]])}
            return {
                "input_ids": mx.array([[1, 2, 3, 4]]),
                "attention_mask": mx.array([[1, 1, 1, 1]]),
            }

        mock_apply_chat_template.side_effect = fake_apply_chat_template
        mock_prepare_inputs.side_effect = fake_prepare_inputs

        result = dataset[0]

        self.assertIn("completion_mask", result)
        self.assertTrue(
            mx.array_equal(result["completion_mask"], mx.array([[0, 0, 1, 1]]))
        )


class TestBatchCollation(unittest.TestCase):
    def test_iterate_batches_concatenates_variable_length_pixel_values(self):
        dataset = [
            {
                "input_ids": mx.array([1, 2, 3]),
                "attention_mask": mx.array([1, 1, 1]),
                "pixel_values": mx.zeros((2, 4)),
                "image_grid_thw": mx.array([[1, 1, 2]]),
            },
            {
                "input_ids": mx.array([4, 5]),
                "attention_mask": mx.array([1, 1]),
                "pixel_values": mx.ones((3, 4)),
                "image_grid_thw": mx.array([[1, 1, 3]]),
            },
        ]

        batch = next(iterate_batches(dataset, batch_size=2, max_seq_length=32))

        self.assertEqual(batch["input_ids"].shape, (2, 32))
        self.assertEqual(batch["pixel_values"].shape, (5, 4))
        self.assertTrue(
            mx.array_equal(batch["image_grid_thw"], mx.array([[1, 1, 2], [1, 1, 3]]))
        )

    def test_iterate_batches_pads_completion_mask(self):
        dataset = [
            {
                "input_ids": mx.array([1, 2, 3]),
                "attention_mask": mx.array([1, 1, 1]),
                "pixel_values": None,
                "completion_mask": mx.array([0, 1, 1]),
            },
            {
                "input_ids": mx.array([4, 5]),
                "attention_mask": mx.array([1, 1]),
                "pixel_values": None,
                "completion_mask": mx.array([0, 1]),
            },
        ]

        batch = next(iterate_batches(dataset, batch_size=2, max_seq_length=32))

        self.assertIn("completion_mask", batch)
        self.assertEqual(batch["completion_mask"].shape, (2, 32))
        self.assertTrue(
            mx.array_equal(batch["completion_mask"][0, :3], mx.array([0, 1, 1]))
        )
        self.assertTrue(
            mx.array_equal(batch["completion_mask"][1, :2], mx.array([0, 1]))
        )


class _VisionDatasetStub(list):
    """A dataset list that also carries a model config, like VisionDataset."""

    def __init__(self, items, config):
        super().__init__(items)
        self.config = config


def _image_example(num_image_tokens, image_token_id=99, num_sub_images=2):
    input_ids = mx.array([1] + [image_token_id] * num_image_tokens)
    return {
        "input_ids": input_ids,
        "attention_mask": mx.ones(input_ids.shape, dtype=mx.int32),
        "pixel_values": mx.zeros((num_sub_images, 4)),
    }


class TestImageTokenTruncationGuard(unittest.TestCase):
    """Truncation must never cut image tokens (#1830).

    ``pixel_values`` still yields one feature per image token, so an example whose
    placeholders are truncated away fails the model's feature/token alignment
    check. Such examples are skipped instead of corrupting the batch.
    """

    config = {"image_token_index": 99}

    def test_raises_when_no_example_fits(self):
        dataset = _VisionDatasetStub(
            [_image_example(10), _image_example(12)], self.config
        )

        with self.assertRaises(ValueError) as ctx:
            next(iterate_batches(dataset, batch_size=1, max_seq_length=8))

        self.assertIn("No trainable examples", str(ctx.exception))

    def test_keeps_examples_that_fit_untouched(self):
        dataset = _VisionDatasetStub([_image_example(3)], self.config)

        batch = next(iterate_batches(dataset, batch_size=1, max_seq_length=32))

        self.assertEqual(int((batch["input_ids"] == 99).sum()), 3)


class TestIdefics3MaskArgument(unittest.TestCase):
    def test_call_accepts_mask_as_third_positional_argument(self):
        """The trainer calls ``model(input_ids, pixel_values, attention_mask)``.

        Idefics3 previously omitted ``mask``, so the attention mask bound to
        ``cache`` and crashed in ``create_attention_mask`` (#1830).
        """
        import inspect

        from mlx_vlm.models.idefics3.idefics3 import Model

        params = list(inspect.signature(Model.__call__).parameters)
        self.assertEqual(params[1:4], ["input_ids", "pixel_values", "mask"])


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

    @patch("mlx_vlm.trainer.sft_trainer.iterate_batches")
    @patch("mlx_vlm.trainer.sft_trainer.mx.save_safetensors")
    def test_train_uses_default_adapter_file_when_missing(
        self, mock_save_safetensors, mock_iterate_batches
    ):
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
            args=TrainingArgs(
                iters=1, batch_size=4, steps_per_save=1, adapter_file=None
            ),
        )

        saved_paths = [call.args[0] for call in mock_save_safetensors.call_args_list]
        self.assertEqual(
            saved_paths,
            [
                "adapters.safetensors",
                "0000001_adapters.safetensors",
                "adapters.safetensors",
            ],
        )

    def test_completion_mask_is_used_without_passing_to_model(self):
        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits

        class CaptureModel:
            model_type = "test_model"

            def __call__(self, input_ids, pixel_values, mask, **kwargs):
                self.mask = mask
                self.kwargs = kwargs
                vocab_size = 10
                return DummyOutput(mx.zeros((*input_ids.shape, vocab_size)))

        model = CaptureModel()
        batch = {
            "input_ids": mx.array([[1, 2, 3, 4]]),
            "attention_mask": mx.array([[1, 1, 1, 1]]),
            "completion_mask": mx.array([[0, 0, 1, 1]]),
            "pixel_values": None,
        }

        vision_language_loss_fn(
            model, batch, train_on_completions=True, assistant_id=999
        )

        self.assertTrue(mx.array_equal(model.mask, mx.array([[1, 1, 1]])))
        self.assertNotIn("completion_mask", model.kwargs)


class TestLoRaScaling(unittest.TestCase):
    """Verify LoRaLayer uses alpha/rank scaling (standard LoRA convention)."""

    def test_b_zero_init_gives_no_lora_contribution(self):
        """When B is zeros (default init), output should equal base linear."""
        linear = nn.Linear(4, 4)
        lora = LoRaLayer(linear, rank=8, alpha=16.0, dropout=0.0)
        # B is already zeros from __init__, don't override it
        x = mx.ones((1, 4))
        base_output = linear(x)
        lora_output = lora(x)
        self.assertTrue(mx.allclose(base_output, lora_output).item())


# Trainer utilities


class TestTrainerUtils(unittest.TestCase):

    @patch("mlx_vlm.trainer.utils.freeze_model")
    @patch("mlx_vlm.trainer.utils.print_trainable_parameters")
    def test_get_peft_model(self, mock_print, mock_freeze):
        class DummyLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(256, 512)
                self.layer2 = nn.QuantizedLinear(256, 512, 8)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = DummyLanguageModel()
                self.config = SimpleNamespace()

        model = DummyModel()

        result = get_peft_model(model, ["layer1", "layer2"])

        self.assertTrue(mock_freeze.called)
        self.assertTrue(mock_print.called)
        self.assertTrue(hasattr(model.config, "lora"))
        self.assertIs(result, model)
        self.assertIsInstance(model.language_model.layer1, LoRALinear)
        self.assertIsInstance(model.language_model.layer2, LoRALinear)
        self.assertEqual(model.config.lora["fine_tune_type"], "lora")
        self.assertEqual(model.config.lora["num_layers"], -1)
        self.assertEqual(model.config.lora["lora_parameters"]["rank"], 10)
        self.assertAlmostEqual(model.config.lora["lora_parameters"]["scale"], 0.01)
        self.assertEqual(
            set(model.config.lora["lora_parameters"]["keys"]),
            {"language_model.layer1", "language_model.layer2"},
        )

    def test_find_all_linear_names(self):
        model = MagicMock()
        model.named_modules.return_value = [
            ("layer1", nn.Linear(256, 512)),
            ("layer2", nn.QuantizedLinear(256, 512, 8)),
            ("mm_projector", nn.Linear(256, 512)),
            ("lm_head", nn.Linear(256, 512)),
        ]

        result = find_all_linear_names(model)
        self.assertEqual(set(result), {"layer1", "layer2"})

    @patch("mlx_vlm.trainer.utils.get_peft_model")
    def test_apply_lora_layers_keeps_vlm_adapter_schema(self, mock_get_peft):
        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                '{"rank": 4, "alpha": 8, "dropout": 0.0}'
            )
            (adapter_dir / "adapters.safetensors").touch()

            model = MagicMock()
            model.language_model.named_modules.return_value = []
            mock_get_peft.return_value = model

            result = apply_lora_layers(model, str(adapter_dir))

            self.assertIs(result, model)
            mock_get_peft.assert_called_once_with(
                model, [], rank=4, alpha=8, dropout=0.0, legacy=True
            )
            model.load_weights.assert_called_once_with(
                str(adapter_dir / "adapters.safetensors"), strict=False
            )

    def test_apply_lora_layers_loads_native_vlm_schema(self):
        class DummyLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 8)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = DummyLanguageModel()
                self.loaded_weights = None

            def load_weights(self, path, strict=True):
                self.loaded_weights = (path, strict)

        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("""
                {
                  "fine_tune_type": "lora",
                  "num_layers": -1,
                  "lora_parameters": {
                    "rank": 4,
                    "dropout": 0.0,
                    "scale": 2.0,
                    "keys": ["language_model.proj"]
                  }
                }
                """)
            (adapter_dir / "adapters.safetensors").touch()

            model = DummyModel()
            result = apply_lora_layers(model, str(adapter_dir))

            self.assertIs(result, model)
            self.assertIsInstance(model.language_model.proj, LoRALinear)
            self.assertEqual(
                model.loaded_weights, (str(adapter_dir / "adapters.safetensors"), False)
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
