import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear

from mlx_vlm.trainer.utils import (
    apply_lora_layers,
    find_all_linear_names,
    get_module_by_name,
    get_peft_model,
    set_module_by_name,
)


class TestTrainerUtils(unittest.TestCase):

    def test_get_module_by_name(self):
        model = MagicMock()
        model.layer1.layer2.layer3 = "test_module"

        result = get_module_by_name(model, "layer1.layer2.layer3")
        self.assertEqual(result, "test_module")

    def test_set_module_by_name(self):
        model = MagicMock()
        new_module = MagicMock()

        set_module_by_name(model, "layer1.layer2.layer3", new_module)
        self.assertEqual(model.layer1.layer2.layer3, new_module)

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

    @patch("mlx_lm.utils.load_adapters")
    def test_apply_lora_layers_dispatches_text_model_adapters(self, mock_load_adapters):
        inner_model = MagicMock()
        loaded_inner_model = MagicMock()
        mock_load_adapters.return_value = loaded_inner_model

        model = MagicMock()
        model._is_text_model = True
        model.language_model._model = inner_model

        result = apply_lora_layers(model, "adapter-dir")

        self.assertIs(result, model)
        mock_load_adapters.assert_called_once_with(inner_model, "adapter-dir")
        self.assertIs(model.language_model._model, loaded_inner_model)

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
            model._is_text_model = False
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

    def test_apply_lora_layers_loads_mlx_lm_vlm_schema(self):
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
            (adapter_dir / "adapter_config.json").write_text(
                """
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
                """
            )
            (adapter_dir / "adapters.safetensors").touch()

            model = DummyModel()
            result = apply_lora_layers(model, str(adapter_dir))

            self.assertIs(result, model)
            self.assertIsInstance(model.language_model.proj, LoRALinear)
            self.assertEqual(
                model.loaded_weights,
                (str(adapter_dir / "adapters.safetensors"), False),
            )


if __name__ == "__main__":
    unittest.main()
