import unittest
from unittest.mock import MagicMock, patch

import mlx.nn as nn

from mlx_vlm.trainer.utils import (
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
        model = MagicMock()
        model.language_model.named_modules.return_value = [
            ("layer1", nn.Linear(256, 512)),
            ("layer2", nn.QuantizedLinear(256, 512, 8)),
        ]

        result = get_peft_model(model, ["layer1", "layer2"])

        self.assertTrue(mock_freeze.called)
        self.assertTrue(mock_print.called)
        self.assertTrue(hasattr(model.config, "lora"))

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


if __name__ == "__main__":
    unittest.main()
