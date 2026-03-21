import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.lora import LoRaLayer
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


class TestLoRaScaling(unittest.TestCase):
    """Verify LoRaLayer uses alpha/rank scaling (standard LoRA convention)."""

    def test_scale_is_alpha_over_rank(self):
        linear = nn.Linear(4, 4)
        lora = LoRaLayer(linear, rank=8, alpha=16.0)
        self.assertAlmostEqual(lora.scale, 2.0)  # 16 / 8 = 2.0

    def test_scale_with_rank_equals_alpha(self):
        linear = nn.Linear(4, 4)
        lora = LoRaLayer(linear, rank=4, alpha=4.0)
        self.assertAlmostEqual(lora.scale, 1.0)  # 4 / 4 = 1.0

    def test_forward_scaling_matches_peft(self):
        """LoRA contribution should equal (alpha/rank) * (x @ A @ B)."""
        linear = nn.Linear(4, 4)
        lora = LoRaLayer(linear, rank=8, alpha=16.0, dropout=0.0)

        # Set deterministic weights
        lora.A = mx.ones((4, 8))
        lora.B = mx.ones((8, 4))
        x = mx.ones((1, 4))

        base_output = linear(x)
        actual_output = lora(x)
        lora_contribution = actual_output - base_output

        # Expected: (alpha / rank) * (x @ A @ B) = 2.0 * (ones(1,4) @ ones(4,8) @ ones(8,4))
        # x @ A = 4 * ones(1,8), then @ B = 32 * ones(1,4), then * 2.0 = 64
        expected_per_element = 2.0 * 4.0 * 8.0  # 64.0
        self.assertAlmostEqual(
            lora_contribution[0, 0].item(), expected_per_element, places=1
        )

    def test_default_alpha_rank_gives_2x(self):
        """Default alpha=16, rank=8 should give 2x scaling, not 16x."""
        linear = nn.Linear(8, 8)
        lora = LoRaLayer(linear, rank=8, alpha=16.0, dropout=0.0)

        lora.A = mx.ones((8, 8))
        lora.B = mx.ones((8, 8))
        x = mx.ones((1, 8))

        base = linear(x)
        actual = lora(x)
        contribution = (actual - base)[0, 0].item()

        raw_delta = (x @ lora.A @ lora.B)[0, 0].item()  # 64.0

        # Should be 2x the raw delta, not 16x
        self.assertAlmostEqual(contribution, 2.0 * raw_delta, places=1)
        self.assertNotAlmostEqual(contribution, 16.0 * raw_delta, places=1)


if __name__ == "__main__":
    unittest.main()
