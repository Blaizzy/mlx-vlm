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

    def test_lora_layer_uses_alpha_over_rank_scaling(self):
        """LoRaLayer must apply the standard alpha/rank scaling factor.

        Regression test for issue #845: previously the layer multiplied
        the LoRA update by raw `alpha`, making the effective scaling
        rank-times too large for the documented defaults.
        """
        rank = 8
        alpha = 16
        linear = nn.Linear(64, 64)
        lora = LoRaLayer(linear, rank=rank, alpha=alpha, dropout=0.0)

        self.assertEqual(lora.rank, rank)
        self.assertEqual(lora.alpha, alpha)
        self.assertAlmostEqual(lora.scaling, alpha / rank)

    def test_lora_layer_forward_matches_alpha_over_rank(self):
        """LoRaLayer.__call__ output should equal base + (alpha/rank) * (x A B).

        Verifies the actual forward pass uses the corrected scaling, not
        just the stored attribute. Sets B to a non-zero value so the
        update is observable (zero-init B is the standard PEFT default).
        """
        rank = 4
        alpha = 8
        linear = nn.Linear(8, 8, bias=False)
        lora = LoRaLayer(linear, rank=rank, alpha=alpha, dropout=0.0)

        # Override B with deterministic non-zero values so update != 0.
        lora.B = mx.ones((rank, 8))
        x = mx.ones((1, 8))

        expected_base = linear(x)
        expected_update = (alpha / rank) * ((x @ lora.A) @ lora.B)
        expected = expected_base + expected_update.astype(x.dtype)

        actual = lora(x)
        mx.eval(actual, expected)
        self.assertTrue(mx.allclose(actual, expected, atol=1e-5).item())


if __name__ == "__main__":
    unittest.main()
