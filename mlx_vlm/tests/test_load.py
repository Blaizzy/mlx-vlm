import unittest

from mlx_vlm.utils import _resolve_skip_vision


class TestResolveSkipVision(unittest.TestCase):
    """Regression tests for the legacy quantization ``skip_vision`` lookup.

    Quantized checkpoints whose ``config.json`` sets ``vision_config`` to ``null``
    (e.g. text-only ``gemma4_unified`` coder quants whose vision/audio tower
    weights were stripped) must not crash the quantization loader. The model
    class skips the tower when ``vision_config`` is ``None``; this helper must
    tolerate ``None`` as well.
    """

    def test_null_vision_config_does_not_raise(self):
        self.assertFalse(_resolve_skip_vision({"vision_config": None}))

    def test_missing_vision_config_defaults_false(self):
        self.assertFalse(_resolve_skip_vision({}))

    def test_empty_vision_config_defaults_false(self):
        self.assertFalse(_resolve_skip_vision({"vision_config": {}}))

    def test_skip_vision_true_is_respected(self):
        self.assertTrue(_resolve_skip_vision({"vision_config": {"skip_vision": True}}))

    def test_skip_vision_false_is_respected(self):
        self.assertFalse(
            _resolve_skip_vision({"vision_config": {"skip_vision": False}})
        )


if __name__ == "__main__":
    unittest.main()
