"""Tests for the loader-side config_overrides mechanism and post-load
processor↔model sync (Gemma 4 image-token budget plumbing)."""

import logging
from types import SimpleNamespace
from unittest import TestCase

from mlx_vlm.models.gemma4.config import VisionConfig
from mlx_vlm.utils import _deep_merge, _sync_processor_to_model_config

# ---------------------------------------------------------------------------
# _deep_merge — pure function tests
# ---------------------------------------------------------------------------


class DeepMergeTests(TestCase):
    def test_adds_missing_keys(self):
        base = {}
        _deep_merge(base, {"a": 1})
        self.assertEqual(base, {"a": 1})

    def test_nested_dicts_recurse_and_preserve_siblings(self):
        # Critical for the vision_config shape: overriding default_output_length
        # must not clobber position_embedding_size / patch_size / etc.
        base = {"vision_config": {"default_output_length": 280, "patch_size": 16}}
        _deep_merge(base, {"vision_config": {"default_output_length": 1120}})
        self.assertEqual(
            base,
            {"vision_config": {"default_output_length": 1120, "patch_size": 16}},
        )

    def test_non_dict_values_replace_wholesale(self):
        # Lists replace (intentional for fields like layer_types).
        base = {"layer_types": ["full_attention", "full_attention", "full_attention"]}
        _deep_merge(base, {"layer_types": ["sliding_window"]})
        self.assertEqual(base, {"layer_types": ["sliding_window"]})


# ---------------------------------------------------------------------------
# VisionConfig — capacity validation in __post_init__
# ---------------------------------------------------------------------------


class VisionConfigCapacityTests(TestCase):
    def test_default_passes(self):
        # 280 * 3**2 = 2520 patches; position_embedding_size = 10240. OK.
        VisionConfig()

    def test_max_supported_passes(self):
        # 1120 * 3**2 = 10080 <= 10240. The maximum Gemma 4 documented budget.
        VisionConfig(default_output_length=1120)

    def test_overflow_raises(self):
        # 2000 * 3**2 = 18000 > 10240.
        with self.assertRaises(ValueError) as cm:
            VisionConfig(default_output_length=2000)
        msg = str(cm.exception)
        self.assertIn("position_embedding_size", msg)
        self.assertIn("default_output_length", msg)


# ---------------------------------------------------------------------------
# _sync_processor_to_model_config — SimpleNamespace mocks (no real model load)
# ---------------------------------------------------------------------------


def _make_model_and_processor(model_budget, processor_budget):
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(default_output_length=model_budget),
        ),
    )
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(max_soft_tokens=processor_budget),
    )
    return model, processor


class SyncProcessorToModelConfigTests(TestCase):
    def test_overwrites_processor_when_model_differs(self):
        model, processor = _make_model_and_processor(
            model_budget=1120, processor_budget=280
        )
        with self.assertLogs("mlx_vlm.utils", level=logging.WARNING) as cm:
            _sync_processor_to_model_config(model, processor)
        self.assertEqual(processor.image_processor.max_soft_tokens, 1120)
        self.assertTrue(
            any("280" in line and "1120" in line for line in cm.output),
            f"expected before/after values in warning, got: {cm.output}",
        )

    def test_no_op_when_already_synced(self):
        model, processor = _make_model_and_processor(
            model_budget=1120, processor_budget=1120
        )
        logger = logging.getLogger("mlx_vlm.utils")
        with self.assertNoLogs("mlx_vlm.utils", level=logging.WARNING):
            _sync_processor_to_model_config(model, processor)
        self.assertEqual(processor.image_processor.max_soft_tokens, 1120)

    def test_no_op_for_processor_without_max_soft_tokens(self):
        # Non-Gemma-shape processor (e.g., Qwen3-VL): image_processor has no
        # ``max_soft_tokens`` attribute. The sync helper must be a no-op.
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(default_output_length=1120),
            ),
        )
        processor = SimpleNamespace(image_processor=SimpleNamespace())
        # Should not raise; should not add the attribute.
        _sync_processor_to_model_config(model, processor)
        self.assertFalse(hasattr(processor.image_processor, "max_soft_tokens"))

    def test_no_op_for_model_without_vision_config(self):
        # Text-only model (no vision_config on its config). No-op, no exception.
        model = SimpleNamespace(config=SimpleNamespace())
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(max_soft_tokens=280)
        )
        _sync_processor_to_model_config(model, processor)
        self.assertEqual(processor.image_processor.max_soft_tokens, 280)
