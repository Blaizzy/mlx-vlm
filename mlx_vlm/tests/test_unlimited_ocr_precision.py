"""Precision and reference-encoder regressions without model downloads."""

import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_vlm.models.deepseekocr.config import VisionConfig as DeepseekVisionConfig
from mlx_vlm.models.deepseekocr.sam import get_rel_pos
from mlx_vlm.models.deepseekocr.vision import MLP, VisionEmbeddings
from mlx_vlm.models.unlimited_ocr.config import TextConfig, VisionConfig
from mlx_vlm.models.unlimited_ocr.language import LanguageModel, RingSlidingKVCache
from mlx_vlm.models.unlimited_ocr.unlimitedocr import Model


class ConstantEmbedding:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, ids):
        return mx.full((*ids.shape, 4), 0.1, dtype=self.dtype)


class FakeSAM:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, images):
        return mx.full((images.shape[0], 1, 1, 2), 0.123456, dtype=self.dtype)


class FakeVision:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, images, patch_embeds=None):
        return mx.full((images.shape[0], 2, 2), 0.234567, dtype=self.dtype)


class TestUnlimitedOCRPrecision(unittest.TestCase):
    def test_fp32_image_features_are_not_rounded_to_text_dtype(self):
        model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=ConstantEmbedding(mx.bfloat16))
            ),
            sam_model=FakeSAM(mx.float32),
            vision_model=FakeVision(mx.float32),
            projector=nn.Identity(),
            image_newline=mx.zeros(4),
            view_separator=mx.zeros(4),
        )
        features = Model.get_input_embeddings(
            model,
            input_ids=mx.array([[0, 1, 1, 1, 2]]),
            pixel_values=[mx.zeros((1, 3, 4, 4)), mx.ones((1, 3, 4, 4))],
            images_spatial_crop=mx.array([[1, 1]]),
            images_seq_mask=mx.array([[False, True, True, True, False]]),
        ).inputs_embeds
        self.assertEqual(features.dtype, mx.float32)
        expected = mx.array([0.234567, 0.234567, 0.123456, 0.123456])
        np.testing.assert_array_equal(np.array(features[0, 1]), np.array(expected))
        self.assertGreater(
            mx.max(
                mx.abs(expected.astype(mx.bfloat16).astype(mx.float32) - expected)
            ).item(),
            1e-5,
        )

    def test_fp32_fusion_does_not_requantize_token_embeddings(self):
        embedding = nn.QuantizedEmbedding(8, 64, group_size=64, bits=8)
        embedding.set_dtype(mx.bfloat16)
        packed = np.array(embedding.weight)
        model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=embedding)
            ),
            sam_model=FakeSAM(mx.float32),
            vision_model=FakeVision(mx.float32),
            projector=nn.Linear(4, 64, bias=False),
            image_newline=mx.zeros(64),
            view_separator=mx.zeros(64),
        )
        result = Model.get_input_embeddings(
            model,
            input_ids=mx.array([[0, 1, 1, 1, 2]]),
            pixel_values=[mx.zeros((1, 3, 4, 4)), mx.ones((1, 3, 4, 4))],
            images_spatial_crop=mx.array([[1, 1]]),
            images_seq_mask=mx.array([[False, True, True, True, False]]),
        ).inputs_embeds
        self.assertEqual(result.dtype, mx.float32)
        self.assertEqual(embedding.weight.dtype, mx.uint32)
        self.assertEqual(embedding.scales.dtype, mx.bfloat16)
        np.testing.assert_array_equal(np.array(embedding.weight), packed)

    def test_text_only_embeddings_keep_native_dtype(self):
        model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=ConstantEmbedding(mx.bfloat16))
            )
        )
        features = Model.get_input_embeddings(
            model, input_ids=mx.array([[0, 1]]), pixel_values=None
        ).inputs_embeds
        self.assertEqual(features.dtype, mx.bfloat16)

    def test_cached_decode_preserves_fp32_prefill_activations(self):
        config = TextConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            n_routed_experts=None,
            n_shared_experts=None,
            sliding_window_size=4,
            sliding_window=4,
        )
        model = LanguageModel(config)
        model.set_dtype(mx.bfloat16)
        model.eval()
        cache = model.make_cache()
        reference_cache = model.make_cache()
        prefix = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 29.0
        for current_cache in (cache, reference_cache):
            mx.eval(model.model(None, inputs_embeds=prefix, cache=current_cache))
        token = mx.array([[4]])
        result = model.model(token, cache=cache)
        expected = model.model(
            None,
            inputs_embeds=model.model.embed_tokens(token).astype(mx.float32),
            cache=reference_cache,
        )
        self.assertEqual(result.dtype, mx.float32)
        self.assertEqual(cache[0].keys.dtype, mx.float32)
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-6)
        self.assertEqual(model.model.embed_tokens.weight.dtype, mx.bfloat16)

    def test_chunked_prefill_keeps_final_prompt_token_in_reference(self):
        config = TextConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            n_routed_experts=None,
            n_shared_experts=None,
            sliding_window_size=2,
            sliding_window=2,
        )
        model = LanguageModel(config)
        model.eval()
        prefix = mx.arange(40, dtype=mx.float32).reshape(1, 5, 8) / 43.0
        for step_size in (1, 4):
            with self.subTest(step_size=step_size):
                cache = model.make_cache()
                processed = 0
                while processed < 4:
                    count = min(step_size, 4 - processed)
                    model(
                        None,
                        inputs_embeds=prefix[:, processed : processed + count],
                        cache=cache,
                        n_to_process=count,
                    )
                    processed += count
                model(None, inputs_embeds=prefix[:, 4:], cache=cache)
                self.assertEqual(cache[0].prefill_length, 5)
                self.assertFalse(cache[0]._prefill_pending)
                self.assertEqual(cache[0].state[0].shape[2], 5)
                saved_prefix = mx.array(cache[0].state[0])
                for token in [2, 3, 4]:
                    mx.eval(model(mx.array([[token]]), cache=cache).logits)
                self.assertEqual(cache[0].state[0].shape[2], 7)
                np.testing.assert_array_equal(
                    np.array(cache[0].state[0][:, :, :5]), np.array(saved_prefix)
                )

    def test_prefill_progress_metadata_roundtrip_and_legacy_restore(self):
        cache = RingSlidingKVCache(2)
        cache.begin_prefill()
        cache.update_and_fetch(mx.ones((1, 1, 3, 4)), mx.ones((1, 1, 3, 4)))
        cache.end_prefill(finished=False)
        restored = RingSlidingKVCache(2)
        restored.state = cache.state
        restored.meta_state = cache.meta_state
        self.assertTrue(restored._prefill_pending)
        self.assertEqual(restored.prefill_length, 3)
        restored.meta_state = cache.meta_state[:4]
        self.assertFalse(restored._prefill_pending)
        self.assertEqual(restored.prefill_length, 3)

    def test_uncached_language_keeps_native_dtype(self):
        config = TextConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            n_routed_experts=None,
            n_shared_experts=None,
        )
        model = LanguageModel(config)
        model.set_dtype(mx.bfloat16)
        self.assertEqual(model.model(mx.array([[1, 2]])).dtype, mx.bfloat16)

    def test_absolute_positions_preserve_channels_and_match_bicubic_reference(self):
        config = DeepseekVisionConfig(
            model_type="vision", hidden_size=8, image_size=56, patch_size=14
        )
        embeddings = VisionEmbeddings(config)
        grid = np.random.default_rng(42).normal(size=(4, 4, 8)).astype(np.float32)
        cls = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
        positions = np.concatenate([cls, grid.reshape(1, 16, 8)], axis=1)
        result = embeddings._get_abs_pos(mx.array(positions), 10)
        expected_grid = np.stack(
            [
                np.array(
                    Image.fromarray(grid[:, :, channel]).resize(
                        (3, 3), Image.Resampling.BICUBIC
                    )
                )
                for channel in range(8)
            ],
            axis=-1,
        )
        expected = np.concatenate([cls, expected_grid.reshape(1, 9, 8)], axis=1)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-5, rtol=1e-5)

    def test_relative_positions_use_half_pixel_centers(self):
        result = get_rel_pos(2, 2, mx.arange(5, dtype=mx.float32)[:, None])
        expected = np.array([[[2], [1 / 3]], [[11 / 3], [2]]], dtype=np.float32)
        np.testing.assert_allclose(np.array(result), expected, atol=1e-6)

    def test_clip_mlp_uses_quick_gelu(self):
        config = DeepseekVisionConfig(
            model_type="vision", hidden_size=4, intermediate_size=4
        )
        x = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        actual = MLP(config).activation_fn(mx.array(x))
        expected = x / (1 + np.exp(-1.702 * x))
        np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)

    def test_unlimited_clip_norm_default_matches_reference(self):
        self.assertEqual(VisionConfig(model_type="vision").layer_norm_eps, 1e-5)
        self.assertEqual(DeepseekVisionConfig(model_type="vision").layer_norm_eps, 1e-6)


if __name__ == "__main__":
    unittest.main()
