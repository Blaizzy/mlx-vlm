import unittest

import mlx.core as mx

from mlx_vlm.models.deepseek_v4.config import ModelConfig
from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model
from mlx_vlm.models.deepseek_v4.vision import Aligner, ViT


class TestDeepseekV4VisionConfig(unittest.TestCase):
    def test_text_only_config_keeps_vision_disabled(self):
        config = ModelConfig()

        self.assertEqual(config.vision_n_layers, 0)

    def test_published_vision_fields_are_loaded(self):
        config = ModelConfig.from_dict(
            {
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_n_heads": 16,
                "vision_inter_dim": 2816,
                "vision_patch_size": 14,
                "vision_rope_theta": 10000,
                "vision_downsample_ratio": 3,
                "vision_max_n_token": 384,
                "vision_min_pixels": 147456,
                "vision_max_wh_ratio": 8,
            }
        )

        self.assertEqual(config.vision_n_layers, 32)
        self.assertEqual(config.vision_dim, 1024)
        self.assertEqual(config.vision_n_heads, 16)
        self.assertEqual(config.vision_inter_dim, 2816)
        self.assertEqual(config.vision_patch_size, 14)
        self.assertEqual(config.vision_rope_theta, 10000)
        self.assertEqual(config.vision_downsample_ratio, 3)
        self.assertEqual(config.vision_max_n_token, 384)
        self.assertEqual(config.vision_min_pixels, 147456)
        self.assertEqual(config.vision_max_wh_ratio, 8)


def tiny_vision_config(**kwargs):
    values = {
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "n_shared_experts": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "q_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "head_dim": 4,
        "o_groups": 2,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 2,
        "index_topk": 2,
        "hc_mult": 2,
        "vision_n_layers": 1,
        "vision_dim": 8,
        "vision_n_heads": 2,
        "vision_inter_dim": 12,
        "vision_patch_size": 2,
        "vision_downsample_ratio": 2,
    }
    values.update(kwargs)
    return ModelConfig(**values)


class TestDeepseekV4VisionTower(unittest.TestCase):
    def test_vit_and_aligner_preserve_published_shapes(self):
        config = tiny_vision_config()
        patches = mx.random.normal((12, 3, 2, 2))

        vision_output = ViT(config)(patches, n_h=3, n_w=4)
        aligned = Aligner(config)(vision_output, n_h=3, n_w=4)
        mx.eval(vision_output, aligned)

        self.assertEqual(vision_output.shape, (12, config.vision_dim))
        self.assertEqual(aligned.shape, (4, config.hidden_size))

    def test_model_replaces_sentinels_after_safe_embedding(self):
        config = tiny_vision_config()
        model = Model(config)
        model.image_start = mx.full((config.hidden_size,), 1.0)
        model.image_pad = mx.full((config.hidden_size,), 2.0)
        model.image_newline = mx.full((config.hidden_size,), 3.0)
        model.image_end = mx.full((config.hidden_size,), 4.0)
        types = mx.array([0, 2, 2, 4], dtype=mx.int32)
        input_ids = mx.array(
            [[1, *(config.vocab_size + types).tolist(), 2]], dtype=mx.int32
        )
        patches = mx.random.normal((8, 3, 2, 2))

        features = model.get_input_embeddings(
            input_ids,
            patches,
            image_grid_hw=mx.array([[2, 4]], dtype=mx.int32),
            image_sample_indices=mx.array([0], dtype=mx.int32),
            image_offsets=mx.array([1], dtype=mx.int32),
            image_types=types,
            image_type_offsets=mx.array([0, 4], dtype=mx.int32),
            image_permutations=mx.array([1, 0], dtype=mx.int32),
        ).inputs_embeds
        expected_images = model.encode_image(patches, 2, 4)[[1, 0]]
        mx.eval(features, expected_images)

        self.assertTrue(mx.allclose(features[0, 1], model.image_start).item())
        self.assertTrue(mx.allclose(features[0, 2:4], expected_images).item())
        self.assertTrue(mx.allclose(features[0, 4], model.image_end).item())
        self.assertTrue(
            mx.allclose(
                features[0, [0, 5]],
                model.language_model.model.embed_tokens(mx.array([1, 2])),
            ).item()
        )

    def test_text_only_model_does_not_construct_vision_modules(self):
        model = Model(tiny_vision_config(vision_n_layers=0))

        self.assertFalse(hasattr(model, "vision"))
        output = model.get_input_embeddings(mx.array([[1, 2]], dtype=mx.int32))
        self.assertEqual(output.inputs_embeds.shape, (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
