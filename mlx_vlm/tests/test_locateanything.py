import unittest

import mlx.core as mx

from mlx_vlm.models.locateanything.config import (
    ModelConfig,
    TextConfig,
    VisionConfig,
)


def tiny_text_config(**overrides):
    params = dict(
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    params.update(overrides)
    return TextConfig(**params)


def tiny_vision_config(**overrides):
    params = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        patch_size=14,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        num_channels=3,
        merge_kernel_size=[2, 2],
    )
    params.update(overrides)
    return VisionConfig(**params)


class TestLocateAnythingConfig(unittest.TestCase):
    def test_vision_config_aliases(self):
        cfg = VisionConfig.from_dict(
            {
                "model_type": "moonvit",
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 14,
                "init_pos_emb_height": 64,
                "init_pos_emb_width": 64,
                "merge_kernel_size": [2, 2],
                "torch_dtype": "bfloat16",  # extra key must be ignored
            }
        )
        self.assertEqual(cfg.embed_dim, 1152)
        self.assertEqual(cfg.depth, 27)
        self.assertEqual(cfg.num_heads, 16)
        self.assertEqual(cfg.spatial_merge_size, 2)

    def test_text_config_defaults(self):
        cfg = TextConfig.from_dict(
            {
                "model_type": "qwen2",
                "hidden_size": 2048,
                "num_hidden_layers": 36,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "intermediate_size": 11008,
                "vocab_size": 152681,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": True,
            }
        )
        self.assertEqual(cfg.num_key_value_heads, 2)
        self.assertEqual(cfg.vocab_size, 152681)
        self.assertTrue(cfg.tie_word_embeddings)

    def test_model_config_token_ids(self):
        cfg = ModelConfig.from_dict(
            {
                "model_type": "locateanything",
                "image_token_index": 151665,
                "box_start_token_id": 151668,
                "box_end_token_id": 151669,
                "ref_start_token_id": 151672,
                "mlp_connector_layers": 2,
                "text_config": {},
                "vision_config": {},
            }
        )
        self.assertEqual(cfg.image_token_index, 151665)
        self.assertEqual(cfg.box_start_token_id, 151668)
        self.assertEqual(cfg.mlp_connector_layers, 2)


class TestLocateAnythingModel(unittest.TestCase):
    def _build_model(self):
        from mlx_vlm.models.locateanything.locateanything import Model

        cfg = ModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            image_token_index=5,
            vocab_size=128,
        )
        return Model(cfg), cfg

    def test_language_only_forward(self):
        from mlx_vlm.models.locateanything.language import LanguageModel

        lm = LanguageModel(tiny_text_config())
        self.assertEqual(len(lm.layers), 2)
        out = lm(mx.array([[0, 1, 2, 3]]))
        self.assertEqual(out.logits.shape, (1, 4, 128))

    def test_vision_forward_shape(self):
        from mlx_vlm.models.locateanything.vision import VisionModel

        vt = VisionModel(tiny_vision_config())
        # 4x4 grid -> 16 patches, NHWC patches [num_patches, p, p, C]
        pixels = mx.random.uniform(shape=(16, 14, 14, 3))
        out = vt(pixels, grid_thw=mx.array([[4, 4]]), grid_shapes=[(4, 4)])
        self.assertIsInstance(out, list)
        # 2x2 merge over 4x4 grid -> 4 merged tokens, each k*k*embed = 4*32
        self.assertEqual(out[0].shape, (4, 4, 32))

    def test_full_forward(self):
        model, _ = self._build_model()
        pixels = mx.random.uniform(shape=(16, 3, 14, 14))
        input_ids = mx.array([[5, 5, 5, 5, 1, 2, 3]])
        out = model(
            input_ids,
            pixel_values=pixels,
            image_grid_hws=mx.array([[4, 4]]),
            _grid_shapes=[(4, 4)],
        )
        self.assertEqual(out.logits.shape, (1, 7, 128))

    def test_sanitize_key_mapping(self):
        model, _ = self._build_model()
        raw = {
            "vision_model.patch_embed.proj.weight": mx.zeros((1,)),
            "vision_model.patch_embed.pos_emb.weight": mx.zeros((1,)),
            "vision_model.encoder.blocks.0.wqkv.weight": mx.zeros((1,)),
            "vision_model.encoder.blocks.0.wo.bias": mx.zeros((1,)),
            "vision_model.encoder.final_layernorm.weight": mx.zeros((1,)),
            "mlp1.0.weight": mx.zeros((1,)),
            "mlp1.1.bias": mx.zeros((1,)),
            "mlp1.3.weight": mx.zeros((1,)),
            "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "language_model.lm_head.weight": mx.zeros((1,)),
        }
        out = model.sanitize(raw)
        self.assertIn("vision_tower.patch_embed.proj.weight", out)
        self.assertIn("vision_tower.blocks.0.wqkv.weight", out)
        self.assertIn("vision_tower.blocks.0.wo.bias", out)
        self.assertIn("vision_tower.final_layernorm.weight", out)
        self.assertIn("multi_modal_projector.layer_norm.weight", out)
        self.assertIn("multi_modal_projector.linear_1.bias", out)
        self.assertIn("multi_modal_projector.linear_2.weight", out)
        self.assertIn(
            "language_model.model.layers.0.self_attn.q_proj.weight", out
        )
        # tied lm_head is dropped
        self.assertNotIn("language_model.lm_head.weight", out)


if __name__ == "__main__":
    unittest.main()
