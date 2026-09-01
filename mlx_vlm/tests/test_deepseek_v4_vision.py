import unittest

from mlx_vlm.models.deepseek_v4.config import ModelConfig


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


if __name__ == "__main__":
    unittest.main()
