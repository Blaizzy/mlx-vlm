import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm import utils as utils_module
from mlx_vlm.utils import load_model


class TestGemma4Regression(unittest.TestCase):
    def _create_minimal_gemma4_checkpoint(self, tmp_path: Path):
        text_config = {
            "model_type": "gemma4_text",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 32,
            "sliding_window_pattern": 2,
            "layer_types": ["full_attention"] * 4,
            "use_double_wide_mlp": False,  # prevent mlp size mismatch
        }
        config = {
            "model_type": "gemma4",
            "text_config": text_config,
            "vision_config": {
                "model_type": "gemma4_vision",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "default_output_length": 280,
                "position_embedding_size": 10240,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 100.0},
            },
            "audio_config": None,
            "image_token_id": 31,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)

        def layer_weights(i) -> Dict[str, mx.array]:
            return {
                f"language_model.model.layers.{i}.self_attn.q_proj.weight": mx.zeros(
                    (16, 16)
                ),
                f"language_model.model.layers.{i}.self_attn.k_proj.weight": mx.zeros(
                    (8, 16)
                ),
                f"language_model.model.layers.{i}.self_attn.v_proj.weight": mx.zeros(
                    (8, 16)
                ),
                f"language_model.model.layers.{i}.self_attn.o_proj.weight": mx.zeros(
                    (16, 16)
                ),
                f"language_model.model.layers.{i}.self_attn.q_norm.weight": mx.zeros((8,)),
                f"language_model.model.layers.{i}.self_attn.k_norm.weight": mx.zeros((8,)),
                f"language_model.model.layers.{i}.input_layernorm.weight": mx.zeros(
                    (16,)
                ),
                f"language_model.model.layers.{i}.post_attention_layernorm.weight": mx.zeros(
                    (16,)
                ),
                f"language_model.model.layers.{i}.pre_feedforward_layernorm.weight": mx.zeros(
                    (16,)
                ),
                f"language_model.model.layers.{i}.post_feedforward_layernorm.weight": mx.zeros(
                    (16,)
                ),
                f"language_model.model.layers.{i}.mlp.gate_proj.weight": mx.zeros(
                    (32, 16)
                ),
                f"language_model.model.layers.{i}.mlp.up_proj.weight": mx.zeros((32, 16)),
                f"language_model.model.layers.{i}.mlp.down_proj.weight": mx.zeros(
                    (16, 32)
                ),
                f"language_model.model.layers.{i}.layer_scalar": mx.ones((1,)),
            }

        weights = {
            "language_model.model.embed_tokens.weight": mx.zeros((32, 16)),
            "language_model.model.norm.weight": mx.zeros((16,)),
        }
        for i in range(4):
            weights.update(layer_weights(i))

        def vision_block_weights(i) -> Dict[str, mx.array]:
            return {
                f"vision_tower.encoder.layers.{i}.self_attn.q_proj.linear.weight": mx.zeros(
                    (16, 16)
                ),
                f"vision_tower.encoder.layers.{i}.self_attn.k_proj.linear.weight": mx.zeros(
                    (16, 16)
                ),
                f"vision_tower.encoder.layers.{i}.self_attn.v_proj.linear.weight": mx.zeros(
                    (16, 16)
                ),
                f"vision_tower.encoder.layers.{i}.self_attn.o_proj.linear.weight": mx.zeros(
                    (16, 16)
                ),
                f"vision_tower.encoder.layers.{i}.self_attn.q_norm.weight": mx.zeros((8,)),
                f"vision_tower.encoder.layers.{i}.self_attn.k_norm.weight": mx.zeros((8,)),
                f"vision_tower.encoder.layers.{i}.mlp.gate_proj.linear.weight": mx.zeros(
                    (32, 16)
                ),
                f"vision_tower.encoder.layers.{i}.mlp.up_proj.linear.weight": mx.zeros(
                    (32, 16)
                ),
                f"vision_tower.encoder.layers.{i}.mlp.down_proj.linear.weight": mx.zeros(
                    (16, 32)
                ),
                f"vision_tower.encoder.layers.{i}.input_layernorm.weight": mx.zeros((16,)),
                f"vision_tower.encoder.layers.{i}.post_attention_layernorm.weight": mx.zeros(
                    (16,)
                ),
                f"vision_tower.encoder.layers.{i}.pre_feedforward_layernorm.weight": mx.zeros(
                    (16,)
                ),
                f"vision_tower.encoder.layers.{i}.post_feedforward_layernorm.weight": mx.zeros(
                    (16,)
                ),
            }

        weights.update(
            {
                "vision_tower.patch_embedder.input_proj.weight": mx.zeros(
                    (16, 3 * 16**2)
                ),
                "vision_tower.patch_embedder.position_embedding_table": mx.zeros(
                    (2, 10240, 16)
                ),
                "embed_vision.embedding_projection.weight": mx.zeros((16, 16)),
            }
        )
        for i in range(2):
            weights.update(vision_block_weights(i))

        return weights

    def _load_model_and_capture_weight_keys(self, tmp_path: Path):
        loaded_weight_keys = []
        original_load_weights = nn.Module.load_weights

        def capture_load_weights(model, file_or_weights, *args, **kwargs):
            loaded_weight_keys.append({key for key, _ in file_or_weights})
            return original_load_weights(model, file_or_weights, *args, **kwargs)

        with patch.object(nn.Module, "load_weights", capture_load_weights):
            model = load_model(tmp_path)

        self.assertEqual(
            1,
            len(loaded_weight_keys),
            "Expected load_model to call load_weights exactly once",
        )
        return model, loaded_weight_keys[0]

    def test_load_mlx_format_filters_legacy_weights(self):
        """
        Verify that MLX-format checkpoints with unused shared-KV weights load
        successfully after filtering against the final Gemma4 parameter tree.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            weights = self._create_minimal_gemma4_checkpoint(tmp_path)

            # Save in MLX format
            mx.save_safetensors(
                str(tmp_path / "model.safetensors"),
                weights,
                metadata={"format": "mlx"},
            )

            model, loaded_weight_keys = self._load_model_and_capture_weight_keys(tmp_path)
            self.assertIsNotNone(model)

            # Verify filtering:
            # In Gemma4 with num_kv_shared_layers=2 and num_hidden_layers=4,
            # layers 2 and 3 share KV projections and thus should have them filtered out.
            self.assertIn(
                "language_model.model.layers.0.self_attn.k_proj.weight",
                loaded_weight_keys,
                "Non-shared KV weight should be present",
            )
            self.assertIn(
                "language_model.model.layers.2.layer_scalar",
                loaded_weight_keys,
                "Valid layer_scalar weight should be preserved",
            )
            self.assertNotIn(
                "language_model.model.layers.2.self_attn.k_proj.weight",
                loaded_weight_keys,
                "Unused shared KV weight should have been filtered from MLX-format checkpoint",
            )

    def test_load_non_mlx_format_sanitizes_legacy_weights(self):
        """
        Verify that models in non-MLX format (e.g. Hugging Face) are correctly
        sanitized using the module-specific sanitize methods.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            weights = self._create_minimal_gemma4_checkpoint(tmp_path)

            # Save in non-MLX format (no metadata)
            mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

            with patch(
                "mlx_vlm.utils.sanitize_weights",
                wraps=utils_module.sanitize_weights,
            ) as sanitize_spy:
                model, loaded_weight_keys = self._load_model_and_capture_weight_keys(
                    tmp_path
                )
            self.assertIsNotNone(model)
            self.assertGreater(
                sanitize_spy.call_count,
                0,
                "Expected non-MLX load path to invoke sanitize_weights",
            )

            # Non-MLX checkpoints rely on Gemma4.LanguageModel.sanitize() to
            # remove per-layer KV weights for shared layers.
            self.assertIn(
                "language_model.model.layers.0.self_attn.k_proj.weight",
                loaded_weight_keys,
                "Non-shared KV weight should be present",
            )
            self.assertIn(
                "language_model.model.layers.2.layer_scalar",
                loaded_weight_keys,
                "Valid layer_scalar weight should be preserved",
            )
            self.assertNotIn(
                "language_model.model.layers.2.self_attn.k_proj.weight",
                loaded_weight_keys,
                "Unused shared KV weight should have been filtered from non-MLX checkpoint",
            )


if __name__ == "__main__":
    unittest.main()
