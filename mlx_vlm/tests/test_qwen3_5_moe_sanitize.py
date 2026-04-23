"""Regression test for Qwen3.6-35B-A3B nested visual layout.

Qwen/Qwen3.6-35B-A3B and its derivatives ship the ViT nested under
`model.language_model.visual.*` in the HF checkpoint rather than the
flat `model.visual.*` layout used by other Qwen VLMs. Before the fix,
the `elif` ordering in `Model.sanitize` mis-routed those keys through
the generic `model.language_model -> language_model.model` rewrite, so
visual weights ended up at `language_model.model.visual.*` and were
orphaned on load (333 tensors silently dropped).
"""

import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model


def _fake(shape=(2, 2)):
    """Tiny mx.array stand-in for a weight value."""
    return mx.zeros(shape)


def _mock_model():
    # Model.sanitize only reads self.config.text_config for
    # tie_word_embeddings and num_hidden_layers; construct the minimum.
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=0,  # skip the MoE expert-split loop
        )
    )
    return SimpleNamespace(config=cfg)


class TestQwen35MoESanitize(unittest.TestCase):
    def test_nested_visual_routes_to_vision_tower(self):
        """Keys like model.language_model.visual.* must land at vision_tower.*"""
        weights = {
            "model.language_model.visual.blocks.0.attn.qkv.weight": _fake(),
            "model.language_model.visual.patch_embed.proj.weight": _fake(),
            "model.language_model.visual.pos_embed.weight": _fake(),
            "model.language_model.visual.merger.linear_fc1.weight": _fake(),
        }
        out = Model.sanitize(_mock_model(), dict(weights))
        self.assertIn("vision_tower.blocks.0.attn.qkv.weight", out)
        self.assertIn("vision_tower.patch_embed.proj.weight", out)
        self.assertIn("vision_tower.pos_embed.weight", out)
        self.assertIn("vision_tower.merger.linear_fc1.weight", out)
        self.assertFalse(
            any(k.startswith("language_model.model.visual") for k in out),
            "nested visual keys must not remain after sanitize",
        )

    def test_language_keys_unchanged(self):
        """Language-only `model.language_model.*` keys must still route to
        `language_model.model.*` (regression guard for the generic rule)."""
        weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": _fake(),
            "model.language_model.embed_tokens.weight": _fake(),
            # 1-D norm weight to exercise the norm_keys +1.0 branch
            "model.language_model.norm.weight": mx.zeros((8,)),
        }
        out = Model.sanitize(_mock_model(), dict(weights))
        self.assertIn("language_model.model.layers.0.self_attn.q_proj.weight", out)
        self.assertIn("language_model.model.embed_tokens.weight", out)
        self.assertIn("language_model.model.norm.weight", out)

    def test_flat_visual_still_routes_to_vision_tower(self):
        """Pre-existing `model.visual.*` flat layout must keep working."""
        weights = {
            "model.visual.blocks.1.attn.qkv.weight": _fake(),
            "model.visual.patch_embed.proj.weight": _fake(),
        }
        out = Model.sanitize(_mock_model(), dict(weights))
        self.assertIn("vision_tower.blocks.1.attn.qkv.weight", out)
        self.assertIn("vision_tower.patch_embed.proj.weight", out)

    def test_lm_head_unchanged(self):
        """lm_head rewrite path must still be reached when no `model` prefix."""
        weights = {"lm_head.weight": _fake()}
        out = Model.sanitize(_mock_model(), dict(weights))
        self.assertIn("language_model.lm_head.weight", out)


if __name__ == "__main__":
    unittest.main()
