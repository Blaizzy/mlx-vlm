import unittest

import mlx.core as mx

from mlx_vlm.models.locateanything.config import ModelConfig, TextConfig, VisionConfig


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

    def test_pbd_config_fields(self):
        tc = TextConfig.from_dict(
            {
                "model_type": "qwen2",
                "block_size": 6,
                "text_mask_token_id": 151676,
                "null_token_id": 152678,
                "switch_token_id": 152679,
            }
        )
        self.assertEqual(tc.block_size, 6)
        self.assertEqual(tc.text_mask_token_id, 151676)
        self.assertEqual(tc.null_token_id, 152678)
        self.assertEqual(tc.switch_token_id, 152679)
        self.assertFalse(tc.causal_attn)

        cfg = ModelConfig(text_config=tc, vision_config=tiny_vision_config())
        self.assertEqual(cfg.n_future_tokens, 6)


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

    def test_block_mask_single_image_is_noop(self):
        # Single image -> cu_seqlens == [0, S] -> mask is entirely True, so the
        # vision attention can safely drop it and use the SDPA flash path.
        from mlx_vlm.models.locateanything.vision import make_block_attention_mask

        m = make_block_attention_mask(mx.array([0, 16]), 16)
        self.assertTrue(bool(mx.all(m)))

    def test_block_mask_multi_image_block_diagonal(self):
        from mlx_vlm.models.locateanything.vision import make_block_attention_mask

        # two blocks: [0:4] and [4:6]
        m = make_block_attention_mask(mx.array([0, 4, 6]), 6)
        self.assertTrue(bool(m[0, 3]) and bool(m[4, 5]))  # intra-block attends
        self.assertFalse(bool(m[0, 4]) or bool(m[3, 5]))  # cross-block blocked

    def test_vision_multi_image_shape(self):
        # Exercises the multi-image branch that still builds the dense mask.
        from mlx_vlm.models.locateanything.vision import VisionModel

        vt = VisionModel(tiny_vision_config())
        pixels = mx.random.uniform(shape=(16 + 8, 14, 14, 3))  # 4x4 + 2x4
        out = vt(
            pixels, grid_thw=mx.array([[4, 4], [2, 4]]), grid_shapes=[(4, 4), (2, 4)]
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (4, 4, 32))  # 4x4 -> 2x2 merged
        self.assertEqual(out[1].shape, (2, 4, 32))  # 2x4 -> 1x2 merged

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
        self.assertIn("language_model.model.layers.0.self_attn.q_proj.weight", out)
        # tied lm_head is dropped
        self.assertNotIn("language_model.lm_head.weight", out)


class TestLocateAnythingPBD(unittest.TestCase):
    TOKEN_IDS = {
        "box_start_token_id": 100,
        "box_end_token_id": 101,
        "coord_start_token_id": 200,
        "coord_end_token_id": 300,
        "ref_start_token_id": 102,
        "ref_end_token_id": 103,
        "none_token_id": 4,
        "null_token_id": 400,
        "switch_token_id": 401,
        "default_mask_token_id": 90,
        "im_end_token_id": 99,
    }

    def test_magi_block_mask_prefill(self):
        from mlx_vlm.models.locateanything.language import build_magi_block_mask

        # Prefill: kv_len == q_len, prefix P=4, block B=3.
        m = build_magi_block_mask(7, 7, 3)
        self.assertEqual(m.shape, (1, 1, 7, 7))
        allowed = (m[0, 0] == 0).tolist()
        # Prefix rows 0..3 are strictly causal.
        for i in range(4):
            for j in range(7):
                self.assertEqual(allowed[i][j], j <= i, f"prefix ({i},{j})")
        # Block rows 4..6 attend to prefix [0,2] (cols 0,1,2), NOT bridge col 3,
        # and bidirectionally within the block (cols 4,5,6).
        blocked_k = 7 - 3 - 1  # = 3
        for i in range(4, 7):
            for j in range(7):
                expect = (j < blocked_k) or (j >= 7 - 3)
                self.assertEqual(allowed[i][j], expect, f"block ({i},{j})")

    def test_magi_block_mask_decode(self):
        from mlx_vlm.models.locateanything.language import build_magi_block_mask

        # Decode: kv_len=10, q_len=6 (r=3 recompute rows + block B=3), prefix cached=4.
        m = build_magi_block_mask(10, 6, 3)
        allowed = (m[0, 0] == 0).tolist()
        qgs = 10 - 6  # global start = 4
        # Recompute rows 0..2 are causal in global coordinates.
        for i in range(3):
            for j in range(10):
                self.assertEqual(allowed[i][j], j <= qgs + i, f"recompute ({i},{j})")
        # Block rows attend prefix [0, blocked_k) and the window [window_start, kv).
        blocked_k = 10 - 3 - 1  # = 6
        window_start = 10 - 3  # = 7
        for i in range(3, 6):
            for j in range(10):
                expect = (j < blocked_k) or (j >= window_start)
                self.assertEqual(allowed[i][j], expect, f"block ({i},{j})")

    def _block_probs(self, rows):
        """Build a [6, vocab] one-hot-ish prob array from per-row {id: p} dicts."""
        vocab = 512
        probs = mx.zeros((6, vocab))
        for i, row in enumerate(rows):
            for tid, p in row.items():
                probs[i, tid] = p
        return probs

    def test_decode_bbox_avg_coord_box(self):
        from mlx_vlm.models.locateanything.pbd import decode_bbox_avg

        t = self.TOKEN_IDS
        rows = [
            {t["box_start_token_id"]: 1.0},
            {210: 1.0},
            {220: 1.0},
            {230: 1.0},
            {240: 1.0},
            {t["box_end_token_id"]: 1.0},
        ]
        box = decode_bbox_avg(self._block_probs(rows), t, generation_mode="fast")
        self.assertEqual(
            box,
            [t["box_start_token_id"], 210, 220, 230, 240, t["box_end_token_id"]],
        )

    def test_decode_bbox_avg_rejects_terminal_block(self):
        from mlx_vlm.models.locateanything.pbd import decode_bbox_avg

        t = self.TOKEN_IDS
        # Bridge says im_end; positions 1-4 are null, position 5 null (would pass
        # end_thresh). The position-0 gate must reject this terminal block.
        rows = [
            {t["im_end_token_id"]: 0.99, t["box_start_token_id"]: 0.001},
            {t["null_token_id"]: 1.0},
            {t["null_token_id"]: 1.0},
            {t["null_token_id"]: 1.0},
            {t["null_token_id"]: 1.0},
            {t["null_token_id"]: 1.0},
        ]
        self.assertIsNone(
            decode_bbox_avg(self._block_probs(rows), t, generation_mode="hybrid")
        )

    def test_handle_pattern_coord_box(self):
        from mlx_vlm.models.locateanything.pbd import handle_pattern

        t = self.TOKEN_IDS
        x0 = [t["box_start_token_id"], 210, 220, 230, 240, t["box_end_token_id"]]
        out = handle_pattern(x0, t, "hybrid")
        self.assertEqual(out["type"], "coord_box")
        self.assertEqual(out["tokens"], x0)
        self.assertFalse(out["need_switch_to_ar"])

    def test_handle_pattern_terminal(self):
        from mlx_vlm.models.locateanything.pbd import handle_pattern

        t = self.TOKEN_IDS
        out = handle_pattern(
            [t["im_end_token_id"]] + [t["null_token_id"]] * 5, t, "hybrid"
        )
        self.assertEqual(out["type"], "im_end")
        self.assertTrue(out["is_terminal"])

    def test_handle_pattern_error_box_switches_to_ar(self):
        from mlx_vlm.models.locateanything.pbd import handle_pattern

        t = self.TOKEN_IDS
        # box_start, one coord, then a non-coord -> malformed box in hybrid mode.
        x0 = [
            t["box_start_token_id"],
            210,
            t["null_token_id"],
            0,
            240,
            t["box_end_token_id"],
        ]
        out = handle_pattern(x0, t, "hybrid")
        self.assertEqual(out["type"], "error_box")
        self.assertTrue(out["need_switch_to_ar"])
        self.assertEqual(out["tokens"], [t["box_start_token_id"], 210])

    def test_handle_pattern_error_box_stays_in_fast(self):
        from mlx_vlm.models.locateanything.pbd import handle_pattern

        t = self.TOKEN_IDS
        x0 = [
            t["box_start_token_id"],
            210,
            t["null_token_id"],
            0,
            240,
            t["box_end_token_id"],
        ]
        out = handle_pattern(x0, t, "fast")
        self.assertEqual(out["type"], "coord_box")
        self.assertFalse(out["need_switch_to_ar"])

    def test_pbd_slow_respects_max_tokens(self):
        from mlx_vlm.models.locateanything.locateanything import Model

        cfg = ModelConfig(
            text_config=tiny_text_config(),
            vision_config=tiny_vision_config(),
            image_token_index=5,
            vocab_size=128,
        )
        model = Model(cfg)
        input_ids = mx.array([[5, 5, 5, 5, 1, 2, 3]])
        kw = dict(
            pixel_values=mx.random.uniform(shape=(16, 3, 14, 14)),
            image_grid_hws=mx.array([[4, 4]]),
            _grid_shapes=[(4, 4)],
        )
        for mt in (0, 1, 4):
            out = model.pbd_generate(
                input_ids, generation_mode="slow", max_tokens=mt, **kw
            )
            self.assertLessEqual(len(out), mt)


class TestLocateAnythingImageProcessor(unittest.TestCase):
    def test_accepts_mx_array_and_pil(self):
        from PIL import Image

        from mlx_vlm.models.locateanything.image_processing_locateanything import (
            LocateAnythingImageProcessor,
        )

        proc = LocateAnythingImageProcessor()
        # 56x56 -> grid 4x4 (divisible by merge*patch = 28)
        arr = proc(mx.zeros((56, 56, 3)))
        self.assertEqual(arr["pixel_values"].shape[1:], (3, 14, 14))
        self.assertEqual(arr["image_grid_hws"].tolist(), [[4, 4]])
        # PIL path still works (regression)
        pil = proc(Image.new("RGB", (56, 56)))
        self.assertEqual(pil["image_grid_hws"].tolist(), [[4, 4]])

    def test_rejects_unknown_type(self):
        from mlx_vlm.models.locateanything.image_processing_locateanything import (
            LocateAnythingImageProcessor,
        )

        with self.assertRaises(ValueError):
            LocateAnythingImageProcessor()("not-an-image")


if __name__ == "__main__":
    unittest.main()
