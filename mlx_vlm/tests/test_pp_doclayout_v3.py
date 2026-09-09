import unittest


class TestPPDocLayoutV3(unittest.TestCase):
    def test_pp_doclayout_v3_rename_rules(self):
        from mlx_vlm.models.pp_doclayout_v3.pp_doclayout_v3 import (
            rename_key,
            should_drop,
        )

        self.assertEqual(
            rename_key("model.backbone.model.embedder.stem1.convolution.weight"),
            "backbone.embedder.stem1.conv.weight",
        )
        self.assertEqual(
            rename_key("model.encoder.encoder.0.layers.0.fc1.weight"),
            "encoder.aifi.0.layers.0.fc1.weight",
        )
        self.assertEqual(
            rename_key("model.encoder.lateral_convs.0.norm.weight"),
            "encoder.lateral_convs.0.bn.weight",
        )
        self.assertEqual(
            rename_key("model.encoder_input_proj.0.1.running_mean"),
            "encoder_input_proj.0.bn.running_mean",
        )
        self.assertEqual(
            rename_key("model.enc_output.0.weight"), "enc_output.fc.weight"
        )
        self.assertEqual(
            rename_key("model.decoder_order_head.3.bias"), "decoder_order_head.3.bias"
        )
        # Converted keys are fixpoints (idempotent sanitize).
        self.assertEqual(
            rename_key("backbone.embedder.stem1.conv.weight"),
            "backbone.embedder.stem1.conv.weight",
        )
        self.assertTrue(
            should_drop(
                "model.backbone.model.embedder.stem1.normalization.num_batches_tracked"
            )
        )
        self.assertTrue(should_drop("model.denoising_class_embed.weight"))
        self.assertFalse(should_drop("model.enc_score_head.weight"))

    def test_pp_doclayout_v3_sanitize_idempotent(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.pp_doclayout_v3 import LayoutModel
        from mlx_vlm.tests.sanitize_invariants import assert_sanitize_idempotent

        hf_weights = {
            "model.backbone.model.embedder.stem1.convolution.weight": mx.random.normal(
                (32, 3, 3, 3)
            ),
            "model.backbone.model.embedder.stem1.normalization.weight": mx.ones((32,)),
            "model.backbone.model.embedder.stem1.normalization.bias": mx.zeros((32,)),
            "model.backbone.model.embedder.stem1.normalization.running_mean": mx.zeros(
                (32,)
            ),
            "model.backbone.model.embedder.stem1.normalization.running_var": mx.ones(
                (32,)
            ),
            "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": mx.array(
                7
            ),
            "model.enc_score_head.weight": mx.random.normal((37, 256)),
            "model.enc_score_head.bias": mx.zeros((37,)),
            "model.denoising_class_embed.weight": mx.random.normal((37, 256)),
        }
        sanitized = assert_sanitize_idempotent(LayoutModel, hf_weights)
        self.assertIn("backbone.embedder.stem1.conv.weight", sanitized)
        self.assertEqual(
            sanitized["backbone.embedder.stem1.conv.weight"].shape, (32, 3, 3, 3)
        )
        self.assertNotIn(
            "backbone.embedder.stem1.normalization.num_batches_tracked",
            sanitized,
        )
        self.assertNotIn("denoising_class_embed.weight", sanitized)

    def test_pp_doclayout_v3_config_routes(self):
        from mlx_vlm.models import pp_doclayout_v3
        from mlx_vlm.utils import get_model_and_args

        model_class, model_type = get_model_and_args(
            config={"model_type": "pp_doclayout_v3"}
        )
        self.assertIs(model_class, pp_doclayout_v3)
        self.assertEqual(model_type, "pp_doclayout_v3")
        self.assertEqual(pp_doclayout_v3.ModelConfig().num_labels, 25)

        cfg = pp_doclayout_v3.ModelConfig.from_dict(
            {
                "model_type": "pp_doclayout_v3",
                "num_labels": 37,
                "id2label": {"0": "Question", "1": "Paragraph"},
                "backbone_config": {"model_type": "hgnet_v2"},
            }
        )
        self.assertEqual(cfg.id2label, {0: "Question", 1: "Paragraph"})
        self.assertEqual(cfg.num_queries, 300)
        self.assertEqual(cfg.decoder_layers, 6)

    def test_pp_doclayout_v3_decode_order(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import decode_order

        # Chain 0 -> 1 -> 2 with strong pairwise scores.
        scores = mx.array(
            [
                [-1e4, 5.0, 5.0],
                [-1e4, -1e4, 5.0],
                [-1e4, -1e4, -1e4],
            ]
        )
        self.assertEqual(decode_order(scores).tolist(), [0, 1, 2])
        # Reference formula cross-check on random input.
        rng_scores = mx.random.normal((7, 7))
        mx.eval(rng_scores)
        got = decode_order(rng_scores).tolist()
        import numpy as np

        arr = np.array(rng_scores.tolist())
        s = 1.0 / (1.0 + np.exp(-arr))
        votes = np.triu(s, 1).sum(0) + np.tril(1.0 - s.T, -1).sum(0)
        self.assertEqual(got, np.argsort(votes).tolist())

    def test_pp_doclayout_v3_mask_to_box(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import mask_to_box_coordinate

        mask = mx.zeros((1, 2, 8, 10))
        mask[0, 0, 2:5, 3:7] = 1.0
        mx.eval(mask)
        boxes = mask_to_box_coordinate(mask)
        mx.eval(boxes)
        # x in [3,7), y in [2,5) over W=10,H=8 -> cxcywh
        self.assertAlmostEqual(float(boxes[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 1]), 0.4375, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 2]), 0.4, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 3]), 0.375, places=5)
        # Empty mask -> zeros.
        self.assertEqual(float(boxes[0, 1].sum()), 0.0)

    def test_pp_doclayout_v3_bilinear_upsample(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.encoder import upsample_bilinear2x

        x = mx.array([[[[0.0], [1.0]], [[2.0], [3.0]]]])
        mx.eval(x)
        y = upsample_bilinear2x(x)
        mx.eval(y)
        # align_corners=False exact values: edges replicate, interior lerps.
        self.assertEqual(tuple(y.shape), (1, 4, 4, 1))
        self.assertAlmostEqual(float(y[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(y[0, 0, 1, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(y[0, 1, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(y[0, 1, 1, 0]), 0.75, places=5)
        self.assertAlmostEqual(float(y[0, 3, 3, 0]), 3.0, places=5)

    def test_pp_doclayout_num_labels_inferred(self):
        from mlx_vlm.models import pp_doclayout_v3

        # Stock configs omit num_labels; it follows the label map.
        cfg = pp_doclayout_v3.ModelConfig.from_dict(
            {
                "model_type": "pp_doclayout_v3",
                "id2label": {"0": "text", "1": "formula", "2": "table"},
                "backbone_config": {"model_type": "hgnet_v2"},
            }
        )
        self.assertEqual(cfg.num_labels, 3)
        # Explicit num_labels still wins when it agrees.
        cfg = pp_doclayout_v3.ModelConfig.from_dict(
            {
                "model_type": "pp_doclayout_v3",
                "num_labels": 37,
                "id2label": {str(i): f"L{i}" for i in range(37)},
                "backbone_config": {"model_type": "hgnet_v2"},
            }
        )
        self.assertEqual(cfg.num_labels, 37)

    def test_pp_doclayout_conversion_without_torch(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.convert import convert

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "source"
                src.mkdir()
                config = {"model_type": "pp_doclayout_v3"}
                (src / "config.json").write_text(json.dumps(config))
                weight = mx.arange(48).reshape(2, 3, 2, 4).astype(dtype)
                mx.save_safetensors(
                    str(src / "model.safetensors"),
                    {
                        "model.backbone.model.embedder.stem1.convolution.weight": weight,
                        "model.denoising_class_embed.weight": mx.ones((2, 2)),
                        "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": mx.array(
                            0
                        ),
                    },
                )
                with (
                    patch.dict(
                        "sys.modules", {"torch": None, "safetensors.torch": None}
                    ),
                    patch("mlx_vlm.models.pp_doclayout_v3.convert._verify") as verify,
                ):
                    out = convert(str(src), str(Path(tmp) / "converted"))
                    again = convert(
                        str(out), str(Path(tmp) / "reconverted"), "bfloat16"
                    )
                self.assertEqual(verify.call_count, 2)
                key = "backbone.embedder.stem1.conv.weight"
                expected = weight.transpose(0, 2, 3, 1)
                converted = mx.load(str(out / "model.safetensors"))
                reconverted = mx.load(str(again / "model.safetensors"))
                self.assertEqual(set(converted), {key})
                self.assertEqual(set(reconverted), {key})
                self.assertEqual(converted[key].dtype, mx.float32)
                self.assertEqual(reconverted[key].dtype, mx.bfloat16)
                self.assertEqual(converted[key].tolist(), expected.tolist())
                self.assertEqual(reconverted[key].tolist(), expected.tolist())
                self.assertEqual(json.loads((out / "config.json").read_text()), config)


if __name__ == "__main__":
    unittest.main()
