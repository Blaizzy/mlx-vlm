import unittest


class TestIndicOCR(unittest.TestCase):
    @staticmethod
    def ocr_config():
        return {
            "model_type": "indic_ocr",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": 262155,
            "video_token_id": 262156,
            "vision_start_token_id": 262153,
            "vision_end_token_id": 262154,
            "tie_word_embeddings": True,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "intermediate_size": 32,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 3,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 64,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
            },
            "vision_config": {
                "model_type": "qwen3_5_vision",
                "patch_size": 2,
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 16,
                "num_heads": 2,
                "num_position_embeddings": 16,
            },
        }

    def test_indic_ocr_config_routes_and_carries_indic_tokens(self):
        from mlx_vlm.models import indic_ocr
        from mlx_vlm.utils import get_model_and_args

        raw = self.ocr_config()

        model_class, _ = get_model_and_args(config=dict(raw))
        self.assertIs(model_class, indic_ocr)

        config = indic_ocr.ModelConfig.from_dict(dict(raw))
        self.assertEqual(config.model_type, "indic_ocr")
        self.assertEqual(config.image_token_id, 262155)
        self.assertEqual(config.video_token_id, 262156)
        self.assertEqual(config.vision_start_token_id, 262153)
        self.assertEqual(config.vision_end_token_id, 262154)
        self.assertEqual(config.image_token_index, 262155)

    def test_ocr_stage_uses_existing_image_prompt_format(self):
        from mlx_vlm.models.indic_ocr import Model, ModelConfig
        from mlx_vlm.prompt_utils import apply_chat_template

        config = ModelConfig.from_dict(self.ocr_config())
        model = Model(config)
        self.assertEqual(config.model_type, "indic_ocr")
        self.assertEqual(model.config.model_type, "qwen3_5")
        messages = apply_chat_template(
            object(), model.config, "Read this page", num_images=1, return_messages=True
        )
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "image"})
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], "Read this page")

    def test_indic_ocr_wrapper_repo_raises_helpful_error(self):
        from mlx_vlm.models import indic_ocr

        with self.assertRaises(ValueError) as ctx:
            indic_ocr.ModelConfig.from_dict(
                {
                    "model_type": "indic_ocr",
                    "pipeline": "two-stage",
                    "stages": {"layout": {}, "ocr": {}},
                }
            )
        self.assertIn("weights/ocr", str(ctx.exception))

    def test_combined_checkpoint_loads_and_quantizes_submodules(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from mlx_vlm.models.indic_ocr import (
            IndicOCRParser,
            LayoutOptions,
            Model,
            ModelConfig,
        )
        from mlx_vlm.models.pp_doclayout_v3 import Model as LayoutModel
        from mlx_vlm.utils import load_model

        class SmallLayout(nn.Module):
            sanitize = staticmethod(LayoutModel.sanitize)

            def __init__(self, config):
                super().__init__()
                self.enc_score_head = nn.Linear(32, config.num_labels)

        config = {
            "model_type": "indic_ocr",
            "layout_config": {"model_type": "pp_doclayout_v3", "num_labels": 2},
            "ocr_config": self.ocr_config(),
        }
        with patch("mlx_vlm.models.pp_doclayout_v3.Model", SmallLayout):
            for quantized in (False, True):
                with (
                    self.subTest(quantized=quantized),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp) / "source"
                    root.mkdir()
                    model = Model(ModelConfig.from_dict(config))
                    self.assertIsInstance(model, IndicOCRParser)
                    self.assertIsInstance(model, nn.Module)
                    saved_config = dict(config)
                    if quantized:
                        nn.quantize(
                            model,
                            group_size=32,
                            bits=4,
                            class_predicate=lambda p, m: (
                                {
                                    "group_size": 32,
                                    "bits": 8 if p.startswith("layout.") else 4,
                                }
                                if hasattr(m, "to_quantized")
                                and m.weight.shape[-1] % 32 == 0
                                else False
                            ),
                        )
                        saved_config["quantization"] = {
                            "group_size": 32,
                            "bits": 4,
                            "layout.model.enc_score_head": {
                                "group_size": 32,
                                "bits": 8,
                            },
                        }
                    weights = dict(tree_flatten(model.parameters()))
                    self.assertTrue(any(k.startswith("layout.model.") for k in weights))
                    self.assertTrue(any(k.startswith("ocr.model.") for k in weights))
                    # Exercise checkpoint prefixes through the real generic loader.
                    checkpoint = {
                        k.replace("layout.model.", "layout_model.", 1).replace(
                            "ocr.model.", "ocr_model.", 1
                        ): v
                        for k, v in weights.items()
                    }
                    (root / "config.json").write_text(json.dumps(saved_config))
                    mx.save_safetensors(str(root / "model.safetensors"), checkpoint)
                    loaded = load_model(root)
                    actual = dict(tree_flatten(loaded.parameters()))
                    self.assertEqual(set(weights), set(actual))
                    for key in weights:
                        self.assertTrue(
                            mx.array_equal(weights[key], actual[key]).item(), key
                        )
                    self.assertFalse(loaded.layout.model.training)
                    self.assertFalse(loaded.ocr.model.training)
                    # Converted paths must be a fixpoint, including quantized tensors.
                    sanitized = loaded.sanitize(actual)
                    self.assertEqual(set(actual), set(sanitized))
                    for key in actual:
                        self.assertTrue(
                            mx.array_equal(actual[key], sanitized[key]).item(), key
                        )
                    mx.save_safetensors(str(root / "model.safetensors"), actual)
                    processor = object()
                    with patch(
                        "mlx_vlm.utils.load_processor", return_value=processor
                    ) as load_processor:
                        with IndicOCRParser.from_pretrained(
                            str(root), lazy=True
                        ) as ready:
                            self.assertIs(ready.ocr.processor, processor)
                            self.assertIsInstance(ready.ocr.model, Model)
                        self.assertIsNone(ready.ocr.processor)
                        load_processor.assert_called_once_with(root, eos_token_ids=None)
                    # Exercise the published shard index and embedded stage configs
                    # through the unmodified standard loader.
                    stages, weight_map, weight_mapping = {}, {}, {}
                    for name in ("layout", "ocr"):
                        stage_path = root / name
                        stage_path.mkdir()
                        stage_config = dict(config[f"{name}_config"])
                        if quantized:
                            stage_config["quantization"] = {
                                "group_size": 32,
                                "bits": 8 if name == "layout" else 4,
                            }
                        (stage_path / "config.json").write_text(
                            json.dumps(stage_config)
                        )
                        prefix = f"{name}.model."
                        stage_weights = {
                            k[len(prefix) :]: v
                            for k, v in actual.items()
                            if k.startswith(prefix)
                        }
                        mx.save_safetensors(
                            str(stage_path / "stage.safetensors"), stage_weights
                        )
                        weight_map.update(
                            (key, f"{name}/stage.safetensors") for key in stage_weights
                        )
                        weight_mapping.update((key, name) for key in stage_weights)
                        stages[name] = {"config": f"{name}/config.json"}
                        if name == "layout":
                            stages[name]["weights"] = f"{name}/stage.safetensors"
                    (root / "model.safetensors").unlink()
                    published_config = dict(
                        saved_config,
                        stages=stages,
                        weight_mapping=weight_mapping,
                        ocr_model_path="ocr",
                    )
                    (root / "config.json").write_text(json.dumps(published_config))
                    index_path = root / "model.safetensors.index.json"
                    index_path.write_text(json.dumps({"weight_map": weight_map}))
                    options = LayoutOptions(conf=0.7)
                    bundled = load_model(root, lazy=True)
                    bundled.layout.options = options
                    self.assertIs(bundled.layout.options, options)
                    bundled_weights = dict(tree_flatten(bundled.parameters()))
                    self.assertEqual(set(actual), set(bundled_weights))
                    for key in actual:
                        self.assertTrue(
                            mx.array_equal(actual[key], bundled_weights[key]).item(),
                            key,
                        )
                    page = object()
                    with (
                        patch(
                            "mlx_vlm.utils.load_processor", return_value=processor
                        ) as load_processor,
                        patch.object(IndicOCRParser, "detect", return_value=page),
                        patch.object(
                            type(bundled.ocr), "run", return_value=page
                        ) as run,
                    ):
                        self.assertIs(bundled.parse("page.png"), page)
                        self.assertIs(bundled.parse("page.png"), page)
                        load_processor.assert_called_once_with(
                            root / "ocr", eos_token_ids=None
                        )
                        run.assert_called_with("page.png", page)
                    override = root / "override"
                    override.mkdir()
                    (override / "config.json").write_text(
                        json.dumps(
                            {
                                "model_type": "pp_doclayout_v3",
                                "num_labels": 3,
                            }
                        )
                    )
                    override_weights = {
                        "enc_score_head.weight": mx.ones((3, 32)),
                        "enc_score_head.bias": mx.zeros((3,)),
                    }
                    mx.save_safetensors(
                        str(override / "model.safetensors"), override_weights
                    )
                    with load_model(root, lazy=True) as overridden:
                        overridden.layout.model = load_model(override, lazy=True)
                        self.assertTrue(
                            mx.array_equal(
                                overridden.layout.model.enc_score_head.weight,
                                override_weights["enc_score_head.weight"],
                            ).item()
                        )
                        self.assertEqual(
                            set(dict(tree_flatten(overridden.ocr.model.parameters()))),
                            set(dict(tree_flatten(bundled.ocr.model.parameters()))),
                        )
                    index_path.unlink()
                    (root / "config.json").write_text(json.dumps(saved_config))
                    checkpoint.pop(next(iter(checkpoint)))
                    mx.save_safetensors(str(root / "model.safetensors"), checkpoint)
                    with self.assertRaises(ValueError):
                        load_model(root)

    def test_parser_sanitize_converts_raw_stage_weights(self):
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_vlm.models.indic_ocr import IndicOCRParser, Model, ModelConfig
        from mlx_vlm.models.pp_doclayout_v3 import Model as LayoutModel

        class Layout(nn.Module):
            sanitize = staticmethod(LayoutModel.sanitize)

        parser = IndicOCRParser(
            Layout(), Model(ModelConfig.from_dict(self.ocr_config())), None
        )
        conv = mx.arange(32 * 3 * 3 * 3).reshape(32, 3, 3, 3)
        embedding = mx.ones((64, 16))
        raw = {
            "layout_model.model.backbone.model.embedder.stem1.convolution.weight": conv,
            "layout_model.model.denoising_class_embed.weight": mx.ones((2, 16)),
            "ocr_model.model.language_model.embed_tokens.weight": embedding,
            "unexpected.weight": mx.ones((1,)),
        }
        sanitized = parser.sanitize(raw)
        self.assertEqual(
            set(sanitized),
            {
                "layout.model.backbone.embedder.stem1.conv.weight",
                "ocr.model.language_model.model.embed_tokens.weight",
                "unexpected.weight",
            },
        )
        self.assertTrue(
            mx.array_equal(
                sanitized["layout.model.backbone.embedder.stem1.conv.weight"],
                conv.transpose(0, 2, 3, 1),
            ).item()
        )
        self.assertEqual(len(raw), 4)
        duplicate = dict(raw)
        duplicate["ocr.model.model.language_model.embed_tokens.weight"] = embedding
        with self.assertRaisesRegex(ValueError, "Duplicate ocr weight"):
            parser.sanitize(duplicate)

    def test_indic_ocr_prompts_and_crops(self):
        from PIL import Image

        from mlx_vlm.models.indic_ocr.processing_indic_ocr import (
            DROP_TYPES,
            crop_for_block,
            map_label,
            prompt_for,
        )

        self.assertEqual(map_label("Paragraph"), "Text")
        self.assertEqual(map_label("equation"), "Equation")
        self.assertEqual(map_label("Table"), "Table")
        self.assertEqual(map_label("page-number"), "PageNumber")
        self.assertIn("Figure", DROP_TYPES)
        self.assertIn("LaTeX", prompt_for("Equation"))
        self.assertIn("HTML", prompt_for("Table"))
        self.assertIn("markdown", prompt_for("Table", "markdown").lower())
        self.assertIn("LaTeX", prompt_for("Text"))

        page = Image.new("RGB", (400, 400), "white")
        crop = crop_for_block([50.0, 50.0, 350.0, 200.0], page)
        self.assertIsNotNone(crop)
        self.assertLessEqual(crop.size[0] * crop.size[1], 2359296)
        tiny = crop_for_block([10.0, 10.0, 12.0, 12.0], page)
        self.assertEqual(tiny.size, (2, 2))
        self.assertIsNone(crop_for_block([10.0, 10.0, 10.0, 12.0], page))
        with self.assertRaises(ValueError):
            prompt_for("Table", "csv")

    def test_indic_ocr_reconstruct(self):
        from mlx_vlm.models.indic_ocr.blocks import Block
        from mlx_vlm.models.indic_ocr.reconstruct import (
            dehyphenate,
            reconstruct,
            repair_math,
        )

        self.assertEqual(dehyphenate("hyphen-\nated"), "hyphenated")

        # Equation without delimiters gets wrapped; delimited text is untouched
        blocks = [
            Block(
                order=1,
                label="Equation",
                type="Equation",
                bbox_xyxy=[0, 0, 10, 10],
                conf=1.0,
                text="\\frac{a}{b}",
            ),
            Block(
                order=0,
                label="Paragraph",
                type="Text",
                bbox_xyxy=[0, 0, 10, 10],
                conf=1.0,
                text="Hello",
            ),
            Block(
                order=2,
                label="Image",
                type="Picture",
                bbox_xyxy=[0, 0, 10, 10],
                conf=1.0,
                text="should not appear",
            ),
        ]
        md = reconstruct(blocks)
        self.assertTrue(md.startswith("Hello"))
        self.assertIn("$$\\frac{a}{b}$$", md)
        self.assertNotIn("should not appear", md)

        # Indic script inside math gets \text{}; prose outside $ is untouched
        self.assertIn("\\text{", repair_math("$$x = 9$$".replace("x", "প্রোটন")))
        self.assertEqual(repair_math("plain prose"), "plain prose")

    def test_indic_ocr_block_records(self):
        from mlx_vlm.models.indic_ocr.blocks import Block, LayoutSchemaError

        rec = {"order": 0, "label": "Paragraph", "bbox_xyxy": [1, 2, 3, 4]}
        block = Block.from_record(rec)
        self.assertEqual(block.type, "Text")
        self.assertEqual(block.conf, 1.0)
        as_rec = block.as_record()
        self.assertEqual(
            list(as_rec.keys()),
            ["order", "label", "type", "bbox_xyxy", "conf"],
        )

        # Foreign taxonomy with explicit type passes; bad type raises
        foreign = {
            "order": 1,
            "label": "para",
            "type": "Text",
            "bbox_xyxy": [1, 2, 3, 4],
        }
        self.assertEqual(Block.from_record(foreign).type, "Text")
        with self.assertRaises(LayoutSchemaError):
            Block.from_record(
                {
                    "order": 2,
                    "label": "para",
                    "type": "Tabel",
                    "bbox_xyxy": [1, 2, 3, 4],
                }
            )
        self.assertTrue(Block.problems({"label": "Paragraph"}))

    def test_indic_ocr_clean_layout(self):
        from mlx_vlm.models.indic_ocr.blocks import (
            Block,
            clean_layout,
            resolve_nested_equations,
        )

        def b(order, label, box, type=None):
            from mlx_vlm.models.indic_ocr.processing_indic_ocr import map_label

            return Block(
                order=order,
                label=label,
                type=type or map_label(label),
                bbox_xyxy=list(box),
                conf=0.9,
            )

        # Nested duplicate of the same group goes
        blocks = [
            b(0, "Paragraph", [0, 0, 100, 100]),
            b(1, "Paragraph", [10, 10, 90, 90]),
        ]
        self.assertEqual(len(clean_layout(blocks)), 1)

        # Only the largest header survives; it wraps the paragraph so kept
        blocks = [
            b(0, "Header", [0, 0, 100, 60]),
            b(1, "Header", [0, 0, 50, 10]),
            b(2, "Paragraph", [10, 25, 90, 55]),
        ]
        kept = clean_layout(blocks)
        self.assertEqual(
            [b_.bbox_xyxy for b_ in kept if b_.label == "Header"],
            [[0, 0, 100, 60]],
        )

        # Equation nested in text is absorbed by default
        blocks = [
            b(0, "Paragraph", [0, 0, 100, 100]),
            b(1, "Equation", [10, 10, 90, 30], type="Equation"),
        ]
        self.assertEqual(len(resolve_nested_equations(blocks)), 1)
        self.assertEqual(len(resolve_nested_equations(blocks, nest=False)), 2)

    def test_indic_ocr_build_ocr_requests(self):
        from PIL import Image

        from mlx_vlm.models.indic_ocr.blocks import Block
        from mlx_vlm.models.indic_ocr.processing_indic_ocr import build_ocr_requests

        page = Image.new("RGB", (400, 400), "white")
        blocks = [
            Block(
                order=0,
                label="Paragraph",
                type="Text",
                bbox_xyxy=[50, 50, 350, 200],
                conf=0.9,
            ),
            Block(
                order=1,
                label="Header",
                type="PageHeader",
                bbox_xyxy=[50, 5, 350, 30],
                conf=0.9,
            ),
            Block(
                order=2,
                label="Image",
                type="Picture",
                bbox_xyxy=[50, 210, 350, 390],
                conf=0.9,
            ),
        ]
        reqs = build_ocr_requests(blocks, page)
        self.assertEqual(len(reqs), 1)
        self.assertIs(reqs[0][0], blocks[0])
        self.assertIn("LaTeX", reqs[0][2])

    def test_indic_ocr_viewer_records_to_blocks(self):
        from mlx_vlm.models.indic_ocr.pipeline import viewer_records_to_blocks

        recs = [
            {
                "bbox": [0.0, 0.0, 500.0, 500.0],
                "label": "Paragraph",
                "reading_order": 2,
                "score": 0.9,
            },
            {
                "bbox": [500.0, 0.0, 1000.0, 1000.0],
                "label": "Page-number",
                "reading_order": 1,
                "score": 0.8,
            },
        ]
        blocks = viewer_records_to_blocks(recs, 200, 400)
        self.assertEqual([b.order for b in blocks], [0, 1])
        self.assertEqual(blocks[0].type, "PageNumber")
        self.assertEqual(blocks[1].bbox_xyxy, [0.0, 0.0, 100.0, 200.0])

    def test_indic_ocr_page_result_round_trip(self):
        import json

        from mlx_vlm.models.indic_ocr.pipeline import PageResult

        record = {
            "image": "page.png",
            "width": 200,
            "height": 400,
            "blocks": [
                {
                    "order": 0,
                    "label": "Paragraph",
                    "type": "Text",
                    "bbox_xyxy": [10, 10, 190, 100],
                    "conf": 0.9,
                    "text": "hi",
                },
                {
                    "order": 1,
                    "label": "Image",
                    "type": "Picture",
                    "bbox_xyxy": [10, 110, 190, 390],
                    "conf": 0.8,
                    "text": "",
                },
            ],
        }
        page = PageResult.from_record(record)
        self.assertEqual(page.blocks[0].type, "Text")
        self.assertIsNone(page.markdown)
        out = page.as_record()
        self.assertEqual(out["image"], "page.png")
        self.assertEqual(len(out["blocks"]), 2)
        self.assertEqual(out["blocks"][0]["text"], "hi")
        # Round-trips through JSON.
        self.assertEqual(
            PageResult.from_record(json.loads(json.dumps(out))).as_record(), out
        )

    def test_indic_ocr_backends(self):
        from PIL import Image

        from mlx_vlm.models.indic_ocr.pipeline import (
            JsonLayoutBackend,
            MLXLayoutBackend,
            PageResult,
        )

        records = [
            {
                "bbox": [0.0, 0.0, 500.0, 1000.0],
                "label": "Paragraph",
                "reading_order": 2,
                "score": 0.9,
            },
            {
                "bbox": [500.0, 0.0, 1000.0, 1000.0],
                "label": "Page-number",
                "reading_order": 1,
                "score": 0.8,
            },
        ]

        class StubDetector:
            def detect(self, image, conf=0.5, img_size=1024):
                return records

        img = Image.new("RGB", (200, 400), "white")
        blocks = MLXLayoutBackend(StubDetector()).detect(img)
        self.assertEqual([b.order for b in blocks], [0, 1])
        self.assertEqual(blocks[0].label, "Page-number")
        self.assertEqual(blocks[0].bbox_xyxy, [0.0, 200.0, 200.0, 400.0])

        page = PageResult(image="p.png", width=200, height=400, blocks=blocks)
        replayed = JsonLayoutBackend(page).detect(img)
        self.assertEqual([b.order for b in replayed], [0, 1])
        self.assertEqual(replayed[0].label, "Page-number")
        # Copies: mutating the replay must not touch the stored page.
        replayed[0].label = "Changed"
        self.assertEqual(page.blocks[0].label, "Page-number")

    def test_indic_ocr_stage_dir_resolution(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.indic_ocr.pipeline import _resolve_repo_root, _stage_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "indic_ocr",
                        "stages": {
                            "layout": {"config": "weights/layout/config.json"},
                            "ocr": {"config": "weights/ocr/config.json"},
                        },
                    }
                )
            )
            resolved, stages = _resolve_repo_root(str(root))
            self.assertEqual(resolved, root)
            self.assertEqual(
                _stage_dir(resolved, stages, "layout", "weights/layout/config.json"),
                root / "weights/layout",
            )
            self.assertEqual(
                _stage_dir(resolved, stages, "ocr", "weights/ocr/config.json"),
                root / "weights/ocr",
            )
            # Defaults when the pointer is absent.
            self.assertEqual(
                _stage_dir(resolved, {}, "ocr", "weights/ocr/config.json"),
                root / "weights/ocr",
            )

    def test_indic_ocr_layout_source_resolution(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.indic_ocr.pipeline import _resolve_layout_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights" / "layout").mkdir(parents=True)
            (root / "weights" / "layout" / "config.json").write_text("{}")
            stages = {"layout": {"config": "weights/layout/config.json"}}

            # Explicit repo/path wins.
            self.assertEqual(
                _resolve_layout_source(root, stages, layout_repo="some-id"), "some-id"
            )
            # Else the single-repo subdir.
            self.assertEqual(
                _resolve_layout_source(root, stages),
                str(root / "weights" / "layout"),
            )
            # Else a helpful error (separate dir without the subdir).
            with tempfile.TemporaryDirectory() as empty:
                with self.assertRaises(ValueError) as ctx:
                    _resolve_layout_source(Path(empty), {})
                self.assertIn("layout_repo", str(ctx.exception))

    def test_indic_ocr_public_loader_supports_flat_and_bundled_stages(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch

        from mlx_vlm.models.indic_ocr.pipeline import IndicOCRParser

        cases = [
            ({"text_config": {}, "vision_config": {}}, "."),
            ({"stages": {"ocr": {"config": "config.json"}}}, "."),
            ({}, "weights/ocr"),
            ({"stages": {"ocr": {"config": "recognizer/config.json"}}}, "recognizer"),
        ]
        for config, subdir in cases:
            with (
                self.subTest(subdir=subdir, config=config),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                (root / "config.json").write_text(json.dumps(config))
                ocr = root / subdir
                if ocr != root:
                    ocr.mkdir(parents=True)
                    (ocr / "config.json").write_text("{}")
                layout = root / "layout"
                layout.mkdir()
                (layout / "config.json").write_text("{}")
                detector, recognizer, processor = Mock(), Mock(), Mock()
                with (
                    patch(
                        "mlx_vlm.load", return_value=(recognizer, processor)
                    ) as load_ocr,
                    patch(
                        "mlx_vlm.utils.load_model", return_value=detector
                    ) as load_layout,
                ):
                    with IndicOCRParser.from_pretrained(
                        str(root), layout_repo=str(layout)
                    ) as parser:
                        self.assertIs(parser.ocr.model, recognizer)
                        self.assertIs(parser.layout.model, detector)
                load_ocr.assert_called_once_with(str(ocr), lazy=False, strict=True)
                load_layout.assert_called_once_with(layout, lazy=False, strict=True)
                detector.eval.assert_called_once_with()

    def test_indic_ocr_public_loader_returns_configured_parser(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch

        from PIL import Image

        from mlx_vlm.models.indic_ocr import (
            CropOptions,
            DedupOptions,
            IndicOCRParser,
            LayoutOptions,
            RecognizerOptions,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stages = {}
            for name, dirname in (("layout", "detector"), ("ocr", "recognizer")):
                stage_path = root / dirname
                stage_path.mkdir()
                (stage_path / "config.json").write_text("{}")
                stages[name] = {"config": f"{dirname}/config.json"}
            (root / "config.json").write_text(
                json.dumps({"model_type": "indic_ocr", "stages": stages})
            )
            image_path = root / "page.png"
            Image.new("RGB", (200, 100)).save(image_path)
            detector, recognizer, processor = Mock(), Mock(), Mock()
            detector.detect.return_value = [
                {
                    "label": "Paragraph",
                    "reading_order": 1,
                    "bbox": [100, 200, 900, 800],
                    "score": 0.9,
                }
            ]
            layout_options = LayoutOptions(conf=0.7)
            recognizer_options = RecognizerOptions(max_tokens=32)
            dedup = DedupOptions(nest=False)
            crop = CropOptions(pad_px=5)
            with (
                patch("mlx_vlm.utils.load_model", return_value=detector) as load_layout,
                patch("mlx_vlm.load", return_value=(recognizer, processor)) as load_ocr,
            ):
                with IndicOCRParser.from_pretrained(
                    root,
                    lazy=True,
                    strict=False,
                    trust_remote_code=False,
                    layout_options=layout_options,
                    recognizer_options=recognizer_options,
                    dedup=dedup,
                    crop=crop,
                ) as parser:
                    self.assertIsInstance(parser, IndicOCRParser)
                    self.assertIs(parser.ocr.model, recognizer)
                    self.assertIs(parser.ocr.processor, processor)
                    self.assertIs(parser.ocr.options, recognizer_options)
                    self.assertIs(parser.ocr.dedup, dedup)
                    self.assertIs(parser.ocr.crop, crop)
                    page = parser.detect(str(image_path))
                    self.assertEqual(page.blocks[0].bbox_xyxy, [40, 10, 160, 90])
                    self.assertEqual(page.blocks[0].type, "Text")
                    self.assertEqual(detector.detect.call_args.kwargs["conf"], 0.7)
                self.assertIsNone(parser.layout.model)
                self.assertIsNone(parser.ocr.model)
                self.assertIsNone(parser.ocr.processor)
            load_layout.assert_called_once_with(
                root / "detector", lazy=True, strict=False, trust_remote_code=False
            )
            load_ocr.assert_called_once_with(
                str(root / "recognizer"),
                lazy=True,
                strict=False,
                trust_remote_code=False,
            )

    def test_indic_ocr_standard_loader_only_dispatches_wrapper_configs(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.indic_ocr import IndicOCRParser
        from mlx_vlm.utils import load_model

        configs = [
            {"model_type": "indic_ocr", "text_config": {}, "stages": {}},
            {"model_type": "pp_doclayout_v3", "stages": {}},
            {"model_type": None, "speculators_model_type": "dflash2", "stages": {}},
        ]
        for config in configs:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "config.json").write_text(json.dumps(config))
                with patch.object(IndicOCRParser, "from_pretrained") as load_parser:
                    with self.assertRaisesRegex(FileNotFoundError, "No safetensors"):
                        load_model(root)
                    load_parser.assert_not_called()

    def test_indic_ocr_public_loader_rejects_wrapper_stages(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.indic_ocr import IndicOCRParser

        for name in ("layout", "ocr"):
            for target in (".", "nested"):
                with (
                    self.subTest(stage=name, target=target),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    stages = {}
                    for stage in ("layout", "ocr"):
                        (root / stage).mkdir()
                        (root / stage / "config.json").write_text("{}")
                        stages[stage] = {"config": f"{stage}/config.json"}
                    stages[name] = {"config": f"{target}/config.json"}
                    wrapper = json.dumps({"model_type": "indic_ocr", "stages": stages})
                    (root / "config.json").write_text(wrapper)
                    if target == "nested":
                        (root / target).mkdir()
                        (root / target / "config.json").write_text(wrapper)
                    with (
                        patch("mlx_vlm.utils.load_model") as load_layout,
                        patch("mlx_vlm.load") as load_ocr,
                        self.assertRaisesRegex(ValueError, f"{name} stage.*two-stage"),
                    ):
                        IndicOCRParser.from_pretrained(str(root))
                    load_layout.assert_not_called()
                    load_ocr.assert_not_called()

    def test_indic_ocr_missing_ocr_config_fails_before_loading_models(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.indic_ocr.pipeline import IndicOCRParser

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}")
            with patch("mlx_vlm.utils.load_model") as load_layout:
                with self.assertRaisesRegex(FileNotFoundError, "No OCR config"):
                    IndicOCRParser.from_pretrained(tmp, layout_repo="example/layout")
                load_layout.assert_not_called()

    def test_indic_ocr_stock_labels_and_foreign_types(self):
        from PIL import Image

        from mlx_vlm.models.indic_ocr.blocks import Block, LayoutSchemaError
        from mlx_vlm.models.indic_ocr.pipeline import viewer_records_to_blocks
        from mlx_vlm.models.indic_ocr.processing_indic_ocr import (
            EQUATION_PROMPT,
            PP_DOCLAYOUT_LABEL_TO_TYPE,
            TABLE_PROMPT_HTML,
            build_ocr_requests,
        )

        for label, expected in PP_DOCLAYOUT_LABEL_TO_TYPE.items():
            block = Block.from_record(
                {"label": label, "order": 0, "bbox_xyxy": [0, 0, 10, 10]}
            )
            self.assertEqual(block.type, expected)
        records = [
            {"label": "formula", "reading_order": 1, "bbox": [0, 0, 400, 1000]},
            {
                "label": "foreign_table",
                "type": "Table",
                "reading_order": 2,
                "bbox": [500, 0, 1000, 1000],
            },
        ]
        blocks = viewer_records_to_blocks(records, 100, 100)
        requests = build_ocr_requests(blocks, Image.new("RGB", (100, 100)))
        self.assertEqual([r[2] for r in requests], [EQUATION_PROMPT, TABLE_PROMPT_HTML])
        with self.assertRaises(LayoutSchemaError):
            viewer_records_to_blocks(
                [{**records[0], "label": "unknown_label"}], 100, 100
            )
        for order in (0, 1.5, "1", True):
            with self.subTest(order=order), self.assertRaises(LayoutSchemaError):
                viewer_records_to_blocks(
                    [{**records[0], "reading_order": order}], 100, 100
                )

    def test_indic_ocr_cleanup_respects_stock_and_foreign_types(self):
        from mlx_vlm.models.indic_ocr.blocks import Block, clean_layout

        def block(order, label, bbox, **extra):
            return Block.from_record(
                {"order": order, "label": label, "bbox_xyxy": bbox, **extra}
            )

        # A page-number block must not be swallowed by a larger text block.
        blocks = [
            block(0, "text", [0, 0, 100, 100]),
            block(1, "number", [10, 80, 20, 90]),
        ]
        self.assertEqual(len(clean_layout(blocks)), 2)
        # Lowercase stock headers follow the same cleanup rules as Indic headers.
        blocks = [
            block(0, "header", [0, 0, 100, 60]),
            block(1, "header", [0, 0, 50, 10]),
            block(2, "text", [10, 25, 90, 55]),
        ]
        self.assertEqual([b.order for b in clean_layout(blocks)], [0, 2])
        # A foreign figure with an explicit type cannot absorb a text caption.
        blocks = [
            block(0, "foreign_figure", [0, 0, 100, 100], type="Figure"),
            block(1, "figure_title", [10, 80, 90, 90]),
        ]
        self.assertEqual(len(clean_layout(blocks)), 2)

    def test_indic_ocr_close_releases_owned_models(self):
        import gc
        import weakref

        from PIL import Image

        from mlx_vlm.models.indic_ocr.pipeline import IndicOCRParser

        class Sentinel:
            pass

        layout, ocr, processor = Sentinel(), Sentinel(), Sentinel()
        references = [weakref.ref(value) for value in (layout, ocr, processor)]
        with IndicOCRParser(layout, ocr, processor) as parser:
            del layout, ocr, processor
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        parser.close()  # Closing twice is safe.
        with self.assertRaisesRegex(RuntimeError, "closed"):
            parser.layout.detect(Image.new("RGB", (10, 10)))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            parser.ocr.run("unused.png", {})

    def test_indic_ocr_rejects_ambiguous_block_orders(self):
        from unittest.mock import patch

        from PIL import Image

        from mlx_vlm.models.indic_ocr.blocks import Block, LayoutSchemaError
        from mlx_vlm.models.indic_ocr.pipeline import BlockOCRRunner, PageResult

        record = {"order": 0, "label": "Paragraph", "bbox_xyxy": [0, 0, 10, 10]}
        for order in (-1, 1.5, "1", True):
            with self.subTest(order=order), self.assertRaises(LayoutSchemaError):
                Block.from_record({**record, "order": order})
        page_record = {
            "image": "unused.png",
            "width": 10,
            "height": 10,
            "blocks": [record, record],
        }
        with self.assertRaisesRegex(LayoutSchemaError, "duplicate order"):
            PageResult.from_record(page_record)
        # A PageResult supplied directly must not bypass strict validation.
        page = PageResult.from_record(page_record, strict=False)
        with patch(
            "mlx_vlm.models.indic_ocr.pipeline._open",
            return_value=Image.new("RGB", (10, 10)),
        ):
            with self.assertRaisesRegex(LayoutSchemaError, "duplicate order"):
                BlockOCRRunner(object(), object()).run("unused.png", page)

    def test_indic_ocr_replay_clears_stale_skipped_text(self):
        from unittest.mock import patch

        from PIL import Image

        from mlx_vlm.models.indic_ocr.blocks import Block
        from mlx_vlm.models.indic_ocr.pipeline import BlockOCRRunner, PageResult
        from mlx_vlm.models.indic_ocr.processing_indic_ocr import transcribe_blocks

        blocks = [
            Block(0, "Image", "Picture", [0, 0, 10, 10], 1.0, "stale image text"),
            Block(1, "Paragraph", "Text", [0, 0, 0, 0], 1.0, "stale empty crop text"),
        ]
        image = Image.new("RGB", (10, 10))
        with patch("mlx_vlm.generate") as generate:
            copied = [b.copy() for b in blocks]
            transcribe_blocks(object(), object(), image, copied, use_tqdm=False)
            self.assertEqual([b.text for b in copied], ["", ""])
            page = PageResult("unused.png", 10, 10, blocks)
            with patch("mlx_vlm.models.indic_ocr.pipeline._open", return_value=image):
                result = BlockOCRRunner(object(), object()).run("unused.png", page)
            self.assertEqual([b.text for b in result.blocks], ["", ""])
            self.assertEqual(
                [b.text for b in page.blocks],
                ["stale image text", "stale empty crop text"],
            )
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
