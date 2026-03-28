"""Tests for custom processor implementations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

# ── Shared mocks ──────────────────────────────────────────────────────────────


def _make_image():
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def _mock_tokenizer(**overrides):
    """Create a mock tokenizer. Pass overrides to replace any default attribute."""
    defaults = dict(
        model_input_names=["input_ids", "attention_mask"],
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        pad_token_id=0,
        image_token="<image>",
        image_token_id=100,
        boi_token="<boi>",
        eoi_token="<eoi>",
        audio_token="<audio>",
        audio_token_id=101,
        boa_token="<boa>",
        eoa_token="<eoa>",
        video_token="<video>",
        video_token_id=102,
    )
    defaults.update(overrides)
    tok = type(
        "MockTok",
        (),
        {
            **defaults,
            "convert_tokens_to_ids": lambda self, t: (
                [0] * len(t) if isinstance(t, list) else 0
            ),
            "__call__": lambda self, text, text_pair=None, return_token_type_ids=False, **kw: (
                (
                    lambda t: {
                        "input_ids": [list(range(10)) for _ in t],
                        "attention_mask": [[1] * 10 for _ in t],
                        **(
                            {"token_type_ids": [[0] * 10 for _ in t]}
                            if return_token_type_ids
                            else {}
                        ),
                    }
                )(t=[text] if isinstance(text, str) else text)
            ),
            "add_special_tokens": lambda self, d: None,
            "encode": lambda self, text, **kw: list(range(10)),
            "init_kwargs": property(lambda self: {}),
            "batch_decode": lambda self, ids, **kw: ["decoded"] * len(ids),
            "decode": lambda self, ids, **kw: "decoded",
        },
    )()
    return tok


def _mock_ip(**extra):
    """Create a mock image processor. Pass extra fields to include in output."""
    pv = np.random.randn(1, 3, 224, 224).astype(np.float32)
    attrs = dict(
        model_input_names=["pixel_values"],
        merge_size=2,
        do_image_splitting=False,
        max_image_tiles=4,
    )
    return type(
        "MockIP",
        (),
        {
            **attrs,
            "__call__": lambda self, images=None, **kw: {"pixel_values": pv, **extra},
            "fetch_images": lambda self, images: (
                [images] if not isinstance(images, list) else images
            ),
        },
    )()


# ── Base class with shared test_with_image / test_text_only ───────────────────


class _ProcessorTestBase:
    """
    Mixin for processor tests. Not a TestCase itself so pytest won't collect it.

    Subclasses must also inherit unittest.TestCase and define:
      - _make_processor() -> processor instance
      - _image_call_args() -> dict of kwargs for __call__ with image
      - _text_call_args()  -> dict of kwargs for __call__ text-only (or None to skip)
    """

    def _make_processor(self):
        raise NotImplementedError

    def _image_call_args(self):
        raise NotImplementedError

    def _text_call_args(self):
        return {"text": ["Hello world"]}

    def _assert_all_mx(self, result):
        import mlx.core as mx

        for key, value in result.items():
            if value is not None:
                self.assertIsInstance(
                    value,
                    mx.array,
                    f"{key}: expected mx.array, got {type(value).__name__}",
                )

    def test_with_image(self):
        result = self._make_processor()(**self._image_call_args())
        self._assert_all_mx(result)
        self.assertIn("pixel_values", result)

    def test_text_only(self):
        args = self._text_call_args()
        if args is None:
            self.skipTest("Processor requires images")
        result = self._make_processor()(**args)
        self._assert_all_mx(result)


# ── Per-model test classes ────────────────────────────────────────────────────


class TestLlavaProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.llava.processing_llava import LlavaProcessor

        p = LlavaProcessor.__new__(LlavaProcessor)
        p.image_token = "<image>"
        p.patch_size = 14
        p.vision_feature_select_strategy = "default"
        p.num_additional_image_tokens = 1
        p.image_processor = _mock_ip()
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> Describe"], "images": [_make_image()]}


class TestLlavaNextProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.llava_next.processing_llava_next import LlavaNextProcessor

        p = LlavaNextProcessor.__new__(LlavaNextProcessor)
        p.image_token = "<image>"
        p.patch_size = 14
        p.vision_feature_select_strategy = "default"
        p.num_additional_image_tokens = 1
        # LLaVA-NeXT needs pixel_values with shape (B, patches, C, H, W) and image_sizes
        p.image_processor = type(
            "IP",
            (),
            {
                "model_input_names": ["pixel_values"],
                "image_grid_pinpoints": [
                    [224, 224],
                    [224, 448],
                    [448, 224],
                    [448, 448],
                ],
                "size": {"height": 224, "width": 224},
                "__call__": lambda self, images=None, **kw: {
                    "pixel_values": np.random.randn(1, 5, 3, 224, 224).astype(
                        np.float32
                    ),
                    "image_sizes": [(224, 224)],
                },
                "fetch_images": lambda self, i: [i] if not isinstance(i, list) else i,
            },
        )()
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> Describe"], "images": [_make_image()]}


class TestPaliGemmaProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.paligemma.processing_paligemma import PaliGemmaProcessor

        p = PaliGemmaProcessor.__new__(PaliGemmaProcessor)
        p.image_token = "<image>"
        p.image_seq_length = 4
        p.bos_token = "<bos>"
        p.image_processor = _mock_ip()
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": "describe", "images": [_make_image()]}

    def _text_call_args(self):
        return None  # PaliGemma requires images

    def test_tokenizer_kwargs_do_not_leak_into_image_processor(self):
        from mlx_vlm.models.paligemma.processing_paligemma import PaliGemmaProcessor

        calls = {}

        class ImageProcessor:
            model_input_names = ["pixel_values"]
            valid_kwargs = type(
                "ImageValidKwargs",
                (),
                {"__annotations__": {"do_resize": bool}},
            )

            def __call__(self, images=None, **kwargs):
                if "padding" in kwargs or "add_special_tokens" in kwargs:
                    raise AssertionError("Tokenizer kwargs reached the image processor")
                calls["image_kwargs"] = kwargs
                return {
                    "pixel_values": np.random.randn(1, 3, 224, 224).astype(np.float32)
                }

        class Tokenizer:
            bos_token = "<bos>"
            eos_token = "<eos>"
            model_input_names = ["input_ids", "attention_mask"]

            def __call__(
                self,
                text,
                text_pair=None,
                return_token_type_ids=False,
                **kwargs,
            ):
                calls["tokenizer_kwargs"] = kwargs
                batch = [text] if isinstance(text, str) else text
                return {
                    "input_ids": [list(range(10)) for _ in batch],
                    "attention_mask": [[1] * 10 for _ in batch],
                    "token_type_ids": [[0] * 10 for _ in batch],
                }

        p = PaliGemmaProcessor.__new__(PaliGemmaProcessor)
        p.image_token = "<image>"
        p.image_seq_length = 4
        p.image_processor = ImageProcessor()
        p.tokenizer = Tokenizer()

        result = p(
            text="describe",
            images=[_make_image()],
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            do_resize=False,
        )

        self._assert_all_mx(result)
        self.assertEqual(calls["image_kwargs"], {"do_resize": False})
        self.assertEqual(
            calls["tokenizer_kwargs"],
            {
                "padding": True,
                "padding_side": "left",
                "add_special_tokens": False,
            },
        )


class TestGemma3Processor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.gemma3.processing_gemma3 import Gemma3Processor

        p = Gemma3Processor.__new__(Gemma3Processor)
        p.image_seq_length = 4
        p.image_token_id = 100
        p.boi_token = "<boi>"
        p.image_token = "<image>"
        p.full_image_sequence = "\n\n<boi>" + "<image>" * 4 + "<eoi>\n\n"
        p.image_processor = _mock_ip(num_crops=[0])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<boi> Cats"], "images": [_make_image()]}


class TestGemma3nProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.gemma3n.processing_gemma3n import Gemma3nProcessor

        p = Gemma3nProcessor.__new__(Gemma3nProcessor)
        p.image_seq_length = 4
        p.image_token_id = 100
        p.boi_token = "<boi>"
        p.image_token = "<image>"
        p.full_image_sequence = "\n\n<boi>" + "<image>" * 4 + "<eoi>\n\n"
        p.audio_seq_length = 4
        p.audio_token_id = 101
        p.boa_token = "<boa>"
        p.audio_token = "<audio>"
        p.full_audio_sequence = "\n\n<boa>" + "<audio>" * 4 + "<eoa>\n\n"
        p.image_processor = _mock_ip(num_crops=[0])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> Cats"], "images": [_make_image()]}


class TestDotsVLProcessor(unittest.TestCase):
    def test_sets_upstream_special_token_ids(self):
        from mlx_vlm.models.dots_ocr.processing_dots_ocr import (
            DotsDummyVideoProcessor,
            DotsVLProcessor,
        )

        def _fake_init(
            self,
            image_processor=None,
            tokenizer=None,
            chat_template=None,
        ):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

        tokenizer = _mock_tokenizer(
            image_token="<|imgpad|>",
            image_token_id=7,
            video_token="<|video_pad|>",
            video_token_id=13,
        )

        with patch(
            "mlx_vlm.models.dots_ocr.processing_dots_ocr.ProcessorMixin.__init__",
            _fake_init,
        ):
            processor = DotsVLProcessor(
                image_processor=_mock_ip(),
                tokenizer=tokenizer,
            )

        self.assertEqual(processor.image_token, "<|imgpad|>")
        self.assertEqual(processor.image_token_id, 151665)
        self.assertEqual(processor.video_token, "<|video_pad|>")
        self.assertEqual(processor.video_token_id, 151656)
        self.assertIsInstance(processor.video_processor, DotsDummyVideoProcessor)
        self.assertEqual(processor.video_processor.temporal_patch_size, 1)

    def test_from_pretrained_uses_slow_image_processor(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.dots_ocr.processing_dots_ocr import (
            DotsDummyVideoProcessor,
            DotsVLProcessor,
        )

        def _fake_init(
            self,
            image_processor=None,
            tokenizer=None,
            chat_template=None,
        ):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "chat_template.json").write_text(
                '{"chat_template": "{{ messages[0].content }}"}'
            )

            tokenizer_from_pretrained = lambda *args, **kwargs: _mock_tokenizer()
            image_from_pretrained = unittest.mock.Mock(return_value=_mock_ip())

            with (
                patch.dict(
                    "transformers.__dict__",
                    {
                        "AutoTokenizer": SimpleNamespace(
                            from_pretrained=tokenizer_from_pretrained
                        ),
                        "AutoImageProcessor": SimpleNamespace(
                            from_pretrained=image_from_pretrained
                        ),
                    },
                ),
                patch(
                    "mlx_vlm.models.dots_ocr.processing_dots_ocr.ProcessorMixin.__init__",
                    _fake_init,
                ),
            ):
                processor = DotsVLProcessor.from_pretrained(tmpdir, use_fast=True)

        image_from_pretrained.assert_called_once()
        self.assertEqual(image_from_pretrained.call_args.kwargs["use_fast"], False)
        self.assertIsInstance(processor.video_processor, DotsDummyVideoProcessor)


class TestSmolVLMProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.smolvlm.processing_smolvlm import SmolVLMProcessor

        p = SmolVLMProcessor.__new__(SmolVLMProcessor)
        p.fake_image_token = "<fake_token_around_image>"
        p.image_token = "<image>"
        p.image_token_id = 100
        p.end_of_utterance_token = "<end_of_utterance>"
        p.global_image_token = "<global-img>"
        p.image_seq_len = 4
        p.video_token = "<video>"
        p.image_processor = _mock_ip(rows=[[0]], cols=[[0]])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> Describe"], "images": [[_make_image()]]}


class TestMllamaProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.mllama.processing_mllama import MllamaProcessor

        p = MllamaProcessor.__new__(MllamaProcessor)
        p.image_token = "<|image|>"
        p.image_token_id = 128256
        p.python_token = "<|python_tag|>"
        p.python_token_id = 0
        p.bos_token = "<bos>"
        p.image_processor = type(
            "IP",
            (),
            {
                "model_input_names": ["pixel_values"],
                "max_image_tiles": 4,
                "__call__": lambda self, images=None, max_image_tiles=4, **kw: {
                    "pixel_values": np.random.randn(1, 4, 3, 560, 560).astype(
                        np.float32
                    ),
                    "num_tiles": [[2]],
                    "aspect_ratio_ids": np.array([[1]]),
                    "aspect_ratio_mask": np.ones((1, 4), dtype=np.int64),
                },
                "fetch_images": lambda self, i: [i] if not isinstance(i, list) else i,
            },
        )()

        # Mllama tokenizer must emit 128256 for <|image|> so n_images_in_ids matches
        def _mllama_tok(text, **kw):
            if isinstance(text, str):
                text = [text]
            ids = []
            for t in text:
                toks = []
                i = 0
                while i < len(t):
                    if t[i:].startswith("<|image|>"):
                        toks.append(128256)
                        i += 9
                    elif t[i] == " ":
                        i += 1
                    else:
                        w = ""
                        while (
                            i < len(t)
                            and t[i] != " "
                            and not t[i:].startswith("<|image|>")
                        ):
                            w += t[i]
                            i += 1
                        toks.append(hash(w) % 30000)
                ids.append(toks)
            ml = max(len(x) for x in ids)
            return {
                "input_ids": [x + [0] * (ml - len(x)) for x in ids],
                "attention_mask": [[1] * len(x) + [0] * (ml - len(x)) for x in ids],
            }

        p.tokenizer = type(
            "MllamaTok",
            (),
            {
                "model_input_names": ["input_ids", "attention_mask"],
                "bos_token": "<bos>",
                "image_token": "<|image|>",
                "image_token_id": 128256,
                "convert_tokens_to_ids": lambda self, t: (
                    128256 if t == "<|image|>" else 0
                ),
                "__call__": lambda self, text, **kw: _mllama_tok(text, **kw),
                "add_special_tokens": lambda self, d: None,
                "init_kwargs": property(lambda self: {}),
                "batch_decode": lambda self, ids, **kw: ["decoded"] * len(ids),
                "decode": lambda self, ids, **kw: "decoded",
            },
        )()
        return p

    def _image_call_args(self):
        return {"text": ["<|image|>Describe"], "images": [[_make_image()]]}

    def test_with_image(self):
        result = self._make_processor()(**self._image_call_args())
        self._assert_all_mx(result)
        self.assertIn("pixel_values", result)
        self.assertIn("cross_attention_mask", result)

    def test_cross_attention_mask_helpers(self):
        from mlx_vlm.models.mllama.processing_mllama import (
            convert_sparse_cross_attention_mask_to_dense,
            get_cross_attention_token_mask,
        )

        ids = [1, 2, 128256, 3, 4, 128256, 5, 6]
        mask = get_cross_attention_token_mask(ids, 128256)
        self.assertEqual(len(mask), 2)
        self.assertEqual(mask[0], [2, 5])
        self.assertEqual(mask[1], [5, 8])
        self.assertEqual(get_cross_attention_token_mask([1, 2, 3], 128256), [])

        dense = convert_sparse_cross_attention_mask_to_dense([mask], [[2, 3]], 4, 8)
        self.assertEqual(dense.shape, (1, 8, 2, 4))
        self.assertEqual(dense[0, 2, 0, 0], 1)
        self.assertEqual(dense[0, 0, 0, 0], 0)

    def test_build_string_from_input(self):
        from mlx_vlm.models.mllama.processing_mllama import build_string_from_input

        self.assertEqual(
            build_string_from_input("Hello", "<bos>", "<|image|>"), "<bos>Hello"
        )
        self.assertEqual(
            build_string_from_input("<|image|>Hello", "<bos>", "<|image|>"),
            "<|image|><bos>Hello",
        )
        self.assertEqual(
            build_string_from_input("<bos>Hello", "<bos>", "<|image|>"), "<bos>Hello"
        )


class TestQwen2VLProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

        p = Qwen2VLProcessor.__new__(Qwen2VLProcessor)
        p.image_token = "<|image_pad|>"
        p.video_token = "<|video_pad|>"
        p.image_token_id = 100
        p.video_token_id = 102
        p.image_processor = _mock_ip(
            image_grid_thw=np.array([[1, 16, 16]], dtype=np.int64)
        )
        p.tokenizer = _mock_tokenizer(
            image_token="<|image_pad|>", video_token="<|video_pad|>"
        )
        return p

    def _image_call_args(self):
        return {"text": ["<|image_pad|> Describe"], "images": [_make_image()]}


class TestQwen2_5VLProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

        p = Qwen2_5_VLProcessor.__new__(Qwen2_5_VLProcessor)
        p.image_token = "<|image_pad|>"
        p.video_token = "<|video_pad|>"
        p.image_token_id = 100
        p.video_token_id = 102
        p.image_processor = _mock_ip(
            image_grid_thw=np.array([[1, 16, 16]], dtype=np.int64)
        )
        p.tokenizer = _mock_tokenizer(
            image_token="<|image_pad|>", video_token="<|video_pad|>"
        )
        return p

    def _image_call_args(self):
        return {"text": ["<|image_pad|> Describe"], "images": [_make_image()]}


class TestQwen3VLProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        p = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
        p.image_token = "<|image_pad|>"
        p.video_token = "<|video_pad|>"
        p.image_token_id = 100
        p.video_token_id = 102
        p.vision_start_token = "<|vs|>"
        p.vision_end_token = "<|ve|>"
        p.vision_start_token_id = 200
        p.vision_end_token_id = 201
        p.image_processor = _mock_ip(
            image_grid_thw=np.array([[1, 16, 16]], dtype=np.int64)
        )
        p.tokenizer = _mock_tokenizer(
            image_token="<|image_pad|>", video_token="<|video_pad|>"
        )
        return p

    def _image_call_args(self):
        return {"text": ["<|image_pad|> Describe"], "images": [_make_image()]}


class TestQwen3OmniMoeProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
            Qwen3OmniMoeProcessor,
        )

        p = Qwen3OmniMoeProcessor.__new__(Qwen3OmniMoeProcessor)
        p.image_token = "<|image|>"
        p.audio_token = "<|audio|>"
        p.video_token = "<|video|>"
        p.vision_bos_token = "<|vb|>"
        p.vision_eos_token = "<|ve|>"
        p.audio_bos_token = "<|ab|>"
        p.audio_eos_token = "<|ae|>"
        p.tokenizer = _mock_tokenizer(
            image_token="<|image|>",
            audio_token="<|audio|>",
            video_token="<|video|>",
            vision_bos_token="<|vb|>",
            vision_eos_token="<|ve|>",
            audio_bos_token="<|ab|>",
            audio_eos_token="<|ae|>",
        )
        p.image_processor = _mock_ip(
            image_grid_thw=np.array([[1, 16, 16]], dtype=np.int64)
        )
        p.video_processor = type("VP", (), {"model_input_names": [], "merge_size": 2})()
        p.feature_extractor = type("FE", (), {"model_input_names": []})()
        return p

    def _image_call_args(self):
        return {"text": "<|image|>Describe", "images": [_make_image()]}

    def _text_call_args(self):
        return {"text": "Hello world"}


class TestIdefics2Processor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.idefics2.processing_idefics2 import Idefics2Processor

        p = Idefics2Processor.__new__(Idefics2Processor)
        p.fake_image_token = "<fake_token_around_image>"
        p.image_token = "<image>"
        p.image_token_id = 100
        p.image_seq_len = 4
        ip = _mock_ip()
        ip.do_image_splitting = False
        p.image_processor = ip
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> What are these?"], "images": [[_make_image()]]}


class TestIdefics3Processor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.idefics3.processing_idefics3 import Idefics3Processor

        p = Idefics3Processor.__new__(Idefics3Processor)
        p.fake_image_token = "<fake_token_around_image>"
        p.image_token = "<image>"
        p.image_token_id = 100
        p.fake_image_token_id = 105
        p.global_image_token_id = 107
        p.global_image_tag = "<global-img>"
        p.image_seq_len = 4
        p.end_of_utterance_token = "<end_of_utterance>"
        p._regex_to_remove_extra_special_tokens = None
        p.image_processor = _mock_ip(rows=[[0]], cols=[[0]])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> What are these?"], "images": [[_make_image()]]}

    def test_image_prompt_string(self):
        from mlx_vlm.models.idefics3.processing_idefics3 import get_image_prompt_string

        result = get_image_prompt_string(0, 0, 4, "<F>", "<I>", "<G>")
        self.assertIn("<I>" * 4, result)
        self.assertIn("<G>", result)
        result = get_image_prompt_string(2, 2, 4, "<F>", "<I>", "<G>")
        self.assertIn("<row_1_col_1>", result)
        self.assertIn("<row_2_col_2>", result)


class TestAyaVisionProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.aya_vision.processing_aya_vision import AyaVisionProcessor

        p = AyaVisionProcessor.__new__(AyaVisionProcessor)
        p.image_token = "<image>"
        p.patch_size = 28
        p.img_size = 364
        p.start_of_img_token = "<|SI|>"
        p.end_of_img_token = "<|EI|>"
        p.img_patch_token = "<|IP|>"
        p.img_line_break_token = "<|LB|>"
        p.tile_token = "TILE"
        p.tile_global_token = "TG"
        p.image_token_id = 0
        p.image_ids = [0] * 5
        p.image_processor = _mock_ip(num_patches=[1])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> Cats"], "images": [_make_image()]}


class TestLlama4Processor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.llama4.processing_llama4 import Llama4Processor

        p = Llama4Processor.__new__(Llama4Processor)
        p.downsample_ratio = 4
        p.patch_size = 14
        p.fake_image_token = "<|image|>"
        p.image_token = "<|image|>"
        p.image_token_id = 100
        p.start_of_img_token = "<|is|>"
        p.end_of_img_token = "<|ie|>"
        p.img_patch_token = "<|p|>"
        p.tile_token = "<|tx|>"
        p.tile_global_token = "<|ty|>"
        p.image_processor = _mock_ip(aspect_ratios=[(1, 1)])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<|image|>Describe"], "images": [_make_image()]}


class TestPixtralProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.pixtral.processing_pixtral import PixtralProcessor

        p = PixtralProcessor.__new__(PixtralProcessor)
        p.patch_size = 16
        p.spatial_merge_size = 1
        p.image_token = "[IMG]"
        p.image_break_token = "[IMG_BREAK]"
        p.image_end_token = "[IMG_END]"
        p.image_token_id = 100
        p.image_break_token_id = 103
        p.image_end_token_id = 104
        p.image_processor = _mock_ip(image_sizes=[[(224, 224)]])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["[IMG]Describe"], "images": [[_make_image()]]}


class TestMistral3Processor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor

        p = Mistral3Processor.__new__(Mistral3Processor)
        p.patch_size = 16
        p.spatial_merge_size = 1
        p.image_token = "[IMG]"
        p.image_break_token = "[IMG_BREAK]"
        p.image_end_token = "[IMG_END]"
        p.image_token_id = 100
        p.image_break_token_id = 103
        p.image_end_token_id = 104
        p.image_processor = _mock_ip(image_sizes=[[(224, 224)]])
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["[IMG]Describe"], "images": [[_make_image()]]}

    def test_from_pretrained_prefers_model_geometry_over_processor_config(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "processor_config.json").write_text(
                json.dumps(
                    {
                        "patch_size": 16,
                        "spatial_merge_size": 1,
                        "image_token": "[IMG]",
                        "image_break_token": "[IMG_BREAK]",
                        "image_end_token": "[IMG_END]",
                    }
                )
            )
            (path / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "patch_size": {"height": 14, "width": 14},
                        "image_processor_type": "PixtralImageProcessor",
                    }
                )
            )
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "mistral3",
                        "spatial_merge_size": 2,
                        "vision_config": {"patch_size": 14},
                    }
                )
            )

            class DummyImageProcessor:
                model_input_names = ["pixel_values"]

            def _fake_init(
                self,
                image_processor=None,
                tokenizer=None,
                patch_size=16,
                spatial_merge_size=1,
                image_token="[IMG]",
                image_break_token="[IMG_BREAK]",
                image_end_token="[IMG_END]",
                chat_template=None,
                **kwargs,
            ):
                self.image_processor = image_processor
                self.tokenizer = tokenizer
                self.patch_size = patch_size
                self.spatial_merge_size = spatial_merge_size
                self.image_token = image_token
                self.image_break_token = image_break_token
                self.image_end_token = image_end_token
                self.chat_template = chat_template

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=_mock_tokenizer(),
                ),
                patch(
                    "mlx_vlm.models.mistral3.processing_mistral3._load_mistral3_image_processor",
                    return_value=DummyImageProcessor(),
                ),
                patch.object(Mistral3Processor, "__init__", _fake_init),
            ):
                processor = Mistral3Processor.from_pretrained(tmpdir)

        self.assertEqual(processor.patch_size, 14)
        self.assertEqual(processor.spatial_merge_size, 2)
        self.assertEqual(processor.image_token, "[IMG]")
        self.assertEqual(processor.image_break_token, "[IMG_BREAK]")
        self.assertEqual(processor.image_end_token, "[IMG_END]")


class TestMultiModalityProcessor(_ProcessorTestBase, unittest.TestCase):
    def _make_processor(self):
        from mlx_vlm.models.multi_modality.processing_multi_modality import (
            MultiModalityProcessor,
        )

        p = MultiModalityProcessor.__new__(MultiModalityProcessor)
        p.image_token = "<image>"
        p.num_image_tokens = 4
        p.image_processor = _mock_ip()
        p.tokenizer = _mock_tokenizer()
        return p

    def _image_call_args(self):
        return {"text": ["<image> Cats"], "images": [_make_image()]}


class TestErnie4_5VLProcessor(_ProcessorTestBase, unittest.TestCase):
    """Test ERNIE 4.5 VL processor components."""

    def _make_processor(self):
        from mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl import (
            Ernie4_5_VLProcessor,
        )

        p = Ernie4_5_VLProcessor.__new__(Ernie4_5_VLProcessor)
        p.spatial_conv_size = 2
        p.temporal_conv_size = 2

        p.image_processor = type(
            "IP",
            (),
            {
                "model_input_names": ["pixel_values"],
                "__call__": lambda self, images, **kw: {
                    "pixel_values": np.random.randn(1, 3, 224, 224).astype(np.float32),
                    "image_grid_thw": np.array([[1, 16, 16]], dtype=np.int64),
                },
            },
        )()

        tok = _mock_tokenizer()
        tok.encode = lambda text, **kw: list(range(10))
        tok.pad_token_id = 0
        p.tokenizer = tok
        return p

    def _image_call_args(self):
        return {
            "text": ["<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>Describe"],
            "images": [_make_image()],
        }

    def test_helper_functions(self):
        from mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl import (
            ceil_by_factor,
            floor_by_factor,
            round_by_factor,
            smart_resize,
        )

        self.assertEqual(round_by_factor(100, 28), 112)
        self.assertEqual(round_by_factor(56, 28), 56)
        self.assertEqual(round_by_factor(42, 28), 56)

        self.assertEqual(ceil_by_factor(100, 28), 112)
        self.assertEqual(ceil_by_factor(56, 28), 56)
        self.assertEqual(ceil_by_factor(57, 28), 84)

        self.assertEqual(floor_by_factor(100, 28), 84)
        self.assertEqual(floor_by_factor(56, 28), 56)
        self.assertEqual(floor_by_factor(55, 28), 28)

        h, w = smart_resize(224, 224, factor=28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

        h, w = smart_resize(10, 10, factor=28, min_pixels=56 * 56)
        self.assertGreaterEqual(h * w, 56 * 56)

        h, w = smart_resize(10000, 10000, factor=28, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(h * w, 28 * 28 * 1280)

    def test_image_processor(self):
        from mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl import (
            ImageProcessor,
        )

        processor = ImageProcessor()

        self.assertEqual(processor.patch_size, 14)
        self.assertEqual(processor.merge_size, 2)
        self.assertEqual(processor.factor, 28)

        (resized_h, resized_w), (grid_h, grid_w) = processor.get_smart_resize(224, 224)
        self.assertEqual(resized_h % 28, 0)
        self.assertEqual(resized_w % 28, 0)
        self.assertEqual(grid_h, resized_h // 14)
        self.assertEqual(grid_w, resized_w // 14)

        image = Image.new("RGB", (224, 224), color="red")
        result = processor.preprocess(image)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)
        self.assertEqual(result["image_grid_thw"].shape[0], 1)
        self.assertEqual(result["image_grid_thw"][0, 0], 1)

        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (448, 448), color="blue"),
        ]
        result = processor.preprocess(images)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)
        self.assertEqual(result["image_grid_thw"].shape[0], 2)

        img_array = np.random.rand(3, 224, 224).astype(np.float32)
        patches = processor._extract_patches(img_array, 16, 16)
        self.assertEqual(patches.shape, ((16 // 2) * (16 // 2) * 4, 3 * 14 * 14))

        image = Image.new("RGB", (224, 224), color="red")
        result = processor(images=image)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)

    def test_processor_class_attributes(self):
        from mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl import (
            Ernie4_5_VLProcessor,
        )

        self.assertEqual(Ernie4_5_VLProcessor.IMG_START, "<|IMAGE_START|>")
        self.assertEqual(Ernie4_5_VLProcessor.IMG_END, "<|IMAGE_END|>")
        self.assertEqual(
            Ernie4_5_VLProcessor.IMAGE_PLACEHOLDER, "<|IMAGE_PLACEHOLDER|>"
        )


class TestToMlxHelper(unittest.TestCase):
    def test_converts_lists_and_numpy(self):
        import mlx.core as mx

        from mlx_vlm.models.base import to_mlx

        result = to_mlx(
            {
                "ids": [[1, 2, 3]],
                "pv": np.zeros((1, 3, 4, 4)),
                "none_val": None,
                "str_val": "hello",
            }
        )
        self.assertIsInstance(result["ids"], mx.array)
        self.assertIsInstance(result["pv"], mx.array)
        self.assertIsNone(result["none_val"])
        self.assertEqual(result["str_val"], "hello")


class TestLfm2VlProcessorPatch(unittest.TestCase):
    def test_num_image_tokens_matches_pixel_unshuffle_padding(self):
        from mlx_vlm.models.lfm2_vl.processing_lfm2_vl import (
            _num_image_tokens_from_patch_grid,
        )

        self.assertEqual(_num_image_tokens_from_patch_grid(16, 16, 2), 64)
        self.assertEqual(_num_image_tokens_from_patch_grid(23, 43, 2), 264)
        self.assertEqual(_num_image_tokens_from_patch_grid(1, 1, 2), 1)
        self.assertEqual(_num_image_tokens_from_patch_grid(7, 9, 4), 6)

    def test_scalar_image_rows_and_cols_are_supported(self):
        from mlx_vlm.models.lfm2_vl.processing_lfm2_vl import _patched_call

        class DummyImageProcessor:
            def __init__(self):
                self.patch_size = 16
                self.downsample_factor = 2
                self.tile_size = 512
                self.max_image_tokens = 256
                self.min_image_tokens = 64
                self.encoder_patch_size = 16
                self.do_image_splitting = False
                self.use_thumbnail = False

            def fetch_images(self, images):
                return [images]

            def __call__(self, images, **kwargs):
                return {
                    "pixel_values": np.zeros((1, 16, 768), dtype=np.float32),
                    "image_rows": np.array([np.int64(1)]),
                    "image_cols": np.array([np.int64(1)]),
                    "image_sizes": [[416, 576]],
                }

        processor = type("DummyProcessor", (), {})()
        processor.image_processor = DummyImageProcessor()
        processor.tokenizer = _mock_tokenizer(image_token="<image>")
        processor.image_token = "<image>"
        processor.image_start_token = "<|image_start|>"
        processor.image_end_token = "<|image_end|>"
        processor._merge_kwargs = lambda *args, **kwargs: {
            "text_kwargs": {},
            "images_kwargs": {},
        }

        result = _patched_call(
            processor,
            images=_make_image(),
            text="<image>Describe this image",
        )

        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

    def test_from_pretrained_uses_slow_image_processor(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "image_processor_type": "Lfm2VlImageProcessorFast",
                        "do_resize": False,
                        "do_image_splitting": True,
                        "do_normalize": True,
                        "do_rescale": True,
                        "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5],
                        "max_num_patches": 1024,
                        "patch_size": 16,
                        "return_row_col_info": True,
                    }
                )
            )

            class DummySiglip2ImageProcessor:
                def __init__(self, **kwargs):
                    self.do_resize = kwargs.get("do_resize", True)
                    self.do_image_splitting = kwargs.get("do_image_splitting", False)
                    self.image_mean = kwargs.get("image_mean")
                    self.image_std = kwargs.get("image_std")
                    self.max_num_patches = kwargs.get("max_num_patches")
                    self.patch_size = kwargs.get("patch_size")

            def _fake_init(
                self, image_processor, tokenizer, chat_template=None, **kwargs
            ):
                self.image_processor = image_processor
                self.tokenizer = tokenizer
                self.chat_template = chat_template

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=_mock_tokenizer(),
                ),
                patch(
                    "mlx_vlm.models.lfm2_vl.processing_lfm2_vl.Siglip2ImageProcessor",
                    DummySiglip2ImageProcessor,
                    create=True,
                ),
                patch(
                    "mlx_vlm.models.lfm2_vl.processing_lfm2_vl._SLOW_PROCESSOR_AVAILABLE",
                    True,
                ),
                patch(
                    "mlx_vlm.models.lfm2_vl.processing_lfm2_vl._original_init",
                    _fake_init,
                ),
            ):
                processor = Lfm2VlProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(processor.image_processor, DummySiglip2ImageProcessor)
        self.assertTrue(processor.image_processor.do_resize)
        self.assertFalse(processor.image_processor.do_image_splitting)


# ── AutoProcessor patch tests ─────────────────────────────────────────────────


def _assert_patch_intercepts(test_case, model_type, module_path, cls_name):
    """Verify the patch routes AutoProcessor to the custom processor class."""
    import importlib
    import json
    import tempfile
    from pathlib import Path

    from transformers import AutoProcessor

    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name, None)
    test_case.assertIsNotNone(cls, f"{cls_name} not found in {module_path}")
    test_case.assertTrue(hasattr(cls, "from_pretrained"))

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "config.json").write_text(
            json.dumps({"model_type": model_type})
        )
        try:
            AutoProcessor.from_pretrained(tmpdir)
        except Exception as e:
            err = str(e).lower()
            test_case.assertNotIn(
                "has no attribute start_image_token",
                err,
                f"{model_type}: patch did not intercept, fell through to HF",
            )


class TestInternVLChatPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "internvl_chat",
            "mlx_vlm.models.internvl_chat",
            "InternVLChatProcessor",
        )


class TestMolmoPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self, "molmo", "mlx_vlm.models.molmo.processing_molmo", "MolmoProcessor"
        )


class TestKimiVLPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "kimi_vl",
            "mlx_vlm.models.kimi_vl.processing_kimi_vl",
            "KimiVLProcessor",
        )


class TestPhi3VPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self, "phi3_v", "mlx_vlm.models.phi3_v.processing_phi3_v", "Phi3VProcessor"
        )


class TestHunYuanVLPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "hunyuan_vl",
            "mlx_vlm.models.hunyuan_vl.processing_hunyuan_vl",
            "HunYuanVLProcessor",
        )


class TestLfm2VlPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        import importlib
        import json
        import tempfile
        from pathlib import Path

        from transformers import AutoProcessor

        importlib.import_module("mlx_vlm.models.lfm2_vl")

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"model_type": "lfm2_vl"})
            )
            with self.assertRaises(Exception) as cm:
                AutoProcessor.from_pretrained(tmpdir)

            self.assertNotIn("requires `torchvision`", str(cm.exception).lower())


class TestErnie4_5VLPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "ernie4_5_moe_vl",
            "mlx_vlm.models.ernie4_5_moe_vl",
            "Ernie4_5_VLProcessor",
        )


class TestQwen3_5Patch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "qwen3_5",
            "mlx_vlm.models.qwen3_vl.processing_qwen3_vl",
            "Qwen3VLProcessor",
        )


class TestQwen3_5MoePatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "qwen3_5_moe",
            "mlx_vlm.models.qwen3_vl.processing_qwen3_vl",
            "Qwen3VLProcessor",
        )


class TestDotsVLPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "dots_ocr",
            "mlx_vlm.models.dots_ocr.processing_dots_ocr",
            "DotsVLProcessor",
        )


class TestPatchChainsForUnknownModelType(unittest.TestCase):
    def test_falls_through(self):
        import importlib
        import json
        import tempfile
        from pathlib import Path

        from transformers import AutoProcessor

        importlib.import_module("mlx_vlm.models.internvl_chat")

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"model_type": "some_unknown_model_xyz"})
            )
            with self.assertRaises(Exception):
                AutoProcessor.from_pretrained(tmpdir)


if __name__ == "__main__":
    unittest.main()
