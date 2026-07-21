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


class TestGemma4UnifiedProcessor(unittest.TestCase):
    # Test fixtures

    class _Tokenizer:
        model_input_names = ["input_ids", "attention_mask"]
        bos_token = "<bos>"
        eos_token = "<eos>"
        pad_token = "<pad>"
        pad_token_id = 0
        image_token = "<|image|>"
        image_token_id = 100
        boi_token = "<boi>"
        eoi_token = "<eoi>"
        audio_token = "<|audio|>"
        audio_token_id = 101
        boa_token = "<boa>"
        eoa_token = "<eoa>"
        video_token = "<|video|>"
        video_token_id = 102
        chat_template = "mock"

        def __init__(self):
            self.last_text = None

        @property
        def init_kwargs(self):
            return {}

        def convert_tokens_to_ids(self, token):
            if isinstance(token, list):
                return [self.convert_tokens_to_ids(t) for t in token]
            return {
                self.image_token: self.image_token_id,
                self.audio_token: self.audio_token_id,
                self.video_token: self.video_token_id,
            }.get(token, 0)

        def add_special_tokens(self, tokens):
            return None

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        ):
            parts = ["<bos>"]
            for message in messages:
                content = message["content"]
                if not isinstance(content, list):
                    parts.append(str(content))
                    continue
                for item in content:
                    if item["type"] == "image":
                        parts.append(self.image_token)
                    elif item["type"] == "audio":
                        parts.append(self.audio_token)
                    elif item["type"] == "video":
                        parts.append(self.video_token)
                    else:
                        parts.append(item.get("text", item.get("content", "")))
            if add_generation_prompt:
                parts.append("<assistant>")
            rendered = "".join(parts)
            return self(rendered) if tokenize else rendered

        def __call__(self, text, **kwargs):
            self.last_text = text
            texts = [text] if isinstance(text, str) else text
            special_tokens = {
                self.image_token: self.image_token_id,
                self.audio_token: self.audio_token_id,
                self.video_token: self.video_token_id,
            }
            input_ids = []
            attention_mask = []
            for item in texts:
                ids = []
                i = 0
                while i < len(item):
                    matched = False
                    for token, token_id in special_tokens.items():
                        if token and item.startswith(token, i):
                            ids.append(token_id)
                            i += len(token)
                            matched = True
                            break
                    if not matched:
                        ids.append((ord(item[i]) % 50) + 1)
                        i += 1
                input_ids.append(ids)
                attention_mask.append([1] * len(ids))
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    # Processor helpers

    def _make_gemma4_unified_processor(
        self, image_processor=None, video_processor=None
    ):
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor,
            Gemma4UnifiedProcessor,
        )

        tokenizer = self._Tokenizer()
        if image_processor is None:
            image_processor = Gemma4UnifiedImageProcessor(
                patch_size=2,
                pooling_kernel_size=2,
                max_soft_tokens=4,
                do_resize=False,
                do_rescale=False,
            )
        processor = Gemma4UnifiedProcessor.__new__(Gemma4UnifiedProcessor)
        processor.tokenizer = tokenizer
        processor.image_processor = image_processor
        processor.feature_extractor = None
        processor.video_processor = video_processor
        processor.image_seq_length = 4
        processor.audio_seq_length = 750
        processor.audio_ms_per_token = 40
        processor.image_token_id = tokenizer.image_token_id
        processor.boi_token = tokenizer.boi_token
        processor.eoi_token = tokenizer.eoi_token
        processor.image_token = tokenizer.image_token
        processor.audio_token_id = tokenizer.audio_token_id
        processor.audio_token = tokenizer.audio_token
        processor.boa_token = tokenizer.boa_token
        processor.eoa_token = tokenizer.eoa_token
        processor.video_token = tokenizer.video_token
        processor.video_token_id = tokenizer.video_token_id
        processor.full_image_sequence = (
            tokenizer.boi_token + tokenizer.image_token * 4 + tokenizer.eoi_token
        )
        processor.full_audio_sequence = (
            tokenizer.boa_token + tokenizer.audio_token * 750 + tokenizer.eoa_token
        )
        return processor, tokenizer

    # Image and video patch processors

    def test_image_processor_outputs_merged_patches_and_positions(self):
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor,
        )

        processor = Gemma4UnifiedImageProcessor(
            patch_size=2,
            pooling_kernel_size=2,
            max_soft_tokens=4,
            do_resize=False,
            do_rescale=False,
        )
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

        data, num_soft_tokens = processor(image)

        self.assertEqual(data["pixel_values"].shape, (1, 4, 48))
        self.assertEqual(data["image_position_ids"].shape, (1, 4, 2))
        self.assertEqual(num_soft_tokens, [4])
        self.assertEqual(
            data["image_position_ids"][0].tolist(),
            [[0, 0], [1, 0], [0, 1], [1, 1]],
        )

    def test_video_processor_outputs_merged_patches_and_positions(self):
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedVideoProcessor,
        )

        processor = Gemma4UnifiedVideoProcessor(
            patch_size=2,
            pooling_kernel_size=2,
            max_soft_tokens=70,
            do_resize=False,
            do_rescale=False,
        )
        video = np.zeros((2, 3, 4, 8), dtype=np.uint8)

        data = processor([video], fps=[1.0])

        self.assertEqual(data["pixel_values_videos"].shape, (2, 70, 48))
        self.assertEqual(data["video_position_ids"].shape, (2, 70, 2))
        self.assertEqual(data["num_frames_per_video"], [2])
        self.assertEqual(data["num_soft_tokens_per_frame"], [2])
        self.assertEqual(
            data["video_position_ids"][0, :4].tolist(),
            [[0, 0], [1, 0], [-1, -1], [-1, -1]],
        )
        self.assertTrue(np.all(data["video_position_ids"][0, 2:] == -1))

    def test_gemma4_video_processor_outputs_padded_patches_and_positions(self):
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4VideoProcessor

        processor = Gemma4VideoProcessor(
            patch_size=2,
            pooling_kernel_size=2,
            max_soft_tokens=70,
            do_resize=False,
            do_rescale=False,
        )
        video = np.zeros((2, 3, 4, 8), dtype=np.uint8)

        data = processor([video], fps=[1.0])

        self.assertEqual(data["pixel_values_videos"].shape, (1, 2, 280, 12))
        self.assertEqual(data["video_position_ids"].shape, (1, 2, 280, 2))
        self.assertEqual(data["num_frames_per_video"], [2])
        self.assertEqual(data["num_soft_tokens_per_frame"], [2])
        self.assertEqual(data["frame_timestamps"], [[0.0, 1.0]])
        self.assertEqual(
            data["video_position_ids"][0, 0, :8].tolist(),
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [3, 1],
            ],
        )
        self.assertTrue(np.all(data["video_position_ids"][0, 0, 8:] == -1))

    def test_video_processor_tolerates_extra_hf_config_keys(self):
        # Regression: extra HF config keys must be ignored, not rejected.
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4VideoProcessor
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedVideoProcessor,
        )

        # The edited base class, and the unified subclass that delegates to it.
        for cls in (Gemma4VideoProcessor, Gemma4UnifiedVideoProcessor):
            processor = cls(
                patch_size=16,
                pooling_kernel_size=3,
                max_soft_tokens=70,
                num_frames=32,
                do_rescale=True,
                rescale_factor=1 / 255,
                do_normalize=True,
                image_mean=[0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0],
                # extra HF keys the processor must ignore
                do_convert_rgb=True,
                do_sample_frames=True,
                resample=3,
                return_metadata=False,
            )

            self.assertEqual(processor.max_soft_tokens, 70)
            self.assertEqual(processor.num_frames, 32)

    # Processor construction and audio feature extraction

    def test_processor_init_declares_video_processor_attribute(self):
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor,
            Gemma4UnifiedProcessor,
            Gemma4UnifiedVideoProcessor,
        )

        self.assertIn("video_processor", Gemma4UnifiedProcessor.get_attributes())

        processor = Gemma4UnifiedProcessor(
            image_processor=Gemma4UnifiedImageProcessor(
                patch_size=2,
                pooling_kernel_size=2,
                max_soft_tokens=4,
                do_resize=False,
                do_rescale=False,
            ),
            tokenizer=self._Tokenizer(),
            image_seq_length=4,
        )

        self.assertIsInstance(processor.video_processor, Gemma4UnifiedVideoProcessor)

    def test_audio_feature_extractor_chunks_waveforms(self):
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedAudioFeatureExtractor,
        )

        extractor = Gemma4UnifiedAudioFeatureExtractor(
            audio_samples_per_token=4,
            feature_size=4,
        )

        result = extractor(
            [
                np.arange(6, dtype=np.float32),
                np.arange(9, dtype=np.float32),
            ]
        )

        self.assertEqual(result["input_features"].shape, (2, 3, 4))
        self.assertEqual(
            result["input_features_mask"].tolist(),
            [[True, True, False], [True, True, True]],
        )

    # Multimodal chat/template integration

    def test_apply_chat_template_returns_multimodal_mlx_inputs(self):
        import mlx.core as mx

        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor,
        )

        processor, tokenizer = self._make_gemma4_unified_processor(
            image_processor=Gemma4UnifiedImageProcessor(
                patch_size=2,
                pooling_kernel_size=2,
                max_soft_tokens=4,
                do_resize=False,
                do_rescale=False,
            ),
        )
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]

        result = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="mlx",
            enable_thinking=False,
        )

        self.assertIsInstance(result["input_ids"], mx.array)
        self.assertIsInstance(result["pixel_values"], mx.array)
        self.assertIn("mm_token_type_ids", result)
        self.assertEqual(int(mx.sum(result["mm_token_type_ids"] == 1).item()), 4)
        self.assertEqual(result["pixel_values"].shape, (1, 4, 48))
        self.assertEqual(result["image_position_ids"].shape, (1, 4, 2))
        self.assertIn(
            "<boi><|image|><|image|><|image|><|image|><eoi>", tokenizer.last_text[0]
        )

    def test_call_returns_patchified_video_inputs(self):
        import mlx.core as mx

        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedVideoProcessor,
        )

        processor, tokenizer = self._make_gemma4_unified_processor(
            video_processor=Gemma4UnifiedVideoProcessor(
                patch_size=2,
                pooling_kernel_size=2,
                max_soft_tokens=70,
                do_resize=False,
                do_rescale=False,
            ),
        )
        video = np.zeros((2, 3, 4, 8), dtype=np.uint8)

        result = processor(
            text=[tokenizer.video_token + "describe"],
            videos=[video],
            fps=[1.0],
        )

        self.assertIsInstance(result["pixel_values_videos"], mx.array)
        self.assertEqual(result["pixel_values_videos"].shape, (2, 70, 48))
        self.assertEqual(result["video_position_ids"].shape, (2, 70, 2))
        self.assertEqual(int(mx.sum(result["mm_token_type_ids"] == 2).item()), 4)
        self.assertIn("<boi><|video|><|video|><eoi>", tokenizer.last_text[0])

    def test_apply_chat_template_renders_media_placeholder_without_tokenizing(self):
        processor, _ = self._make_gemma4_unified_processor()
        rendered = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "https://example.com/rabbit.jpg"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            tokenize=False,
            enable_thinking=False,
        )

        self.assertIn("<|image|>", rendered)

    def test_apply_chat_template_renders_video_placeholder_without_tokenizing(self):
        processor, _ = self._make_gemma4_unified_processor()
        rendered = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "clip.mp4"},
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
            tokenize=False,
            enable_thinking=False,
        )

        self.assertIn("<|video|>", rendered)

    def test_call_returns_hf_compatible_mm_token_type_ids(self):
        processor, tokenizer = self._make_gemma4_unified_processor()

        result = processor(
            text=[
                tokenizer.image_token
                + tokenizer.video_token
                + tokenizer.audio_token
                + "describe"
            ]
        )

        self.assertEqual(result["mm_token_type_ids"].tolist()[0][:3], [1, 2, 3])

    # Utility integration

    def test_prepare_inputs_respects_mm_token_type_ids_override(self):
        from mlx_vlm.utils import prepare_inputs

        processor, tokenizer = self._make_gemma4_unified_processor()
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

        result = prepare_inputs(
            processor,
            images=[image],
            prompts=tokenizer.image_token + "describe",
            return_mm_token_type_ids=False,
        )

        self.assertNotIn("mm_token_type_ids", result)


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


class TestMiniCPMVProcessor(unittest.TestCase):
    class _Tokenizer:
        model_input_names = ["input_ids", "attention_mask"]
        eos_token = "<eos>"
        pad_token = "<pad>"
        pad_token_id = 0
        unk_token_id = 99
        image_token = "<|image_pad|>"
        image_token_id = 101
        video_token = "<|video_pad|>"
        video_token_id = 102

        _ids = {
            "<image>": 11,
            "</image>": 12,
            "<slice>": 13,
            "</slice>": 14,
            "<image_id>": 15,
            "</image_id>": 16,
            "<unk>": 99,
            "<|image_pad|>": 101,
            "<|video_pad|>": 102,
            "<|listen|>": 99,
            "\n": 2,
        }

        def convert_tokens_to_ids(self, token):
            return self._ids.get(token, 1)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            ids = []
            specials = sorted(self._ids, key=len, reverse=True)
            index = 0
            while index < len(text):
                for token in specials:
                    if text.startswith(token, index):
                        ids.append(self._ids[token])
                        index += len(token)
                        break
                else:
                    if not text[index].isspace():
                        ids.append(1)
                    index += 1
            return ids

        def build_inputs_with_special_tokens(self, ids):
            return ids

        def batch_decode(self, ids, **kwargs):
            return ["decoded"] * len(ids)

        def decode(self, ids, **kwargs):
            return "decoded"

    def _make_processor(self):
        from mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6 import (
            MiniCPMVImageProcessor,
            MiniCPMVProcessor,
            MiniCPMVVideoProcessor,
        )

        p = MiniCPMVProcessor.__new__(MiniCPMVProcessor)
        p.image_processor = MiniCPMVImageProcessor(
            slice_mode=False,
            use_image_id=False,
            scale_resolution=56,
            patch_size=14,
        )
        p.video_processor = MiniCPMVVideoProcessor(
            slice_mode=False,
            use_image_id=False,
            scale_resolution=56,
            patch_size=14,
        )
        p.tokenizer = self._Tokenizer()
        p.image_feature_size = p.image_processor.image_feature_size
        p._ensure_tokenizer_attrs()
        p.image_token = p.tokenizer.image_token
        p.image_token_id = p.tokenizer.image_token_id
        p.video_token = p.tokenizer.video_token
        p.video_token_id = p.tokenizer.video_token_id
        return p

    def test_video_marker_expands_to_frame_bounds(self):
        p = self._make_processor()
        video = np.zeros((2, 3, 16, 16), dtype=np.uint8)

        result = p(
            text=["<|video_pad|> Describe this."],
            videos=[video],
            slice_mode=False,
            max_num_frames=2,
            padding=False,
        )

        self.assertEqual(len(result["pixel_values"][0]), 2)
        self.assertEqual(result["tgt_sizes"][0].shape, (2, 2))
        self.assertEqual(result["image_bound"][0].shape, (2, 2))
        self.assertEqual(result["num_frames_per_video"], [[2]])
        self.assertEqual(result["num_patches_per_frame"], [[1, 1]])
        for start, end in result["image_bound"][0]:
            self.assertTrue(np.all(result["input_ids"][0, start:end] == 102))

    def test_prompt_utils_routes_minicpm_video_messages(self):
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = apply_chat_template(
            processor=None,
            config={"model_type": "minicpmv4_6"},
            prompt="Describe this video",
            return_messages=True,
            video=["clip.mp4"],
            fps=1,
        )

        self.assertEqual(messages[0]["content"][0]["type"], "video")
        self.assertEqual(messages[0]["content"][1]["type"], "text")


class TestGlmOcrProcessor(unittest.TestCase):
    def test_from_pretrained_uses_local_numpy_image_processor(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.glm_ocr.processing import (
            Glm46VImageProcessor,
            GlmOcrProcessor,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "processor_config.json").write_text(
                json.dumps(
                    {
                        "image_processor": {
                            "patch_size": 14,
                            "temporal_patch_size": 2,
                            "merge_size": 2,
                            "size": {
                                "shortest_edge": 12544,
                                "longest_edge": 9633792,
                            },
                            "image_mean": [0.48145466, 0.4578275, 0.40821073],
                            "image_std": [0.26862954, 0.26130258, 0.27577711],
                        },
                        "processor_class": "GlmOcrProcessor",
                    }
                )
            )

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=_mock_tokenizer(image_token="<|image|>"),
                ),
                patch("mlx_vlm.models.base.load_chat_template"),
            ):
                processor = GlmOcrProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(processor.image_processor, Glm46VImageProcessor)
        self.assertEqual(processor.image_processor.patch_size, 14)
        self.assertEqual(processor.image_processor.max_pixels, 9633792)

    def test_image_processor_matches_glm_patch_shape(self):
        from mlx_vlm.models.glm_ocr.processing import Glm46VImageProcessor

        processor = Glm46VImageProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=14 * 14 * 2 * 2,
            max_pixels=14 * 14 * 2 * 2 * 64,
        )

        image = Image.fromarray(np.zeros((28, 56, 3), dtype=np.uint8))
        result = processor(images=image)

        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 4]])
        self.assertEqual(result["pixel_values"].shape, (8, 1176))

    def test_image_processor_matches_transformers_backend_bit_exactly(self):
        try:
            import torch
            from transformers.models.glm46v.image_processing_glm46v import (
                Glm46VImageProcessor as HFGlm46VImageProcessor,
            )
        except Exception as exc:
            self.skipTest(f"Transformers torch image backend unavailable: {exc}")

        from mlx_vlm.models.glm_ocr.processing import Glm46VImageProcessor

        image = Image.fromarray(np.zeros((28, 56, 3), dtype=np.uint8))
        hf_processor = HFGlm46VImageProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            size={
                "shortest_edge": 14 * 14 * 2 * 2,
                "longest_edge": 14 * 14 * 2 * 2 * 64,
            },
        )
        processor = Glm46VImageProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=14 * 14 * 2 * 2,
            max_pixels=14 * 14 * 2 * 2 * 64,
        )

        expected = hf_processor(images=image)
        actual = processor(images=image)

        expected_pixels = expected["pixel_values"]
        if isinstance(expected_pixels, torch.Tensor):
            expected_pixels = expected_pixels.detach().cpu().numpy()

        self.assertTrue(np.array_equal(expected_pixels, actual["pixel_values"]))
        self.assertTrue(
            np.array_equal(
                expected["image_grid_thw"].detach().cpu().numpy(),
                actual["image_grid_thw"],
            )
        )


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

    def test_forwards_image_pixel_kwargs_to_image_processor(self):
        p = self._make_processor()
        seen = {}

        class ImageProcessor:
            model_input_names = ["pixel_values"]
            merge_size = 2

            def __call__(self, images=None, **kwargs):
                seen.update(kwargs)
                return {
                    "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32),
                    "image_grid_thw": np.array([[1, 16, 16]], dtype=np.int64),
                }

        class Tokenizer:
            model_input_names = ["input_ids", "attention_mask"]

            def __call__(self, text, **kwargs):
                if "max_pixels" in kwargs:
                    raise AssertionError("max_pixels leaked into tokenizer kwargs")
                return {
                    "input_ids": [list(range(10)) for _ in text],
                    "attention_mask": [[1] * 10 for _ in text],
                }

        p.image_processor = ImageProcessor()
        p.tokenizer = Tokenizer()

        p(
            text=["<|image_pad|> Describe"],
            images=[_make_image()],
            max_pixels=1280 * 28 * 28,
        )

        self.assertEqual(seen["max_pixels"], 1280 * 28 * 28)


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

    def test_image_processor_honors_per_call_max_pixels(self):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        image = np.zeros((3, 1200, 1200), dtype=np.uint8)
        processor = Qwen3VLImageProcessor(
            patch_size=14,
            merge_size=2,
            max_pixels=12845056,
        )

        default_grid = processor(images=[image])["image_grid_thw"][0]
        capped_grid = processor(
            images=[image],
            max_pixels=1280 * 28 * 28,
        )[
            "image_grid_thw"
        ][0]

        self.assertGreater(
            default_grid[1] * default_grid[2],
            capped_grid[1] * capped_grid[2],
        )
        self.assertLessEqual(
            capped_grid[1] * 14 * capped_grid[2] * 14,
            1280 * 28 * 28,
        )

    def test_video_processor_accepts_pil_frame_lists(self):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLVideoProcessor

        frames = [
            Image.new("RGB", (224, 224), color=(i * 40, 128, 128)) for i in range(4)
        ]
        processor = Qwen3VLVideoProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            do_rescale=False,
            do_normalize=False,
        )

        output = processor(videos=[frames])

        np.testing.assert_array_equal(
            output["video_grid_thw"], np.array([[2, 16, 16]], dtype=np.int64)
        )
        self.assertEqual(output["pixel_values_videos"].shape, (512, 1176))

        direct_output = processor(videos=frames)
        np.testing.assert_array_equal(
            direct_output["video_grid_thw"], np.array([[2, 16, 16]], dtype=np.int64)
        )
        self.assertEqual(direct_output["pixel_values_videos"].shape, (512, 1176))

    def test_video_processor_accepts_channel_last_arrays(self):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLVideoProcessor

        video = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        processor = Qwen3VLVideoProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            do_rescale=False,
            do_normalize=False,
        )

        output = processor(videos=[video])

        np.testing.assert_array_equal(
            output["video_grid_thw"], np.array([[2, 16, 16]], dtype=np.int64)
        )
        self.assertEqual(output["pixel_values_videos"].shape, (512, 1176))


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


class TestPixtralImageProcessor(unittest.TestCase):
    def test_preprocess_resizes_to_patch_multiple_and_pads(self):
        from mlx_vlm.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

        image_processor = PixtralImageProcessor(
            size={"longest_edge": 40},
            patch_size=14,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
        )
        wide = Image.fromarray(np.zeros((31, 55, 3), dtype=np.uint8))
        square = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))

        output = image_processor([[wide, square]])

        self.assertEqual(output["image_sizes"], [(28, 42), (28, 28)])
        self.assertEqual(output["pixel_values"].shape, (2, 3, 28, 42))

    def test_split_image_sizes_by_sample_handles_flat_sizes(self):
        from mlx_vlm.models.pixtral.image_processing_pixtral import (
            split_image_sizes_by_sample,
        )

        images = [[_make_image(), _make_image()], [_make_image()]]
        sizes = [(28, 42), (28, 28), (56, 56)]

        self.assertEqual(
            split_image_sizes_by_sample(sizes, images),
            [[(28, 42), (28, 28)], [(56, 56)]],
        )


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

    def test_from_pretrained_uses_torch_free_pixtral_image_processor(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor
        from mlx_vlm.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

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
            self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
            self.image_break_token_id = tokenizer.convert_tokens_to_ids(
                image_break_token
            )
            self.image_end_token_id = tokenizer.convert_tokens_to_ids(image_end_token)
            self.chat_template = chat_template

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
                        "image_processor": {
                            "image_processor_type": "PixtralImageProcessorFast",
                            "patch_size": 14,
                            "size": {"longest_edge": 64},
                        },
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

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=_mock_tokenizer(),
                ),
                patch.object(Mistral3Processor, "__init__", _fake_init),
            ):
                processor = Mistral3Processor.from_pretrained(
                    tmpdir, trust_remote_code=True
                )

        self.assertIsInstance(processor.image_processor, PixtralImageProcessor)
        self.assertEqual(processor.patch_size, 14)
        self.assertEqual(processor.spatial_merge_size, 2)

        output = processor(text=["[IMG]Describe"], images=[[_make_image()]])
        self.assertEqual(output["pixel_values"].shape[0], 1)
        self.assertEqual(output["pixel_values"].shape[1], 3)
        self.assertEqual(int(output["image_sizes"][0, 0].item()) % 28, 0)
        self.assertEqual(int(output["image_sizes"][0, 1].item()) % 28, 0)


class TestStep3VLProcessor(unittest.TestCase):
    def test_from_pretrained_uses_fixed_tokenizer(self):
        from mlx_vlm.models.step3p7.processing_step3p7 import Step3VLProcessor
        from mlx_vlm.tokenizer_utils import BPEStreamingDetokenizer

        tokenizer = _mock_tokenizer(
            chat_template="template",
            vocab={"Got": 0, "Ġit": 1},
            backend_tokenizer=SimpleNamespace(decoder="bad"),
        )

        def _fake_init(self, tokenizer=None, chat_template=None, **kwargs):
            self.tokenizer = tokenizer
            self.chat_template = chat_template

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
            ) as from_pretrained,
            patch.object(Step3VLProcessor, "__init__", _fake_init),
        ):
            processor = Step3VLProcessor.from_pretrained(
                "step-model", trust_remote_code=True
            )

        from_pretrained.assert_called_once_with(
            "step-model",
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        self.assertIs(processor.tokenizer, tokenizer)
        self.assertIs(processor.detokenizer_class, BPEStreamingDetokenizer)
        self.assertIn("ByteLevel", repr(tokenizer.backend_tokenizer.decoder))

        processor.detokenizer = object()
        processor.detokenizer.add_token(0)
        processor.detokenizer.add_token(1)
        processor.detokenizer.finalize()
        self.assertEqual(processor.detokenizer.text, "Got it")


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


class TestPaddleOCRVLProcessor(unittest.TestCase):
    """Regression tests for PaddleOCR-VL processor loading."""

    def test_from_pretrained_loads_preprocessor_geometry(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl import (
            PaddleOCRVLProcessor,
        )

        def _fake_init(
            self,
            image_processor=None,
            tokenizer=None,
            chat_template=None,
            **kwargs,
        ):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.chat_template = chat_template
            self.image_token = (
                "<|IMAGE_PLACEHOLDER|>"
                if not hasattr(tokenizer, "image_token")
                else tokenizer.image_token
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "config.json").write_text(
                json.dumps({"model_type": "paddleocr_vl"})
            )
            (path / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "min_pixels": 64,
                        "max_pixels": 4096,
                        "patch_size": 16,
                        "temporal_patch_size": 2,
                        "merge_size": 4,
                        "image_mean": [0.1, 0.2, 0.3],
                        "image_std": [0.9, 0.8, 0.7],
                        "do_convert_rgb": False,
                    }
                )
            )

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=_mock_tokenizer(image_token="<paddle-image>"),
                ),
                patch.object(PaddleOCRVLProcessor, "__init__", _fake_init),
            ):
                processor = PaddleOCRVLProcessor.from_pretrained(tmpdir)

        self.assertEqual(processor.image_token, "<paddle-image>")
        self.assertEqual(processor.image_processor.min_pixels, 64)
        self.assertEqual(processor.image_processor.max_pixels, 4096)
        self.assertEqual(processor.image_processor.patch_size, 16)
        self.assertEqual(processor.image_processor.temporal_patch_size, 2)
        self.assertEqual(processor.image_processor.merge_size, 4)
        self.assertEqual(processor.image_processor.image_mean, [0.1, 0.2, 0.3])
        self.assertEqual(processor.image_processor.image_std, [0.9, 0.8, 0.7])
        self.assertFalse(processor.image_processor.do_convert_rgb)

    def test_load_image_processor_returns_none(self):
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.utils import load_image_processor

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "config.json").write_text(
                json.dumps({"model_type": "paddleocr_vl"})
            )
            image_processor = load_image_processor(path)

        self.assertIsNone(image_processor)


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

    def test_numpy_image_processor_outputs_packed_patches(self):
        from mlx_vlm.models.lfm2_vl.processing_lfm2_vl import (
            Lfm2VlNumpyImageProcessor,
            _num_image_tokens_from_patch_grid,
        )

        processor = Lfm2VlNumpyImageProcessor(
            encoder_patch_size=16,
            downsample_factor=2,
            min_image_tokens=64,
            max_image_tokens=256,
            max_num_patches=1024,
        )

        result = processor(_make_image(), return_tensors="np")

        self.assertEqual(result["pixel_values"].shape, (1, 1024, 768))
        self.assertEqual(result["pixel_attention_mask"].shape, (1, 1024))
        self.assertEqual(result["spatial_shapes"].tolist(), [[16, 16]])
        self.assertEqual(int(result["pixel_attention_mask"].sum()), 256)
        self.assertEqual(_num_image_tokens_from_patch_grid(16, 16, 2), 64)

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
            (Path(tmpdir) / "processor_config.json").write_text(
                json.dumps(
                    {
                        "image_processor": {
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
                        },
                        "processor_class": "Lfm2VlProcessor",
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
                ) as tokenizer_from_pretrained,
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
        tokenizer_from_pretrained.assert_called_once_with(
            tmpdir,
            trust_remote_code=False,
            local_files_only=True,
        )


class TestMolmoPointProcessor(unittest.TestCase):
    def test_processor_exposes_image_processor(self):
        from mlx_vlm.models.molmo_point.processing_molmo_point import (
            MolmoPointImageProcessor,
            MolmoPointProcessor,
        )

        processor = MolmoPointProcessor(_mock_tokenizer())

        self.assertIsInstance(processor.image_processor, MolmoPointImageProcessor)

    def test_processor_uses_image_processor_for_images(self):
        from mlx_vlm.models.molmo_point.processing_molmo_point import (
            IMAGE_PROMPT,
            MolmoPointProcessor,
        )

        class DummyImageProcessor:
            def preprocess(self, images):
                return {
                    "pixel_values": np.zeros((1, 729, 588), dtype=np.float32),
                    "image_token_pooling": np.zeros((1, 729), dtype=np.int64),
                    "image_grids": np.array([[1, 1, 1, 1]], dtype=np.int64),
                    "image_num_crops": np.array([1], dtype=np.int64),
                }

        processor = MolmoPointProcessor(
            _mock_tokenizer(bos_token_id=1, eos_token_id=2),
            image_processor=DummyImageProcessor(),
        )
        result = processor(text=IMAGE_PROMPT, images=_make_image())

        self.assertIn("pixel_values", result)
        self.assertIn("image_token_pooling", result)
        self.assertIn("image_grids", result)
        self.assertIn("image_num_crops", result)


class TestNemotronHNanoOmniProcessor(unittest.TestCase):
    def test_native_processor_handles_stripped_auto_map(self):
        import importlib
        import json
        import tempfile
        from pathlib import Path

        from mlx_vlm.utils import load_processor, prepare_inputs

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "NemotronH_Nano_Omni_Reasoning_V3"})
            )
            (model_dir / "processor_config.json").write_text(
                json.dumps(
                    {"processor_class": "NemotronH_Nano_Omni_Reasoning_V3Processor"}
                )
            )
            (model_dir / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "image_processor_type": (
                            "NemotronH_Nano_Omni_Reasoning_V3ImageProcessor"
                        ),
                        "patch_size": 16,
                        "downsample_ratio": 0.5,
                        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
                        "norm_std": [0.26862954, 0.26130258, 0.27577711],
                        "min_num_patches": 64,
                        "max_num_patches": 64,
                        "max_model_len": 128,
                    }
                )
            )

            importlib.import_module("mlx_vlm.models.nemotron_h_nano_omni")

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=_mock_tokenizer(image_token_id=18),
            ):
                processor = load_processor(tmpdir, add_detokenizer=False)

            result = prepare_inputs(
                processor,
                images=[_make_image()],
                prompts="<image>\nDescribe this image.",
                image_token_index=processor.image_token_id,
            )

        self.assertEqual(
            processor.__class__.__name__,
            "NemotronHNanoOmniProcessor",
        )
        self.assertIn("pixel_values", result)
        self.assertIn("num_tokens", result)
        self.assertGreater(int(result["num_tokens"][0].item()), 0)


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


class TestLagunaProcessor(unittest.TestCase):
    @staticmethod
    def _fast_tokenizer():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(
            WordLevel(
                {"<unk>": 0, "<eos>": 1, "<pad>": 2, "prompt": 3},
                unk_token="<unk>",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<pad>",
        )
        fast_tokenizer.chat_template = "template"
        return fast_tokenizer

    def test_from_pretrained_loads_fast_tokenizer_directly(self):
        from mlx_vlm.models.laguna.processing_laguna import LagunaProcessor

        tokenizer = self._fast_tokenizer()
        with patch(
            "mlx_vlm.models.laguna.processing_laguna."
            "PreTrainedTokenizerFast.from_pretrained",
            return_value=tokenizer,
        ) as from_pretrained:
            processor = LagunaProcessor.from_pretrained(
                "/tmp/model",
                processor_kwargs={"local_files_only": True},
                quantize_activations=True,
                trust_remote_code=True,
            )

        self.assertIs(processor.tokenizer, tokenizer)
        args, kwargs = from_pretrained.call_args
        self.assertEqual(args, ("/tmp/model",))
        self.assertTrue(kwargs["fix_mistral_regex"])
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertNotIn("processor_kwargs", kwargs)
        self.assertNotIn("quantize_activations", kwargs)

    def test_auto_processor_patch_intercepts_laguna(self):
        import importlib
        import json
        import tempfile
        from pathlib import Path

        from transformers import AutoProcessor

        from mlx_vlm.models.laguna.processing_laguna import LagunaProcessor

        importlib.import_module("mlx_vlm.models.laguna")

        tokenizer = self._fast_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "laguna",
                        "rope_parameters": {
                            "sliding_attention": {"rope_type": "yarn"}
                        },
                    }
                )
            )
            with patch(
                "mlx_vlm.models.laguna.processing_laguna."
                "PreTrainedTokenizerFast.from_pretrained",
                return_value=tokenizer,
            ):
                processor = AutoProcessor.from_pretrained(
                    tmpdir, quantize_activations=True
                )

        self.assertIsInstance(processor, LagunaProcessor)
        self.assertIs(processor.tokenizer, tokenizer)


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


class TestPaddleOCRVLPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "paddleocr_vl",
            "mlx_vlm.models.paddleocr_vl",
            "PaddleOCRVLProcessor",
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


class TestQwen3OmniMoePatch(unittest.TestCase):
    def test_patch_intercepts_without_hf_video_processor(self):
        import json
        import tempfile
        from pathlib import Path

        from transformers import AutoProcessor

        from mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
            Qwen3OmniMoeProcessor,
        )

        tokenizer = _mock_tokenizer(
            image_token="<|image_pad|>",
            audio_token="<|audio_pad|>",
            video_token="<|video_pad|>",
            vision_bos_token="<|vision_start|>",
            vision_eos_token="<|vision_end|>",
            audio_bos_token="<|audio_bos|>",
            audio_eos_token="<|audio_eos|>",
        )
        feature_extractor = type(
            "FE",
            (),
            {"model_input_names": ["input_features"], "sampling_rate": 16000},
        )()

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"model_type": "qwen3_omni_moe"}),
                encoding="utf-8",
            )
            (Path(tmpdir) / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "feature_extractor_type": "WhisperFeatureExtractor",
                        "image_processor_type": "Qwen2VLImageProcessor",
                        "processor_class": "Qwen3OmniMoeProcessor",
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "transformers.AutoFeatureExtractor.from_pretrained",
                    return_value=feature_extractor,
                ),
            ):
                processor = AutoProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(processor, Qwen3OmniMoeProcessor)
        self.assertEqual(
            type(processor.video_processor).__name__, "Qwen3VLVideoProcessor"
        )


class TestDotsVLPatch(unittest.TestCase):
    def test_patch_intercepts(self):
        _assert_patch_intercepts(
            self,
            "dots_ocr",
            "mlx_vlm.models.dots_ocr.processing_dots_ocr",
            "DotsVLProcessor",
        )


class TestDeepseekV4Processor(unittest.TestCase):
    class MockTokenizer:
        chat_template = None
        model_input_names = ["input_ids", "attention_mask"]

        def __call__(self, text, **kwargs):
            return {"input_ids": [0], "attention_mask": [1]}

        def apply_chat_template(self, *args, **kwargs):
            return "templated"

        def encode(self, text, **kwargs):
            return [0]

        def decode(self, ids, **kwargs):
            return "decoded"

        def batch_decode(self, ids, **kwargs):
            return ["decoded"] * len(ids)

    def test_loads_local_chat_template_jinja(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
            load_deepseek_v4_chat_template,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "chat_template.jinja").write_text(
                "{{ messages[0]['content'] }}",
                encoding="utf-8",
            )

            self.assertEqual(
                load_deepseek_v4_chat_template(tmpdir),
                "{{ messages[0]['content'] }}",
            )

    def test_from_pretrained_sets_local_chat_template(self):
        import tempfile
        from pathlib import Path

        from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
            DeepseekV4Processor,
        )

        tokenizer = self.MockTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "chat_template.jinja").write_text(
                "{{ messages[0]['content'] }}",
                encoding="utf-8",
            )
            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
                ),
                patch.object(
                    DeepseekV4Processor,
                    "check_argument_for_proper_class",
                    return_value=None,
                ),
            ):
                processor = DeepseekV4Processor.from_pretrained(tmpdir)

        self.assertEqual(processor.chat_template, "{{ messages[0]['content'] }}")

    def test_from_pretrained_prefers_explicit_chat_template(self):
        from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
            DeepseekV4Processor,
        )

        tokenizer = self.MockTokenizer()

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch.object(
                DeepseekV4Processor,
                "check_argument_for_proper_class",
                return_value=None,
            ),
        ):
            processor = DeepseekV4Processor.from_pretrained(
                "repo/name",
                chat_template="{{ explicit }}",
            )

        self.assertEqual(processor.chat_template, "{{ explicit }}")

    def test_from_pretrained_uses_default_chat_template_when_missing(self):
        import tempfile

        from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
            DEFAULT_CHAT_TEMPLATE,
            DeepseekV4Processor,
        )

        tokenizer = self.MockTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
                ),
                patch.object(
                    DeepseekV4Processor,
                    "check_argument_for_proper_class",
                    return_value=None,
                ),
            ):
                processor = DeepseekV4Processor.from_pretrained(tmpdir)

        self.assertEqual(processor.chat_template, DEFAULT_CHAT_TEMPLATE)
        self.assertIn("<｜Assistant｜></think>", processor.chat_template)

    def test_patch_intercepts(self):
        import json
        import tempfile
        from pathlib import Path

        from transformers import AutoProcessor

        from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
            DeepseekV4Processor,
        )

        tokenizer = self.MockTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"}),
                encoding="utf-8",
            )
            with (
                patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
                ),
                patch.object(
                    DeepseekV4Processor,
                    "check_argument_for_proper_class",
                    return_value=None,
                ),
            ):
                processor = AutoProcessor.from_pretrained(tmpdir)

        self.assertIsInstance(processor, DeepseekV4Processor)


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


class TestLocateAnythingProcessor(unittest.TestCase):
    def test_save_pretrained_round_trips_custom_config(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from transformers import PreTrainedTokenizerBase

        from mlx_vlm.models.locateanything.image_processing_locateanything import (
            LocateAnythingImageProcessor,
        )
        from mlx_vlm.models.locateanything.processing_locateanything import (
            LocateAnythingProcessor,
        )

        class DummyTokenizer(PreTrainedTokenizerBase):
            model_input_names = ["input_ids", "attention_mask"]
            vocab_files_names = {}

            def __init__(self, chat_template=None):
                super().__init__(chat_template=chat_template)
                self.eos_token = "<eos>"
                self.pad_token = "<pad>"

            def save_pretrained(self, save_directory, **kwargs):
                path = Path(save_directory) / "tokenizer_config.json"
                path.write_text(
                    json.dumps({"tokenizer_class": "DummyTokenizer"}),
                    encoding="utf-8",
                )
                return (str(path),)

            def batch_decode(self, *args, **kwargs):
                return []

            def decode(self, *args, **kwargs):
                return ""

            def convert_tokens_to_ids(self, token):
                return 1

            def __call__(self, *args, **kwargs):
                return {"input_ids": [[1]], "attention_mask": [[1]]}

        chat_template = "{{ messages }}"
        processor = LocateAnythingProcessor(
            image_processor=LocateAnythingImageProcessor(
                patch_size=28,
                merge_kernel_size=[2, 4],
                in_token_limit=1234,
            ),
            tokenizer=DummyTokenizer(chat_template=chat_template),
            chat_template=chat_template,
        )

        with TemporaryDirectory() as tmp:
            saved_files = processor.save_pretrained(tmp)

            processor_config = json.loads(
                (Path(tmp) / "processor_config.json").read_text(encoding="utf-8")
            )
            preprocessor_config = json.loads(
                (Path(tmp) / "preprocessor_config.json").read_text(encoding="utf-8")
            )
            chat_template_config = json.loads(
                (Path(tmp) / "chat_template.json").read_text(encoding="utf-8")
            )

            self.assertIn(str(Path(tmp) / "processor_config.json"), saved_files)
            self.assertEqual(
                processor_config["processor_class"],
                "LocateAnythingProcessor",
            )
            self.assertEqual(processor_config["chat_template"], chat_template)
            self.assertEqual(preprocessor_config["patch_size"], 28)
            self.assertEqual(preprocessor_config["merge_kernel_size"], [2, 4])
            self.assertEqual(preprocessor_config["in_token_limit"], 1234)
            self.assertEqual(chat_template_config["chat_template"], chat_template)

            with patch(
                "mlx_vlm.models.locateanything.processing_locateanything."
                "AutoTokenizer.from_pretrained",
                return_value=DummyTokenizer(),
            ):
                reloaded = LocateAnythingProcessor.from_pretrained(tmp)

            self.assertEqual(reloaded.image_processor.patch_size, 28)
            self.assertEqual(reloaded.image_processor.merge_kernel_size, [2, 4])
            self.assertEqual(reloaded.image_processor.in_token_limit, 1234)
            self.assertEqual(reloaded.chat_template, chat_template)
            self.assertEqual(reloaded.tokenizer.chat_template, chat_template)


class TestProcessorRegistration(unittest.TestCase):
    _AFFECTED_MODULES = (
        "mlx_vlm.models.glm4v.glm4v",
        "mlx_vlm.models.glm4v_moe.glm4v_moe",
        "mlx_vlm.models.deepseek_vl_v2.deepseek_vl_v2",
        "mlx_vlm.models.deepseekocr.deepseekocr",
        "mlx_vlm.models.deepseekocr_2.deepseekocr_2",
        "mlx_vlm.models.unlimited_ocr.unlimitedocr",
        "mlx_vlm.models.jina_vlm.jina_vlm",
    )

    def test_no_string_first_autoprocessor_register(self):
        import re
        from pathlib import Path

        import mlx_vlm

        models_dir = Path(mlx_vlm.__file__).parent / "models"
        pattern = re.compile(r"""AutoProcessor\.register\(\s*['"]""")
        offenders = [
            str(path.relative_to(models_dir))
            for path in models_dir.rglob("*.py")
            if pattern.search(path.read_text())
        ]
        self.assertEqual(offenders, [], f"string-first register calls: {offenders}")

    def test_affected_modules_import_cleanly(self):
        import importlib

        for module in self._AFFECTED_MODULES:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
