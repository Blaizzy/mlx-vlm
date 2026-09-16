"""Shared processor contracts, loader routing, and multimodal integration tests."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoProcessor, PreTrainedTokenizerBase, PreTrainedTokenizerFast

import mlx_vlm.models.aya_vision.processing_aya_vision as aya_vision
import mlx_vlm.models.deepseek_v4.processing_deepseek_v4 as deepseek
import mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma as diffusion
import mlx_vlm.models.dots_ocr.processing_dots_ocr as dots_ocr
import mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl as ernie
import mlx_vlm.models.gemma4.processing_gemma4 as g4
import mlx_vlm.models.gemma4_unified.processing_gemma4_unified as g4u
import mlx_vlm.models.glm4v_moe.processing as glm4v_moe
import mlx_vlm.models.glm_ocr.processing as glm_ocr
import mlx_vlm.models.idefics3.processing_idefics3 as idefics3
import mlx_vlm.models.kimi_k3.processing_kimi_k3 as kimi_k3
import mlx_vlm.models.kimi_vl.processing_kimi_vl as kimi_vl
import mlx_vlm.models.laguna.processing_laguna as laguna
import mlx_vlm.models.lfm2_vl.processing_lfm2_vl as lfm
import mlx_vlm.models.locateanything.image_processing_locateanything as locate_ip
import mlx_vlm.models.locateanything.processing_locateanything as locateanything
import mlx_vlm.models.mage_vl.config as mage_config
import mlx_vlm.models.mage_vl.mage_vl as mage_model
import mlx_vlm.models.mage_vl.processing_mage_vl as mage
import mlx_vlm.models.mage_vl.vision as mage_vision
import mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6 as minicpm
import mlx_vlm.models.mistral3.processing_mistral3 as mistral3
import mlx_vlm.models.mllama.processing_mllama as mllama
import mlx_vlm.models.molmo_point.processing_molmo_point as molmo_point
import mlx_vlm.models.muse_glimmer.processing_muse_glimmer as muse_glimmer
import mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl as paddleocr_vl
import mlx_vlm.models.pixtral.image_processing_pixtral as pixtral_ip
import mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe as omni
import mlx_vlm.models.qwen3_vl.processing_qwen3_vl as qwen3
import mlx_vlm.models.smolvlm.processing_smolvlm as smolvlm
import mlx_vlm.models.step3p7.processing_step3p7 as step3p7
import mlx_vlm.models.unlimited_ocr.processing_unlimitedocr as ocr
from mlx_vlm.generate import GenerationResult, generate
from mlx_vlm.generate.video import processor_handles_video
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.tokenizer_utils import BPEStreamingDetokenizer
from mlx_vlm.utils import (
    StoppingCriteria,
    VideoMetadata,
    load_image_processor,
    load_processor,
    prepare_inputs,
    resolve_video_sampling,
    should_add_special_tokens,
)


class _RecordingTokenizer(PreTrainedTokenizerFast):
    def __call__(self, text, **kwargs):
        self.last_text = text
        return super().__call__(text, **kwargs)


class KimiTokenizer(PreTrainedTokenizerBase):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__()
        self.encode_calls = []

    def convert_tokens_to_ids(self, token):
        return 0

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return [1, 2, 3]

    def apply_chat_template(
        self, conversation, tokenize=False, add_generation_prompt=True
    ):
        return "rendered"


def _assert_shapes(data, **shapes):
    assert {key: data[key].shape for key in shapes} == shapes


def _message(*items):
    return [{"role": "user", "content": list(items)}]


def _load_processor(cls, path, tokenizer=None, **kwargs):
    with _loader_mocks(cls, tokenizer):
        return cls.from_pretrained(path, **kwargs)


def _fast_tokenizer(
    tokens=("[UNK]", "[PAD]", "hello"),
    *,
    tokenizer_class=PreTrainedTokenizerFast,
    split=True,
    **kwargs,
):
    backend = Tokenizer(
        WordLevel({t: i for i, t in enumerate(tokens)}, unk_token=tokens[0])
    )
    if split:
        backend.pre_tokenizer = Whitespace()
    return tokenizer_class(
        tokenizer_object=backend, unk_token=tokens[0], pad_token=tokens[1], **kwargs
    )


def _gemma_image(cls=g4u.Gemma4UnifiedImageProcessor, max_soft_tokens=4):
    return cls(
        patch_size=2,
        pooling_kernel_size=2,
        max_soft_tokens=max_soft_tokens,
        do_resize=False,
        do_rescale=False,
    )


def _lfm_processor(image_processor=None):
    return SimpleNamespace(
        image_processor=image_processor or lfm.Lfm2VlNumpyImageProcessor(),
        tokenizer=_Tokenizer(),
        image_token="<image>",
        image_start_token="<|image_start|>",
        image_end_token="<|image_end|>",
        image_thumbnail_token="<|img_thumbnail|>",
        _merge_kwargs=lambda *a, **kw: {"text_kwargs": {}, "images_kwargs": {}},
    )


def _write_configs(path, **configs):
    for name, config in configs.items():
        (Path(path) / f"{name}.json").write_text(json.dumps(config))


def _assert_attrs(obj, **expected):
    assert {key: getattr(obj, key) for key in expected} == expected


@contextmanager
def _loader_mocks(cls, tokenizer=None):
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer or _Tokenizer(),
        ) as loader,
        patch.object(cls, "check_argument_for_proper_class", return_value=None),
    ):
        yield loader


@pytest.mark.parametrize(
    "model_type,module_path,class_name",
    [
        ("internvl_chat", "internvl_chat", "InternVLChatProcessor"),
        ("molmo", "molmo.processing_molmo", "MolmoProcessor"),
        ("kimi_vl", "kimi_vl.processing_kimi_vl", "KimiVLProcessor"),
        ("kimi_k3", "kimi_k3.processing_kimi_k3", "KimiK3Processor"),
        ("phi3_v", "phi3_v.processing_phi3_v", "Phi3VProcessor"),
        ("hunyuan_vl", "hunyuan_vl.processing_hunyuan_vl", "HunYuanVLProcessor"),
        ("ernie4_5_moe_vl", "ernie4_5_moe_vl", "Ernie4_5_VLProcessor"),
        ("qwen4_exp", "qwen4_exp", "Qwen3VLProcessor"),
    ],
)
def test_auto_processor_routes_to_custom_loader(
    tmp_path, model_type, module_path, class_name
):
    cls = getattr(importlib.import_module("mlx_vlm.models." + module_path), class_name)
    _write_configs(tmp_path, config={"model_type": model_type})
    sentinel = object()
    with patch.object(cls, "from_pretrained", return_value=sentinel) as loader:
        assert (
            AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False) is sentinel
        )
    loader.assert_called_once_with(tmp_path, trust_remote_code=False)
    if model_type in ("hunyuan_vl", "qwen4_exp"):
        assert isinstance(AutoProcessor.from_pretrained(tmp_path), cls)
    else:
        with pytest.raises(ValueError, match="Unrecognized processing class"):
            AutoProcessor.from_pretrained(tmp_path)


class _Tokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    bos_token, eos_token, pad_token = "<bos>", "<eos>", "<pad>"
    pad_token_id = unk_token_id = 0
    image_token, image_token_id = "<image>", 100
    audio_token, audio_token_id = "<audio>", 101
    video_token, video_token_id = "<video>", 102
    boi_token, eoi_token = "<boi>", "<eoi>"
    boa_token, eoa_token = "<boa>", "<eoa>"

    def __init__(self, tokens=None, skip_spaces=False, pad=True, **attrs):
        self.init_kwargs = {}
        self.__dict__.update(attrs)
        self.special_tokens = tokens
        self.skip_spaces = skip_spaces
        self.pad = pad

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        return (self.special_tokens or {}).get(token, self.unk_token_id)

    def encode(self, text, **kwargs):
        if self.special_tokens is None:
            return list(range(10))
        ids, index = ([], 0)
        while index < len(text):
            for token in sorted(self.special_tokens, key=len, reverse=True):
                if text.startswith(token, index):
                    ids.append(self.special_tokens[token])
                    index += len(token)
                    break
            else:
                if not self.skip_spaces or not text[index].isspace():
                    ids.append(1)
                index += 1
        return ids

    def __call__(self, text, text_pair=None, return_token_type_ids=False, **kwargs):
        self.last_text, self.last_kwargs = (text, kwargs)
        rows = [self.encode(t) for t in ([text] if isinstance(text, str) else text)]
        width = max(map(len, rows)) if self.pad else 0
        result = {
            "input_ids": [
                row + [self.pad_token_id] * (width - len(row)) for row in rows
            ],
            "attention_mask": [
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ],
        }
        if return_token_type_ids:
            result["token_type_ids"] = [[0] * width for row in rows]
        return result

    def add_special_tokens(self, tokens):
        pass

    def build_inputs_with_special_tokens(self, ids):
        return ids


class GemmaTokenizer(_Tokenizer):
    def __init__(self):
        super().__init__(
            {"<|image|>": 100, "<|audio|>": 101, "<|video|>": 102},
            image_token="<|image|>",
            audio_token="<|audio|>",
            video_token="<|video|>",
            chat_template="mock",
        )

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        parts = ["<bos>"]
        for message in messages:
            for item in message["content"]:
                parts.append(
                    item["text"]
                    if item["type"] == "text"
                    else getattr(self, item["type"] + "_token")
                )
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "".join(parts)
        return self(rendered) if tokenize else rendered


class _ImageStub:
    model_input_names = ["pixel_values"]
    merge_size, max_image_tiles = 2, 4
    do_image_splitting = False

    def __init__(self, outputs=None, **attrs):
        self.__dict__.update(attrs)
        self.outputs = outputs or {
            "pixel_values": np.zeros((1, 3, 224, 224), np.float32)
        }

    def __call__(self, images=None, **kwargs):
        self.last_kwargs = kwargs
        return self.outputs

    def fetch_images(self, images):
        return images if isinstance(images, list) else [images]


def _mock_ip(**extra):
    return _ImageStub({"pixel_values": np.zeros((1, 3, 224, 224), np.float32), **extra})


def _tiled_ip(size, image_size, extra_grid=()):
    grid = [[size * h, size * w] for h, w in [(1, 1), (1, 2), (2, 1), (2, 2)]]
    return _ImageStub(
        dict(
            pixel_values=np.zeros((1, 5, 3, size, size), np.float32),
            image_sizes=[image_size],
        ),
        image_grid_pinpoints=grid + list(extra_grid),
        size={"height": size, "width": size},
    )


def _make_image(height=224, width=224):
    return Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )


def _bare(cls, **attrs):
    obj = cls.__new__(cls)
    obj.__dict__.update(attrs)
    return obj


def _assert_all_mx(result, *media):
    assert {"input_ids", "attention_mask", *media} <= result.keys()
    for key, value in result.items():
        if value is not None:
            assert isinstance(value, mx.array), f"{key}: {type(value).__name__}"


SMOKE_PROCESSORS = {
    "llava": "LlavaProcessor",
    "llava_next": "LlavaNextProcessor",
    "llava_onevision": "LlavaOnevisionProcessor",
    "paligemma": "PaliGemmaProcessor",
    "gemma3": "Gemma3Processor",
    "gemma3n": "Gemma3nProcessor",
    "smolvlm": "SmolVLMProcessor",
    "mllama": "MllamaProcessor",
    "qwen2_vl": "Qwen2VLProcessor",
    "qwen2_5_vl": "Qwen2_5_VLProcessor",
    "qwen3_vl": "Qwen3VLProcessor",
    "qwen3_omni_moe": "Qwen3OmniMoeProcessor",
    "idefics2": "Idefics2Processor",
    "idefics3": "Idefics3Processor",
    "aya_vision": "AyaVisionProcessor",
    "llama4": "Llama4Processor",
    "pixtral": "PixtralProcessor",
    "mistral3": "Mistral3Processor",
    "multi_modality": "MultiModalityProcessor",
    "ernie4_5_moe_vl": "Ernie4_5_VLProcessor",
}


def _make_processor(name):
    module = importlib.import_module(f"mlx_vlm.models.{name}.processing_{name}")
    cls = getattr(module, SMOKE_PROCESSORS[name])
    ip, tok, kwargs = _mock_ip(), _Tokenizer(), {}
    if name in ("llava", "llava_next"):
        kwargs = dict(
            patch_size=14,
            vision_feature_select_strategy="default",
            num_additional_image_tokens=1,
        )
        if name == "llava_next":
            ip = _tiled_ip(224, (224, 224))
    elif name == "llava_onevision":
        ip = _tiled_ip(384, [768, 768], [[1536, 1536]])
        kwargs = dict(
            num_image_tokens=729,
            vision_aspect_ratio="anyres_max_9",
            vision_feature_select_strategy="full",
        )
    elif name == "paligemma":
        ip.image_seq_length = 4
        tok.add_tokens = lambda *a, **kw: None
    elif name in ("gemma3", "gemma3n"):
        ip, kwargs = (_mock_ip(num_crops=[0]), dict(image_seq_length=4))
        if name == "gemma3n":
            kwargs.update(feature_extractor=None, audio_seq_length=4)
    elif name in ("smolvlm", "idefics2", "idefics3"):
        kwargs["image_seq_len"] = 4
        tok.image_boundary_token = "<fake_token_around_image>"
        if name != "idefics2":
            ip = _mock_ip(rows=[[0]], cols=[[0]])
    elif name == "mllama":
        tok = _Tokenizer(
            {"<|image|>": 128256},
            skip_spaces=True,
            image_token="<|image|>",
            image_token_id=128256,
        )
        ip = _mock_ip(
            pixel_values=np.zeros((1, 4, 3, 560, 560), np.float32),
            num_tiles=[[2]],
            aspect_ratio_ids=np.array([[1]]),
            aspect_ratio_mask=np.ones((1, 4), np.int64),
        )
    elif name in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
        ip = _mock_ip(image_grid_thw=np.array([[1, 16, 16]], np.int64))
        tok = _Tokenizer(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
            vision_start_token="<|vs|>",
            vision_end_token="<|ve|>",
            vision_start_token_id=200,
            vision_end_token_id=201,
        )
    elif name == "qwen3_omni_moe":
        tok = _Tokenizer(
            image_token="<|image|>",
            audio_token="<|audio|>",
            video_token="<|video|>",
            vision_bos_token="<|vb|>",
            vision_eos_token="<|ve|>",
            audio_bos_token="<|ab|>",
            audio_eos_token="<|ae|>",
        )
        ip = _mock_ip(image_grid_thw=np.array([[1, 16, 16]], np.int64))
        kwargs = dict(
            video_processor=SimpleNamespace(model_input_names=[], merge_size=2),
            feature_extractor=SimpleNamespace(model_input_names=[]),
        )
    elif name == "aya_vision":
        ip = _mock_ip(num_patches=[1])
    elif name == "llama4":
        ip = _mock_ip(aspect_ratios=[(1, 1)])
    elif name in ("pixtral", "mistral3"):
        ip = _mock_ip(image_sizes=[[(224, 224)]])
    elif name == "multi_modality":
        kwargs["num_image_tokens"] = 4
    elif name == "ernie4_5_moe_vl":
        ip = _mock_ip(image_grid_thw=np.array([[1, 16, 16]], np.int64))
    with patch.object(
        cls, "check_argument_for_proper_class", return_value=None, create=True
    ):
        return cls(image_processor=ip, tokenizer=tok, **kwargs)


@pytest.mark.parametrize(
    "name,with_image",
    [
        pytest.param(
            name, with_image, id=f"{name}-{('image' if with_image else 'text')}"
        )
        for name in SMOKE_PROCESSORS
        for with_image in (True, False)
        if (name, with_image)
        not in {("paligemma", False), ("qwen3_vl", False), ("mistral3", True)}
    ],
)
def test_processor_mlx_outputs(name, with_image):
    p = _make_processor(name)
    kwargs = dict(text=["Hello world"])
    if with_image:
        token = {
            "gemma3": "<boi>",
            "ernie4_5_moe_vl": "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>",
        }.get(name, getattr(p, "image_token", "<image>"))
        images = [_make_image()]
        if name in ("smolvlm", "mllama", "idefics2", "idefics3", "pixtral", "mistral3"):
            images = [images]
        kwargs.update(text=[token + " Describe"], images=images)
        if name == "paligemma":
            kwargs["text"] = "describe"
    if name == "qwen3_omni_moe":
        kwargs["text"] = kwargs["text"][0]
    result = p(**kwargs)
    _assert_all_mx(result, *(["pixel_values"] if with_image else []))
    if name == "mllama" and with_image:
        assert "cross_attention_mask" in result


def test_unlimited_ocr_default_chat_template_omits_trailing_space():
    Template = pytest.importorskip("jinja2").Template
    p = object.__new__(ocr.UnlimitedOCRProcessor)
    rendered = Template(p.default_chat_template).render(
        messages=[
            {"role": "user", "content": "<image>document parsing."},
            {"role": "assistant", "content": "partial"},
            {"role": "user", "content": "continue"},
        ],
        add_generation_prompt=True,
    )
    assert rendered == "<image>document parsing. partial continue"


class TestOutputControlTokens:
    @pytest.mark.parametrize(
        "cls,markers",
        [
            (glm4v_moe.Glm46VMoEProcessor, ("<|begin_of_box|>", "<|end_of_box|>")),
            (aya_vision.AyaVisionProcessor, ("<|START_RESPONSE|>", "<|END_RESPONSE|>")),
            (
                aya_vision.AyaVisionOutputProcessor,
                ("<|START_RESPONSE|>", "<|END_RESPONSE|>"),
            ),
        ],
    )
    def test_clean_output(self, cls, markers):
        if cls is aya_vision.AyaVisionOutputProcessor:
            from transformers.models.aya_vision.processing_aya_vision import (
                AyaVisionProcessor as Native,
            )

            assert issubclass(cls, Native)
        assert _bare(cls).clean_output(markers[0] + "answer" + markers[1]) == "answer"

    def test_kimi_vl_stops_on_assistant_marker(self):
        p = _bare(
            kimi_vl.KimiVLProcessor, tokenizer=_Tokenizer({"<|im_assistant|>": 163586})
        )
        assert p.additional_eos_token_ids == [163586]


class TestGemma4UnifiedProcessor:
    def _processor(self, video=False):
        video_processor = (
            _gemma_image(g4u.Gemma4UnifiedVideoProcessor, max_soft_tokens=70)
            if video
            else None
        )
        return g4u.Gemma4UnifiedProcessor(
            image_processor=_gemma_image(),
            tokenizer=GemmaTokenizer(),
            video_processor=video_processor,
            image_seq_length=4,
        )

    def test_merged_image_patches_and_positions(self):
        data, soft_tokens = _gemma_image()(Image.new("RGB", (8, 8)))
        _assert_shapes(data, pixel_values=(1, 4, 48), image_position_ids=(1, 4, 2))
        assert soft_tokens == [4]
        assert data["image_position_ids"][0].tolist() == [
            [x, y] for y in range(2) for x in range(2)
        ]

    def test_video_padding_and_positions(self):
        video = np.zeros((2, 3, 4, 8), np.uint8)
        data = _gemma_image(g4.Gemma4VideoProcessor, max_soft_tokens=70)(
            [video], fps=[1.0]
        )
        _assert_shapes(
            data, pixel_values_videos=(1, 2, 280, 12), video_position_ids=(1, 2, 280, 2)
        )
        assert data["num_frames_per_video"] == data["num_soft_tokens_per_frame"] == [2]
        assert data["frame_timestamps"] == [[0.0, 1.0]]
        positions = data["video_position_ids"][0, 0]
        assert positions[:8].tolist() == [[x, y] for y in range(2) for x in range(4)]
        assert np.all(positions[8:] == -1)

    def test_constructor_declares_video_processor(self):
        cls = g4u.Gemma4UnifiedProcessor
        assert "video_processor" in cls.get_attributes()
        p = cls(
            image_processor=_gemma_image(),
            tokenizer=GemmaTokenizer(),
            image_seq_length=4,
        )
        assert isinstance(p.video_processor, g4u.Gemma4UnifiedVideoProcessor)

    def test_audio_chunk_masks(self):
        extractor = g4u.Gemma4UnifiedAudioFeatureExtractor(
            audio_samples_per_token=4, feature_size=4
        )
        result = extractor([np.arange(n, dtype=np.float32) for n in (6, 9)])
        assert result["input_features"].shape == (2, 3, 4)
        assert result["input_features_mask"].tolist() == [
            [True, True, False],
            [True] * 3,
        ]

    @pytest.mark.parametrize("video", [False, True], ids=["image-chat", "video-call"])
    def test_multimodal_inputs(self, video):
        p = self._processor(video)
        if video:
            result = p(
                text=[p.video_token + "describe"],
                videos=[np.zeros((2, 3, 4, 8), np.uint8)],
                fps=[1.0],
            )
            _assert_shapes(
                result, pixel_values_videos=(2, 70, 48), video_position_ids=(2, 70, 2)
            )
            assert isinstance(result["pixel_values_videos"], mx.array)
            assert "<boi><|video|><|video|><eoi>" in p.tokenizer.last_text[0]
        else:
            messages = _message(
                {"type": "image", "image": Image.new("RGB", (8, 8))},
                {"type": "text", "text": "Describe this image in detail."},
            )
            result = p.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="mlx",
                enable_thinking=False,
            )
            _assert_shapes(
                result, pixel_values=(1, 4, 48), image_position_ids=(1, 4, 2)
            )
            assert all(
                isinstance(result[k], mx.array) for k in ("input_ids", "pixel_values")
            )
            assert "<boi>" + "<|image|>" * 4 + "<eoi>" in p.tokenizer.last_text[0]
        assert mx.sum(result["mm_token_type_ids"] == (2 if video else 1)).item() == 4

    def test_video_placeholder_without_tokenizing(self):
        rendered = self._processor().apply_chat_template(
            _message(
                {"type": "video", "video": "clip.mp4"},
                {"type": "text", "text": "Describe this video."},
            ),
            tokenize=False,
            enable_thinking=False,
        )
        assert "<|video|>" in rendered


class TestLlavaOnevisionProcessor:
    def test_video_expansion(self):
        p = _make_processor("llava_onevision")
        video = np.zeros((4, 3, 384, 384), np.float32)
        expanded = p._expand_placeholders(
            "<video> Describe", iter(()), iter([video]), (384, 384)
        )
        assert expanded.count("<video>") == 4 * 14 * 14 + 1

    def test_excess_image_placeholders(self):
        with pytest.raises(ValueError):
            _make_processor("llava_onevision")._expand_placeholders(
                "<image> <image>", iter([[384, 384]]), iter(()), (384, 384)
            )

    @pytest.mark.parametrize(
        "channels_first", [False, True], ids=["hwc-white", "chw-red"]
    )
    def test_video_normalization(self, channels_first):
        frames = np.full((2, 40, 60, 3), 255, np.uint8)
        if channels_first:
            frames = np.zeros((2, 3, 40, 60), np.uint8)
            frames[:, 0] = 255
        pixels = _make_processor("llava_onevision").video_processor([frames])
        assert pixels.shape == (1, 2, 3, 384, 384)
        if channels_first:
            assert np.allclose(pixels[:, :, 0], 1.0, atol=1e-5)
            assert np.allclose(pixels[:, :, 1], -1.0, atol=1e-5)
        else:
            assert np.allclose(pixels, 1.0, atol=1e-5)


def test_pali_gemma_tokenizer_kwargs_do_not_leak_into_image_processor():
    p = _make_processor("paligemma")
    p.image_processor.valid_kwargs = type(
        "ImageKwargs", (), {"__annotations__": {"do_resize": bool}}
    )
    result = p(
        text="describe",
        images=[_make_image()],
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        do_resize=False,
    )
    _assert_all_mx(result)
    assert p.image_processor.last_kwargs == {"do_resize": False}
    assert p.tokenizer.last_kwargs == dict(
        padding=True, padding_side="left", add_special_tokens=False
    )


def test_dots_vl_from_pretrained_uses_slow_image_processor(tmp_path):
    _write_configs(
        tmp_path, chat_template={"chat_template": "{{ messages[0].content }}"}
    )
    loader = Mock(return_value=_mock_ip())
    with (
        _loader_mocks(dots_ocr.DotsVLProcessor),
        patch.dict(
            "transformers.__dict__",
            {"AutoImageProcessor": SimpleNamespace(from_pretrained=loader)},
        ),
    ):
        p = dots_ocr.DotsVLProcessor.from_pretrained(tmp_path, use_fast=True)
    loader.assert_called_once()
    assert loader.call_args.kwargs["use_fast"] is False
    assert isinstance(p.video_processor, dots_ocr.DotsDummyVideoProcessor)


def test_minicpmv_video_marker_expands_to_frame_bounds():
    tokenizer = _Tokenizer(
        {
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
        },
        skip_spaces=True,
        unk_token_id=99,
        image_token="<|image_pad|>",
        image_token_id=101,
        video_token="<|video_pad|>",
    )
    options = dict(
        slice_mode=False, use_image_id=False, scale_resolution=56, patch_size=14
    )
    p = _bare(
        minicpm.MiniCPMVProcessor,
        tokenizer=tokenizer,
        image_processor=minicpm.MiniCPMVImageProcessor(**options),
        video_processor=minicpm.MiniCPMVVideoProcessor(**options),
    )
    p.image_feature_size = p.image_processor.image_feature_size
    p._ensure_tokenizer_attrs()
    for key in ("image_token", "image_token_id", "video_token", "video_token_id"):
        setattr(p, key, getattr(tokenizer, key))
    result = p(
        text=["<|video_pad|> Describe this."],
        videos=[np.zeros((2, 3, 16, 16), np.uint8)],
        slice_mode=False,
        max_num_frames=2,
        padding=False,
    )
    assert len(result["pixel_values"][0]) == 2
    assert result["tgt_sizes"][0].shape == result["image_bound"][0].shape == (2, 2)
    assert result["num_frames_per_video"] == [[2]]
    assert result["num_patches_per_frame"] == [[1, 1]]
    for start, end in result["image_bound"][0]:
        assert np.all(result["input_ids"][0, start:end] == 102)


class TestGlmOcrProcessor:
    GEOMETRY = dict(patch_size=14, temporal_patch_size=2, merge_size=2)

    def test_local_numpy_loader(self, tmp_path):
        ip_config = dict(
            **self.GEOMETRY,
            size={"shortest_edge": 12544, "longest_edge": 9633792},
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        _write_configs(
            tmp_path,
            processor_config=dict(
                image_processor=ip_config, processor_class="GlmOcrProcessor"
            ),
        )
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=_Tokenizer(image_token="<|image|>"),
            ),
            patch("mlx_vlm.models.base.load_chat_template"),
        ):
            p = glm_ocr.GlmOcrProcessor.from_pretrained(tmp_path)
        assert isinstance(p.image_processor, glm_ocr.Glm46VImageProcessor)
        _assert_attrs(p.image_processor, patch_size=14, max_pixels=9633792)

    @pytest.mark.parametrize("reference", [False, True], ids=["shape", "torch-parity"])
    def test_image_patches(self, reference):
        p = glm_ocr.Glm46VImageProcessor(
            **self.GEOMETRY, min_pixels=784, max_pixels=50176
        )
        image = Image.new("RGB", (56, 28))
        if reference:
            try:
                import torch
                from transformers.models.glm46v.image_processing_glm46v import (
                    Glm46VImageProcessor as HFProcessor,
                )
            except Exception as exc:
                pytest.skip(f"Transformers torch image backend unavailable: {exc}")
            expected = HFProcessor(
                **self.GEOMETRY, size={"shortest_edge": 784, "longest_edge": 50176}
            )(images=image)
        actual = p(images=image)
        if reference:
            for key in ("pixel_values", "image_grid_thw"):
                value = expected[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                np.testing.assert_array_equal(value, actual[key])
        else:
            assert actual["image_grid_thw"].tolist() == [[1, 2, 4]]
            assert actual["pixel_values"].shape == (8, 1176)


def test_smolvlm_split_image_prompt_matches_flattened_feature_rows():
    image_seq_len = 81
    single = smolvlm.get_image_prompt_string(
        0, 0, image_seq_len, "<F>", "<image>", "<G>"
    )
    split = smolvlm.get_image_prompt_string(
        3, 4, image_seq_len, "<F>", "<image>", "<G>"
    )
    assert single.count("<image>") == image_seq_len
    assert split.count("<image>") == 13 * image_seq_len
    assert "<row_1_col_1>" in split
    assert "<row_3_col_4>" in split


class TestMllamaProcessor:
    def test_cross_attention_mask_helpers(self):
        mask = mllama.get_cross_attention_token_mask(
            [1, 2, 128256, 3, 4, 128256, 5, 6], 128256
        )
        assert mask == [[2, 5], [5, 8]]
        assert mllama.get_cross_attention_token_mask([1, 2, 3], 128256) == []
        dense = mllama.convert_sparse_cross_attention_mask_to_dense(
            [mask], [[2, 3]], 4, 8
        )
        assert dense.shape == (1, 8, 2, 4)
        assert dense[0, 2, 0, 0] == 1 and dense[0, 0, 0, 0] == 0

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello", "<bos>Hello"),
            ("<|image|>Hello", "<|image|><bos>Hello"),
            ("<bos>Hello", "<bos>Hello"),
        ],
    )
    def test_build_string_from_input(self, text, expected):
        assert mllama.build_string_from_input(text, "<bos>", "<|image|>") == expected


class TestQwen3VLProcessor:
    def _capturing_processor(self, grids):
        p = _make_processor("qwen3_vl")
        p.image_processor = _mock_ip(
            pixel_values=np.zeros((len(grids), 3, 224, 224), np.float32),
            image_grid_thw=np.array(grids, np.int64),
        )
        p.tokenizer = _Tokenizer({}, pad=False)
        return p

    def test_surplus_tokens_follow_prompt_order(self):
        text = "old <|image_pad|> new <|vs|><|image_pad|><|ve|>"
        result = qwen3._drop_surplus_image_tokens(
            text,
            image_token="<|image_pad|>",
            vision_start_token="<|vs|>",
            vision_end_token="<|ve|>",
            count=1,
        )
        assert result == "old  new <|vs|><|image_pad|><|ve|>"

    def test_surplus_tokens_stay_within_batch_entries(self):
        p = self._capturing_processor([[1, 4, 4], [1, 4, 8]])
        p(
            text=[
                "first <|vs|><|image_pad|><|ve|>",
                "stale <|vs|><|image_pad|><|ve|> current <|vs|><|image_pad|><|ve|>",
            ],
            images=[_make_image(), _make_image()],
        )
        assert p.tokenizer.last_text == [
            prefix + "<|vs|>" + "<|image_pad|>" * n + "<|ve|>"
            for prefix, n in [("first ", 4), ("stale  current ", 8)]
        ]

    @pytest.mark.parametrize(
        "grouped", [False, True], ids=["ambiguous-flat", "extra-grouped"]
    )
    def test_invalid_image_counts(self, grouped):
        p = self._capturing_processor([[1, 4, 4], [1, 4, 8], [1, 4, 12]])
        images = [_make_image() for _ in range(3)]
        text = [
            "first <|image_pad|>",
            "stale <|image_pad|> first <|image_pad|> second <|image_pad|>",
        ]
        error = "Cannot unambiguously map"
        if grouped:
            images, text = (
                [images[:2], images[2:]],
                ["first <|image_pad|>", "second <|image_pad|>"],
            )
            error = (
                "Text entry 0 contains 1 image placeholders, but 2 images were supplied"
            )
        with pytest.raises(ValueError, match=error):
            p(text=text, images=images)

    @pytest.mark.parametrize("layout", ["nested-pil", "flat-pil", "hwc-array"])
    def test_video_frame_layouts(self, layout):
        frames = [Image.new("RGB", (224, 224), (i * 40, 128, 128)) for i in range(4)]
        video = {
            "nested-pil": [frames],
            "flat-pil": frames,
            "hwc-array": [np.zeros((4, 224, 224, 3), np.uint8)],
        }[layout]
        p = qwen3.Qwen3VLVideoProcessor(
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
            do_rescale=False,
            do_normalize=False,
        )
        output = p(videos=video)
        np.testing.assert_array_equal(output["video_grid_thw"], [[2, 16, 16]])
        assert output["pixel_values_videos"].shape == (512, 1176)


class TestIdefics3Processor:
    def test_image_prompt_string(self):
        result = idefics3.get_image_prompt_string(0, 0, 4, "<F>", "<I>", "<G>")
        assert "<I>" * 4 in result
        assert "<G>" in result
        result = idefics3.get_image_prompt_string(2, 2, 4, "<F>", "<I>", "<G>")
        assert "<row_1_col_1>" in result
        assert "<row_2_col_2>" in result

    def test_end_of_utterance_is_an_additional_eos_token(self):
        p = _make_processor("idefics3")
        p.tokenizer.convert_tokens_to_ids = lambda token: (
            128258 if token == "<end_of_utterance>" else None
        )
        assert p.additional_eos_token_ids == [128258]


def test_pixtral_image_preprocess_resizes_to_patch_multiple_and_pads():
    image_processor = pixtral_ip.PixtralImageProcessor(
        size={"longest_edge": 40},
        patch_size=14,
        image_mean=[0, 0, 0],
        image_std=[1, 1, 1],
    )
    wide = Image.fromarray(np.zeros((31, 55, 3), dtype=np.uint8))
    square = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    output = image_processor([[wide, square]])
    assert output["image_sizes"] == [(28, 42), (28, 28)]
    assert output["pixel_values"].shape == (2, 3, 28, 42)


def test_mistral3_from_pretrained_uses_torch_free_pixtral_image_processor(tmp_path):
    _write_configs(
        tmp_path,
        processor_config=dict(
            patch_size=16,
            spatial_merge_size=1,
            image_token="[IMG]",
            image_break_token="[IMG_BREAK]",
            image_end_token="[IMG_END]",
            image_processor=dict(
                image_processor_type="PixtralImageProcessorFast",
                patch_size=14,
                size={"longest_edge": 64},
            ),
        ),
        config=dict(
            model_type="mistral3",
            spatial_merge_size=2,
            vision_config={"patch_size": 14},
        ),
    )
    p = _load_processor(mistral3.Mistral3Processor, tmp_path, trust_remote_code=True)
    assert isinstance(p.image_processor, pixtral_ip.PixtralImageProcessor)
    _assert_attrs(p, patch_size=14, spatial_merge_size=2)
    output = p(text=["[IMG]Describe"], images=[[_make_image()]])
    _assert_all_mx(output, "pixel_values")
    assert output["pixel_values"].shape[:2] == (1, 3)
    assert all(int(size) % 28 == 0 for size in output["image_sizes"][0].tolist())


def test_step3_v_l_from_pretrained_uses_fixed_tokenizer():
    tokenizer = _Tokenizer(
        chat_template="template",
        vocab={"Got": 0, "Ġit": 1},
        backend_tokenizer=SimpleNamespace(decoder="bad"),
    )
    with _loader_mocks(step3p7.Step3VLProcessor, tokenizer) as loader:
        p = step3p7.Step3VLProcessor.from_pretrained(
            "step-model", trust_remote_code=True
        )
    loader.assert_called_once_with(
        "step-model", trust_remote_code=True, fix_mistral_regex=True
    )
    assert p.tokenizer is tokenizer
    assert p.detokenizer_class is BPEStreamingDetokenizer
    assert "ByteLevel" in repr(tokenizer.backend_tokenizer.decoder)
    p.detokenizer = object()
    for token in (0, 1):
        p.detokenizer.add_token(token)
    p.detokenizer.finalize()
    assert p.detokenizer.text == "Got it"


class TestErnie4_5VLProcessor:
    def test_helper_functions(self):
        for fn, inputs, expected in [
            (ernie.round_by_factor, [100, 56, 42], [112, 56, 56]),
            (ernie.ceil_by_factor, [100, 56, 57], [112, 56, 84]),
            (ernie.floor_by_factor, [100, 56, 55], [84, 56, 28]),
        ]:
            assert [fn(x, 28) for x in inputs] == expected
        h, w = ernie.smart_resize(224, 224, factor=28)
        assert h % 28 == w % 28 == 0
        h, w = ernie.smart_resize(10, 10, factor=28, min_pixels=56 * 56)
        assert h * w >= 56 * 56
        h, w = ernie.smart_resize(10000, 10000, factor=28, max_pixels=28 * 28 * 1280)
        assert h * w <= 28 * 28 * 1280

    def test_image_processor(self):
        p = ernie.ImageProcessor()
        _assert_attrs(p, patch_size=14, merge_size=2, factor=28)
        (h, w), (gh, gw) = p.get_smart_resize(224, 224)
        assert h % 28 == w % 28 == 0 and (gh, gw) == (h // 14, w // 14)
        image = Image.new("RGB", (224, 224), "red")
        for inputs, count in [
            (image, 1),
            ([image, Image.new("RGB", (448, 448), "blue")], 2),
        ]:
            result = p.preprocess(inputs)
            assert {"pixel_values", "image_grid_thw"} <= result.keys()
            assert result["image_grid_thw"].shape[0] == count
            if count == 1:
                assert result["image_grid_thw"][0, 0] == 1
        pixels = np.random.rand(3, 224, 224).astype(np.float32)
        assert p._extract_patches(pixels, 16, 16).shape == (256, 588)
        assert {"pixel_values", "image_grid_thw"} <= p(images=image).keys()


class TestPaddleOCRVLProcessor:
    def test_from_pretrained_loads_preprocessor_geometry(self, tmp_path):
        geometry = dict(
            min_pixels=64,
            max_pixels=4096,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=4,
            image_mean=[0.1, 0.2, 0.3],
            image_std=[0.9, 0.8, 0.7],
            do_convert_rgb=False,
        )
        _write_configs(
            tmp_path,
            config={"model_type": "paddleocr_vl"},
            preprocessor_config=geometry,
        )
        p = _load_processor(
            paddleocr_vl.PaddleOCRVLProcessor,
            tmp_path,
            _Tokenizer(image_token="<paddle-image>"),
        )
        assert p.image_token == "<paddle-image>"
        _assert_attrs(p.image_processor, **geometry)

    def test_load_image_processor_returns_none(self, tmp_path):
        _write_configs(tmp_path, config={"model_type": "paddleocr_vl"})
        assert load_image_processor(tmp_path) is None


class TestLfm2VlProcessorPatch:
    @pytest.mark.parametrize(
        "batched", [False, True], ids=["tiles", "flat-image-batch"]
    )
    def test_marker_expansion(self, batched):
        p = _lfm_processor()
        images = (
            [_make_image(540, 960) for _ in range(3)]
            if batched
            else [_make_image(1440, 2560)]
        )
        prompts = (
            ["<image>First", "<image>Second", "<image>Third"]
            if batched
            else "<image>Describe this image"
        )
        result = lfm._patched_call(p, images=images, text=prompts)
        assert result["pixel_values"].shape == (3 if batched else 9, 1024, 768)
        expanded = p.tokenizer.last_text
        if batched:
            assert len(expanded) == 3
            for text, suffix in zip(expanded, ("First", "Second", "Third")):
                assert text.count("<|image_start|>") == 1
                assert text.count("<image>") == 252 and text.endswith(suffix)
        else:
            text = expanded[0]
            markers = [
                f"<|img_row_{row}_col_{col}|>" for row in (1, 2) for col in (1, 2, 3, 4)
            ]
            assert text.startswith("<|image_start|>" + markers[0])
            assert all(marker in text for marker in markers)
            assert text.index(markers[-1]) < text.index("<|img_thumbnail|>")
            assert "<|img_thumbnail|>" + "<image>" * 252 + "<|image_end|>" in text
            assert (
                text.endswith("Describe this image")
                and text.count("<image>") == 8 * 256 + 252
            )

    def test_scalar_image_rows_and_cols_are_supported(self):
        ip = _ImageStub(
            dict(
                pixel_values=np.zeros((1, 16, 768), np.float32),
                image_rows=np.array([np.int64(1)]),
                image_cols=np.array([np.int64(1)]),
                image_sizes=[[416, 576]],
            ),
            patch_size=16,
            downsample_factor=2,
            tile_size=512,
            max_image_tokens=256,
            min_image_tokens=64,
            encoder_patch_size=16,
            use_thumbnail=False,
        )
        result = lfm._patched_call(
            _lfm_processor(ip), images=_make_image(), text="<image>Describe this image"
        )
        assert {"input_ids", "attention_mask"} <= result.keys()

    @pytest.mark.parametrize(
        "override", [False, True], ids=["config-merge", "explicit-splitting"]
    )
    def test_from_pretrained(self, tmp_path, override):
        _write_configs(
            tmp_path,
            processor_config=dict(
                processor_class="Lfm2VlProcessor", use_image_special_tokens=True
            ),
        )
        if not override:
            _write_configs(
                tmp_path,
                preprocessor_config=dict(
                    image_processor_type="Lfm2VlImageProcessorFast",
                    resample=3,
                    do_resize=False,
                    max_image_tokens=128,
                    max_pixels_tolerance=1.5,
                    image_mean=[0.4] * 3,
                ),
            )
        with _loader_mocks(lfm.Lfm2VlProcessor):
            if override:
                with (
                    patch.object(
                        lfm, "Siglip2ImageProcessor", SimpleNamespace, create=True
                    ),
                    patch.object(lfm, "_SLOW_PROCESSOR_AVAILABLE", True),
                ):
                    p = lfm.Lfm2VlProcessor.from_pretrained(
                        tmp_path, do_image_splitting=False
                    )
                assert not p.image_processor.do_image_splitting
                assert p.image_processor.use_thumbnail
            else:
                p = lfm.Lfm2VlProcessor.from_pretrained(tmp_path)
                _assert_attrs(
                    p.image_processor,
                    resample=Image.Resampling.BICUBIC,
                    max_image_tokens=128,
                    max_pixels_tolerance=1.5,
                    image_mean=[0.4] * 3,
                    do_resize=True,
                )

    def test_resample_filter_follows_the_checkpoint(self):
        for kwargs, expected in [
            ({}, 3),
            ({"resample": 3}, 3),
            ({"resample": 2}, 2),
            ({"resample": Image.Resampling.LANCZOS}, 1),
            ({"resample": "nonsense"}, 3),
        ]:
            assert lfm.Lfm2VlNumpyImageProcessor(**kwargs).resample is Image.Resampling(
                expected
            )
        image = _make_image(540, 960)
        outputs = [
            lfm.Lfm2VlNumpyImageProcessor(resample=r)([image], return_tensors="np")[
                "pixel_values"
            ]
            for r in (3, 2)
        ]
        assert not np.allclose(*outputs)

    @pytest.mark.parametrize(
        "force_resize", [False, True], ids=["patch-budget", "forced-resize"]
    )
    def test_patch_geometry(self, force_resize):
        p = lfm.Lfm2VlNumpyImageProcessor(
            **{"do_resize": False} if force_resize else {"max_num_patches": 256}
        )
        assert p.do_resize if force_resize else p.max_num_patches == 1024
        image = _make_image(1440, 2560) if force_resize else _make_image(480, 640)
        result = p(
            [image], return_tensors="np", **{"do_resize": False} if force_resize else {}
        )
        assert result["pixel_values"].shape == (9 if force_resize else 1, 1024, 768)
        if force_resize:
            assert result["spatial_shapes"].tolist() == [[32, 32]] * 8 + [[24, 42]]
        else:
            rows, cols = result["spatial_shapes"][0].tolist()
            assert int(result["pixel_attention_mask"].sum()) == rows * cols

    @pytest.mark.parametrize("layout", ["lists", "tuples", "arrays"])
    def test_mismatched_nested_image_groups(self, layout):
        images = [Image.new("RGB", (64, 64), (i * 60, 0, 0)) for i in range(4)]
        groups = [images[:1], images[1:]]
        if layout == "tuples":
            groups = tuple(tuple(group) for group in groups)
        elif layout == "arrays":
            groups = [np.stack(group) for group in groups]
        with pytest.raises(ValueError, match="text \\[2, 2\\] and images \\[1, 3\\]"):
            lfm._patched_call(
                _lfm_processor(),
                images=groups,
                text=["<image><image>Prompt A", "<image><image>Prompt B"],
            )


def test_molmo_point_processor_uses_image_processor_for_images():
    fields = dict(
        pixel_values=np.zeros((1, 729, 588), np.float32),
        image_token_pooling=np.zeros((1, 729), np.int64),
        image_grids=np.array([[1, 1, 1, 1]], np.int64),
        image_num_crops=np.array([1], np.int64),
    )
    p = molmo_point.MolmoPointProcessor(
        _Tokenizer(bos_token_id=1, eos_token_id=2),
        image_processor=SimpleNamespace(preprocess=lambda images: fields),
    )
    assert (
        fields.keys() <= p(text=molmo_point.IMAGE_PROMPT, images=_make_image()).keys()
    )


def test_nemotron_h_nano_omni_native_processor_handles_stripped_auto_map(tmp_path):
    model_type = "NemotronH_Nano_Omni_Reasoning_V3"
    _write_configs(
        tmp_path,
        config={"model_type": model_type},
        processor_config={"processor_class": model_type + "Processor"},
        preprocessor_config={
            "image_processor_type": model_type + "ImageProcessor",
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711],
            "min_num_patches": 64,
            "max_num_patches": 64,
            "max_model_len": 128,
        },
    )
    importlib.import_module("mlx_vlm.models.nemotron_h_nano_omni")
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=_Tokenizer(image_token_id=18),
    ):
        p = load_processor(str(tmp_path), add_detokenizer=False)
    result = prepare_inputs(
        p,
        images=[_make_image()],
        prompts="<image>\nDescribe this image.",
        image_token_index=p.image_token_id,
    )
    assert p.__class__.__name__ == "NemotronHNanoOmniProcessor"
    assert "pixel_values" in result
    assert "num_tokens" in result
    assert int(result["num_tokens"][0].item()) > 0


def test_laguna_special_tokens_chat_template_owns_laguna_special_tokens():
    p = SimpleNamespace(chat_template="{{ messages }}")
    assert not should_add_special_tokens("laguna", p)
    assert should_add_special_tokens("llama", p)


class TestKimiK3Processor:
    @pytest.fixture
    def p(self):
        return kimi_k3.KimiK3Processor(tokenizer=KimiTokenizer())

    def test_literal_control_tokens(self, p):
        text = "literal <|end_of_msg|> marker"
        p.apply_chat_template([{"role": "user", "content": text}], tokenize=True)
        calls = p.tokenizer.encode_calls
        for source, split in [(text, True), ("<|end_of_msg|>", False)]:
            matches = [kwargs for value, kwargs in calls if value == source]
            assert matches and matches[0]["split_special_tokens"] is split

    def test_python_chat_renderer(self, p):
        result = apply_chat_template(
            p, {"model_type": "kimi_k3"}, "Describe this image.", num_images=1
        )
        assert "Describe this image.<|kimi_image_placeholder|>" in result
        assert '<|open|>message role="assistant"' in result

    def test_unsupported_video(self, p):
        with pytest.raises(ValueError, match="unsupported media type: video"):
            p(videos=[np.zeros((2, 16, 24, 3), np.uint8)], text="Describe this video.")

    def test_fast_tokenizer_round_trip(self, tmp_path):
        kimi_k3.KimiK3Processor(tokenizer=_fast_tokenizer()).save_pretrained(tmp_path)
        assert (tmp_path / "tokenizer.json").is_file()
        with patch.object(kimi_k3, "_convert_kimi_k3_tiktoken") as convert:
            loaded = kimi_k3.KimiK3Processor.from_pretrained(tmp_path)
        convert.assert_not_called()
        assert loaded.tokenizer.encode("hello", add_special_tokens=False) == [2]

    @pytest.mark.parametrize(
        "remote", [False, True], ids=["local-tiktoken", "remote-fast"]
    )
    def test_tokenizer_loading(self, tmp_path, remote):
        tokenizer = KimiTokenizer()
        _write_configs(
            tmp_path,
            tokenizer={},
            tokenizer_config={"added_tokens_decoder": {}},
            preprocessor_config=(
                {"media_proc_cfg": {"patch_size": 18}} if remote else {}
            ),
        )
        vocab = tmp_path / "tiktoken.model"
        vocab.write_text("tiktoken ranks")
        if not remote:
            (tmp_path / "tokenizer.json").unlink()

        def download(repo_id, filename, **kwargs):
            assert repo_id == "moonshotai/Kimi-K3"
            assert kwargs["revision"] == "model-revision"
            assert filename in ("tokenizer.json", "preprocessor_config.json")
            return tmp_path / filename

        with (
            patch("huggingface_hub.hf_hub_download", side_effect=download),
            patch.object(
                PreTrainedTokenizerFast, "from_pretrained", return_value=tokenizer
            ) as fast,
            patch.object(
                kimi_k3, "_convert_kimi_k3_tiktoken", return_value=tokenizer
            ) as convert,
        ):
            source = "moonshotai/Kimi-K3" if remote else tmp_path
            kwargs = {"revision": "model-revision"} if remote else {}
            loaded = kimi_k3.KimiK3Processor.from_pretrained(source, **kwargs)
        assert isinstance(loaded, kimi_k3.KimiK3Processor)
        if remote:
            assert fast.call_args.args[0] == "moonshotai/Kimi-K3"
            assert fast.call_args.kwargs["revision"] == "model-revision"
            assert not fast.call_args.kwargs["trust_remote_code"]
            assert loaded.image_processor.patch_size == 18
            convert.assert_not_called()
        else:
            convert.assert_called_once_with(vocab, tmp_path / "tokenizer_config.json")
            fast.assert_not_called()

    def test_conversion_requires_tiktoken(self):
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(ImportError, match="Install `tiktoken`.*tokenizer.json"),
        ):
            kimi_k3._convert_kimi_k3_tiktoken("tiktoken.model", "tokenizer_config.json")

    def test_restores_all_control_slots(self):
        supplied = {100: "[BOS]", 103: "<|open|>", 355: "[PAD]"}
        config = {
            "added_tokens_decoder": {
                str(i): {"content": t} for i, t in supplied.items()
            }
        }
        tokens = kimi_k3._kimi_k3_control_tokens(config, base_vocab_size=100)
        assert len(tokens) == 256 and tokens[1] == "<|reserved_token_101|>"
        assert {i: tokens[i - 100] for i in supplied} == supplied


def test_laguna_from_pretrained_loads_fast_tokenizer_directly():
    tokenizer = _fast_tokenizer(
        ("<unk>", "<pad>", "<eos>", "prompt"),
        eos_token="<eos>",
        chat_template="template",
    )
    with patch.object(
        laguna.PreTrainedTokenizerFast, "from_pretrained", return_value=tokenizer
    ) as loader:
        p = laguna.LagunaProcessor.from_pretrained(
            "/tmp/model",
            processor_kwargs={"local_files_only": True},
            quantize_activations=True,
            trust_remote_code=True,
        )
    assert p.tokenizer is tokenizer
    args, kwargs = loader.call_args
    assert args == ("/tmp/model",)
    assert all(
        kwargs[k]
        for k in ("fix_mistral_regex", "local_files_only", "trust_remote_code")
    )
    assert not {"processor_kwargs", "quantize_activations"} & kwargs.keys()


def test_qwen3_omni_moe_patch_intercepts_without_hf_video_processor(tmp_path):
    tokenizer = _Tokenizer(
        image_token="<|image_pad|>",
        audio_token="<|audio_pad|>",
        video_token="<|video_pad|>",
        vision_bos_token="<|vision_start|>",
        vision_eos_token="<|vision_end|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
    )
    feature_extractor = type(
        "FE", (), {"model_input_names": ["input_features"], "sampling_rate": 16000}
    )()
    _write_configs(tmp_path, config={"model_type": "qwen3_omni_moe"})
    _write_configs(
        tmp_path,
        preprocessor_config={
            "feature_extractor_type": "WhisperFeatureExtractor",
            "image_processor_type": "Qwen2VLImageProcessor",
            "processor_class": "Qwen3OmniMoeProcessor",
        },
    )
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=feature_extractor,
        ),
    ):
        p = AutoProcessor.from_pretrained(str(tmp_path))
    assert isinstance(p, omni.Qwen3OmniMoeProcessor)
    assert type(p.video_processor).__name__ == "Qwen3VLVideoProcessor"


class TestDeepseekV4Processor:
    def test_loads_local_chat_template_jinja(self, tmp_path):
        template = "{{ messages[0]['content'] }}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert deepseek.load_deepseek_v4_chat_template(tmp_path) == template

    @pytest.mark.parametrize(
        "auto", [False, True], ids=["explicit-template", "auto-loader"]
    )
    def test_from_pretrained(self, tmp_path, auto):
        cls = deepseek.DeepseekV4Processor
        _write_configs(tmp_path, config={"model_type": "deepseek_v4"})
        tokenizer = _Tokenizer(
            chat_template=None, apply_chat_template=lambda *a, **kw: "templated"
        )
        with _loader_mocks(cls, tokenizer):
            p = (
                AutoProcessor.from_pretrained(tmp_path)
                if auto
                else cls.from_pretrained("repo/name", chat_template="{{ explicit }}")
            )
        if auto:
            assert isinstance(p, cls)
        else:
            assert p.chat_template == "{{ explicit }}"


def test_locate_anything_save_pretrained_round_trips_custom_config(tmp_path):
    template = "{{ messages }}"
    geometry = dict(patch_size=28, merge_kernel_size=[2, 4], in_token_limit=1234)
    tokenizer = _fast_tokenizer(chat_template=template)
    p = locateanything.LocateAnythingProcessor(
        image_processor=locate_ip.LocateAnythingImageProcessor(**geometry),
        tokenizer=tokenizer,
        chat_template=template,
    )
    saved = p.save_pretrained(tmp_path)
    configs = {
        name: json.loads((tmp_path / f"{name}.json").read_text())
        for name in ("processor_config", "preprocessor_config", "chat_template")
    }
    assert str(tmp_path / "processor_config.json") in saved
    assert configs["processor_config"]["processor_class"] == "LocateAnythingProcessor"
    for key in ("processor_config", "chat_template"):
        assert configs[key]["chat_template"] == template
    assert {k: configs["preprocessor_config"][k] for k in geometry} == geometry
    with patch.object(
        locateanything.AutoTokenizer, "from_pretrained", return_value=_fast_tokenizer()
    ):
        loaded = locateanything.LocateAnythingProcessor.from_pretrained(tmp_path)
    _assert_attrs(loaded.image_processor, **geometry)
    assert loaded.chat_template == loaded.tokenizer.chat_template == template


def test_muse_glimmer_from_pretrained_attaches_model_config(tmp_path):
    _write_configs(
        tmp_path,
        config=dict(
            model_type="muse_glimmer",
            text_config=dict(vocab_size=1234, eos_token_id=99),
        ),
    )
    p = _load_processor(
        muse_glimmer.MuseGlimmerProcessor,
        tmp_path,
        _Tokenizer(chat_template="{{ messages }}"),
    )
    _assert_attrs(
        p.config,
        model_type="muse_glimmer",
        vocab_size=1234,
        eos_token_id=99,
        thinking_start_token="to=self<|message|>",
        thinking_end_token="<|eom|>",
    )


class TestProcessorRegistration:
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

        import mlx_vlm

        models_dir = Path(mlx_vlm.__file__).parent / "models"
        pattern = re.compile("AutoProcessor\\.register\\(\\s*['\"]")
        offenders = [
            str(path.relative_to(models_dir))
            for path in models_dir.rglob("*.py")
            if pattern.search(path.read_text())
        ]
        assert offenders == [], f"string-first register calls: {offenders}"

    @pytest.mark.parametrize("module", _AFFECTED_MODULES)
    def test_affected_modules_import_cleanly(self, module):
        importlib.import_module(module)


class TestTrustRemoteCodePassthrough:
    """Regression test for #1724 — an explicit trust_remote_code must not be overridden."""

    def test_molmo_point_honors_explicit_false(self):
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as from_pretrained,
            patch.object(molmo_point, "load_chat_template"),
        ):
            molmo_point.MolmoPointProcessor.from_pretrained(
                "/tmp/model", trust_remote_code=False
            )
        _, kwargs = from_pretrained.call_args
        assert not kwargs["trust_remote_code"]


def test_qwen3_vl_video_timestamp_video_prompt_falls_back_to_processor_fps():
    p = _make_processor("qwen3_vl")
    p.tokenizer = _Tokenizer({"<|video_pad|>": 102}, video_token="<|video_pad|>")
    p.image_processor = None
    p.video_processor = _ImageStub(
        dict(
            pixel_values_videos=np.zeros((1, 4), np.float32),
            video_grid_thw=np.array([[2, 2, 2]], np.int64),
        ),
        temporal_patch_size=2,
        fps=2.0,
    )
    p.vision_start_token, p.vision_end_token = ("<|vision_start|>", "<|vision_end|>")
    p.vision_start_token_id, p.vision_end_token_id = (58, 59)
    p(
        text=["<|vision_start|><|video_pad|><|vision_end|>Describe the clip."],
        videos=["clip.mp4"],
    )
    rendered = p.tokenizer.last_text[0]
    assert rendered.count(" seconds>") == 2
    assert "<0.2 seconds>" in rendered and "<1.2 seconds>" in rendered


class TestMageVLProcessor:
    """Mage VL image/video processing and processor-to-model compatibility."""

    VIDEO_BLOCK = "<|vision_start|><|video_pad|><|vision_end|>"
    IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
    SHORT_METADATA = [VideoMetadata(total_num_frames=10, fps=30, frames_indices=[0, 9])]
    CHAT_TEMPLATE = (
        "{% for message in messages %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
    )
    IMAGE_CONFIG = dict(
        patch_size=16,
        temporal_patch_size=1,
        merge_size=2,
        min_pixels=1024,
        max_pixels=8192,
    )

    @pytest.fixture
    def p(self):
        tokens = [
            "[UNK]",
            "[PAD]",
            mage.IMAGE_PAD,
            mage.VIDEO_PAD,
            mage.VISION_START,
            mage.VISION_END,
        ]
        tokenizer = _fast_tokenizer(
            tokens,
            tokenizer_class=_RecordingTokenizer,
            split=False,
            additional_special_tokens=tokens[2:],
            chat_template=self.CHAT_TEMPLATE,
        )
        return mage.MageVLProcessor(
            image_processor=qwen3.Qwen3VLImageProcessor(**self.IMAGE_CONFIG),
            tokenizer=tokenizer,
        )

    @staticmethod
    def frames(count=3, width=32, value=0):
        return np.full((count, 3, 32, width), value, dtype=np.uint8)

    @staticmethod
    def image_counts(output):
        return (np.array(output["input_ids"]) == 2).sum(axis=1).tolist()

    def test_image_batch_keeps_distinct_counts(self, p):
        images = [Image.new("RGB", (32, 32)), Image.new("RGB", (64, 32))]
        output = p(text=[self.IMAGE_BLOCK, self.IMAGE_BLOCK], images=images)
        assert self.image_counts(output) == [1, 2]
        assert "patch_positions" not in output
        expected = p.image_processor(images)
        np.testing.assert_array_equal(output["pixel_values"], expected["pixel_values"])

    def test_video_uses_reference_timestamps_positions_and_frame_attention(self, p):
        metadata = VideoMetadata(
            total_num_frames=61, fps=30, frames_indices=[0, 15, 60]
        )
        output = p(
            text=self.VIDEO_BLOCK, videos=[self.frames()], video_metadata=[metadata]
        )
        assert p.tokenizer.last_text == [
            "".join(f"<{sec:.1f} seconds>{self.IMAGE_BLOCK}" for sec in [0, 0.5, 2])
        ]
        assert self.image_counts(output) == [3]
        assert "pixel_values_videos" not in output
        assert "video_grid_thw" not in output
        assert output["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
        positions = np.array(output["patch_positions"]).reshape(3, 4, 3)
        np.testing.assert_array_equal(positions[:, :, 0], [[0] * 4, [15] * 4, [60] * 4])
        np.testing.assert_array_equal(
            positions[1, :, 1:], [[0, 0], [0, 1], [1, 0], [1, 1]]
        )
        assert mage_vision.build_cu_seqlens(
            output["image_grid_thw"].tolist(), 12, 4
        ) == [0, 4, 8, 12]

    @pytest.mark.parametrize(
        "prompt,videos",
        [(VIDEO_BLOCK * 2, [frames()]), (VIDEO_BLOCK, [frames(), frames()])],
    )
    def test_video_placeholder_mismatch_is_rejected(self, p, prompt, videos):
        with pytest.raises(ValueError, match="placeholder"):
            p(text=prompt, videos=videos)

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"video_metadata": []}, "one video_metadata"),
            ({"video_metadata": SHORT_METADATA}, "frame count"),
            ({"fps": 0}, "positive and finite"),
        ],
    )
    def test_bad_metadata_is_rejected(self, p, kwargs, error):
        with pytest.raises(ValueError, match=error):
            p(text=self.VIDEO_BLOCK, videos=[self.frames()], **kwargs)

    def test_prepare_inputs_accepts_supplied_metadata(self, p):
        metadata = {"total_num_frames": 60, "fps": 10, "frames_indices": [0, 20, 59]}
        output = prepare_inputs(
            p,
            videos=[self.frames()],
            prompts=self.VIDEO_BLOCK,
            video_metadata=[metadata],
        )
        assert "<5.9 seconds>" in p.tokenizer.last_text[0]
        assert np.array(output["patch_positions"])[-1, 0] == 59

    def test_shared_decoder_supports_odd_frame_counts(self, p, tmp_path):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32)
        )
        if not writer.isOpened():
            pytest.skip("mp4v encoder unavailable")
        for value in (0, 100, 200):
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        writer.release()
        output = prepare_inputs(p, videos=[path], prompts=self.VIDEO_BLOCK, nframes=3)
        assert output["image_grid_thw"].shape == (3, 3)
        assert np.array(output["patch_positions"])[-1, 0] == 2

    def test_numpy_processor_loads_through_auto_processor(self, p, tmp_path):
        _write_configs(
            tmp_path,
            config={"model_type": "mage_vl"},
            preprocessor_config=self.IMAGE_CONFIG,
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=p.tokenizer,
        ):
            loaded = AutoProcessor.from_pretrained(tmp_path)
        assert isinstance(loaded, mage.MageVLProcessor)
        assert processor_handles_video(loaded)
        sampling = resolve_video_sampling(loaded, {})
        assert (sampling.min_frames, sampling.frame_factor, sampling.max_frames) == (
            1,
            1,
            32,
        )
        assert self.image_counts(
            loaded(text=self.VIDEO_BLOCK, videos=[self.frames()])
        ) == [3]

    def test_mixed_video_pixels_affect_only_their_visual_embeddings(self, p):
        common = dict(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
        )
        config = mage_config.ModelConfig(
            text_config=mage_config.TextConfig(
                **common, num_key_value_heads=1, head_dim=32, vocab_size=32
            ),
            vision_config=mage_config.VisionConfig(
                **common, out_hidden_size=64, text_hidden_size=64
            ),
            image_token_id=2,
            video_token_id=3,
        )
        model = mage_model.Model(config)
        kwargs = {
            "text": self.IMAGE_BLOCK + self.VIDEO_BLOCK,
            "images": [Image.new("RGB", (32, 32), "red")],
        }
        first = p(**kwargs, videos=[self.frames(2, value=0)])
        second = p(**kwargs, videos=[self.frames(2, value=255)])
        a = model.get_input_embeddings(**first).inputs_embeds
        b = model.get_input_embeddings(**second).inputs_embeds
        mx.eval(a, b)
        assert mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item()
        a, b = np.array(a)[0], np.array(b)[0]
        visual_indices = np.flatnonzero(np.array(first["input_ids"])[0] == 2)
        np.testing.assert_array_equal(a[visual_indices[0]], b[visual_indices[0]])
        assert not np.allclose(a[visual_indices[1:]], b[visual_indices[1:]])

    def test_grid_coercion_accepts_processor_shapes(self):
        raw = np.array([1, 4, 4])
        assert mage_model._as_grid_list(raw) == [(1, 4, 4)]


class TinyDiffusionGemma4Tokenizer(_Tokenizer):
    TOOL_ATTRS = ("stc_token", "etc_token", "escape_token", "soc_token", "eoc_token")

    def __init__(self):
        super().__init__(
            {
                "<image>": 60,
                "<video>": 61,
                "<boi>": 62,
                "<eoi>": 63,
                "<pad>": 0,
                "<eos>": 1,
            },
            skip_spaces=True,
            image_token_id=60,
            video_token_id=61,
            eos_token_id=1,
            unk_token_id=2,
            audio_token=None,
            audio_token_id=None,
            boa_token=None,
            eoa_token=None,
        )
        self.additional_special_tokens = []
        for key in self.TOOL_ATTRS:
            setattr(self, key, None)

    @property
    def all_special_ids(self):
        tokens = self.additional_special_tokens + [
            getattr(self, key)
            for key in self.TOOL_ATTRS
            if getattr(self, key) is not None
        ]
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def add_special_tokens(self, tokens):
        for token in tokens.get("additional_special_tokens", []):
            self.special_tokens[token] = self.video_token_id


class TinyVideoProcessor:
    model_input_names = ["pixel_values_videos"]

    def __call__(self, videos, fps=None):
        return {
            "pixel_values_videos": np.zeros((2, 3, 4, 4), dtype=np.float32),
            "num_frames_per_video": [2],
            "num_soft_tokens_per_frame": [1],
            "frame_timestamps": [[0.0, 1.0]],
        }


class TinyImageProcessor:
    model_input_names = ["pixel_values"]

    def fetch_images(self, images):
        return images

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        return {"pixel_values": np.stack(images).astype(np.float32)}, [1] * len(images)


def tiny_diffusion_gemma_processor(image_processor=None):
    return diffusion.DiffusionGemma4Processor(
        image_processor=image_processor,
        tokenizer=TinyDiffusionGemma4Tokenizer(),
        video_processor=TinyVideoProcessor(),
    )


class TestDiffusionGemma4Processor:
    def test_auto_processor_loads_multimodal_processor(self, tmp_path):
        _write_configs(
            tmp_path,
            config={"model_type": "diffusion_gemma"},
            processor_config=dict(
                audio_ms_per_token=40,
                audio_seq_length=750,
                image_seq_length=140,
                processor_class="DiffusionGemma4Processor",
                image_processor=dict(
                    do_normalize=False,
                    image_processor_type="Gemma4ImageProcessor",
                    max_soft_tokens=140,
                    patch_size=16,
                    pooling_kernel_size=3,
                ),
                video_processor=dict(
                    max_soft_tokens=70,
                    num_frames=8,
                    video_processor_type="Gemma4VideoProcessor",
                ),
            ),
        )
        tok = TinyDiffusionGemma4Tokenizer()
        tok.chat_template = None
        with _loader_mocks(g4.Gemma4Processor, tok):
            p = AutoProcessor.from_pretrained(tmp_path)
        assert isinstance(p, diffusion.DiffusionGemma4Processor)
        assert isinstance(p.image_processor, g4.Gemma4ImageProcessor)
        assert isinstance(p.video_processor, g4.Gemma4VideoProcessor)
        assert (p.image_processor.max_soft_tokens, p.video_processor.num_frames) == (
            140,
            8,
        )
        assert p.get_attributes() == ["image_processor", "tokenizer", "video_processor"]

    def test_demotes_tool_parser_tokens(self):
        tok = TinyDiffusionGemma4Tokenizer()
        tok.special_tokens.update(
            {t: 70 + i for i, t in enumerate(diffusion._TOOL_PARSER_TOKENS)}
        )
        tok.special_tokens["<extra_special>"] = 90
        tokens = ("<|tool_call>", "<tool_call|>", '<|"|>', "<|channel>", "<channel|>")
        for attr, token in zip(tok.TOOL_ATTRS, tokens):
            setattr(tok, attr, token)
        tok.additional_special_tokens = ["<extra_special>"]
        with patch.object(
            g4.Gemma4Processor,
            "from_pretrained",
            return_value=SimpleNamespace(tokenizer=tok),
        ):
            p = diffusion.DiffusionGemma4Processor.from_pretrained("demo")
        assert p.tokenizer is tok
        assert tok.additional_special_tokens == [
            "<extra_special>"
        ] and tok.all_special_ids == [90]
        assert all(getattr(tok, attr) is None for attr in tok.TOOL_ATTRS)
        assert all(
            tok.convert_tokens_to_ids(t) != tok.unk_token_id
            for t in diffusion._TOOL_PARSER_TOKENS
        )

    def test_strip_channel_scaffolding_is_noop_without_markers(self):
        plain = "Title: A calm river cruise\nKeywords: boat, river"
        assert diffusion._strip_channel_scaffolding(plain) == plain

    def test_generate_strips_channel_scaffolding(self):
        dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="diffusion_gemma", eos_token_id=999999)
        )
        p = tiny_diffusion_gemma_processor()
        p.tokenizer.stopping_criteria = StoppingCriteria([999999], p.tokenizer)
        chunk = GenerationResult(
            text="<|channel>thought\n<channel|>Title: A calm river cruise",
            token=1,
            prompt_tokens=3,
            generation_tokens=8,
            total_tokens=11,
            prompt_tps=10.0,
            generation_tps=5.0,
        )
        with patch.object(dispatch, "stream_generate", return_value=iter([chunk])):
            assert generate(model, p, "").text == "Title: A calm river cruise"

    def test_video_outputs_cross_thread_boundary(self):

        def produce():
            return tiny_diffusion_gemma_processor()(
                text="<video> describe", videos=[np.zeros((2, 3, 4, 4), np.uint8)]
            )

        def consume(result):
            mx.eval(result)
            mask = result["mm_token_type_ids"]
            return result["pixel_values"].shape, mx.sum(mask == 2).item()

        with ThreadPoolExecutor(1) as producer, ThreadPoolExecutor(1) as consumer:
            result = producer.submit(produce).result(timeout=5)
            expected = ((2, 3, 4, 4), 2)
            assert consumer.submit(consume, result).result(timeout=5) == expected

    def test_chat_template_includes_video_token(self):
        p = tiny_diffusion_gemma_processor()
        rendered = apply_chat_template(
            p,
            SimpleNamespace(model_type="diffusion_gemma"),
            "Describe this video.",
            video=["clip.mp4"],
        )
        assert p.video_token in rendered
        result = p(text=rendered, videos=[np.zeros((2, 3, 4, 4), np.uint8)])
        assert mx.sum(result["mm_token_type_ids"] == 2).item() == 2

    def test_mixed_media_pixel_order(self):
        p = tiny_diffusion_gemma_processor(image_processor=TinyImageProcessor())
        result = p(
            text="<video> then <image>",
            images=[np.ones((3, 4, 4), np.float32)],
            videos=[np.zeros((2, 3, 4, 4), np.uint8)],
        )
        assert "pixel_values_videos" not in result
        pixels = result["pixel_values"]
        assert pixels.shape == (3, 3, 4, 4)
        assert mx.all(pixels[:2] == 0).item() and mx.all(pixels[2] == 1).item()


@pytest.mark.parametrize(
    "cls,cap",
    [
        (g4.Gemma4VideoProcessor, "num_frames"),
        (minicpm.MiniCPMVVideoProcessor, "max_num_frames"),
    ],
)
def test_video_frame_caps(cls, cap):
    p = cls()
    assert p.video_sampling_defaults() == {"max_frames": getattr(p, cap)}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("to=user<|message|>Final answer.<|return|>", "Final answer."),
        (
            "Two tabby cats sleep on a pink couch.",
            "Two tabby cats sleep on a pink couch.",
        ),
    ],
)
def test_muse_clean_output(raw, expected):
    assert muse_glimmer._extract_final_channel(raw) == expected
