"""Shared processor contracts, loader routing, and multimodal integration tests."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

from mlx_vlm.generate import GenerationResult, generate
from mlx_vlm.generate.video import processor_handles_video
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.tests.test_tokenizer_utils import (
    GemmaTokenizer,
    KimiTokenizer,
    ProcessorTokenizer,
    RecordingTokenizer,
    TinyDiffusionGemma4Tokenizer,
    fast_tokenizer,
)
from mlx_vlm.utils import (
    StoppingCriteria,
    VideoMetadata,
    load_image_processor,
    load_processor,
    prepare_inputs,
    resolve_video_sampling,
)

equal = np.testing.assert_array_equal
DATA = json.loads(Path(__file__).with_name("processor_cases.json").read_text())
PROFILES = DATA["profiles"]
# Importing the registry also installs each family's AutoProcessor routing patch.
m, c = NS(), NS()
for name, (path, cls) in DATA["modules"].items():
    if "." not in path:
        path += f".processing_{path}"
    module = importlib.import_module("mlx_vlm.models." + path)
    setattr(m, name, module)
    if cls:
        setattr(c, name, getattr(module, cls))


def _ignore_types(cls):
    return patch.object(
        cls, "check_argument_for_proper_class", return_value=None, create=True
    )


def _assert_shapes(data, **shapes):
    assert {key: data[key].shape for key in shapes} == shapes


def _message(*items):
    return [{"role": "user", "content": list(items)}]


def _gemma_image(cls=m.g4u.Gemma4UnifiedImageProcessor, max_soft_tokens=4):
    return cls(**PROFILES["gemma_image"], max_soft_tokens=max_soft_tokens)


def _lfm_processor(image_processor=None):
    return NS(
        image_processor=image_processor or m.lfm.Lfm2VlNumpyImageProcessor(),
        tokenizer=ProcessorTokenizer(),
        **PROFILES["lfm_tokens"],
        _merge_kwargs=lambda *a, **kw: {"text_kwargs": {}, "images_kwargs": {}},
    )


def _write_configs(path, **configs):
    for name, config in configs.items():
        (Path(path) / f"{name}.json").write_text(json.dumps(config))


def _assert_attrs(obj, **expected):
    assert {key: attrgetter(key)(obj) for key in expected} == expected


@pytest.mark.parametrize("model_type,module_path,class_name", DATA["routes"])
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


def _stub(name):
    case = deepcopy(DATA["stubs"][name])
    outputs = {
        k: np.zeros(shape, np.float32) for k, shape in case.get("shapes", {}).items()
    }
    outputs.update(
        {k: np.array(v, np.int64) for k, v in case.get("arrays", {}).items()}
    )
    return _ImageStub({**outputs, **case.get("values", {})}, **case.get("attrs", {}))


def _make_image(height=224, width=224):
    return Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )


def _assert_all_mx(result, *media):
    assert {"input_ids", "attention_mask", *media} <= result.keys()
    for key, value in result.items():
        if value is not None:
            assert isinstance(value, mx.array), f"{key}: {type(value).__name__}"


SMOKE_PROCESSORS = DATA["smoke"]


def _make_processor(name):
    case = SMOKE_PROCESSORS[name]
    defaults = DATA["smoke_defaults"].get(case.get("family"), {})
    case = deepcopy({**defaults, **case})
    module = importlib.import_module(f"mlx_vlm.models.{name}.processing_{name}")
    cls = getattr(module, case["class"])
    ip = _stub(case["stub"]) if "stub" in case else _mock_ip(**case.get("outputs", {}))
    tok = ProcessorTokenizer(**case.get("tokenizer", {}))
    kwargs = dict(case.get("kwargs", {}))
    if name == "paligemma":
        ip.image_seq_length, tok.add_tokens = 4, lambda *a, **kw: None
    if name.startswith("qwen") or name == "ernie4_5_moe_vl":
        ip = _mock_ip(image_grid_thw=np.array([[1, 16, 16]], np.int64))
    if name == "qwen3_omni_moe":
        kwargs.update(
            video_processor=NS(model_input_names=[], merge_size=2),
            feature_extractor=NS(model_input_names=[]),
        )
    with _ignore_types(cls):
        return cls(image_processor=ip, tokenizer=tok, **kwargs)


@pytest.mark.parametrize(
    "name,with_image",
    [
        (name, mode == "image")
        for name, case in SMOKE_PROCESSORS.items()
        for mode in case.get("modes", ["image", "text"])
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
    p = object.__new__(c.ocr)
    rendered = Template(p.default_chat_template).render(
        messages=DATA["ocr_messages"], add_generation_prompt=True
    )
    assert rendered == "<image>document parsing. partial continue"


@pytest.mark.parametrize("name,start,end", DATA["cleanup"])
def test_output_control_tokens(name, start, end):
    cls = (
        m.aya_vision.AyaVisionOutputProcessor
        if name == "aya_output"
        else getattr(c, name)
    )
    if name == "aya_output":
        native = importlib.import_module(
            "transformers.models.aya_vision.processing_aya_vision"
        )
        assert issubclass(cls, native.AyaVisionProcessor)
    text = "Title: A calm river cruise\nKeywords: boat, river"
    assert cls.__new__(cls).clean_output(start + text + end) == text


@pytest.mark.parametrize("name,token,token_id", DATA["eos_tokens"])
def test_additional_eos_tokens(name, token, token_id):
    cls = getattr(c, name)
    p = _make_processor(name) if name == "idefics3" else cls.__new__(cls)
    p.tokenizer = ProcessorTokenizer({token: token_id})
    assert p.additional_eos_token_ids == [token_id]


class TestGemma4UnifiedProcessor:
    def _processor(self, video=False):
        video_cls = m.g4u.Gemma4UnifiedVideoProcessor
        video_processor = _gemma_image(video_cls, 70) if video else None
        p = c.g4u(
            image_processor=_gemma_image(),
            tokenizer=GemmaTokenizer(),
            video_processor=video_processor,
            image_seq_length=4,
        )
        assert "video_processor" in p.get_attributes()
        assert isinstance(p.video_processor, m.g4u.Gemma4UnifiedVideoProcessor)
        return p

    def test_merged_image_patches_and_positions(self):
        data, soft_tokens = _gemma_image()(Image.new("RGB", (8, 8)))
        _assert_shapes(data, pixel_values=(1, 4, 48), image_position_ids=(1, 4, 2))
        assert soft_tokens == [4]
        assert data["image_position_ids"][0].tolist() == [
            [x, y] for y in range(2) for x in range(2)
        ]

    def test_video_padding_and_positions(self):
        video = np.zeros((2, 3, 4, 8), np.uint8)
        p = _gemma_image(m.g4.Gemma4VideoProcessor, max_soft_tokens=70)
        assert p.video_sampling_defaults() == {"max_frames": p.num_frames}
        data = p([video], fps=[1.0])
        _assert_shapes(
            data, pixel_values_videos=(1, 2, 280, 12), video_position_ids=(1, 2, 280, 2)
        )
        assert data["num_frames_per_video"] == data["num_soft_tokens_per_frame"] == [2]
        assert data["frame_timestamps"] == [[0.0, 1.0]]
        positions = data["video_position_ids"][0, 0]
        assert positions[:8].tolist() == [[x, y] for y in range(2) for x in range(4)]
        assert np.all(positions[8:] == -1)

    def test_audio_chunk_masks(self):
        extractor = m.g4u.Gemma4UnifiedAudioFeatureExtractor(
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
        else:
            messages = _message(
                {"type": "image", "image": Image.new("RGB", (8, 8))},
                {"type": "text", "text": "Describe this image in detail."},
            )
            result = p.apply_chat_template(messages, **PROFILES["chat_kwargs"])
            assert isinstance(result["input_ids"], mx.array)
        case = DATA["gemma_multimodal"][video]
        _assert_shapes(result, **{k: tuple(v) for k, v in case["shapes"].items()})
        assert isinstance(result[case["pixels"]], mx.array)
        assert case["marker"] in p.tokenizer.last_text[0]
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
        with pytest.raises(ValueError):
            p._expand_placeholders(
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


def test_minicpmv_video_marker_expands_to_frame_bounds():
    tokenizer = ProcessorTokenizer(**PROFILES["minicpm_tokenizer"])
    options = dict(**PROFILES["minicpm_image"])
    with _ignore_types(c.minicpm):
        p = c.minicpm(
            tokenizer=tokenizer,
            image_processor=m.minicpm.MiniCPMVImageProcessor(**options),
            video_processor=m.minicpm.MiniCPMVVideoProcessor(**options),
        )
    result = p(
        text=["<|video_pad|> Describe this."],
        videos=[np.zeros((2, 3, 16, 16), np.uint8)],
        slice_mode=False,
        max_num_frames=2,
        padding=False,
    )
    vp = p.video_processor
    assert vp.video_sampling_defaults() == {"max_frames": vp.max_num_frames}
    assert len(result["pixel_values"][0]) == 2
    assert result["tgt_sizes"][0].shape == result["image_bound"][0].shape == (2, 2)
    assert result["num_frames_per_video"] == [[2]]
    assert result["num_patches_per_frame"] == [[1, 1]]
    for start, end in result["image_bound"][0]:
        assert np.all(result["input_ids"][0, start:end] == 102)


class TestGlmOcrProcessor:
    GEOMETRY = dict(patch_size=14, temporal_patch_size=2, merge_size=2)

    @pytest.mark.parametrize("reference", [False, True], ids=["shape", "torch-parity"])
    def test_image_patches(self, reference):
        p = m.glm_ocr.Glm46VImageProcessor(
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
                equal(value, actual[key])
        else:
            assert actual["image_grid_thw"].tolist() == [[1, 2, 4]]
            assert actual["pixel_values"].shape == (8, 1176)


@pytest.mark.parametrize(
    "module,rows,cols,length", [(m.smolvlm, 3, 4, 81), (m.idefics3, 2, 2, 4)]
)
def test_tiled_image_prompt(module, rows, cols, length):
    for h, w in [(0, 0), (rows, cols)]:
        text = module.get_image_prompt_string(h, w, length, "<F>", "<image>", "<G>")
        assert text.count("<image>") == (h * w + 1) * length
        assert "<G>" in text
        if h:
            assert "<row_1_col_1>" in text
            assert f"<row_{h}_col_{w}>" in text


class TestMllamaProcessor:
    def test_cross_attention_mask_helpers(self):
        mask = m.mllama.get_cross_attention_token_mask(
            [1, 2, 128256, 3, 4, 128256, 5, 6], 128256
        )
        assert mask == [[2, 5], [5, 8]]
        assert m.mllama.get_cross_attention_token_mask([1, 2, 3], 128256) == []
        dense = m.mllama.convert_sparse_cross_attention_mask_to_dense(
            [mask], [[2, 3]], 4, 8
        )
        assert dense.shape == (1, 8, 2, 4)
        assert dense[0, 2, 0, 0] == 1 and dense[0, 0, 0, 0] == 0

    @pytest.mark.parametrize("text,expected", DATA["mllama_prompts"])
    def test_build_string_from_input(self, text, expected):
        assert m.mllama.build_string_from_input(text, "<bos>", "<|image|>") == expected


class TestQwen3VLProcessor:
    def _capturing_processor(self, grids):
        p = _make_processor("qwen3_vl")
        p.image_processor = _mock_ip(
            pixel_values=np.zeros((len(grids), 3, 224, 224), np.float32),
            image_grid_thw=np.array(grids, np.int64),
        )
        p.tokenizer = ProcessorTokenizer({}, pad=False)
        return p

    def test_surplus_tokens_follow_prompt_and_batch_order(self):
        block = "<|vs|><|image_pad|><|ve|>"
        kwargs = dict(
            image_token="<|image_pad|>",
            vision_start_token="<|vs|>",
            vision_end_token="<|ve|>",
            count=1,
        )
        assert (
            m.qwen3._drop_surplus_image_tokens(
                "old <|image_pad|> new " + block, **kwargs
            )
            == "old  new " + block
        )
        p = self._capturing_processor([[1, 4, 4], [1, 4, 8]])
        p(
            text=["first " + block, "stale " + block + " current " + block],
            images=[_make_image(), _make_image()],
        )
        assert p.tokenizer.last_text == [
            prefix + "<|vs|>" + "<|image_pad|>" * n + "<|ve|>"
            for prefix, n in [("first ", 4), ("stale  current ", 8)]
        ]

    @pytest.mark.parametrize(
        "case", DATA["qwen_image_errors"], ids=["ambiguous-flat", "extra-grouped"]
    )
    def test_invalid_image_counts(self, case):
        p = self._capturing_processor([[1, 4, 4], [1, 4, 8], [1, 4, 12]])
        images = [_make_image() for _ in range(3)]
        if case["grouped"]:
            images = [images[:2], images[2:]]
        with pytest.raises(ValueError, match=case["error"]):
            p(text=case["text"], images=images)

    @pytest.mark.parametrize("layout", ["nested-pil", "flat-pil", "hwc-array"])
    def test_video_frame_layouts(self, layout):
        frames = [Image.new("RGB", (224, 224), (i * 40, 128, 128)) for i in range(4)]
        video = {
            "nested-pil": [frames],
            "flat-pil": frames,
            "hwc-array": [np.zeros((4, 224, 224, 3), np.uint8)],
        }[layout]
        p = m.qwen3.Qwen3VLVideoProcessor(**PROFILES["qwen_video"])
        output = p(videos=video)
        equal(output["video_grid_thw"], [[2, 16, 16]])
        assert output["pixel_values_videos"].shape == (512, 1176)


def test_pixtral_image_preprocess_resizes_to_patch_multiple_and_pads():
    image_processor = m.pixtral_ip.PixtralImageProcessor(**PROFILES["pixtral_image"])
    wide = Image.fromarray(np.zeros((31, 55, 3), dtype=np.uint8))
    square = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    output = image_processor([[wide, square]])
    assert output["image_sizes"] == [(28, 42), (28, 28)]
    assert output["pixel_values"].shape == (2, 3, 28, 42)


class TestErnie4_5VLProcessor:
    def test_helper_functions(self):
        for fn, inputs, expected in [
            (m.ernie.round_by_factor, [100, 56, 42], [112, 56, 56]),
            (m.ernie.ceil_by_factor, [100, 56, 57], [112, 56, 84]),
            (m.ernie.floor_by_factor, [100, 56, 55], [84, 56, 28]),
        ]:
            assert [fn(x, 28) for x in inputs] == expected
        h, w = m.ernie.smart_resize(224, 224, factor=28)
        assert h % 28 == w % 28 == 0
        h, w = m.ernie.smart_resize(10, 10, factor=28, min_pixels=56 * 56)
        assert h * w >= 56 * 56
        h, w = m.ernie.smart_resize(10000, 10000, factor=28, max_pixels=28 * 28 * 1280)
        assert h * w <= 28 * 28 * 1280

    def test_image_processor(self):
        p = m.ernie.ImageProcessor()
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


class TestLfm2VlProcessorPatch:
    @pytest.mark.parametrize(
        "batched", [False, True], ids=["tiles", "flat-image-batch"]
    )
    def test_marker_expansion(self, batched):
        ip = m.lfm.Lfm2VlNumpyImageProcessor(max_num_patches=256, do_resize=False)
        assert ip.max_num_patches == 1024 and ip.do_resize
        p = _lfm_processor(ip)
        images = (
            [_make_image(540, 960) for _ in range(3)]
            if batched
            else [_make_image(1440, 2560)]
        )
        suffixes = ["First", "Second", "Third"] if batched else ["Describe this image"]
        prompts = ["<image>" + text for text in suffixes]
        result = m.lfm._patched_call(
            p, images=images, text=prompts if batched else prompts[0]
        )
        assert result["pixel_values"].shape == (3 if batched else 9, 1024, 768)
        expanded = p.tokenizer.last_text
        assert len(expanded) == len(suffixes)
        for text, suffix in zip(expanded, suffixes):
            assert text.count("<|image_start|>") == 1 and text.endswith(suffix)
            assert text.count("<image>") == (252 if batched else 8 * 256 + 252)
        if not batched:
            markers = [f"<|img_row_{r}_col_{c}|>" for r in (1, 2) for c in (1, 2, 3, 4)]
            assert text.startswith("<|image_start|>" + markers[0])
            assert all(marker in text for marker in markers)
            assert text.index(markers[-1]) < text.index("<|img_thumbnail|>")
            assert "<|img_thumbnail|>" + "<image>" * 252 + "<|image_end|>" in text
        raw = ip(images, return_tensors="np", do_resize=False)
        if batched:
            assert int(raw["pixel_attention_mask"].sum()) == sum(
                r * c for r, c in raw["spatial_shapes"]
            )
        else:
            assert raw["spatial_shapes"].tolist() == [[32, 32]] * 8 + [[24, 42]]

    def test_scalar_image_rows_and_cols_are_supported(self):
        result = m.lfm._patched_call(
            _lfm_processor(_stub("lfm_scalar")),
            images=_make_image(),
            text="<image>Describe this image",
        )
        assert {"input_ids", "attention_mask"} <= result.keys()

    def test_resample_filter_follows_the_checkpoint(self):
        for kwargs, expected in [
            ({}, 3),
            ({"resample": 3}, 3),
            ({"resample": 2}, 2),
            ({"resample": Image.Resampling.LANCZOS}, 1),
            ({"resample": "nonsense"}, 3),
        ]:
            assert m.lfm.Lfm2VlNumpyImageProcessor(
                **kwargs
            ).resample is Image.Resampling(expected)
        image = _make_image(540, 960)
        outputs = [
            m.lfm.Lfm2VlNumpyImageProcessor(resample=r)([image], return_tensors="np")[
                "pixel_values"
            ]
            for r in (3, 2)
        ]
        assert not np.allclose(*outputs)

    @pytest.mark.parametrize("layout", ["lists", "tuples", "arrays"])
    def test_mismatched_nested_image_groups(self, layout):
        images = [Image.new("RGB", (64, 64), (i * 60, 0, 0)) for i in range(4)]
        groups = [images[:1], images[1:]]
        if layout == "tuples":
            groups = tuple(tuple(group) for group in groups)
        elif layout == "arrays":
            groups = [np.stack(group) for group in groups]
        with pytest.raises(ValueError, match="text \\[2, 2\\] and images \\[1, 3\\]"):
            m.lfm._patched_call(
                _lfm_processor(),
                images=groups,
                text=["<image><image>Prompt A", "<image><image>Prompt B"],
            )


def test_molmo_point_processor_uses_image_processor_for_images():
    fields = _stub("molmo").outputs
    fields["image_token_pooling"] = fields["image_token_pooling"].astype(np.int64)
    p = c.molmo_point(
        ProcessorTokenizer(bos_token_id=1, eos_token_id=2),
        image_processor=NS(preprocess=lambda images: fields),
    )
    assert (
        fields.keys() <= p(text=m.molmo_point.IMAGE_PROMPT, images=_make_image()).keys()
    )


def test_kimi_chat_tokens_and_media():
    p = c.kimi_k3(tokenizer=KimiTokenizer())
    text = "literal <|end_of_msg|> marker"
    p.apply_chat_template([{"role": "user", "content": text}], tokenize=True)
    for source, split in [(text, True), ("<|end_of_msg|>", False)]:
        matches = [
            kwargs for value, kwargs in p.tokenizer.encode_calls if value == source
        ]
        assert matches and matches[0]["split_special_tokens"] is split
    result = apply_chat_template(
        p, {"model_type": "kimi_k3"}, "Describe this image.", num_images=1
    )
    assert "Describe this image.<|kimi_image_placeholder|>" in result
    assert '<|open|>message role="assistant"' in result
    with pytest.raises(ValueError, match="unsupported media type: video"):
        p(videos=[np.zeros((2, 16, 24, 3), np.uint8)], text="Describe this video.")


def test_locate_anything_save_pretrained_round_trips_custom_config(tmp_path):
    template = "{{ messages }}"
    geometry = dict(patch_size=28, merge_kernel_size=[2, 4], in_token_limit=1234)
    tokenizer = fast_tokenizer(chat_template=template)
    p = c.locateanything(
        image_processor=m.locate_ip.LocateAnythingImageProcessor(**geometry),
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
        m.locateanything.AutoTokenizer, "from_pretrained", return_value=fast_tokenizer()
    ):
        loaded = c.locateanything.from_pretrained(tmp_path)
    _assert_attrs(loaded.image_processor, **geometry)
    assert loaded.chat_template == loaded.tokenizer.chat_template == template


def test_qwen3_vl_video_timestamp_video_prompt_falls_back_to_processor_fps():
    p = _make_processor("qwen3_vl")
    p.tokenizer = ProcessorTokenizer(
        {"<|video_pad|>": 102}, video_token="<|video_pad|>"
    )
    p.image_processor = None
    p.video_processor = _stub("qwen_video")
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
    CHAT_TEMPLATE = DATA["mage_chat_template"]
    IMAGE_CONFIG = DATA["checkpoints"]["mage"]["preprocessor_config"]

    def processor(self):
        tokens = [
            "[UNK]",
            "[PAD]",
            m.mage.IMAGE_PAD,
            m.mage.VIDEO_PAD,
            m.mage.VISION_START,
            m.mage.VISION_END,
        ]
        tokenizer = fast_tokenizer(
            tokens,
            tokenizer_class=RecordingTokenizer,
            split=False,
            additional_special_tokens=tokens[2:],
            chat_template=self.CHAT_TEMPLATE,
        )
        return c.mage(
            image_processor=m.qwen3.Qwen3VLImageProcessor(**self.IMAGE_CONFIG),
            tokenizer=tokenizer,
        )

    @pytest.fixture
    def p(self):
        return self.processor()

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
        equal(output["pixel_values"], expected["pixel_values"])

    @pytest.mark.parametrize("prepare", [False, True], ids=["direct", "prepare-inputs"])
    def test_video_timestamps_positions_and_attention(self, p, prepare):
        indices, fps = ([0, 20, 59], 10) if prepare else ([0, 15, 60], 30)
        metadata = dict(
            total_num_frames=indices[-1] + 1, fps=fps, frames_indices=indices
        )
        call = prepare_inputs if prepare else p
        kwargs = {
            "prompts" if prepare else "text": self.VIDEO_BLOCK,
            "videos": [self.frames()],
            "video_metadata": [metadata if prepare else VideoMetadata(**metadata)],
        }
        output = call(*([p] if prepare else []), **kwargs)
        assert p.tokenizer.last_text == [
            "".join(f"<{i / fps:.1f} seconds>{self.IMAGE_BLOCK}" for i in indices)
        ]
        assert self.image_counts(output) == [3]
        assert not {"pixel_values_videos", "video_grid_thw"} & output.keys()
        assert output["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
        positions = np.array(output["patch_positions"]).reshape(3, 4, 3)
        equal(positions[:, :, 0], [[i] * 4 for i in indices])
        equal(positions[1, :, 1:], [[0, 0], [0, 1], [1, 0], [1, 1]])
        assert m.mage_vision.build_cu_seqlens(
            output["image_grid_thw"].tolist(), 12, 4
        ) == [0, 4, 8, 12]

    @pytest.mark.parametrize("prompts,videos,kwargs,error", DATA["mage_errors"])
    def test_invalid_video(self, p, prompts, videos, kwargs, error):
        if "video_metadata" in kwargs and kwargs["video_metadata"]:
            kwargs = {
                "video_metadata": [VideoMetadata(**v) for v in kwargs["video_metadata"]]
            }
        with pytest.raises(ValueError, match=error):
            p(
                text=self.VIDEO_BLOCK * prompts,
                videos=[self.frames()] * videos,
                **kwargs,
            )

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

    def test_mixed_video_pixels_affect_only_their_visual_embeddings(self, p):
        config = m.mage_config.ModelConfig(
            text_config=m.mage_config.TextConfig(**PROFILES["mage_text"]),
            vision_config=m.mage_config.VisionConfig(**PROFILES["mage_vision"]),
            image_token_id=2,
            video_token_id=3,
        )
        model = m.mage_model.Model(config)
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
        equal(a[visual_indices[0]], b[visual_indices[0]])
        assert not np.allclose(a[visual_indices[1:]], b[visual_indices[1:]])
        assert m.mage_model._as_grid_list(np.array([1, 4, 4])) == [(1, 4, 4)]


def tiny_diffusion_gemma_processor(image_processor=None):
    return c.diffusion(
        image_processor=image_processor,
        tokenizer=TinyDiffusionGemma4Tokenizer(PROFILES["diffusion_tokenizer"]),
        video_processor=_stub("diffusion_video"),
    )


class TestDiffusionGemma4Processor:
    def test_generate_strips_channel_scaffolding(self):
        dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
        model = NS(config=NS(model_type="diffusion_gemma", eos_token_id=999999))
        p = tiny_diffusion_gemma_processor()
        p.tokenizer.stopping_criteria = StoppingCriteria([999999], p.tokenizer)
        chunk = GenerationResult(**PROFILES["generation_chunk"])
        with patch.object(dispatch, "stream_generate", return_value=iter([chunk])):
            assert generate(model, p, "").text == "Title: A calm river cruise"

    def test_video_chat_cross_thread(self):
        def produce():
            p = tiny_diffusion_gemma_processor()
            rendered = apply_chat_template(
                p,
                NS(model_type="diffusion_gemma"),
                "Describe this video.",
                video=["clip.mp4"],
            )
            assert p.video_token in rendered
            return p(text=rendered, videos=[np.zeros((2, 3, 4, 4), np.uint8)])

        def consume(result):
            mx.eval(result)
            return (
                result["pixel_values"].shape,
                mx.sum(result["mm_token_type_ids"] == 2).item(),
            )

        with ThreadPoolExecutor(1) as producer, ThreadPoolExecutor(1) as consumer:
            result = producer.submit(produce).result(timeout=5)
            assert consumer.submit(consume, result).result(timeout=5) == (
                (2, 3, 4, 4),
                2,
            )

    def test_mixed_media_pixel_order(self):
        image = np.ones((3, 4, 4), np.float32)
        ip = _ImageStub(({"pixel_values": image[None]}, [1]))
        p = tiny_diffusion_gemma_processor(image_processor=ip)
        result = p(
            text="<video> then <image>",
            images=[image],
            videos=[np.zeros((2, 3, 4, 4), np.uint8)],
        )
        assert "pixel_values_videos" not in result
        pixels = result["pixel_values"]
        assert pixels.shape == (3, 3, 4, 4)
        assert mx.all(pixels[:2] == 0).item() and mx.all(pixels[2] == 1).item()


@pytest.mark.parametrize("name", DATA["checkpoints"])
def test_checkpoint_loading(tmp_path, name):
    _write_configs(tmp_path, **DATA["checkpoints"][name])
    family = "lfm" if name == "lfm_override" else name
    cls = getattr(c, family, None)
    tok = ProcessorTokenizer(
        **DATA["loader_tokenizers"].get(name, {"chat_template": "{{ messages }}"})
    )
    kwargs = DATA["loader_kwargs"].get(name, {})
    if name == "paddleocr_vl":
        assert load_image_processor(tmp_path) is None
    if name == "mage":
        tok = TestMageVLProcessor().processor().tokenizer
    elif name == "diffusion":
        tok = TinyDiffusionGemma4Tokenizer(PROFILES["diffusion_tokenizer"])
        tok.chat_template = None
    elif name == "deepseek":
        tok = ProcessorTokenizer(
            chat_template=None, apply_chat_template=lambda *a, **kw: "templated"
        )
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(patch("transformers.AutoTokenizer.from_pretrained", return_value=tok))
        if name not in ("glm_ocr", "omni", "mage", "nemotron"):
            base = c.g4 if name == "diffusion" else cls
            enter(_ignore_types(base))
        if name == "glm_ocr":
            enter(patch("mlx_vlm.models.base.load_chat_template"))
        elif family == "lfm":
            enter(patch.object(m.lfm, "Siglip2ImageProcessor", NS, create=True))
            enter(patch.object(m.lfm, "_SLOW_PROCESSOR_AVAILABLE", True))
        elif name == "dots_ocr":
            image_loader = Mock(return_value=_mock_ip())
            enter(
                patch.dict(
                    "transformers.__dict__",
                    {"AutoImageProcessor": NS(from_pretrained=image_loader)},
                )
            )
        elif name == "omni":
            fe = NS(model_input_names=["input_features"], sampling_rate=16000)
            enter(
                patch(
                    "transformers.AutoFeatureExtractor.from_pretrained", return_value=fe
                )
            )
        if name == "nemotron":
            p = load_processor(str(tmp_path), add_detokenizer=False)
        elif name in ("omni", "mage", "diffusion", "deepseek"):
            p = AutoProcessor.from_pretrained(tmp_path)
        else:
            p = cls.from_pretrained(tmp_path, **kwargs)
        if name == "deepseek":
            explicit = cls.from_pretrained("repo/name", chat_template="{{ explicit }}")
            assert explicit.chat_template == "{{ explicit }}"
            template = "{{ messages[0]['content'] }}"
            (tmp_path / "chat_template.jinja").write_text(template)
            assert m.deepseek.load_deepseek_v4_chat_template(tmp_path) == template
    assert isinstance(p, cls)
    _assert_attrs(p, **DATA["loader_attributes"].get(name, {}))
    for attr, expected in DATA["loader_components"].get(name, {}).items():
        assert type(getattr(p, attr)).__name__ == expected
    if name == "dots_ocr":
        image_loader.assert_called_once()
        assert image_loader.call_args.kwargs["use_fast"] is False
    elif name == "diffusion":
        assert p.get_attributes() == ["image_processor", "tokenizer", "video_processor"]
    elif name == "mistral3":
        out = p(text=["[IMG]Describe"], images=[[_make_image()]])
        _assert_all_mx(out, "pixel_values")
        assert out["pixel_values"].shape[:2] == (1, 3)
        assert all(int(size) % 28 == 0 for size in out["image_sizes"][0].tolist())
    elif name == "nemotron":
        out = prepare_inputs(
            p,
            images=[_make_image()],
            prompts="<image>\nDescribe this image.",
            image_token_index=p.image_token_id,
        )
        assert "pixel_values" in out and int(out["num_tokens"][0].item()) > 0
    elif name == "mage":
        assert processor_handles_video(p)
        _assert_attrs(
            resolve_video_sampling(p, {}), min_frames=1, frame_factor=1, max_frames=32
        )
        assert TestMageVLProcessor.image_counts(
            p(text=m.mage.VIDEO_PAD, videos=[TestMageVLProcessor.frames()])
        ) == [3]
