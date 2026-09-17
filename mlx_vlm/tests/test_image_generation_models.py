"""Image generation/editing contracts and model-specific numerical checks."""

from __future__ import annotations

import importlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

import mlx_vlm.models.bonsai.config as bonsai_config
import mlx_vlm.models.bonsai.download as bonsai_download
import mlx_vlm.models.bonsai.pipeline as bonsai_pipeline
import mlx_vlm.models.ernie_image.config as ernie_config
import mlx_vlm.models.ernie_image.convert as ernie_convert
import mlx_vlm.models.ernie_image.download as ernie_download
import mlx_vlm.models.ernie_image.model as ernie_model
import mlx_vlm.models.ernie_image.pipeline as ernie_pipeline
import mlx_vlm.models.ernie_image.text_encoder as ernie_text_encoder
import mlx_vlm.models.ernie_image.transformer as ernie_transformer
import mlx_vlm.models.ernie_image.weights as ernie_weights
import mlx_vlm.models.flux2.config as flux_config
import mlx_vlm.models.flux2.download as flux_download
import mlx_vlm.models.flux2.model as flux_model
import mlx_vlm.models.flux2.pipeline as flux_pipeline
import mlx_vlm.models.flux2.vae as flux_vae
import mlx_vlm.models.flux2.weights as flux_weights
import mlx_vlm.models.ideogram4.config as ideogram_config
import mlx_vlm.models.ideogram4.download as ideogram_download
import mlx_vlm.models.ideogram4.pipeline as ideogram_pipeline
import mlx_vlm.models.ideogram4.prompting as ideogram_prompting
import mlx_vlm.models.ideogram4.transformer as ideogram_transformer
import mlx_vlm.models.ideogram4.weights as ideogram_weights
import mlx_vlm.models.mage_flow.config as mage_config
import mlx_vlm.models.mage_flow.convert as mage_convert
import mlx_vlm.models.mage_flow.download as mage_download
import mlx_vlm.models.mage_flow.model as mage_model
import mlx_vlm.models.mage_flow.scheduler as mage_scheduler
import mlx_vlm.models.mage_flow.transformer as mage_transformer
import mlx_vlm.models.mage_flow.weights as mage_weights
import mlx_vlm.models.z_image.config as z_config
import mlx_vlm.models.z_image.convert as z_convert
import mlx_vlm.models.z_image.model as z_model
import mlx_vlm.models.z_image.pipeline as z_pipeline
import mlx_vlm.models.z_image.vae as z_vae
from mlx_vlm.generate.edit_image import (
    ImageEditRequest,
    image_edit_model_class,
    is_image_edit_model,
    load_image_edit_model,
)
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    generate_image,
    image_generation_model_class,
    is_image_generation_model,
)

image_module = importlib.import_module("mlx_vlm.generate.image")
dispatch_module = importlib.import_module("mlx_vlm.generate.dispatch")
prompt_utils_module = importlib.import_module("mlx_vlm.prompt_utils")
structured_module = importlib.import_module("mlx_vlm.structured")
utils_module = importlib.import_module("mlx_vlm.utils")

IMAGE_CASES = json.loads(
    Path(__file__).with_name("image_generation_cases.json").read_text()
)
FAMILY_PREFIX = dict(
    bonsai="Bonsai",
    flux2="Flux2",
    ideogram4="Ideogram4",
    z_image="ZImage",
    ernie_image="ErnieImage",
    mage_flow="MageFlow",
)


def _family_module(family, component):
    return importlib.import_module(f"mlx_vlm.models.{family}.{component}")


def _model_class(family, edit=False):
    name = FAMILY_PREFIX[family].removesuffix("Image") + "Image"
    return getattr(
        _family_module(family, "model"),
        name + ("EditModel" if edit else "GenerationModel"),
    )


@pytest.mark.parametrize(
    "family,alias,probe",
    [
        ("z_image", None, None),
        ("ernie_image", None, "baidu/ERNIE-Image-Turbo"),
        ("mage_flow", "microsoft/Mage-Flow-Edit-Turbo", "mage-flow-edit-base"),
    ],
)
def test_edit_model_dispatch(tmp_path, family, alias, probe):
    _write_layout(tmp_path, family)
    cls = _model_class(family, edit=True)
    # Mage's edit ID is deliberately distinct from its generation checkpoint.
    model_id = alias or str(tmp_path)
    assert cls.supports_model(model_id)
    assert image_edit_model_class(model_id) is cls
    assert is_image_edit_model(probe or model_id)
    if family == "mage_flow":
        assert not is_image_generation_model("mage-flow-edit")
        assert not is_image_edit_model("mage-flow-turbo")


def _component(family, check, config):
    component, suffix = {
        "transformer": ("transformer", "Transformer"),
        "text_encoder": ("text_encoder", "TextEncoder"),
        "vae_decode": ("vae", "VAE"),
    }[check]
    name = FAMILY_PREFIX[family] + suffix
    cls = getattr(_family_module(family, component), name)
    if family == "mage_flow":
        return cls(**config)
    config_cls = getattr(_family_module(family, "config"), name + "Config")
    return cls(config_cls(**config))


class _CaptureLength(nn.Module):
    def __call__(self, hidden, *args, **kwargs):
        self.length = hidden.shape[1]
        return hidden


def _component_output(family, check, model):
    if check == "text_encoder":
        return model(mx.array([[1, 2, 3, 4, 5]])), (1, 5, 64)
    if check == "vae_decode":
        return model.decode(mx.random.normal((1, 4, 4, 4))), (1, 8, 8, 3)
    if family == "ideogram4":
        return model(
            llm_features=mx.zeros((1, 3, 8)),
            x=mx.zeros((1, 3, 4)),
            t=mx.array([0.5]),
            position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
            segment_ids=mx.ones((1, 3), dtype=mx.int32),
            indicator=mx.array(
                [
                    [
                        ideogram_transformer.LLM_TOKEN_INDICATOR,
                        ideogram_transformer.OUTPUT_IMAGE_INDICATOR,
                        ideogram_transformer.OUTPUT_IMAGE_INDICATOR,
                    ]
                ]
            ),
        ), (1, 3, 4)
    if family == "ernie_image":
        return model(
            mx.zeros((2, 8, 2, 2), dtype=mx.bfloat16),
            timestep=mx.array([1000.0, 500.0], dtype=mx.bfloat16),
            text_hidden_states=mx.zeros((2, 3, 16), dtype=mx.bfloat16),
            text_lengths=mx.array([1, 3]),
        ), (2, 8, 2, 2)
    if family == "mage_flow":
        return model(
            img=mx.zeros((1, 4, 8)),
            txt=mx.zeros((1, 3, 16)),
            timesteps=mx.array([1.0]),
            img_shapes=[(1, 2, 2)],
        ), (1, 4, 8)
    assert family == "z_image", f"Missing forward adapter for {family}"
    # Z-Image checks padding at all three stages, in addition to output shape.
    noise, context, unified = [_CaptureLength() for _ in range(3)]
    model.noise_refiner, model.context_refiner, model.layers = (
        [noise],
        [context],
        [unified],
    )
    output = model(
        mx.random.normal((1, 16, 1, 4, 4)),
        mx.array([0.5]),
        mx.random.normal((1, 8, 32)),
    )
    assert (noise.length, context.length, unified.length) == (32, 32, 64)
    return output, (1, 16, 1, 4, 4)


@pytest.mark.parametrize(
    "case,check",
    [
        pytest.param(case, check, id=f"{case['id']}-{check}")
        for case in IMAGE_CASES["models"]
        for check in case["checks"]
    ],
)
def test_image_model_contract(case, check):
    model = _component(case["id"], check, case["config"][check])
    output, shape = _component_output(case["id"], check, model)
    mx.eval(output)
    assert output.shape == shape
    assert bool(mx.all(mx.isfinite(output)))


EXPANDED_CAPTION = ideogram_prompting.format_caption(
    {
        "high_level_description": "A detailed photo of a red cube.",
        "style_description": {
            "aesthetics": "minimal product photography",
            "lighting": "soft studio lighting",
            "photo": "close-up product photo",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": "A quiet studio with a matte grey backdrop.",
            "elements": [{"type": "obj", "desc": "A translucent red glass cube."}],
        },
    }
)


def _pipeline(family, **components):
    cls = getattr(
        _family_module(family, "pipeline"),
        FAMILY_PREFIX[family].removesuffix("Image") + "ImagePipeline",
    )
    pipeline = cls.__new__(cls)
    for name, value in components.items():
        setattr(pipeline, name, value)
    return pipeline


def _flux_edit_pipeline(
    variant: str = "flux2-klein-9b-kv",
) -> flux_pipeline.Flux2ImageEdit:
    pipeline = flux_pipeline.Flux2ImageEdit.__new__(flux_pipeline.Flux2ImageEdit)
    pipeline.variant = flux_config.get_variant(variant)
    pipeline.model_path = None
    pipeline.runtime_config = flux_pipeline.Flux2RuntimeConfig(tiled_vae="off")
    pipeline.tokenizer = type("Tokenizer", (), {"count_tokens": lambda self, text: 7})()
    pipeline.edit_array = lambda *args, **kwargs: mx.zeros((20, 24, 3), dtype=mx.uint8)
    return pipeline


class _CaptionTokenizer:
    def apply_chat_template(
        self, messages, add_generation_prompt, tokenize
    ):  # noqa: ANN001, ARG002
        return messages[0]["content"][0]["text"]

    def __call__(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        return {"input_ids": [11, 22, 33]}


class _IdeogramVAE:
    def __init__(self) -> None:
        self.latents = None

    def decode(self, latents):
        self.latents = latents
        return mx.zeros((1, 3, 32, 32), dtype=mx.float32)

    def decode_packed_latents(self, packed):  # noqa: ANN001
        raise AssertionError("Ideogram decode should use latent_norm + vae.decode")


class _RecordingPipeline:
    """Record wrapper arguments while supplying each family's return convention."""

    model_path = Path("/tmp/image-model")
    quantization_config = last_revised_prompt = None

    def __init__(self, family, variant):
        self.family, self.calls = family, []
        if family == "z_image":
            steps, guidance, shift = (
                (50, 4.0, 6.0) if variant == "base" else (9, 0.0, 3.0)
            )
            self.config = z_config.ZImageConfig(
                default_steps=steps,
                default_guidance=guidance,
                scheduler_shift=shift,
                variant=variant,
            )
        elif family == "ernie_image":
            self.variant = ernie_config.get_variant(variant)
            self.runtime_config = ernie_pipeline.ErnieImageRuntimeConfig(
                use_prompt_enhancer=False
            )
        else:
            self.variant = SimpleNamespace(name=variant)

    def generate_array(self, prompt, **kwargs):
        self.calls.append(dict(prompt=prompt, **kwargs))
        if self.family == "ideogram4":
            return mx.zeros((8, 10, 3), dtype=mx.uint8), dict(
                steps=kwargs["steps"], guidance=kwargs["guidance"], prompt_tokens=3
            )
        return mx.zeros((16, 16, 3), dtype=mx.uint8)

    def edit_array(self, prompt, image, **kwargs):
        self.calls.append(dict(prompt=prompt, image=image, **kwargs))
        return mx.zeros((16, 32, 3), dtype=mx.uint8)

    def count_prompt_tokens(self, prompt):
        return 1 if self.family == "z_image" else 3

    def _should_enhance_prompt(self, *, for_edit=False):
        return False


@pytest.mark.parametrize("case", IMAGE_CASES["wrappers"], ids=lambda case: case["id"])
def test_image_wrapper_result(case):
    family, variant = case["family"], case["variant"]
    edit = case["task"] == "edit"
    options = dict(case["request"])
    if edit:
        options["image_paths"] = tuple(options["image_paths"])
    image_request = (ImageEditRequest if edit else ImageGenerationRequest)(**options)
    expected, forwarded, metadata = (
        case.get(k, {}) for k in ("expected", "forwarded", "metadata")
    )
    pipeline = (
        _flux_edit_pipeline(variant)
        if family == "flux2"
        else _RecordingPipeline(family, variant)
    )
    cls = _model_class(family, edit)
    model = cls(pipeline=pipeline, model_id=variant)
    result = (
        model.edit(image_request)
        if edit
        else (
            generate_image(model, image_request)
            if family == "ernie_image"
            else model.generate(image_request)
        )
    )
    assert {k: getattr(result, k) for k in expected} == expected
    for key, value in metadata.items():
        assert result.metadata[key] == value
        if isinstance(value, bool):
            assert result.metadata[key] is value
    if forwarded:
        assert {k: pipeline.calls[-1][k] for k in forwarded} == forwarded


class _ErnieTransformer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, hidden_states, **kwargs):
        self.calls.append((hidden_states.shape, kwargs["text_lengths"]))
        batch = hidden_states.shape[0]
        values = mx.arange(batch, dtype=hidden_states.dtype).reshape(batch, 1, 1, 1)
        return mx.broadcast_to(values, hidden_states.shape)


class _ErnieVAE:
    quantization_config = None

    def __init__(self) -> None:
        self.encoder = object()
        self.bn = SimpleNamespace(
            running_mean=mx.zeros((128,)), running_var=mx.ones((128,))
        )

    def encode(self, pixels):
        return mx.zeros(
            (pixels.shape[0], 32, pixels.shape[2] // 8, pixels.shape[3] // 8),
            dtype=mx.bfloat16,
        )

    def decode_packed_latents(self, latents):
        return mx.zeros((1, 3, latents.shape[2] * 16, latents.shape[3] * 16))


def _ernie_runtime_pipeline(
    variant: str, *, evict: bool = False
) -> ernie_pipeline.ErnieImagePipeline:
    pipeline = _pipeline("ernie_image")
    pipeline.variant = ernie_config.get_variant(variant)
    pipeline.model_path = Path("/tmp/ernie")
    pipeline.runtime_config = ernie_pipeline.ErnieImageRuntimeConfig(
        evict_text_encoder=False, evict_transformer=evict, use_prompt_enhancer=False
    )
    pipeline.tokenizer = SimpleNamespace(count_tokens=lambda prompt: len(prompt))
    pipeline.text_encoder = None
    pipeline.prompt_enhancer = None
    pipeline.component_quantization = {}
    pipeline.transformer = _ErnieTransformer()
    pipeline.vae = _ErnieVAE()
    pipeline.prompt_cache = {}
    pipeline._encode_prompts = lambda prompts: (
        mx.zeros((len(prompts), 3, 4), dtype=mx.bfloat16),
        mx.array([1, 3] if len(prompts) == 2 else [3]),
    )
    pipeline._ensure_components = lambda **kwargs: None
    return pipeline


def _write_files(root, files=(), metadata=None):
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    for relative, content in (metadata or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))


def _write_layout(root, family, *, turbo=True):
    components = ["transformer", "text_encoder", "vae"]
    if family == "bonsai":
        files = list(bonsai_download.REQUIRED_FILES)
    elif family == "ernie_image":
        files = [f"{c}/0.safetensors" for c in components] + [
            "tokenizer/tokenizer.json"
        ]
    elif family == "z_image":
        files = [
            f"{c}/{f}" for c in components for f in ("config.json", "model.safetensors")
        ]
        files += ["scheduler/scheduler_config.json", "tokenizer/tokenizer.json"]
    else:
        if family == "ideogram4":
            components.append("unconditional_transformer")
        files = [
            f"{c}/"
            + (
                "model.safetensors"
                if c == "text_encoder"
                else "diffusion_pytorch_model.safetensors"
            )
            for c in components
        ]
        if family != "flux2":
            files += [f"{c}/config.json" for c in components]
        tokenizer_dir = "text_encoder" if family == "mage_flow" else "tokenizer"
        files.append(f"{tokenizer_dir}/tokenizer.json")
    classes = dict(
        ideogram4="Ideogram4Pipeline",
        ernie_image="ErnieImagePipeline",
        mage_flow="MageFlowPipeline",
        z_image="ZImagePipeline",
    )
    metadata = (
        {"model_index.json": {"_class_name": classes[family]}}
        if family in classes
        else {}
    )
    if family == "mage_flow":
        metadata.update({f: {} for f in files if f.endswith(".json")})
    elif family == "z_image":
        metadata.update({f: {"_class_name": "ZImagePipeline"} for f in files})
    elif family == "ernie_image":
        metadata["mlx_ernie_image.json"] = dict(
            model_type="ernie_image",
            variant="ernie-image-turbo" if turbo else "ernie-image",
        )
    _write_files(root, files, metadata)


class _PackedTransformer:
    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["hidden_states"])


class _PackedVAE:
    def decode_packed_latents(self, packed, tiling_config=None):
        return mx.zeros(
            (1, 3, packed.shape[2] * 16, packed.shape[3] * 16), dtype=mx.bfloat16
        )


def _packed_pipeline(family):
    cls, runtime, variant = {
        "bonsai": (
            bonsai_pipeline.BonsaiImage,
            bonsai_pipeline.BonsaiRuntimeConfig,
            bonsai_config.get_variant("ternary"),
        ),
        "flux2": (
            flux_pipeline.Flux2Image,
            flux_pipeline.Flux2RuntimeConfig,
            flux_config.get_variant("flux2-klein-4b"),
        ),
    }[family]
    pipeline = cls.__new__(cls)
    pipeline.variant, pipeline.model_path = variant, None
    pipeline.runtime_config = runtime(tiled_vae="off")
    pipeline.transformer, pipeline.vae = _PackedTransformer(), _PackedVAE()
    pipeline.tokenizer = None
    pipeline._encode_prompt = lambda prompt, max_sequence_length: (
        mx.zeros((1, 32, 7680), dtype=mx.bfloat16),
        mx.zeros((1, 32, 4), dtype=mx.int32),
    )
    pipeline._ensure_transformer_and_vae = lambda: None
    return pipeline


class _TinyLinear(nn.Module):
    def __init__(self, output_size=32):
        super().__init__()
        self.proj = nn.Linear(64, output_size, bias=False)


def _quantization_metadata(mode, bits, group_size):
    return dict(
        quantization_mode=mode,
        quantization_level=str(bits),
        quantization_group_size=str(group_size),
    )


def _weight_index(root, keys):
    _write_files(
        root,
        metadata={
            "model.safetensors.index.json": {
                "weight_map": {k: "model.safetensors" for k in keys}
            }
        },
    )


@pytest.mark.parametrize("family,steps", [("bonsai", 4), ("flux2", 1)])
def test_packed_pipeline_image_output(family, steps):
    image = _packed_pipeline(family).generate(
        "prompt", seed=7, steps=steps, width=512, height=512
    )
    assert image.size == (512, 512)


@pytest.mark.parametrize("family", ["bonsai", "flux2"])
def test_packed_pipeline_empty_prompt(family):
    with pytest.raises(ValueError, match="prompt"):
        _packed_pipeline(family).generate("", width=512, height=512)


@pytest.mark.parametrize(
    "family,width,height",
    [
        pytest.param(family, w, h, id=f"{family}-{w}x{h}")
        for family, sizes in [
            ("bonsai", [(255, 512), (512, 2050), (513, 512)]),
            ("flux2", [(255, 512), (512, 2050), (513, 512)]),
            ("ideogram4", [(255, 512), (512, 2050), (2048, 256)]),
            ("z_image", [(0, 512), (513, 512), (512, 513)]),
            ("ernie_image", [(15, 16), (1025, 1024), (1024, 1000)]),
            ("mage_flow", [(511, 512), (512, 2049), (513, 512), (512, 496)]),
        ]
        for w, h in sizes
    ],
)
def test_invalid_dimensions(family, width, height):
    if family == "z_image":
        pipeline = _pipeline("z_image")
        with pytest.raises(ValueError, match="positive multiple of 16"):
            pipeline.generate_array("prompt", width=width, height=height)
    else:
        config = dict(
            bonsai=bonsai_config,
            flux2=flux_config,
            ideogram4=ideogram_config,
            ernie_image=ernie_config,
            mage_flow=mage_config,
        )[family]
        with pytest.raises(ValueError):
            config.validate_dimensions(width=width, height=height)


@pytest.mark.parametrize(
    "family,local",
    [("bonsai", True), ("bonsai", False), ("flux2", False), ("mage_flow", False)],
)
def test_model_download(monkeypatch, tmp_path, family, local):
    module, variant, repo = {
        "bonsai": (
            bonsai_download,
            "ternary",
            "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        ),
        "flux2": (flux_download, "flux2-klein-9b", "black-forest-labs/FLUX.2-klein-9B"),
        "mage_flow": (
            mage_download,
            "mage-flow-turbo",
            "mage-flow-community/Mage-Flow-Turbo",
        ),
    }[family]
    cached = (
        tmp_path / "bonsai-image-4B-ternary-mlx"
        if local
        else tmp_path / "hf-cache" / "snapshot"
    )
    _write_layout(cached, family)
    snapshot = MagicMock(return_value=str(cached))
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    if family == "flux2":
        monkeypatch.setattr(module, "find_valid_cached_snapshot", lambda variant: None)
    options = dict(models_dir=tmp_path, max_workers=3) if local else dict(max_workers=2)
    assert module.download_model(variant, **options) == cached
    kwargs = snapshot.call_args.kwargs
    assert kwargs["repo_id"] == repo and kwargs["max_workers"] == options["max_workers"]
    if local:
        assert kwargs["local_dir"] == str(cached)
    elif family != "mage_flow":
        assert "local_dir" not in kwargs
    if family != "bonsai":
        assert kwargs["allow_patterns"] == list(module.DOWNLOAD_PATTERNS)
        if family == "flux2":
            assert "model_index.json" in kwargs["allow_patterns"]


@pytest.mark.parametrize(
    "family,mode,bits,group_size",
    [
        pytest.param(family, mode, bits, group, id=f"{family}-{mode}")
        for family in ("ernie_image", "z_image")
        for mode, bits, group in [
            ("mxfp4", 4, 32),
            ("mxfp8", 8, 32),
            ("nvfp4", 4, 16),
            ("affine", 4, 32 if family == "ernie_image" else 64),
        ]
    ],
)
def test_quantized_weight_loading(family, mode, bits, group_size):
    options = dict(mode=mode, bits=bits, group_size=group_size)
    if family == "ernie_image":
        model = _TinyLinear(16)
        dense = mx.arange(16 * 64, dtype=mx.float32).reshape(16, 64)
        arrays = mx.quantize(dense, **options)
        weights = dict(zip(["proj.weight", "proj.scales", "proj.biases"], arrays))
        load = ernie_weights.apply_weights
    else:
        model = _TinyLinear()
        quantized = _TinyLinear()
        nn.quantize(quantized, **options)
        weights = dict(tree_flatten(quantized.parameters()))
        from mlx_vlm.models.z_image.weights import _apply_weights as load
    loaded = load(model, weights, _quantization_metadata(mode, bits, group_size))
    assert loaded.quantization_config == options
    if family == "ernie_image":
        assert isinstance(loaded.proj, nn.QuantizedLinear)
    elif mode == "affine":
        assert hasattr(loaded.proj, "biases")


@pytest.mark.parametrize(
    "convert", [ernie_convert, mage_convert], ids=["ernie_image", "mage_flow"]
)
def test_incompatible_quantization_options(convert):
    with pytest.raises(ValueError, match="requires"):
        convert._quantization_parameters("mxfp8", 64, 8)


@pytest.mark.parametrize(
    "family,aliases,probe",
    [
        ("bonsai", ["bonsai-ternary"], "bonsai-ternary"),
        (
            "flux2",
            ["flux2-klein-4b", "black-forest-labs/FLUX.2-klein-9B"],
            "klein-base-9b",
        ),
        ("ideogram4", ["ideogram-ai/ideogram-4-fp8"], None),
        ("z_image", ["Tongyi-MAI/Z-Image"], None),
        ("ernie_image", ["baidu/ERNIE-Image-Turbo"], "ernie-image"),
        ("mage_flow", ["microsoft/Mage-Flow"], "mage-flow-base"),
    ],
    ids=list(FAMILY_PREFIX),
)
def test_generation_model_dispatch(monkeypatch, tmp_path, family, aliases, probe):
    cls = _model_class(family)
    _write_layout(tmp_path, family)
    if family == "bonsai":
        _write_files(
            tmp_path,
            metadata={
                "manifest.json": {
                    "files": [
                        {"remote_path": p}
                        for p in (
                            "transformer-packed-mflux/diffusion_pytorch_model.safetensors",
                            "text_encoder-mlx-4bit/model.safetensors",
                            "tokenizer/tokenizer.json",
                        )
                    ]
                }
            },
        )
    elif family == "flux2":
        _write_files(
            tmp_path,
            metadata={
                "text_encoder/config.json": {"hidden_size": 4096},
                "model_index.json": {"_class_name": "Flux2KleinPipeline"},
            },
        )
    lookup = MagicMock(return_value=tmp_path)
    monkeypatch.setattr(image_module, "get_model_path", lookup)
    assert cls.is_image_generation_model and cls.model_type == family
    assert cls.supports_model(str(tmp_path))
    for model_id in [*aliases, str(tmp_path)]:
        assert image_generation_model_class(model_id) is cls
    assert is_image_generation_model(probe or str(tmp_path))
    if family == "bonsai":
        assert not is_image_generation_model("mlx-community/nanoLLaVA-1.5-8bit")
    elif family in {"flux2", "ideogram4"}:
        assert all(c.args == (aliases[-1],) for c in lookup.call_args_list)
        if family == "ideogram4":
            assert all(
                c.kwargs["allow_patterns"]
                == [
                    "model_index.json",
                    "config.json",
                    "manifest.json",
                    "**/config.json",
                ]
                for c in lookup.call_args_list
            )


@pytest.mark.parametrize(
    "family,module,missing",
    [
        ("bonsai", bonsai_download, "transformer-packed-mflux"),
        ("flux2", flux_download, "transformer"),
        ("ideogram4", ideogram_download, "transformer"),
        ("mage_flow", mage_download, "transformer"),
    ],
    ids=["bonsai", "flux2", "ideogram4", "mage_flow"],
)
def test_missing_checkpoint_files(tmp_path, family, module, missing):
    if family == "ideogram4":
        _write_files(
            tmp_path,
            metadata={"model_index.json": {"_class_name": "Ideogram4Pipeline"}},
        )
    with pytest.raises(FileNotFoundError, match=missing):
        module.validate_model_layout(tmp_path)
    if family == "mage_flow":
        _write_layout(tmp_path, family)
        assert module.validate_model_layout(tmp_path) == tmp_path


@pytest.mark.parametrize(
    "family,module,method,repo",
    [
        ("z_image", z_convert, "convert_z_image", "Tongyi-MAI/Z-Image-Turbo"),
        (
            "ernie_image",
            ernie_convert,
            "convert_ernie_image",
            "baidu/ERNIE-Image-Turbo",
        ),
    ],
    ids=["z_image", "ernie_image"],
)
@pytest.mark.parametrize("mode,bits,group_size", [("affine", 4, 64), ("mxfp8", 8, 32)])
def test_conversion_resolves_hub_model(
    monkeypatch, tmp_path, family, module, method, repo, mode, bits, group_size
):
    source, output = tmp_path / "source", tmp_path / "output"
    if family == "z_image":
        _write_files(
            source, metadata={"model_index.json": {"_class_name": "ZImagePipeline"}}
        )
    else:
        _write_layout(source, family)
    monkeypatch.setattr(module, "get_model_path", lambda *a, **kw: source)
    convert = MagicMock(return_value=output)
    monkeypatch.setattr(module, method, convert)
    extra = dict(q_group_size=None, q_bits=None) if family == "ernie_image" else {}
    assert module.convert(repo, output, quantize=True, q_mode=mode, **extra) == output
    assert convert.call_args.args == (source, output)
    options = convert.call_args.kwargs
    assert options["q_mode"] == mode
    if family == "z_image":
        assert options["quantize_vae"] is False
        assert (options["q_bits"], options["q_group_size"]) == (bits, group_size)
    else:
        assert options["source_id"] == repo
        params = module._quantization_parameters(
            mode, options["q_group_size"], options["q_bits"]
        )
        assert (params["bits"], params["group_size"]) == (bits, group_size)


@pytest.mark.parametrize(
    "family,override,expected",
    [
        ("flux2", False, "flux2-klein-9b"),
        ("flux2", True, "flux2-klein-9b-kv"),
        ("ernie_image", False, "ernie-image"),
        ("ernie_image", True, "ernie-image-turbo"),
    ],
)
def test_local_variant_precedence(tmp_path, family, override, expected):
    _write_layout(tmp_path, family, turbo=False)
    if family == "flux2":
        _write_files(
            tmp_path,
            metadata={
                "transformer/config.json": {"num_layers": 8, "num_attention_heads": 32}
            },
        )
        if override:
            _write_files(tmp_path, ["flux-2-klein-9b-kv.safetensors"])
    elif override:
        metadata_path = tmp_path / "mlx_ernie_image.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["source"] = "baidu/ERNIE-Image-Turbo"
        metadata_path.write_text(json.dumps(metadata))
    assert (
        _family_module(family, "config").variant_from_local_path(tmp_path).name
        == expected
    )


def test_bonsai_variant_aliases_are_ternary_only():
    assert bonsai_config.get_variant("bonsai").precision == "2bit"
    assert bonsai_config.get_variant("bonsai-ternary").name == "ternary"
    assert (
        bonsai_config.get_variant("prism-ml/bonsai-image-ternary-4B-mlx-2bit").name
        == "ternary"
    )
    with pytest.raises(ValueError, match="Unknown Bonsai variant"):
        bonsai_config.get_variant("binary")


def test_bonsai_parse_size():
    assert bonsai_config.parse_size("1248x832") == (1248, 832)
    assert bonsai_config.parse_size("832x1248") == (832, 1248)


def test_flux2_remote_component_index_is_a_metadata_fallback(monkeypatch, tmp_path):
    metadata_path = tmp_path / "metadata"
    metadata_path.mkdir()
    component_index_path = tmp_path / "component-index"
    _write_layout(component_index_path, "flux2")
    _weight_index(
        component_index_path / "transformer",
        [
            "time_guidance_embed.linear_1.weight",
            "double_stream_modulation_img.linear.weight",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        ],
    )
    calls = []

    def fake_get_model_path(repo_id: str, **kwargs):
        assert repo_id == "example/custom-quantized-model"
        calls.append(kwargs["allow_patterns"])
        return metadata_path if len(calls) == 1 else component_index_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)

    assert (
        image_generation_model_class("example/custom-quantized-model")
        is flux_model.Flux2ImageGenerationModel
    )
    assert calls == [
        ["model_index.json", "config.json", "manifest.json", "**/config.json"],
        [
            "model_index.json",
            "config.json",
            "manifest.json",
            "**/config.json",
            "**/model.safetensors.index.json",
        ],
    ]


def test_flux2_text_encoder_accepts_native_quantized_keys(monkeypatch, tmp_path):
    class TinyTextEncoder(nn.Module):
        def __init__(self, **kwargs) -> None:  # noqa: ARG002
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 64)

    dense = mx.arange(4 * 64, dtype=mx.float32).reshape(4, 64)
    packed, scales, biases = mx.quantize(dense, group_size=32, bits=8)
    monkeypatch.setattr(flux_weights, "Qwen3TextEncoder", TinyTextEncoder)
    monkeypatch.setattr(
        flux_weights,
        "_load_safetensors",
        lambda directory: (  # noqa: ARG005
            {
                "embed_tokens.weight": packed,
                "embed_tokens.scales": scales,
                "embed_tokens.biases": biases,
            },
            {},
        ),
    )

    model = flux_weights.load_text_encoder(
        tmp_path, flux_config.get_variant("flux2-klein-4b")
    )

    assert isinstance(model.embed_tokens, nn.QuantizedEmbedding)
    assert model.quantization_config["bits"] == 8


@pytest.mark.parametrize(
    "family,shape", [("flux2", (8, 4, 3, 3)), ("ernie_image", (2, 3, 3, 3))]
)
def test_conv_layout_source_and_native(family, shape):
    weights = _family_module(family, "weights")
    match = (
        weights._match_conv_layout if family == "flux2" else weights.match_conv_layout
    )
    source = mx.arange(np.prod(shape).item()).reshape(shape)
    native = source.transpose(0, 2, 3, 1)
    for value, layout in [(source, "pytorch_nchw"), (native, "mlx_nhwc")]:
        options = dict(source_layout=layout) if family == "ernie_image" else {}
        result = match(
            value, target_shape=native.shape, key="encoder.conv_in.weight", **options
        )
        np.testing.assert_array_equal(np.array(result), np.array(native))


def test_flux2_quantized_repo_routes_through_resolved_layout(monkeypatch, tmp_path):
    model_path = (
        tmp_path
        / "models--mlx-community--flux2-klein-4b-8bit"
        / "snapshots"
        / "9beac1a3ad296d9e5e3f8845674e6577fa8654ec"
    )
    _write_layout(model_path, "flux2")
    _weight_index(
        model_path / "transformer",
        [
            "time_guidance_embed.linear_1.weight",
            "double_stream_modulation_img.linear.weight",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        ],
    )
    calls = {}

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        calls["repo_id"] = repo_id
        return model_path

    def fake_from_pretrained(cls, variant, **kwargs):  # noqa: ARG001
        calls["variant"] = variant.name
        calls["model_path"] = kwargs["model_path"]
        return _flux_edit_pipeline("flux2-klein-4b")

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)
    monkeypatch.setattr(
        flux_pipeline.Flux2ImageEdit,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    model = load_image_edit_model("mlx-community/flux2-klein-4b-8bit")

    assert model.variant == "flux2-klein-4b"
    assert calls == {
        "repo_id": "mlx-community/flux2-klein-4b-8bit",
        "variant": "flux2-klein-4b",
        "model_path": model_path,
    }


def test_flux2_reference_image_array_keeps_float32_input():
    image = Image.new("RGB", (1, 1), color=(255, 127, 0))
    array = flux_pipeline._reference_image_array(image)

    assert array.dtype == mx.float32
    assert np.array(array).shape == (1, 3, 1, 1)


def test_ideogram4_variant_resolution_is_exact():
    assert (
        ideogram_config.get_variant(ideogram_config.IDEOGRAM_4_FP8_REPO_ID).repo_id
        == ideogram_config.IDEOGRAM_4_FP8_REPO_ID
    )

    for shorthand in ("ideogram4", "ideogram-4", "ideogram-4-fp8"):
        with pytest.raises(ValueError):
            ideogram_config.get_variant(shorthand)


def test_ideogram4_plain_prompt_wraps_as_minimal_json_caption():
    prepared = ideogram_prompting.normalize_prompt(
        "A red cube on a marble plinth.", warn=False
    )
    caption = json.loads(prepared.text)

    assert prepared.was_wrapped
    assert prepared.is_json_caption
    assert prepared.is_structured_caption
    assert caption["high_level_description"] == "A red cube on a marble plinth."
    assert caption["compositional_deconstruction"]["elements"] == [
        {"type": "obj", "desc": "A red cube on a marble plinth."}
    ]
    assert ideogram_prompting.is_structured_caption(prepared.text)


def test_ideogram4_caption_warnings_cover_elements_and_bounding_boxes():
    prompt = ideogram_prompting.format_caption(
        {
            "compositional_deconstruction": {
                "background": "A studio.",
                "elements": [
                    {"type": "text", "desc": "A title.", "bbox": [900, 100, 100, 800]}
                ],
            }
        }
    )

    with pytest.warns(UserWarning) as records:
        prepared = ideogram_prompting.normalize_prompt(prompt)

    messages = [str(record.message) for record in records]
    assert not prepared.is_structured_caption
    assert any(".text" in message for message in messages)
    assert any("y_min < y_max" in message for message in messages)


def test_ideogram4_prompt_expansion_uses_structured_generation(monkeypatch):
    tokenizer, logits_processor = object(), object()
    model, processor = SimpleNamespace(config={}), SimpleNamespace(tokenizer=tokenizer)
    load = MagicMock(return_value=(model, processor))
    template = MagicMock(return_value="formatted prompt")
    schema = MagicMock(return_value=logits_processor)
    generate = MagicMock(return_value=SimpleNamespace(text=EXPANDED_CAPTION))
    for module, name, replacement in [
        (utils_module, "load", load),
        (prompt_utils_module, "apply_chat_template", template),
        (structured_module, "build_json_schema_logits_processor", schema),
        (dispatch_module, "generate", generate),
    ]:
        monkeypatch.setattr(module, name, replacement)
    result = ideogram_prompting.generate_prompt_expansion_caption(
        "A red cube.", model="tiny-text-model", aspect_ratio="1:1"
    )
    load.assert_called_once_with("tiny-text-model")
    assert template.call_args.args[:2] == (processor, {})
    messages = template.call_args.args[2]
    assert isinstance(messages, list)
    assert "visible wording" in messages[0]["content"]
    assert "do not add an aspect_ratio field" in messages[1]["content"]
    schema.assert_called_once_with(
        tokenizer, ideogram_prompting.IDEOGRAM4_CAPTION_SCHEMA
    )
    assert generate.call_args.args == (model, processor, "formatted prompt")
    assert generate.call_args.kwargs["logits_processors"] == [logits_processor]
    assert result.text == EXPANDED_CAPTION and result.model == "tiny-text-model"


def test_ideogram4_dequantizes_weight_only_fp8():
    scale = mx.array([0.5, 2.0], dtype=mx.float32)
    expected = mx.array([[1.0, -2.0], [0.5, 4.0]], dtype=mx.float32)
    raw = {
        "linear.weight": mx.to_fp8(expected / mx.expand_dims(scale, axis=-1)),
        "linear.weight_scale": scale,
        "linear.bias": mx.array([1.0, -1.0], dtype=mx.float32),
    }

    converted = ideogram_weights.dequantize_fp8_weight_only(raw, precision=mx.float32)

    assert "linear.weight_scale" not in converted
    np.testing.assert_allclose(
        np.array(converted["linear.weight"]), np.array(expected), rtol=0, atol=0
    )
    assert converted["linear.bias"].dtype == mx.float32


def test_ideogram4_build_inputs_packs_text_and_image_tokens():
    pipeline = _pipeline("ideogram4")
    pipeline.tokenizer = _CaptionTokenizer()

    inputs = pipeline._build_inputs("prompt", height=256, width=256)

    assert inputs["text_token_ids"].shape == (1, 3)
    assert inputs["num_text_tokens"] == 3
    assert inputs["num_image_tokens"] == 16 * 16
    assert inputs["position_ids"].shape == (1, 3 + 16 * 16, 3)
    assert (
        int(inputs["indicator"][0, 0].item())
        == ideogram_transformer.LLM_TOKEN_INDICATOR
    )
    assert (
        int(inputs["indicator"][0, -1].item())
        == ideogram_transformer.OUTPUT_IMAGE_INDICATOR
    )


def test_ideogram4_pipeline_uses_prepared_prompt_and_reports_metadata():
    pipeline = _pipeline("ideogram4")
    pipeline.model_path = Path("/tmp/fake-ideogram")
    pipeline.runtime_config = ideogram_pipeline.Ideogram4RuntimeConfig(
        evict_text_encoder=False, evict_transformers=False
    )
    pipeline.text_encoder = object()
    pipeline.conditional_transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
    pipeline.unconditional_transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
    pipeline.vae = object()

    pipeline.prepare_prompt = MagicMock(
        return_value=ideogram_prompting.NormalizedPrompt(
            text=EXPANDED_CAPTION,
            is_json_caption=True,
            is_structured_caption=True,
            was_wrapped=False,
            prompt_expansion_model="tiny-text-model",
            prompt_expansion_used=True,
        )
    )
    pipeline._build_inputs = MagicMock(
        return_value={
            "text_token_ids": mx.array([[1]], dtype=mx.int32),
            "position_ids": mx.zeros((1, 2, 3), dtype=mx.int32),
            "segment_ids": mx.ones((1, 2), dtype=mx.int32),
            "indicator": mx.ones((1, 2), dtype=mx.int32),
            "num_text_tokens": 1,
            "num_image_tokens": 1,
            "grid_h": 1,
            "grid_w": 1,
        }
    )
    pipeline._encode_text = lambda token_ids, num_image_tokens: mx.zeros((1, 2, 4))
    pipeline._ensure_transformers_and_vae = lambda: None
    pipeline._decode = lambda z, grid_h, grid_w: mx.zeros((16, 16, 3), dtype=mx.uint8)

    array, metadata = pipeline.generate_array(
        "plain prompt",
        steps=1,
        width=256,
        height=256,
        guidance=7.0,
        prompt_expansion_model="tiny-text-model",
    )

    assert array.shape == (16, 16, 3)
    assert pipeline.prepare_prompt.call_args.args == ("plain prompt",)
    assert (
        pipeline.prepare_prompt.call_args.kwargs["prompt_expansion_model"]
        == "tiny-text-model"
    )
    assert pipeline._build_inputs.call_args.args == (EXPANDED_CAPTION,)
    assert metadata["revised_prompt"] == EXPANDED_CAPTION
    assert metadata["prompt_expansion_model"] == "tiny-text-model"
    assert metadata["prompt_expansion_used"]
    assert metadata["prompt_is_structured_caption"]


def test_ideogram4_decode_uses_ideogram_latent_norm_path():
    pipeline = _pipeline("ideogram4")
    pipeline.vae = _IdeogramVAE()

    array = pipeline._decode(mx.zeros((1, 16 * 16, 128)), grid_h=16, grid_w=16)

    assert array.shape == (32, 32, 3)
    assert pipeline.vae.latents.shape == (1, 32, 32, 32)


def test_detects_diffusers_z_image_model_index(tmp_path):
    (tmp_path / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}')
    assert z_convert.is_z_image_model_path(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name":"FluxPipeline"}')
    assert not z_convert.is_z_image_model_path(tmp_path)


@pytest.mark.parametrize(
    "steps,strength,expected", [(9, 0.6, 3), (9, 0.5, 4), (8, 0.6, 3), (8, 0.3, 5)]
)
def test_z_image_img2img_start_index_matches_diffusers(steps, strength, expected):
    assert z_pipeline._img2img_start_index(steps, strength) == expected


def test_z_image_generation_evicts_components_before_reloading_encoder(
    monkeypatch,
):
    pipeline = _pipeline("z_image")
    pipeline.evict_text_encoder = True
    pipeline.transformer = object()
    pipeline.vae = object()
    pipeline.text_encoder = None
    reloaded = False

    def reload_encoder() -> None:
        nonlocal reloaded
        assert pipeline.transformer is None
        assert pipeline.vae is None
        reloaded = True

    monkeypatch.setattr(pipeline, "_reload_encoder", reload_encoder)
    monkeypatch.setattr(
        pipeline,
        "_encode_prompt",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("stop after reload")),
    )

    with pytest.raises(RuntimeError, match="stop after reload"):
        pipeline.generate_array("fox", steps=2, width=16, height=16)
    assert reloaded


def test_z_image_rejects_classifier_free_guidance():
    model = object.__new__(z_model.ZImageGenerationModel)
    model.pipeline = SimpleNamespace(config=z_config.ZImageConfig())
    with pytest.raises(ValueError, match="does not support classifier-free guidance"):
        model.generate(ImageGenerationRequest(prompt="test", guidance=2.0))


def test_z_image_config_detects_base_variant(tmp_path):
    configs = {
        "transformer/config.json": {},
        "text_encoder/config.json": {},
        "vae/config.json": {},
        "scheduler/scheduler_config.json": {"shift": 6.0},
    }
    _write_files(tmp_path, metadata=configs)
    config = z_config.ZImageConfig.from_model_path(tmp_path)
    assert config.variant == "base"
    assert config.default_steps == 50
    assert config.default_guidance == 4.0


@pytest.mark.parametrize("case", IMAGE_CASES["sanitizers"], ids=lambda case: case["id"])
def test_weight_key_sanitization(case):
    family, component, keys = case["family"], case["component"], case["keys"]
    # Each row records the expected destination (None means drop) and source shape.
    weights = {key: mx.zeros(value["shape"]) for key, value in keys.items()}
    sanitize = getattr(
        _family_module(family, "weights"), f"sanitize_{component}_weights"
    )
    actual = sanitize(weights)
    expected = {
        value["target"]: weights[key] for key, value in keys.items() if value["target"]
    }
    assert actual.keys() == expected.keys()
    for key in expected:
        assert bool(mx.array_equal(actual[key], expected[key]))


def test_z_image_sanitize_vae_weights():
    conv = mx.zeros((8, 4, 3, 3))
    sanitized = z_vae.sanitize_vae_weights(
        {"encoder.conv_in.weight": conv, "decoder.conv_norm_out.weight": mx.zeros((8,))}
    )
    assert sanitized["encoder.conv_in.weight"].shape == (8, 3, 3, 4)
    assert "decoder.conv_norm_out.weight" in sanitized

    native = z_vae.sanitize_vae_weights({"encoder.conv_in.conv2d.weight": conv})
    assert set(native) == {"encoder.conv_in.weight"}

    converted = z_vae.sanitize_vae_weights(sanitized, source_layout=False)
    assert mx.array_equal(
        converted["encoder.conv_in.weight"], sanitized["encoder.conv_in.weight"]
    )


def test_z_image_conversion_preserves_native_vae_layout(tmp_path):
    native = mx.zeros((8, 3, 3, 4))
    vae_path = tmp_path / "vae"
    vae_path.mkdir()
    (vae_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"mlx_vlm_format": "z_image"}})
    )

    converted = z_convert._sanitize_vae_for_conversion(
        vae_path, {"encoder.conv_in.weight": native}
    )

    assert converted["encoder.conv_in.weight"].shape == native.shape
    assert mx.array_equal(converted["encoder.conv_in.weight"], native)


def test_z_image_saved_affine_metadata_marks_native_layout(tmp_path):

    model = _TinyLinear()
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    z_convert._save_component(
        tmp_path,
        "transformer",
        model,
        {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}},
    )
    index = json.loads(
        (tmp_path / "transformer" / "model.safetensors.index.json").read_text()
    )
    assert index["metadata"]["mlx_vlm_format"] == "z_image"
    assert index["metadata"]["quantization_group_size"] == "64"
    assert index["metadata"]["quantization_level"] == "4"
    assert index["metadata"]["quantization_mode"] == "affine"


def test_ernie_dispatches_from_weight_index(tmp_path):
    _write_layout(tmp_path, "ernie_image")
    (tmp_path / "model_index.json").unlink()
    (tmp_path / "mlx_ernie_image.json").unlink()
    index = {
        "weight_map": {
            "adaln_modulation.weight": "0.safetensors",
            "final_norm.linear.weight": "0.safetensors",
            "layers.0.adaLN_sa_ln.weight": "0.safetensors",
        }
    }
    (tmp_path / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps(index)
    )
    assert (
        image_generation_model_class(tmp_path.as_posix())
        is ernie_model.ErnieImageGenerationModel
    )


def test_ernie_layout_accepts_mflux_checkpoint_without_configs(tmp_path):
    _write_layout(tmp_path, "ernie_image")
    assert ernie_download.validate_model_layout(tmp_path) == tmp_path


def test_ernie_layout_requires_complete_prompt_enhancer(tmp_path):
    _write_layout(tmp_path, "ernie_image")
    (tmp_path / "pe" / "model.safetensors").parent.mkdir()
    (tmp_path / "pe" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="pe_tokenizer"):
        ernie_download.validate_model_layout(tmp_path)


def test_ernie_image_transformer_config_parses_official_fields():
    config = ernie_config.ErnieImageTransformerConfig.from_dict(
        {
            "_class_name": "ErnieImageTransformer2DModel",
            "hidden_size": 32,
            "ffn_hidden_size": 64,
            "num_attention_heads": 4,
            "rope_axes_dim": [2, 2, 4],
            "unknown": 1,
        }
    )
    assert config.head_dim == 8
    assert config.rope_axes_dim == (2, 2, 4)


def test_ernie_rope_matches_reference_hybrid_convention():
    ids = np.array([[[3.0, 1.0, 2.0], [4.0, 2.0, 1.0]]], dtype=np.float32)
    axes = (2, 2, 4)
    angles = []
    for axis, dim in enumerate(axes):
        omega = 1.0 / (256.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles.append(ids[..., axis, None] * omega)
    angles = np.concatenate(angles, axis=-1)
    angles = np.stack([angles, angles], axis=-1).reshape(1, 2, 1, 8)

    cos, sin = ernie_transformer.rope_frequencies(
        mx.array(ids), axes_dim=axes, theta=256.0
    )
    np.testing.assert_allclose(
        np.array(cos.transpose(0, 2, 1, 3)), np.cos(angles), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.array(sin.transpose(0, 2, 1, 3)), np.sin(angles), rtol=1e-6
    )

    values = mx.arange(16, dtype=mx.float32).reshape(1, 1, 2, 8)
    expected_rotated = np.concatenate(
        [-np.array(values)[..., 4:], np.array(values)[..., :4]], axis=-1
    )
    np.testing.assert_array_equal(
        np.array(ernie_transformer.rotate_half(values)), expected_rotated
    )


def test_ernie_image_conditioning_skips_last_text_block_and_final_norm():
    class FakeEmbedding(nn.Module):
        def __call__(self, input_ids):
            return mx.zeros((*input_ids.shape, 2))

    class AddOne(nn.Module):
        def __call__(self, hidden_states, mask, cache=None):  # noqa: ARG002
            return hidden_states + 1

    class TimesTen(nn.Module):
        def __call__(self, hidden_states):
            return hidden_states * 10

    encoder = ernie_text_encoder.ErnieImageTextEncoder(
        ernie_text_encoder.ErnieImageTextConfig(
            vocab_size=4,
            hidden_size=2,
            intermediate_size=4,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=2,
            rope_parameters=None,
        )
    )
    encoder.embed_tokens = FakeEmbedding()
    encoder.layers = [AddOne(), AddOne(), AddOne()]
    encoder.norm = TimesTen()
    ids = mx.array([[1, 2]])

    np.testing.assert_array_equal(np.array(encoder(ids)), np.full((1, 2, 2), 2))
    np.testing.assert_array_equal(
        np.array(encoder(ids, normalize=True)), np.full((1, 2, 2), 30)
    )


def test_ernie_image_pad_text_preserves_cfg_order_and_lengths():
    negative = mx.ones((1, 1, 2))
    positive = mx.full((1, 3, 2), 2)
    padded, lengths = ernie_pipeline._pad_text([negative, positive])
    assert tuple(padded.shape) == (2, 3, 2)
    np.testing.assert_array_equal(np.array(lengths), [1, 3])
    np.testing.assert_array_equal(np.array(padded[0, 1:]), np.zeros((2, 2)))


def test_ernie_weight_sanitizers_and_layouts():
    native = mx.zeros((8, 1, 1, 4))
    source = native.transpose(0, 3, 1, 2)
    sanitized = ernie_weights.sanitize_transformer_weights(
        {
            "adaLN_modulation.1.weight": mx.zeros((6, 4)),
            "layers.0.self_attention.to_out.0.weight": mx.zeros((4, 4)),
            "x_embedder.proj.weight": source,
        },
        target_shapes={
            "adaln_modulation.weight": (6, 4),
            "layers.0.self_attention.to_out.weight": (4, 4),
            "x_embedder.proj.weight": tuple(native.shape),
        },
    )
    assert "adaln_modulation.weight" in sanitized
    assert "layers.0.self_attention.to_out.weight" in sanitized
    assert sanitized["x_embedder.proj.weight"].shape == native.shape
    assert (
        ernie_weights.match_conv_layout(
            native, target_shape=tuple(native.shape), key="conv.weight"
        ).shape
        == native.shape
    )


def test_ernie_image_conversion_quantizes_compatible_vae_attention():
    vae = flux_vae.Flux2VAE(
        decoder_block_out_channels=(32, 32),
        include_encoder=True,
        encoder_block_out_channels=(32, 32),
    )
    ernie_convert._quantize_component(
        vae,
        {"mode": "mxfp8", "group_size": 32, "bits": 8},
        lambda path, module: hasattr(module, "to_quantized"),
    )
    assert isinstance(vae.decoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
    assert isinstance(vae.encoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
    assert isinstance(vae.decoder.conv_in, nn.Conv2d)


def test_generation_request_converts_to_edit_request():
    class FakeEditModel:
        def edit(self, request):
            assert isinstance(request, ImageEditRequest)
            assert request.image_paths == ("reference.png",)
            return SimpleNamespace(path=None)

    generate_image(
        FakeEditModel(),
        ImageGenerationRequest(prompt="edit"),
        task="edit",
        image_paths=("reference.png",),
    )


def test_ernie_edit_model_defaults_prompt_enhancer_off(
    monkeypatch,
):
    import mlx_vlm.models.ernie_image.model as ernie_model

    captured: dict[str, object] = {}

    class _StubPipeline:
        def __init__(self) -> None:
            self.variant = ernie_config.get_variant("ernie-image-turbo")
            self.model_path = Path("/tmp/ernie")

        @classmethod
        def from_pretrained(cls, variant, **kwargs):  # noqa: ARG003
            captured["use_prompt_enhancer"] = kwargs.get("use_prompt_enhancer")
            return cls()

    monkeypatch.setattr(ernie_model, "ErnieImagePipeline", _StubPipeline)
    ernie_model.ErnieImageEditModel.from_model_id("ernie-image-turbo")
    assert captured["use_prompt_enhancer"] is False
    ernie_model.ErnieImageEditModel.from_model_id(
        "ernie-image-turbo", use_prompt_enhancer=True
    )
    assert captured["use_prompt_enhancer"] is True


def test_ernie_image_prompt_cache_evicts_least_recently_used_entry():
    pipeline = _pipeline("ernie_image")
    pipeline.runtime_config = ernie_pipeline.ErnieImageRuntimeConfig(
        prompt_cache_size=2
    )
    pipeline.prompt_cache = OrderedDict()
    pipeline.tokenizer = SimpleNamespace(encode=lambda prompt: prompt)
    pipeline.text_encoder = lambda prompt: mx.array([len(prompt)])

    first = pipeline._encode_prompt("first")
    pipeline._encode_prompt("second")
    assert pipeline._encode_prompt("first") is first
    pipeline._encode_prompt("third")

    assert list(pipeline.prompt_cache) == ["first", "third"]


def test_ernie_image_turbo_skips_cfg_and_evicts_large_components():
    pipeline = _ernie_runtime_pipeline("ernie-image-turbo", evict=True)
    pipeline.generate_array("prompt", seed=1, steps=1, width=16, height=16)
    assert pipeline.transformer is None
    assert pipeline.vae is None


def test_ernie_image_img2img_uses_strength_to_select_denoising_steps(
    tmp_path,
):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (32, 16), color="navy").save(image_path)
    pipeline = _ernie_runtime_pipeline("ernie-image-turbo")
    result = pipeline.edit_array(
        "make it a convertible",
        image_path,
        seed=1,
        steps=8,
        width=32,
        height=16,
        image_strength=0.5,
    )
    assert result.shape == (16, 32, 3)
    # Turbo editing defaults to CFG guidance rather than Turbo generation's 1.0.
    assert pipeline.transformer.calls[0][0][0] == 2
    assert len(pipeline.transformer.calls) == 4


@pytest.mark.parametrize(
    "size,expected", [((300, 200), (384, 256)), ((4000, 250), (2048, 128))]
)
def test_ernie_image_edit_auto_size_preserves_aspect_ratio(tmp_path, size, expected):
    image_path = tmp_path / "source.png"
    Image.new("RGB", size, color="navy").save(image_path)
    _, width, height = ernie_pipeline._load_edit_image(
        image_path, width=None, height=None
    )
    assert (width, height) == expected


def test_ernie_image_img2img_rejects_decoder_only_converted_vae():
    with pytest.raises(ValueError, match="VAE encoder weights"):
        ernie_weights._require_vae_encoder_weights(
            {"decoder.conv_in.weight": mx.zeros((1,))},
            target_shapes={
                "encoder.conv_in.weight": (2, 3, 3, 3),
                "quant_conv.weight": (4, 1, 1, 4),
                "decoder.conv_in.weight": (2, 3, 3, 4),
            },
        )


def test_ernie_image_conversion_detects_decoder_only_vae_index(tmp_path):
    for keys, expected in [
        (["decoder.conv_in.weight", "post_quant_conv.weight"], False),
        (["encoder.conv_in.weight", "quant_conv.weight"], True),
    ]:
        _weight_index(tmp_path / "vae", keys)
        assert ernie_convert._vae_checkpoint_has_encoder(tmp_path) is expected


def test_ernie_image_prompt_enhancement_auto_detects_optional_components(
    tmp_path,
):
    pipeline = _ernie_runtime_pipeline("ernie-image-turbo")
    pipeline.model_path = tmp_path
    assert not pipeline._should_enhance_prompt()
    (tmp_path / "pe").mkdir()
    (tmp_path / "pe" / "model.safetensors").write_bytes(b"x")
    (tmp_path / "pe_tokenizer").mkdir()
    (tmp_path / "pe_tokenizer" / "tokenizer.json").write_text("{}")
    pipeline.runtime_config = ernie_pipeline.ErnieImageRuntimeConfig(
        use_prompt_enhancer=None
    )
    assert pipeline._should_enhance_prompt()
    # Auto-detected enhancement applies to generation only: img2img needs the
    # source-aware prompt preserved unless callers explicitly opt in.
    assert not pipeline._should_enhance_prompt(for_edit=True)
    pipeline.runtime_config = ernie_pipeline.ErnieImageRuntimeConfig(
        use_prompt_enhancer=True
    )
    assert pipeline._should_enhance_prompt(for_edit=True)


def test_ernie_image_conversion_detection_and_layout_metadata(tmp_path):
    _write_layout(tmp_path, "ernie_image")
    assert ernie_convert.is_ernie_image_checkpoint(tmp_path)
    assert ernie_convert._source_layout(tmp_path) == "mlx_nhwc"
    output = tmp_path / "output"
    for component in ("transformer", "text_encoder", "vae"):
        (output / component).mkdir(parents=True, exist_ok=True)
    ernie_convert._write_missing_configs(output)
    assert (
        json.loads((output / "transformer" / "config.json").read_text())["_class_name"]
        == "ErnieImageTransformer2DModel"
    )
    assert (
        json.loads((output / "model_index.json").read_text())["_class_name"]
        == "ErnieImagePipeline"
    )


def test_mage_flow_load_prefers_local_metadata(tmp_path):
    _write_layout(tmp_path, "mage_flow")
    (tmp_path / "mlx_mage_flow.json").write_text('{"variant":"mage-flow-edit-turbo"}')
    assert (
        mage_model._resolve_load_variant("community/custom-name", tmp_path).name
        == "mage-flow-edit-turbo"
    )


def test_mage_flow_local_variant_uses_cache_parent_name(tmp_path):
    snapshot = (
        tmp_path / "models--microsoft--Mage-Flow-Edit-Turbo" / "snapshots" / "hash"
    )
    _write_layout(snapshot, "mage_flow")
    assert mage_config.variant_from_local_path(snapshot).name == "mage-flow-edit-turbo"


def test_mage_flow_scheduler_matches_static_shift():
    scheduler = mage_scheduler.FlowMatchEulerDiscreteScheduler(
        num_inference_steps=4, shift=6.0
    )
    expected = np.array([1.0, 4.5 / 4.75, 3.0 / 3.5, 1.5 / 2.25, 0.0])
    np.testing.assert_allclose(np.array(scheduler.sigmas), expected, rtol=1e-6)


def test_mage_flow_rope_covers_target_and_references():
    cosine, sine = mage_transformer.image_rope_frequencies(
        [(1, 2, 3), (1, 2, 3)], axes_dim=(2, 2, 4)
    )
    assert cosine.shape == (12, 4)
    assert sine.shape == (12, 4)
    assert not np.allclose(np.array(cosine[:6, 0]), np.array(cosine[6:, 0]))
    np.testing.assert_allclose(np.array(cosine[:6, 1:]), np.array(cosine[6:, 1:]))


def test_mage_flow_weight_sanitizers():
    vae = mage_weights.sanitize_vae_weights(
        {
            "student.dconv_encoder.blocks.0.ca.1.weight": mx.zeros((2, 2, 1, 1)),
            "pipeline.dec_net.res_blocks.0.mlp.2.weight": mx.zeros((2, 2)),
            "pipeline.y_embedder.encoder.encoder.conv_in.weight": mx.zeros(
                (2, 2, 3, 3)
            ),
        }
    )
    assert vae["dconv_encoder.blocks.0.ca_conv.weight"].shape == (2, 1, 1, 2)
    assert "decoder_model.dec_net.res_blocks.0.linear_2.weight" in vae
    assert len(vae) == 2


def test_mage_flow_native_vae_layout_is_not_transposed():
    weight = mx.zeros((2, 3, 3, 4))
    sanitized = mage_weights.sanitize_vae_weights(
        {"decoder_model.conv_in.weight": weight}, source_layout="mlx_nhwc"
    )
    assert sanitized["decoder_model.conv_in.weight"].shape == weight.shape


def test_mage_flow_quantizes_only_compatible_layers():
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compatible = nn.Linear(64, 32, bias=False)
            self.incompatible = nn.Linear(63, 32, bias=False)

    model = TinyModel()
    config = mage_convert._quantization_parameters("affine", 64, 4)
    mage_convert._quantize_component(model, config)
    assert isinstance(model.compatible, nn.QuantizedLinear)
    assert isinstance(model.incompatible, nn.Linear)
    assert model.quantization_config == config


def test_mage_flow_quantization_skips_sensitive_transformer_layers():
    module = nn.Linear(64, 32, bias=False)
    assert mage_convert._transformer_quantization_predicate(
        "transformer_blocks.0.attn.to_q", module
    )
    assert not mage_convert._transformer_quantization_predicate(
        "transformer_blocks.0.img_mod.linear", module
    )
    assert not mage_convert._transformer_quantization_predicate(
        "transformer_blocks.0.txt_mod.linear", module
    )
    assert not mage_convert._transformer_quantization_predicate("proj_out", module)


def test_mage_flow_saved_quantization_metadata(tmp_path):

    model = _TinyLinear()
    config = mage_convert._quantization_parameters("affine", 64, 4)
    mage_convert._quantize_component(model, config)
    mage_convert._save_component(tmp_path, model, config)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["mlx_vlm_format"] == "mage_flow"
    assert index["metadata"]["tensor_layout"] == "mlx_nhwc"
    component_config = json.loads((tmp_path / "config.json").read_text())
    assert component_config["quantization"] == config
    assert component_config["quantization_config"] == config
    weights, metadata = mage_weights._load_safetensors(tmp_path)
    loaded = mage_weights._apply_weights(
        _TinyLinear(), weights, {**component_config, **metadata}
    )
    assert isinstance(loaded.proj, nn.QuantizedLinear)
    assert loaded.quantization_config == config


def test_mage_flow_quantized_weights_require_config():

    quantized = _TinyLinear()
    nn.quantize(quantized, group_size=64, bits=4, mode="affine")
    with pytest.raises(ValueError, match="quantization mode"):
        mage_weights._apply_weights(
            _TinyLinear(), dict(tree_flatten(quantized.parameters())), {}
        )


def test_mage_flow_conversion_rejects_output_inside_source(tmp_path):
    _write_layout(tmp_path, "mage_flow")
    with pytest.raises(ValueError, match="inside its source"):
        mage_convert.convert_mage_flow(tmp_path, tmp_path / "converted")


def test_mage_flow_conversion_rejects_ambiguous_local_variant(tmp_path):
    _write_layout(tmp_path, "mage_flow")
    output = tmp_path.parent / f"{tmp_path.name}-converted"
    with pytest.raises(ValueError, match="--variant"):
        mage_convert.convert_mage_flow(tmp_path, output)


def test_mage_flow_conversion_prefers_source_id_variant(monkeypatch, tmp_path):
    _write_layout(tmp_path, "mage_flow")

    component = SimpleNamespace(
        quantization_config=None, parameters=lambda: {}, update=lambda parameters: None
    )
    text_encoder = SimpleNamespace(model=component)
    for name, result in [
        ("text_encoder", text_encoder),
        ("transformer", component),
        ("vae", component),
    ]:
        monkeypatch.setattr(
            mage_convert, "load_" + name, MagicMock(return_value=result)
        )
    monkeypatch.setattr(mage_convert, "_cast_component", lambda model, dtype: None)
    monkeypatch.setattr(
        mage_convert,
        "_save_component",
        lambda directory, model, quantization: directory.mkdir(
            parents=True, exist_ok=True
        ),
    )

    output = tmp_path.parent / f"{tmp_path.name}-converted"
    mage_convert.convert_mage_flow(
        tmp_path, output, source_id="mage-flow-community/Mage-Flow-Edit-Turbo"
    )
    metadata = json.loads((output / "mlx_mage_flow.json").read_text())
    assert metadata["variant"] == "mage-flow-edit-turbo"


@pytest.mark.parametrize("case", ["expand", "structured", "invalid"])
def test_ideogram_prompt_expansion_policy(monkeypatch, case):
    failure = ideogram_prompting.PromptExpansionCaptionError("bad json")
    expand = MagicMock(
        return_value=ideogram_prompting.PromptExpansionResult(
            text=EXPANDED_CAPTION, raw_text=EXPANDED_CAPTION, model="tiny-text-model"
        ),
        side_effect=failure if case == "invalid" else None,
    )
    monkeypatch.setattr(ideogram_prompting, "generate_prompt_expansion_caption", expand)
    if case == "invalid":
        with pytest.warns(UserWarning, match="falling back"):
            result = ideogram_prompting.prepare_prompt(
                "A red cube.", prompt_expansion_model="bad-model"
            )
        assert result.was_wrapped and not result.prompt_expansion_used
        assert result.prompt_expansion_error == "bad json"
    else:
        prompt = EXPANDED_CAPTION if case == "structured" else "A red cube."
        result = ideogram_prompting.prepare_prompt(
            prompt,
            prompt_expansion_model="tiny-text-model",
            width=1024,
            height=1024,
            warn=False,
        )
        assert result.text == EXPANDED_CAPTION and not result.was_wrapped
        assert result.prompt_expansion_used is (case == "expand")
        if case == "expand":
            assert result.prompt_expansion_model == "tiny-text-model"
            expand.assert_called_once_with(
                "A red cube.", model="tiny-text-model", aspect_ratio="1:1"
            )
        else:
            expand.assert_not_called()
