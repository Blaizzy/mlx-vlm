"""Image generation/editing contracts and model-specific numerical checks."""

from __future__ import annotations

import importlib
import json
from collections import OrderedDict
from dataclasses import dataclass
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
import mlx_vlm.models.bonsai.model as bonsai_model
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
import mlx_vlm.models.ideogram4.model as ideogram_model
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
import mlx_vlm.models.z_image.text_encoder as z_text_encoder
import mlx_vlm.models.z_image.transformer as z_transformer
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


class _IdeogramPipeline:
    variant = type("Variant", (), {"name": "ideogram-4-fp8"})()

    def generate_array(self, prompt: str, **kwargs):  # noqa: ANN001
        return mx.zeros((8, 10, 3), dtype=mx.uint8), {
            "steps": kwargs["steps"],
            "guidance": kwargs["guidance"],
            "prompt_tokens": 3,
        }


class _ErniePipeline:
    def __init__(self, variant: str) -> None:
        self.variant = ernie_config.get_variant(variant)
        self.model_path = Path("/tmp/ernie")
        self.runtime_config = ernie_pipeline.ErnieImageRuntimeConfig(
            use_prompt_enhancer=False
        )
        self.calls = []
        self.quantization_config = None
        self.last_revised_prompt = None

    def generate_array(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return mx.zeros((16, 16, 3), dtype=mx.uint8)

    def edit_array(self, prompt, image, **kwargs):
        self.calls.append((prompt, image, kwargs))
        return mx.zeros((16, 32, 3), dtype=mx.uint8)

    def count_prompt_tokens(self, prompt):  # noqa: ARG002
        return 3

    def _should_enhance_prompt(self, *, for_edit: bool = False):  # noqa: ARG002
        return False


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
    pipeline = ernie_pipeline.ErnieImagePipeline.__new__(
        ernie_pipeline.ErnieImagePipeline
    )
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
        pipeline = z_pipeline.ZImagePipeline.__new__(z_pipeline.ZImagePipeline)
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
    "family,cls,aliases,probe",
    [
        (
            "bonsai",
            bonsai_model.BonsaiImageGenerationModel,
            ["bonsai-ternary"],
            "bonsai-ternary",
        ),
        (
            "flux2",
            flux_model.Flux2ImageGenerationModel,
            ["flux2-klein-4b", "black-forest-labs/FLUX.2-klein-9B"],
            "klein-base-9b",
        ),
        (
            "ideogram4",
            ideogram_model.Ideogram4ImageGenerationModel,
            ["ideogram-ai/ideogram-4-fp8"],
            None,
        ),
    ],
    ids=["bonsai", "flux2", "ideogram4"],
)
def test_generation_model_dispatch(monkeypatch, tmp_path, family, cls, aliases, probe):
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
    for model_id in [*aliases, str(tmp_path)]:
        assert image_generation_model_class(model_id) is cls
    assert is_image_generation_model(probe or str(tmp_path))
    if family == "bonsai":
        assert not is_image_generation_model("mlx-community/nanoLLaVA-1.5-8bit")
    else:
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


def test_bonsai_variant_aliases_are_ternary_only() -> None:
    assert bonsai_config.get_variant("bonsai").precision == "2bit"
    assert bonsai_config.get_variant("bonsai-ternary").name == "ternary"
    assert (
        bonsai_config.get_variant("prism-ml/bonsai-image-ternary-4B-mlx-2bit").name
        == "ternary"
    )
    with pytest.raises(ValueError, match="Unknown Bonsai variant"):
        bonsai_config.get_variant("binary")


def test_bonsai_parse_size() -> None:
    assert bonsai_config.parse_size("1248x832") == (1248, 832)
    assert bonsai_config.parse_size("832x1248") == (832, 1248)


def test_flux2_remote_component_index_is_a_metadata_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_flux2_variant_from_local_path_reads_config(tmp_path: Path) -> None:
    _write_layout(tmp_path, "flux2")
    config = tmp_path / "transformer" / "config.json"
    config.write_text('{"num_layers": 8, "num_attention_heads": 32}')

    assert flux_config.variant_from_local_path(tmp_path).name == "flux2-klein-9b"


def test_flux2_variant_from_local_path_prefers_kv_marker(tmp_path: Path) -> None:
    _write_layout(tmp_path, "flux2")
    config = tmp_path / "transformer" / "config.json"
    config.write_text('{"num_layers": 8, "num_attention_heads": 32}')
    (tmp_path / "flux-2-klein-9b-kv.safetensors").write_bytes(b"x")

    assert flux_config.variant_from_local_path(tmp_path).name == "flux2-klein-9b-kv"


def test_flux2_edit_model_returns_image_result() -> None:
    model = flux_model.Flux2ImageEditModel(
        pipeline=_flux_edit_pipeline(), model_id="kv"
    )
    result = model.edit(
        ImageEditRequest(
            prompt="add sunglasses",
            image_paths=("reference.png",),
            seed=9,
            steps=2,
            guidance=1.0,
        )
    )

    assert result.width == 24
    assert result.height == 20
    assert result.metadata["uses_reference_kv_cache"] is True
    assert result.metadata["reference_count"] == 1


def test_flux2_text_encoder_accepts_native_quantized_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_flux2_vae_conv_layout_accepts_source_and_native_shapes() -> None:
    source = mx.arange(8 * 4 * 3 * 3).reshape(8, 4, 3, 3)
    native = source.transpose(0, 2, 3, 1)

    converted = flux_weights._match_conv_layout(
        source, target_shape=tuple(native.shape), key="decoder.conv.weight"
    )
    unchanged = flux_weights._match_conv_layout(
        native, target_shape=tuple(native.shape), key="decoder.conv.weight"
    )

    assert np.array_equal(np.array(converted), np.array(native))
    assert np.array_equal(np.array(unchanged), np.array(native))


def test_flux2_quantized_repo_routes_through_resolved_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_flux2_reference_image_array_keeps_float32_input() -> None:
    image = Image.new("RGB", (1, 1), color=(255, 127, 0))
    array = flux_pipeline._reference_image_array(image)

    assert array.dtype == mx.float32
    assert np.array(array).shape == (1, 3, 1, 1)


def test_ideogram4_variant_resolution_is_exact() -> None:
    assert (
        ideogram_config.get_variant(ideogram_config.IDEOGRAM_4_FP8_REPO_ID).repo_id
        == ideogram_config.IDEOGRAM_4_FP8_REPO_ID
    )

    for shorthand in ("ideogram4", "ideogram-4", "ideogram-4-fp8"):
        with pytest.raises(ValueError):
            ideogram_config.get_variant(shorthand)


def test_ideogram4_caption_schema_matches_prompting_contract() -> None:
    properties = ideogram_prompting.IDEOGRAM4_CAPTION_SCHEMA["properties"]
    composition = properties["compositional_deconstruction"]
    elements = composition["properties"]["elements"]["items"]["anyOf"]
    object_element, text_element = elements
    style_variants = properties["style_description"]["anyOf"]
    photo_style, art_style = style_variants

    assert ideogram_prompting.IDEOGRAM4_CAPTION_SCHEMA["required"] == [
        "compositional_deconstruction"
    ]
    assert composition["required"] == ["background", "elements"]
    assert object_element["required"] == ["type", "desc"]
    assert text_element["required"] == ["type", "text", "desc"]
    assert object_element["properties"]["bbox"]["minItems"] == 4
    assert object_element["properties"]["bbox"]["maxItems"] == 4
    assert "photo" in photo_style["properties"]
    assert "art_style" not in photo_style["properties"]
    assert "art_style" in art_style["properties"]
    assert "photo" not in art_style["properties"]


def test_ideogram4_plain_prompt_wraps_as_minimal_json_caption() -> None:
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


def test_ideogram4_caption_warnings_cover_elements_and_bounding_boxes() -> None:
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


def test_ideogram4_dequantizes_weight_only_fp8() -> None:
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


def test_ideogram4_build_inputs_packs_text_and_image_tokens() -> None:
    pipeline = ideogram_pipeline.Ideogram4ImagePipeline.__new__(
        ideogram_pipeline.Ideogram4ImagePipeline
    )
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


def test_ideogram4_pipeline_uses_prepared_prompt_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ideogram_pipeline.Ideogram4ImagePipeline.__new__(
        ideogram_pipeline.Ideogram4ImagePipeline
    )
    pipeline.model_path = Path("/tmp/fake-ideogram")
    pipeline.runtime_config = ideogram_pipeline.Ideogram4RuntimeConfig(
        evict_text_encoder=False, evict_transformers=False
    )
    pipeline.text_encoder = object()
    pipeline.conditional_transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
    pipeline.unconditional_transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
    pipeline.vae = object()
    captured: dict[str, object] = {}

    def fake_prepare_prompt(prompt: str, **kwargs):
        assert prompt == "plain prompt"
        assert kwargs["prompt_expansion_model"] == "tiny-text-model"
        captured["prepare_kwargs"] = kwargs
        return ideogram_prompting.NormalizedPrompt(
            text=EXPANDED_CAPTION,
            is_json_caption=True,
            is_structured_caption=True,
            was_wrapped=False,
            prompt_expansion_model="tiny-text-model",
            prompt_expansion_used=True,
        )

    def fake_build_inputs(prompt: str, **kwargs):
        captured["tokenized_prompt"] = prompt
        return {
            "text_token_ids": mx.array([[1]], dtype=mx.int32),
            "position_ids": mx.zeros((1, 2, 3), dtype=mx.int32),
            "segment_ids": mx.ones((1, 2), dtype=mx.int32),
            "indicator": mx.ones((1, 2), dtype=mx.int32),
            "num_text_tokens": 1,
            "num_image_tokens": 1,
            "grid_h": 1,
            "grid_w": 1,
        }

    monkeypatch.setattr(pipeline, "prepare_prompt", fake_prepare_prompt)
    monkeypatch.setattr(pipeline, "_build_inputs", fake_build_inputs)
    monkeypatch.setattr(
        pipeline,
        "_encode_text",
        lambda token_ids, num_image_tokens: mx.zeros((1, 2, 4)),
    )
    monkeypatch.setattr(pipeline, "_ensure_transformers_and_vae", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "_decode",
        lambda z, grid_h, grid_w: mx.zeros((16, 16, 3), dtype=mx.uint8),
    )

    array, metadata = pipeline.generate_array(
        "plain prompt",
        steps=1,
        width=256,
        height=256,
        guidance=7.0,
        prompt_expansion_model="tiny-text-model",
    )

    assert array.shape == (16, 16, 3)
    assert captured["tokenized_prompt"] == EXPANDED_CAPTION
    assert metadata["revised_prompt"] == EXPANDED_CAPTION
    assert metadata["prompt_expansion_model"] == "tiny-text-model"
    assert metadata["prompt_expansion_used"]
    assert metadata["prompt_is_structured_caption"]


def test_ideogram4_tiny_transformer_forward_shape() -> None:
    config = ideogram_config.Ideogram4TransformerConfig(
        emb_dim=12,
        num_layers=1,
        num_heads=3,
        intermediate_size=16,
        adanln_dim=4,
        in_channels=4,
        llm_features_dim=8,
        mrope_section=(1, 1, 0),
    )
    model = ideogram_transformer.Ideogram4Transformer(config)

    out = model(
        llm_features=mx.zeros((1, 3, 8), dtype=mx.float32),
        x=mx.zeros((1, 3, 4), dtype=mx.float32),
        t=mx.array([0.5], dtype=mx.float32),
        position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
        segment_ids=mx.ones((1, 3), dtype=mx.int32),
        indicator=mx.array(
            [
                [
                    ideogram_transformer.LLM_TOKEN_INDICATOR,
                    ideogram_transformer.OUTPUT_IMAGE_INDICATOR,
                    ideogram_transformer.OUTPUT_IMAGE_INDICATOR,
                ]
            ],
            dtype=mx.int32,
        ),
    )

    assert out.shape == (1, 3, 4)


def test_ideogram4_decode_uses_ideogram_latent_norm_path() -> None:
    pipeline = ideogram_pipeline.Ideogram4ImagePipeline.__new__(
        ideogram_pipeline.Ideogram4ImagePipeline
    )
    pipeline.vae = _IdeogramVAE()

    array = pipeline._decode(mx.zeros((1, 16 * 16, 128)), grid_h=16, grid_w=16)

    assert array.shape == (32, 32, 3)
    assert pipeline.vae.latents.shape == (1, 32, 32, 32)


def test_ideogram4_model_wrapper_returns_image_result() -> None:
    model = ideogram_model.Ideogram4ImageGenerationModel(
        pipeline=_IdeogramPipeline(), model_id="ideogram-ai/ideogram-4-fp8"
    )

    result = model.generate(
        ImageGenerationRequest(
            prompt="caption", seed=9, steps=2, width=10, height=8, guidance=7.0
        )
    )

    assert result.width == 10
    assert result.height == 8
    assert result.prompt_tokens == 3
    assert result.variant == "ideogram-4-fp8"


def test_detects_diffusers_z_image_model_index(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}')
    assert z_convert.is_z_image_model_path(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name":"FluxPipeline"}')
    assert not z_convert.is_z_image_model_path(tmp_path)


def test_z_image_edit_model_forwards_img2img_options() -> None:
    calls = {}

    class FakePipeline:
        config = z_config.ZImageConfig(
            default_steps=9, default_guidance=0.0, scheduler_shift=3.0, variant="turbo"
        )
        model_path = Path("/tmp/z-image")

        def edit_array(self, prompt: str, image_paths, **kwargs):
            calls.update(prompt=prompt, image_paths=image_paths, **kwargs)
            return mx.zeros((16, 32, 3), dtype=mx.uint8)

        def count_prompt_tokens(self, prompt: str) -> int:
            return 1

    model = z_model.ZImageEditModel(
        pipeline=FakePipeline(), model_id="Tongyi-MAI/Z-Image-Turbo"
    )
    result = model.edit(
        ImageEditRequest(
            prompt="replace the cart",
            image_paths=("source.png",),
            extra={"strength": 0.55},
        )
    )
    assert calls["steps"] == 8
    assert calls["guidance"] == 0.0
    assert calls["strength"] == 0.55
    assert result.width == 32
    assert result.height == 16

    model.edit(
        ImageEditRequest(
            prompt="replace the cart",
            image_paths=("source.png",),
            steps=4,
            guidance=1.0,
        )
    )
    assert calls["steps"] == 4
    assert calls["guidance"] == 1.0


@pytest.mark.parametrize(
    "steps,strength,expected", [(9, 0.6, 3), (9, 0.5, 4), (8, 0.6, 3), (8, 0.3, 5)]
)
def test_z_image_img2img_start_index_matches_diffusers(
    steps: int, strength: float, expected: int
) -> None:
    assert z_pipeline._img2img_start_index(steps, strength) == expected


def test_z_image_base_model_preserves_explicit_generic_values() -> None:
    calls = {}

    class FakePipeline:
        config = z_config.ZImageConfig(
            default_steps=50, default_guidance=4.0, scheduler_shift=6.0, variant="base"
        )
        model_path = Path("/tmp/z-image")

        def generate_array(self, prompt: str, **kwargs):
            calls.update(prompt=prompt, **kwargs)
            return mx.zeros((16, 16, 3), dtype=mx.uint8)

        def count_prompt_tokens(self, prompt: str) -> int:
            return 1

    model = z_model.ZImageGenerationModel(
        pipeline=FakePipeline(), model_id="Tongyi-MAI/Z-Image"
    )
    result = model.generate(ImageGenerationRequest(prompt="fox", steps=4, guidance=1.0))
    assert calls["steps"] == 4
    assert calls["guidance"] == 1.0
    assert result.metadata["guidance_mode"] == "disabled"


def test_z_image_generation_evicts_components_before_reloading_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object.__new__(z_pipeline.ZImagePipeline)
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


def test_z_image_transformer_forward_shape() -> None:
    cfg = z_config.ZImageTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        in_channels=16,
        text_embed_dim=32,
        num_hidden_layers=2,
        n_refiner_layers=1,
        n_context_refiner_layers=1,
        adaln_embed_dim=256,
        rope_sections=(4, 6, 6),
    )
    model = z_transformer.ZImageTransformer(cfg)

    class CaptureLength(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.length = 0

        def __call__(self, hidden: mx.array, *args, **kwargs) -> mx.array:
            self.length = hidden.shape[1]
            return hidden

    noise_refiner = CaptureLength()
    context_refiner = CaptureLength()
    unified_layer = CaptureLength()
    model.noise_refiner = [noise_refiner]
    model.context_refiner = [context_refiner]
    model.layers = [unified_layer]
    # Input: [B=1, C=16, F=1, H=4, W=4] (patch_size=2 → 2x2 grid)
    x = mx.random.normal((1, 16, 1, 4, 4))
    t = mx.array([0.5])
    cap = mx.random.normal((1, 8, 32))
    out = model(x, t, cap)
    mx.eval(out)
    assert out.shape == (1, 16, 1, 4, 4)
    assert noise_refiner.length == 32
    assert context_refiner.length == 32
    assert unified_layer.length == 64


def test_z_image_text_encoder_forward_shape() -> None:
    cfg = z_config.ZImageTextEncoderConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
    )
    model = z_text_encoder.ZImageTextEncoder(cfg)
    ids = mx.array([[1, 2, 3, 4, 5]])
    out = model(ids)
    mx.eval(out)
    assert out.shape == (1, 5, 64)


def test_z_image_vae_decoder_shape() -> None:
    cfg = z_config.ZImageVAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
    )
    vae = z_vae.ZImageVAE(cfg)
    # Latent: [B=1, H=4, W=4, C=4]
    z = mx.random.normal((1, 4, 4, 4))
    out = vae.decode(z)
    mx.eval(out)
    # After 2 up_blocks with upsample (first block upsamples, second doesn't)
    # input 4×4 → 8×8 after first upsample → stays 8×8 (last block no upsample)
    assert out.shape[0] == 1
    assert out.shape[-1] == 3  # 3 output channels


def test_z_image_rejects_classifier_free_guidance() -> None:
    model = object.__new__(z_model.ZImageGenerationModel)
    model.pipeline = SimpleNamespace(config=z_config.ZImageConfig())
    with pytest.raises(ValueError, match="does not support classifier-free guidance"):
        model.generate(ImageGenerationRequest(prompt="test", guidance=2.0))


def test_z_image_config_detects_base_variant(tmp_path: Path) -> None:
    configs = {
        "transformer/config.json": {},
        "text_encoder/config.json": {},
        "vae/config.json": {},
        "scheduler/scheduler_config.json": {"shift": 6.0},
    }
    for relative, content in configs.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    config = z_config.ZImageConfig.from_model_path(tmp_path)
    assert config.variant == "base"
    assert config.default_steps == 50
    assert config.default_guidance == 4.0


def test_z_image_sanitize_transformer_weights() -> None:
    weights = {
        "all_final_layer.2-1.linear.weight": mx.zeros((3,)),
        "all_final_layer.2-1.adaLN_modulation.1.weight": mx.zeros((3,)),
        "all_x_embedder.2-1.weight": mx.zeros((3,)),
        "layers.0.attention.to_q.weight": mx.zeros((3,)),
        "t_embedder.mlp.0.weight": mx.zeros((3,)),
    }
    sanitized = z_transformer.sanitize_transformer_weights(weights)
    assert "final_layer.linear.weight" in sanitized
    assert "final_layer.adaLN_modulation.0.weight" in sanitized
    assert "x_embedder.weight" in sanitized
    assert "layers.0.attention.to_q.weight" in sanitized
    assert "t_embedder.linear1.weight" in sanitized


def test_z_image_sanitize_text_encoder_weights() -> None:
    sanitized = z_text_encoder.sanitize_text_encoder_weights(
        {
            "model.embed_tokens.weight": mx.zeros((2, 2)),
            "model.rotary_emb.inv_freq": mx.zeros((2,)),
        }
    )
    assert set(sanitized) == {"embed_tokens.weight"}


def test_z_image_sanitize_vae_weights() -> None:
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


def test_z_image_conversion_preserves_native_vae_layout(tmp_path: Path) -> None:
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


def test_z_image_saved_affine_metadata_marks_native_layout(tmp_path: Path) -> None:

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


def test_ernie_local_variant_uses_native_metadata(tmp_path: Path) -> None:
    _write_layout(tmp_path, "ernie_image", turbo=False)
    assert ernie_config.variant_from_local_path(tmp_path).name == "ernie-image"


def test_ernie_local_variant_prefers_recorded_source(tmp_path: Path) -> None:
    _write_layout(tmp_path, "ernie_image", turbo=False)
    metadata_path = tmp_path / "mlx_ernie_image.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source"] = "baidu/ERNIE-Image-Turbo"
    metadata_path.write_text(json.dumps(metadata))

    assert ernie_config.variant_from_local_path(tmp_path).name == "ernie-image-turbo"


def test_ernie_dispatches_ids_metadata_and_mflux_indexes(tmp_path: Path) -> None:
    _write_layout(tmp_path, "ernie_image")
    assert (
        image_generation_model_class(tmp_path.as_posix())
        is ernie_model.ErnieImageGenerationModel
    )
    assert (
        image_generation_model_class("baidu/ERNIE-Image-Turbo")
        is ernie_model.ErnieImageGenerationModel
    )
    assert is_image_generation_model("ernie-image")
    assert (
        image_edit_model_class(tmp_path.as_posix()) is ernie_model.ErnieImageEditModel
    )
    assert is_image_edit_model("baidu/ERNIE-Image-Turbo")

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


def test_ernie_layout_accepts_mflux_checkpoint_without_configs(tmp_path: Path) -> None:
    _write_layout(tmp_path, "ernie_image")
    assert ernie_download.validate_model_layout(tmp_path) == tmp_path


def test_ernie_layout_requires_complete_prompt_enhancer(tmp_path: Path) -> None:
    _write_layout(tmp_path, "ernie_image")
    (tmp_path / "pe" / "model.safetensors").parent.mkdir()
    (tmp_path / "pe" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="pe_tokenizer"):
        ernie_download.validate_model_layout(tmp_path)


def test_ernie_image_transformer_config_parses_official_fields() -> None:
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


def test_ernie_rope_matches_reference_hybrid_convention() -> None:
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


def test_tiny_ernie_transformer_forward() -> None:
    config = ernie_config.ErnieImageTransformerConfig(
        hidden_size=32,
        ffn_hidden_size=64,
        in_channels=8,
        out_channels=8,
        num_layers=2,
        num_attention_heads=4,
        rope_axes_dim=(2, 2, 4),
        text_in_dim=16,
    )
    transformer = ernie_transformer.ErnieImageTransformer(config)
    output = transformer(
        mx.zeros((2, 8, 2, 2), dtype=mx.bfloat16),
        timestep=mx.array([1000.0, 500.0], dtype=mx.bfloat16),
        text_hidden_states=mx.zeros((2, 3, 16), dtype=mx.bfloat16),
        text_lengths=mx.array([1, 3]),
    )
    mx.eval(output)
    assert output.shape == (2, 8, 2, 2)
    assert bool(mx.all(mx.isfinite(output)))


def test_ernie_image_conditioning_skips_last_text_block_and_final_norm() -> None:
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


def test_ernie_image_pad_text_preserves_cfg_order_and_lengths() -> None:
    negative = mx.ones((1, 1, 2))
    positive = mx.full((1, 3, 2), 2)
    padded, lengths = ernie_pipeline._pad_text([negative, positive])
    assert tuple(padded.shape) == (2, 3, 2)
    np.testing.assert_array_equal(np.array(lengths), [1, 3])
    np.testing.assert_array_equal(np.array(padded[0, 1:]), np.zeros((2, 2)))


def test_ernie_weight_sanitizers_and_layouts() -> None:
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

    text = ernie_weights.sanitize_text_encoder_weights(
        {
            "language_model.model.embed_tokens.weight": mx.zeros((4, 2)),
            "language_model.model.rotary_emb.inv_freq": mx.zeros((1,)),
            "vision_tower.weight": mx.zeros((1,)),
        }
    )
    assert set(text) == {"embed_tokens.weight"}


def test_ernie_image_ambiguous_rgb_conv_uses_source_layout_metadata() -> None:
    source = mx.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
    expected = source.transpose(0, 2, 3, 1)
    converted = ernie_weights.match_conv_layout(
        source,
        target_shape=(2, 3, 3, 3),
        key="encoder.conv_in.weight",
        source_layout="pytorch_nchw",
    )
    unchanged = ernie_weights.match_conv_layout(
        expected,
        target_shape=(2, 3, 3, 3),
        key="encoder.conv_in.weight",
        source_layout="mlx_nhwc",
    )
    np.testing.assert_array_equal(np.array(converted), np.array(expected))
    np.testing.assert_array_equal(np.array(unchanged), np.array(expected))


def test_ernie_image_conversion_quantizes_compatible_vae_attention() -> None:
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


@pytest.mark.parametrize(
    "variant,steps,guidance,cfg",
    [("ernie-image", 50, 4.0, True), ("ernie-image-turbo", 8, 1.0, False)],
)
def test_ernie_image_model_request_defaults_follow_variant(
    variant: str, steps: int, guidance: float, cfg: bool
) -> None:
    pipeline = _ErniePipeline(variant)
    model = ernie_model.ErnieImageGenerationModel(pipeline=pipeline, model_id=variant)
    result = generate_image(
        model,
        ImageGenerationRequest(
            prompt="a lighthouse", seed=7, extra={"negative_prompt": "fog"}
        ),
    )
    assert result.steps == steps
    assert result.guidance == guidance
    assert result.metadata["classifier_free_guidance"] is cfg
    assert result.width == result.height == 512
    assert pipeline.calls[0][1]["negative_prompt"] == "fog"


def test_ernie_image_generation_request_converts_to_edit_request() -> None:
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


def test_ernie_base_edit_uses_variant_defaults() -> None:
    pipeline = _ErniePipeline("ernie-image")
    model = ernie_model.ErnieImageEditModel(pipeline=pipeline, model_id="ernie")
    result = model.edit(
        ImageEditRequest(
            prompt="make it a convertible", image_paths=("source.png",), seed=3
        )
    )
    assert result.steps == 50
    assert result.guidance == 4.0


def test_ernie_edit_model_defaults_prompt_enhancer_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ernie_image_prompt_cache_evicts_least_recently_used_entry() -> None:
    pipeline = ernie_pipeline.ErnieImagePipeline.__new__(
        ernie_pipeline.ErnieImagePipeline
    )
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


def test_ernie_image_turbo_skips_cfg_and_evicts_large_components() -> None:
    pipeline = _ernie_runtime_pipeline("ernie-image-turbo", evict=True)
    pipeline.generate_array("prompt", seed=1, steps=1, width=16, height=16)
    assert pipeline.transformer is None
    assert pipeline.vae is None


def test_ernie_image_img2img_uses_strength_to_select_denoising_steps(
    tmp_path: Path,
) -> None:
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
def test_ernie_image_edit_auto_size_preserves_aspect_ratio(
    tmp_path: Path, size: tuple[int, int], expected: tuple[int, int]
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", size, color="navy").save(image_path)
    _, width, height = ernie_pipeline._load_edit_image(
        image_path, width=None, height=None
    )
    assert (width, height) == expected


def test_ernie_image_img2img_rejects_decoder_only_converted_vae() -> None:
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
    tmp_path: Path,
) -> None:
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


def test_ernie_image_conversion_detection_and_layout_metadata(tmp_path: Path) -> None:
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


def test_mage_flow_registers_generation_and_edit_families() -> None:
    assert (
        image_generation_model_class("microsoft/Mage-Flow")
        is mage_model.MageFlowImageGenerationModel
    )
    assert (
        image_edit_model_class("microsoft/Mage-Flow-Edit-Turbo")
        is mage_model.MageFlowImageEditModel
    )
    assert is_image_generation_model("mage-flow-base")
    assert not is_image_generation_model("mage-flow-edit")
    assert is_image_edit_model("mage-flow-edit-base")
    assert not is_image_edit_model("mage-flow-turbo")


def test_mage_flow_load_prefers_local_metadata(tmp_path: Path) -> None:
    _write_layout(tmp_path, "mage_flow")
    (tmp_path / "mlx_mage_flow.json").write_text('{"variant":"mage-flow-edit-turbo"}')
    assert (
        mage_model._resolve_load_variant("community/custom-name", tmp_path).name
        == "mage-flow-edit-turbo"
    )


def test_mage_flow_local_variant_uses_cache_parent_name(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--microsoft--Mage-Flow-Edit-Turbo" / "snapshots" / "hash"
    )
    _write_layout(snapshot, "mage_flow")
    assert mage_config.variant_from_local_path(snapshot).name == "mage-flow-edit-turbo"


def test_mage_flow_scheduler_matches_static_shift() -> None:
    scheduler = mage_scheduler.FlowMatchEulerDiscreteScheduler(
        num_inference_steps=4, shift=6.0
    )
    expected = np.array([1.0, 4.5 / 4.75, 3.0 / 3.5, 1.5 / 2.25, 0.0])
    np.testing.assert_allclose(np.array(scheduler.sigmas), expected, rtol=1e-6)


def test_mage_flow_tiny_transformer_forward() -> None:
    transformer = mage_transformer.MageFlowTransformer(
        in_channels=8,
        out_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=4,
        depth=2,
        axes_dim=(2, 2, 4),
    )
    output = transformer(
        img=mx.zeros((1, 4, 8), dtype=mx.float32),
        txt=mx.zeros((1, 3, 16), dtype=mx.float32),
        timesteps=mx.array([1.0]),
        img_shapes=[(1, 2, 2)],
    )
    mx.eval(output)
    assert output.shape == (1, 4, 8)
    assert bool(mx.all(mx.isfinite(output)))


def test_mage_flow_rope_covers_target_and_references() -> None:
    cosine, sine = mage_transformer.image_rope_frequencies(
        [(1, 2, 3), (1, 2, 3)], axes_dim=(2, 2, 4)
    )
    assert cosine.shape == (12, 4)
    assert sine.shape == (12, 4)
    assert not np.allclose(np.array(cosine[:6, 0]), np.array(cosine[6:, 0]))
    np.testing.assert_allclose(np.array(cosine[:6, 1:]), np.array(cosine[6:, 1:]))


def test_mage_flow_weight_sanitizers() -> None:
    transformer = mage_weights.sanitize_transformer_weights(
        {
            "transformer_blocks.0.img_mod.1.weight": mx.zeros((6, 1)),
            "transformer_blocks.0.attn.to_out.0.bias": mx.zeros((1,)),
            "transformer_blocks.0.txt_mlp.net.0.proj.weight": mx.zeros((4, 1)),
        }
    )
    assert "transformer_blocks.0.img_mod.linear.weight" in transformer
    assert "transformer_blocks.0.attn.to_out.bias" in transformer
    assert "transformer_blocks.0.txt_mlp.linear_in.weight" in transformer

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


def test_mage_flow_native_vae_layout_is_not_transposed() -> None:
    weight = mx.zeros((2, 3, 3, 4))
    sanitized = mage_weights.sanitize_vae_weights(
        {"decoder_model.conv_in.weight": weight}, source_layout="mlx_nhwc"
    )
    assert sanitized["decoder_model.conv_in.weight"].shape == weight.shape


def test_mage_flow_quantizes_only_compatible_layers() -> None:
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


def test_mage_flow_quantization_skips_sensitive_transformer_layers() -> None:
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


def test_mage_flow_saved_quantization_metadata(tmp_path: Path) -> None:

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


def test_mage_flow_quantized_weights_require_config() -> None:

    quantized = _TinyLinear()
    nn.quantize(quantized, group_size=64, bits=4, mode="affine")
    with pytest.raises(ValueError, match="quantization mode"):
        mage_weights._apply_weights(
            _TinyLinear(), dict(tree_flatten(quantized.parameters())), {}
        )


def test_mage_flow_conversion_rejects_output_inside_source(tmp_path: Path) -> None:
    _write_layout(tmp_path, "mage_flow")
    with pytest.raises(ValueError, match="inside its source"):
        mage_convert.convert_mage_flow(tmp_path, tmp_path / "converted")


def test_mage_flow_conversion_rejects_ambiguous_local_variant(tmp_path: Path) -> None:
    _write_layout(tmp_path, "mage_flow")
    output = tmp_path.parent / f"{tmp_path.name}-converted"
    with pytest.raises(ValueError, match="--variant"):
        mage_convert.convert_mage_flow(tmp_path, output)


def test_mage_flow_conversion_prefers_source_id_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path, "mage_flow")

    @dataclass
    class FakeComponent:
        quantization_config = None

        def parameters(self):
            return {}

        def update(self, parameters):  # noqa: ARG002
            pass

    class FakeTextEncoder:
        model = FakeComponent()

    import mlx_vlm.models.mage_flow.convert as convert_module

    monkeypatch.setattr(
        convert_module, "load_text_encoder", lambda path: FakeTextEncoder()
    )
    monkeypatch.setattr(
        convert_module, "load_transformer", lambda path: FakeComponent()
    )
    monkeypatch.setattr(convert_module, "load_vae", lambda path: FakeComponent())
    monkeypatch.setattr(convert_module, "_cast_component", lambda model, dtype: None)
    monkeypatch.setattr(
        convert_module,
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


def test_mage_flow_all_six_variants_are_present() -> None:
    assert set(mage_config.VARIANTS) == {
        "mage-flow-base",
        "mage-flow",
        "mage-flow-turbo",
        "mage-flow-edit-base",
        "mage-flow-edit",
        "mage-flow-edit-turbo",
    }


@pytest.mark.parametrize("edit", [False, True], ids=["generate", "edit"])
def test_z_image_dispatch(tmp_path, edit):
    _write_layout(tmp_path, "z_image")
    cls = z_model.ZImageEditModel if edit else z_model.ZImageGenerationModel
    resolve = image_edit_model_class if edit else image_generation_model_class
    supports = is_image_edit_model if edit else is_image_generation_model
    assert cls.supports_model(str(tmp_path))
    assert resolve(str(tmp_path)) is cls
    assert supports(str(tmp_path))
    if not edit:
        assert resolve("Tongyi-MAI/Z-Image") is cls


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
