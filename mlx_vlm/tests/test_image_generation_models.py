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
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image

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
IMAGE_FAMILIES = IMAGE_CASES["families"]


def _assert_attributes(value, expected):
    assert {name: getattr(value, name) for name in expected} == expected


def _family_module(family, component):
    return importlib.import_module(f"mlx_vlm.models.{family}.{component}")


class _ModelFamily:
    """Load a real model submodule when a behavioral check accesses it."""

    def __init__(self, family):
        self.family = family

    def __getattr__(self, component):
        return _family_module(self.family, component)


bonsai = _ModelFamily("bonsai")
flux = _ModelFamily("flux2")
ideogram = _ModelFamily("ideogram4")
z = _ModelFamily("z_image")
ernie = _ModelFamily("ernie_image")
mage = _ModelFamily("mage_flow")


def _model_class(family, edit=False):
    name = IMAGE_FAMILIES[family]["prefix"].removesuffix("Image") + "Image"
    return getattr(
        _family_module(family, "model"),
        name + ("EditModel" if edit else "GenerationModel"),
    )


@pytest.mark.parametrize(
    "family", [family for family, spec in IMAGE_FAMILIES.items() if "edit" in spec]
)
def test_edit_model_dispatch(tmp_path, family):
    _write_layout(tmp_path, family)
    cls = _model_class(family, edit=True)
    settings = IMAGE_FAMILIES[family]["edit"]
    model_id = settings.get("alias", str(tmp_path))
    assert cls.supports_model(model_id)
    assert image_edit_model_class(model_id) is cls
    assert is_image_edit_model(settings.get("probe", model_id))
    if family == "mage_flow":
        assert not is_image_generation_model("mage-flow-edit")
        assert not is_image_edit_model("mage-flow-turbo")


def _component(case):
    family, component = case["module"].split(".")
    suffix = {
        "transformer": "Transformer",
        "text_encoder": "TextEncoder",
        "vae": "VAE",
    }[component]
    module = _family_module(family, component)
    name = IMAGE_FAMILIES[family]["prefix"] + suffix
    model_cls = getattr(module, name)
    config_name = case.get("config_class", name + "Config")
    if config_name is None:
        return model_cls(**case["config"])
    return model_cls(getattr(module, config_name)(**case["config"]))


class ForwardInputs:
    """Construct tensors for the component APIs supported by the JSON cases."""

    @staticmethod
    def text_encoder(model, inputs):
        return model(mx.array(inputs["tokens"]))

    @staticmethod
    def vae_decode(model, inputs):
        return model.decode(mx.random.normal(inputs["latents"]))

    @staticmethod
    def ideogram4(model, inputs):
        return model(
            llm_features=mx.zeros(inputs["features"]),
            x=mx.zeros(inputs["image"]),
            t=mx.array([0.5]),
            position_ids=mx.zeros(inputs["positions"], dtype=mx.int32),
            segment_ids=mx.ones(inputs["segments"], dtype=mx.int32),
            indicator=mx.array(
                [
                    [
                        ideogram.transformer.LLM_TOKEN_INDICATOR,
                        ideogram.transformer.OUTPUT_IMAGE_INDICATOR,
                        ideogram.transformer.OUTPUT_IMAGE_INDICATOR,
                    ]
                ]
            ),
        )

    @staticmethod
    def ernie_image(model, inputs):
        return model(
            mx.zeros(inputs["image"], dtype=mx.bfloat16),
            timestep=mx.array(inputs["timesteps"], dtype=mx.bfloat16),
            text_hidden_states=mx.zeros(inputs["text"], dtype=mx.bfloat16),
            text_lengths=mx.array(inputs["text_lengths"]),
        )

    @staticmethod
    def mage_flow(model, inputs):
        return model(
            img=mx.zeros(inputs["image"]),
            txt=mx.zeros(inputs["text"]),
            timesteps=mx.array([1.0]),
            img_shapes=[tuple(shape) for shape in inputs["image_shapes"]],
        )

    @staticmethod
    def z_image(model, inputs):
        # Preserve the original padding check at all three transformer stages.
        stages = [
            MagicMock(side_effect=lambda hidden, *args, **kwargs: hidden)
            for _ in range(3)
        ]
        noise, context, unified = stages
        model.noise_refiner, model.context_refiner, model.layers = (
            [noise],
            [context],
            [unified],
        )
        output = model(
            mx.random.normal(inputs["image"]),
            mx.array([0.5]),
            mx.random.normal(inputs["context"]),
        )
        assert tuple(stage.call_args.args[0].shape[1] for stage in stages) == tuple(
            inputs["padded_lengths"]
        )
        return output


def _check_download(tmp_path, monkeypatch, case):
    module = importlib.import_module(f"mlx_vlm.models.{case['module']}")
    options = dict(case["config"])
    directory_option = case.get("directory_option")
    if directory_option:
        options[directory_option] = tmp_path
    cached = tmp_path / case.get("directory_name", "snapshot")
    snapshot = MagicMock(return_value=str(cached))
    validate = MagicMock(return_value=cached)
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    monkeypatch.setattr(
        module, case.get("validator", "validate_model_layout"), validate
    )
    if hasattr(module, "find_valid_cached_snapshot"):
        monkeypatch.setattr(
            module, "find_valid_cached_snapshot", MagicMock(return_value=None)
        )

    assert module.download_model(**options) == cached
    snapshot.assert_called_once()
    validate.assert_called_once_with(cached, **case.get("validation_options", {}))
    kwargs = snapshot.call_args.kwargs
    assert kwargs["repo_id"] == case["repo_id"]
    assert kwargs["max_workers"] == options["max_workers"]
    if directory_option:
        assert kwargs["local_dir"] == str(cached)
        assert cached.is_dir()
    else:
        assert "local_dir" not in kwargs
    if patterns := getattr(module, "DOWNLOAD_PATTERNS", None):
        assert kwargs["allow_patterns"] == list(patterns)
    assert set(case.get("required_patterns", ())) <= set(
        kwargs.get("allow_patterns", ())
    )
    assert not set(case.get("excluded_patterns", ())) & set(
        kwargs.get("allow_patterns", ())
    )
    assert {key: kwargs[key] for key in case.get("forwarded", {})} == case.get(
        "forwarded", {}
    )


class ImageChecks:
    """Shared image contracts; model calls and fixture construction stay in Python."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch

    def download(self, case):
        _check_download(self.tmp_path, self.monkeypatch, case)

    def variant(self, case):
        get_variant = _family_module(case["family"], "config").get_variant
        for alias in case["aliases"]:
            if "error" in case:
                with pytest.raises(ValueError, match=case["error"]):
                    get_variant(alias)
            else:
                _assert_attributes(get_variant(alias), case["expected"])

    def local_variant(self, case):
        family, root = case["family"], self.tmp_path / case.get("path", "")
        if case.get("layout", True):
            _write_layout(root, family, turbo=case.get("turbo", True))
        _write_files(root, case.get("files", ()), case.get("metadata"))
        config = _family_module(family, "config")
        if "source_id" in case:
            actual = _family_module(family, "model")._resolve_load_variant(
                case["source_id"], root
            )
        elif family == "z_image":
            actual = config.ZImageConfig.from_model_path(root)
        else:
            actual = config.variant_from_local_path(root)
        _assert_attributes(actual, case["expected"])

    def wrapper(self, case):
        family, variant = case["family"], case["variant"]
        edit = case["task"] == "edit"
        options = dict(case["request"])
        if edit:
            options["image_paths"] = tuple(options["image_paths"])
        image_request = (ImageEditRequest if edit else ImageGenerationRequest)(
            **options
        )
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

    def forward(self, case):
        output = getattr(ForwardInputs, case["adapter"])(
            _component(case), case["input"]
        )
        mx.eval(output)
        assert output.shape == tuple(case["expected_shape"])
        assert bool(mx.all(mx.isfinite(output)))

    def sanitize(self, case):
        keys = case["keys"]
        weights = {
            key: mx.arange(np.prod(row["shape"]).item(), dtype=mx.float32).reshape(
                row["shape"]
            )
            for key, row in keys.items()
        }
        expected = {
            row.get("target", key): (
                weights[key].transpose(row["axes"]) if "axes" in row else weights[key]
            )
            for key, row in keys.items()
            if row.get("target", key) is not None
        }
        options = dict(case.get("options", {}))
        if case.get("target_shapes"):
            options["target_shapes"] = {
                key: value.shape for key, value in expected.items()
            }
        sanitize = getattr(
            _family_module(case["family"], case.get("module", "weights")),
            f"sanitize_{case['component']}_weights",
        )
        actual = sanitize(weights, **options)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert bool(mx.array_equal(actual[key], expected[key]))
        if "roundtrip_options" in case:
            again = sanitize(actual, **case["roundtrip_options"])
            assert again.keys() == actual.keys()
            for key in actual:
                assert bool(mx.array_equal(again[key], actual[key]))

    def layout(self, case):
        family, root = case["family"], self.tmp_path
        module = _family_module(family, "download")
        if family == "ideogram4":
            _write_files(
                root,
                metadata={
                    "model_index.json": {
                        "_class_name": IMAGE_FAMILIES[family]["index_class"]
                    }
                },
            )
        if case.get("valid") or case.get("files"):
            _write_layout(root, family)
            _write_files(root, case.get("files", []))
        if case.get("missing"):
            with pytest.raises(FileNotFoundError, match=case["missing"]):
                module.validate_model_layout(root)
        if family == "mage_flow":
            _write_layout(root, family)
        if case.get("valid") or family == "mage_flow":
            assert module.validate_model_layout(root) == root

    def quantized_load(self, case):
        family = case["family"]
        options = {key: case[key] for key in ("mode", "bits", "group_size")}
        model = _TinyLinear(16 if family == "ernie_image" else 32)
        if family == "ernie_image":
            dense = mx.arange(16 * 64, dtype=mx.float32).reshape(16, 64)
            weights = dict(
                zip(
                    ["proj.weight", "proj.scales", "proj.biases"],
                    mx.quantize(dense, **options),
                )
            )
        else:
            quantized = _TinyLinear()
            nn.quantize(quantized, **options)
            weights = dict(tree_flatten(quantized.parameters()))
        module = _family_module(family, "weights")
        load = (
            module.apply_weights if family == "ernie_image" else module._apply_weights
        )
        loaded = load(model, weights, _quantization_metadata(**options))
        assert loaded.quantization_config == options
        assert isinstance(loaded.proj, nn.QuantizedLinear)
        if family == "z_image" and case["mode"] == "affine":
            assert hasattr(loaded.proj, "biases")

    def conversion(self, case):
        family, mode = case["family"], case["mode"]
        repo, bits, group_size = case["repo"], case["bits"], case["group_size"]
        module = _family_module(family, "convert")
        method = "convert_" + family
        source, output = self.tmp_path / "source", self.tmp_path / "output"
        if family == "z_image":
            _write_files(
                source, metadata={"model_index.json": {"_class_name": "ZImagePipeline"}}
            )
        else:
            _write_layout(source, family)
        self.monkeypatch.setattr(module, "get_model_path", lambda *a, **kw: source)
        convert = MagicMock(return_value=output)
        self.monkeypatch.setattr(module, method, convert)
        extra = dict(q_group_size=None, q_bits=None) if family == "ernie_image" else {}
        assert (
            module.convert(repo, output, quantize=True, q_mode=mode, **extra) == output
        )
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

    def save_reload(self, case):
        family, root = case["family"], self.tmp_path
        convert, weights_module = (
            _family_module(family, component) for component in ("convert", "weights")
        )
        model = _TinyLinear()
        config = dict(mode="affine", bits=4, group_size=64)
        if family == "z_image":
            nn.quantize(model, **config)
            convert._save_component(
                root, "transformer", model, {"quantization": config}
            )
            root = root / "transformer"
        else:
            config = convert._quantization_parameters("affine", 64, 4)
            convert._quantize_component(model, config)
            convert._save_component(root, model, config)
        index = json.loads((root / "model.safetensors.index.json").read_text())
        expected = case["metadata"]
        assert {key: index["metadata"][key] for key in expected} == expected
        if case.get("reload"):
            component_config = json.loads((root / "config.json").read_text())
            for key in ("quantization", "quantization_config"):
                assert component_config[key] == config
            weights, metadata = weights_module._load_safetensors(root)
            loaded = weights_module._apply_weights(
                _TinyLinear(), weights, component_config | metadata
            )
            assert isinstance(loaded.proj, nn.QuantizedLinear)
            assert loaded.quantization_config == config


@pytest.mark.parametrize(
    "check,case",
    [
        pytest.param(check, case, id=f"{check}-{case['id']}")
        for check, cases in IMAGE_CASES["checks"].items()
        for case in cases
    ],
)
def test_image_contract(check, case, tmp_path, monkeypatch):
    getattr(ImageChecks(tmp_path, monkeypatch), check)(case)


EXPANDED_CAPTION = ideogram.prompting.format_caption(IMAGE_CASES["caption"])


def _pipeline(family, **components):
    cls = getattr(
        _family_module(family, "pipeline"), IMAGE_FAMILIES[family]["pipeline_class"]
    )
    pipeline = cls.__new__(cls)
    for name, value in components.items():
        setattr(pipeline, name, value)
    return pipeline


def _flux_edit_pipeline(
    variant: str = "flux2-klein-9b-kv",
) -> flux.pipeline.Flux2ImageEdit:
    pipeline = flux.pipeline.Flux2ImageEdit.__new__(flux.pipeline.Flux2ImageEdit)
    pipeline.variant = flux.config.get_variant(variant)
    pipeline.model_path = None
    pipeline.runtime_config = flux.pipeline.Flux2RuntimeConfig(tiled_vae="off")
    pipeline.tokenizer = SimpleNamespace(count_tokens=lambda text: 7)
    pipeline.edit_array = lambda *args, **kwargs: mx.zeros((20, 24, 3), dtype=mx.uint8)
    return pipeline


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
            self.config = z.config.ZImageConfig(
                default_steps=steps,
                default_guidance=guidance,
                scheduler_shift=shift,
                variant=variant,
            )
        elif family == "ernie_image":
            self.variant = ernie.config.get_variant(variant)
            self.runtime_config = ernie.pipeline.ErnieImageRuntimeConfig(
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


def _ernie_runtime_pipeline(variant, *, evict=False):
    return _pipeline(
        "ernie_image",
        variant=ernie.config.get_variant(variant),
        model_path=Path("/tmp/ernie"),
        runtime_config=ernie.pipeline.ErnieImageRuntimeConfig(
            evict_text_encoder=False, evict_transformer=evict, use_prompt_enhancer=False
        ),
        tokenizer=SimpleNamespace(count_tokens=lambda prompt: len(prompt)),
        text_encoder=None,
        prompt_enhancer=None,
        component_quantization={},
        prompt_cache={},
        transformer=_ErnieTransformer(),
        vae=_ErnieVAE(),
        _encode_prompts=lambda prompts: (
            mx.zeros((len(prompts), 3, 4), dtype=mx.bfloat16),
            mx.array([1, 3] if len(prompts) == 2 else [3]),
        ),
        _ensure_components=lambda **kwargs: None,
    )


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
    layout = IMAGE_CASES["layouts"][family]
    metadata = dict(layout["metadata"])
    if family == "ernie_image" and not turbo:
        metadata["mlx_ernie_image.json"] = dict(
            model_type=family, variant="ernie-image"
        )
    _write_files(root, layout["files"], metadata)


class _PackedVAE:
    def decode_packed_latents(self, packed, tiling_config=None):
        return mx.zeros(
            (1, 3, packed.shape[2] * 16, packed.shape[3] * 16), dtype=mx.bfloat16
        )


def _packed_pipeline(family):
    spec = IMAGE_FAMILIES[family]
    runtime = getattr(
        _family_module(family, "pipeline"), spec["prefix"] + "RuntimeConfig"
    )
    return _pipeline(
        family,
        variant=_family_module(family, "config").get_variant(spec["packed_variant"]),
        model_path=None,
        runtime_config=runtime(tiled_vae="off"),
        tokenizer=None,
        transformer=lambda **kwargs: mx.zeros_like(kwargs["hidden_states"]),
        vae=_PackedVAE(),
        _encode_prompt=lambda prompt, max_sequence_length: (
            mx.zeros((1, 32, 7680), dtype=mx.bfloat16),
            mx.zeros((1, 32, 4), dtype=mx.int32),
        ),
        _ensure_transformer_and_vae=lambda: None,
    )


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


def _weight_index(root, keys, shard="model.safetensors"):
    _write_files(
        root,
        metadata={
            "model.safetensors.index.json": {"weight_map": {k: shard for k in keys}}
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
        pytest.param(family, width, height, id=f"{family}-{width}x{height}")
        for family, spec in IMAGE_FAMILIES.items()
        for width, height in spec["invalid_dimensions"]
    ],
)
def test_invalid_dimensions(family, width, height):
    if family == "z_image":
        with pytest.raises(ValueError, match="positive multiple of 16"):
            _pipeline(family).generate_array("prompt", width=width, height=height)
    else:
        with pytest.raises(ValueError):
            _family_module(family, "config").validate_dimensions(
                width=width, height=height
            )


@pytest.mark.parametrize(
    "convert", [ernie.convert, mage.convert], ids=["ernie_image", "mage_flow"]
)
def test_incompatible_quantization_options(convert):
    with pytest.raises(ValueError, match="requires"):
        convert._quantization_parameters("mxfp8", 64, 8)


@pytest.mark.parametrize("family", IMAGE_FAMILIES)
def test_generation_model_dispatch(monkeypatch, tmp_path, family):
    settings = IMAGE_FAMILIES[family]
    aliases, probe = settings["aliases"], settings.get("probe")
    cls = _model_class(family)
    _write_layout(tmp_path, family)
    _write_files(tmp_path, metadata=IMAGE_CASES["discovery_metadata"].get(family))
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
                c.kwargs["allow_patterns"] == DISCOVERY_PATTERNS
                for c in lookup.call_args_list
            )


def test_bonsai_parse_size():
    assert bonsai.config.parse_size("1248x832") == (1248, 832)
    assert bonsai.config.parse_size("832x1248") == (832, 1248)


DISCOVERY_PATTERNS = [
    "model_index.json",
    "config.json",
    "manifest.json",
    "**/config.json",
]


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
        is flux.model.Flux2ImageGenerationModel
    )
    assert calls == [
        DISCOVERY_PATTERNS,
        DISCOVERY_PATTERNS + ["**/model.safetensors.index.json"],
    ]


def test_flux2_text_encoder_accepts_native_quantized_keys(monkeypatch, tmp_path):
    class TinyTextEncoder(nn.Module):
        def __init__(self, **kwargs) -> None:  # noqa: ARG002
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 64)

    dense = mx.arange(4 * 64, dtype=mx.float32).reshape(4, 64)
    packed, scales, biases = mx.quantize(dense, group_size=32, bits=8)
    monkeypatch.setattr(flux.weights, "Qwen3TextEncoder", TinyTextEncoder)
    monkeypatch.setattr(
        flux.weights,
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

    model = flux.weights.load_text_encoder(
        tmp_path, flux.config.get_variant("flux2-klein-4b")
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
    cases = [(source, "pytorch_nchw"), (native, "mlx_nhwc")]
    if family == "ernie_image":
        cases.append((native, None))
    for value, layout in cases:
        options = dict(source_layout=layout) if family == "ernie_image" else {}
        result = match(
            value, target_shape=native.shape, key="encoder.conv_in.weight", **options
        )
        assert_array_equal(result, native)


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
    resolve = MagicMock(return_value=model_path)
    load = MagicMock(return_value=_flux_edit_pipeline("flux2-klein-4b"))
    monkeypatch.setattr(image_module, "get_model_path", resolve)
    monkeypatch.setattr(flux.pipeline.Flux2ImageEdit, "from_pretrained", load)

    model = load_image_edit_model("mlx-community/flux2-klein-4b-8bit")

    assert model.variant == "flux2-klein-4b"
    assert resolve.call_args.args == ("mlx-community/flux2-klein-4b-8bit",)
    assert load.call_args.args[0].name == "flux2-klein-4b"
    assert load.call_args.kwargs["model_path"] == model_path


def test_flux2_reference_image_array_keeps_float32_input():
    image = Image.new("RGB", (1, 1), color=(255, 127, 0))
    array = flux.pipeline._reference_image_array(image)

    assert array.dtype == mx.float32
    assert np.array(array).shape == (1, 3, 1, 1)


def test_ideogram4_plain_prompt_wraps_as_minimal_json_caption():
    prepared = ideogram.prompting.normalize_prompt(
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
    assert ideogram.prompting.is_structured_caption(prepared.text)


def test_ideogram4_caption_warnings_cover_elements_and_bounding_boxes():
    prompt = ideogram.prompting.format_caption(
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
        prepared = ideogram.prompting.normalize_prompt(prompt)

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
    result = ideogram.prompting.generate_prompt_expansion_caption(
        "A red cube.", model="tiny-text-model", aspect_ratio="1:1"
    )
    load.assert_called_once_with("tiny-text-model")
    assert template.call_args.args[:2] == (processor, {})
    messages = template.call_args.args[2]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert all(
        isinstance(message["content"], str) and message["content"]
        for message in messages
    )
    assert "A red cube." in messages[1]["content"] and "1:1" in messages[1]["content"]
    schema.assert_called_once_with(
        tokenizer, ideogram.prompting.IDEOGRAM4_CAPTION_SCHEMA
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

    converted = ideogram.weights.dequantize_fp8_weight_only(raw, precision=mx.float32)

    assert "linear.weight_scale" not in converted
    assert_allclose(converted["linear.weight"], expected, rtol=0, atol=0)
    assert converted["linear.bias"].dtype == mx.float32


def test_ideogram4_build_inputs_packs_text_and_image_tokens():
    tokenizer = MagicMock(return_value={"input_ids": [11, 22, 33]})
    tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0][
        "content"
    ][0]["text"]
    pipeline = _pipeline("ideogram4", tokenizer=tokenizer)

    inputs = pipeline._build_inputs("prompt", height=256, width=256)

    assert inputs["text_token_ids"].shape == (1, 3)
    assert inputs["num_text_tokens"] == 3
    assert inputs["num_image_tokens"] == 16 * 16
    assert inputs["position_ids"].shape == (1, 3 + 16 * 16, 3)
    assert (
        int(inputs["indicator"][0, 0].item())
        == ideogram.transformer.LLM_TOKEN_INDICATOR
    )
    assert (
        int(inputs["indicator"][0, -1].item())
        == ideogram.transformer.OUTPUT_IMAGE_INDICATOR
    )


def test_ideogram4_pipeline_uses_prepared_prompt_and_reports_metadata():
    pipeline = _pipeline(
        "ideogram4",
        model_path=Path("/tmp/fake-ideogram"),
        runtime_config=ideogram.pipeline.Ideogram4RuntimeConfig(
            evict_text_encoder=False, evict_transformers=False
        ),
        text_encoder=object(),
        conditional_transformer=lambda **kwargs: mx.zeros_like(kwargs["x"]),
        unconditional_transformer=lambda **kwargs: mx.zeros_like(kwargs["x"]),
        vae=object(),
        prepare_prompt=MagicMock(
            return_value=ideogram.prompting.NormalizedPrompt(
                text=EXPANDED_CAPTION,
                is_json_caption=True,
                is_structured_caption=True,
                was_wrapped=False,
                prompt_expansion_model="tiny-text-model",
                prompt_expansion_used=True,
            )
        ),
        _build_inputs=MagicMock(
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
        ),
        _encode_text=lambda token_ids, num_image_tokens: mx.zeros((1, 2, 4)),
        _ensure_transformers_and_vae=lambda: None,
        _decode=lambda z, grid_h, grid_w: mx.zeros((16, 16, 3), dtype=mx.uint8),
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
    decode = MagicMock(return_value=mx.zeros((1, 3, 32, 32), dtype=mx.float32))
    vae = SimpleNamespace(
        decode=decode,
        decode_packed_latents=MagicMock(
            side_effect=AssertionError(
                "Ideogram decode should use latent_norm + vae.decode"
            )
        ),
    )
    array = _pipeline("ideogram4", vae=vae)._decode(
        mx.zeros((1, 16 * 16, 128)), grid_h=16, grid_w=16
    )
    assert array.shape == (32, 32, 3)
    assert decode.call_args.args[0].shape == (1, 32, 32, 32)


def test_detects_diffusers_z_image_model_index(tmp_path):
    (tmp_path / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}')
    assert z.convert.is_z_image_model_path(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name":"FluxPipeline"}')
    assert not z.convert.is_z_image_model_path(tmp_path)


@pytest.mark.parametrize(
    "steps,strength,expected", [(9, 0.6, 3), (9, 0.5, 4), (8, 0.6, 3), (8, 0.3, 5)]
)
def test_z_image_img2img_start_index_matches_diffusers(steps, strength, expected):
    assert z.pipeline._img2img_start_index(steps, strength) == expected


def test_z_image_generation_evicts_components_before_reloading_encoder(monkeypatch):
    pipeline = _pipeline(
        "z_image",
        evict_text_encoder=True,
        transformer=object(),
        vae=object(),
        text_encoder=None,
    )

    def reload_encoder():
        assert pipeline.transformer is None
        assert pipeline.vae is None

    reload = MagicMock(side_effect=reload_encoder)
    monkeypatch.setattr(pipeline, "_reload_encoder", reload)
    monkeypatch.setattr(
        pipeline,
        "_encode_prompt",
        MagicMock(side_effect=RuntimeError("stop after reload")),
    )
    with pytest.raises(RuntimeError, match="stop after reload"):
        pipeline.generate_array("fox", steps=2, width=16, height=16)
    reload.assert_called_once_with()


def test_z_image_rejects_classifier_free_guidance():
    model = object.__new__(z.model.ZImageGenerationModel)
    model.pipeline = SimpleNamespace(config=z.config.ZImageConfig())
    with pytest.raises(ValueError, match="does not support classifier-free guidance"):
        model.generate(ImageGenerationRequest(prompt="test", guidance=2.0))


def test_z_image_conversion_preserves_native_vae_layout(tmp_path):
    native = mx.zeros((8, 3, 3, 4))
    vae_path = tmp_path / "vae"
    vae_path.mkdir()
    (vae_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"mlx_vlm_format": "z_image"}})
    )

    converted = z.convert._sanitize_vae_for_conversion(
        vae_path, {"encoder.conv_in.weight": native}
    )

    assert converted["encoder.conv_in.weight"].shape == native.shape
    assert mx.array_equal(converted["encoder.conv_in.weight"], native)


def test_ernie_dispatches_from_weight_index(tmp_path):
    _write_layout(tmp_path, "ernie_image")
    for name in ("model_index.json", "mlx_ernie_image.json"):
        (tmp_path / name).unlink()
    _weight_index(
        tmp_path / "transformer",
        [
            "adaln_modulation.weight",
            "final_norm.linear.weight",
            "layers.0.adaLN_sa_ln.weight",
        ],
        shard="0.safetensors",
    )
    assert (
        image_generation_model_class(tmp_path.as_posix())
        is ernie.model.ErnieImageGenerationModel
    )


def test_ernie_image_transformer_config_parses_official_fields():
    config = ernie.config.ErnieImageTransformerConfig.from_dict(
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

    cos, sin = ernie.transformer.rope_frequencies(
        mx.array(ids), axes_dim=axes, theta=256.0
    )
    assert_allclose(cos.transpose(0, 2, 1, 3), np.cos(angles), rtol=1e-6)
    assert_allclose(sin.transpose(0, 2, 1, 3), np.sin(angles), rtol=1e-6)

    values = mx.arange(16, dtype=mx.float32).reshape(1, 1, 2, 8)
    expected_rotated = np.concatenate(
        [-np.array(values)[..., 4:], np.array(values)[..., :4]], axis=-1
    )
    assert_array_equal(ernie.transformer.rotate_half(values), expected_rotated)


def test_ernie_image_conditioning_skips_last_text_block_and_final_norm():
    encoder = _component(IMAGE_CASES["conditioning_encoder"])
    encoder.embed_tokens = lambda ids: mx.zeros((*ids.shape, 2))
    encoder.layers = [lambda hidden, mask, cache=None: hidden + 1] * 3
    encoder.norm = lambda hidden: hidden * 10
    ids = mx.array([[1, 2]])

    assert_array_equal(encoder(ids), np.full((1, 2, 2), 2))
    assert_array_equal(encoder(ids, normalize=True), np.full((1, 2, 2), 30))


def test_ernie_image_pad_text_preserves_cfg_order_and_lengths():
    negative = mx.ones((1, 1, 2))
    positive = mx.full((1, 3, 2), 2)
    padded, lengths = ernie.pipeline._pad_text([negative, positive])
    assert tuple(padded.shape) == (2, 3, 2)
    assert_array_equal(lengths, [1, 3])
    assert_array_equal(padded[0, 1:], np.zeros((2, 2)))


def test_ernie_image_conversion_quantizes_compatible_vae_attention():
    vae = flux.vae.Flux2VAE(
        decoder_block_out_channels=(32, 32),
        include_encoder=True,
        encoder_block_out_channels=(32, 32),
    )
    ernie.convert._quantize_component(
        vae,
        {"mode": "mxfp8", "group_size": 32, "bits": 8},
        lambda path, module: hasattr(module, "to_quantized"),
    )
    assert isinstance(vae.decoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
    assert isinstance(vae.encoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
    assert isinstance(vae.decoder.conv_in, nn.Conv2d)


def test_generation_request_converts_to_edit_request():
    edit = MagicMock(return_value=SimpleNamespace(path=None))
    generate_image(
        SimpleNamespace(edit=edit),
        ImageGenerationRequest(prompt="edit"),
        task="edit",
        image_paths=("reference.png",),
    )
    request = edit.call_args.args[0]
    assert isinstance(request, ImageEditRequest)
    assert request.image_paths == ("reference.png",)


def test_ernie_edit_model_defaults_prompt_enhancer_off(monkeypatch):
    load = MagicMock(
        return_value=_RecordingPipeline("ernie_image", "ernie-image-turbo")
    )
    monkeypatch.setattr(ernie.model.ErnieImagePipeline, "from_pretrained", load)
    for options, expected in [({}, False), ({"use_prompt_enhancer": True}, True)]:
        ernie.model.ErnieImageEditModel.from_model_id("ernie-image-turbo", **options)
        assert load.call_args.kwargs["use_prompt_enhancer"] is expected


def test_ernie_image_prompt_cache_evicts_least_recently_used_entry():
    pipeline = _pipeline(
        "ernie_image",
        runtime_config=ernie.pipeline.ErnieImageRuntimeConfig(prompt_cache_size=2),
        prompt_cache=OrderedDict(),
        tokenizer=SimpleNamespace(encode=lambda prompt: prompt),
        text_encoder=lambda prompt: mx.array([len(prompt)]),
    )

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
    _, width, height = ernie.pipeline._load_edit_image(
        image_path, width=None, height=None
    )
    assert (width, height) == expected


def test_ernie_image_img2img_rejects_decoder_only_converted_vae():
    with pytest.raises(ValueError, match="VAE encoder weights"):
        ernie.weights._require_vae_encoder_weights(
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
        assert ernie.convert._vae_checkpoint_has_encoder(tmp_path) is expected


def test_ernie_image_prompt_enhancement_auto_detects_optional_components(
    tmp_path,
):
    pipeline = _ernie_runtime_pipeline("ernie-image-turbo")
    pipeline.model_path = tmp_path
    assert not pipeline._should_enhance_prompt()
    _write_files(
        tmp_path, ["pe/model.safetensors"], {"pe_tokenizer/tokenizer.json": {}}
    )
    pipeline.runtime_config = ernie.pipeline.ErnieImageRuntimeConfig(
        use_prompt_enhancer=None
    )
    assert pipeline._should_enhance_prompt()
    # Auto-detected enhancement applies to generation only: img2img needs the
    # source-aware prompt preserved unless callers explicitly opt in.
    assert not pipeline._should_enhance_prompt(for_edit=True)
    pipeline.runtime_config = ernie.pipeline.ErnieImageRuntimeConfig(
        use_prompt_enhancer=True
    )
    assert pipeline._should_enhance_prompt(for_edit=True)


def test_ernie_image_conversion_detection_and_layout_metadata(tmp_path):
    _write_layout(tmp_path, "ernie_image")
    assert ernie.convert.is_ernie_image_checkpoint(tmp_path)
    assert ernie.convert._source_layout(tmp_path) == "mlx_nhwc"
    output = tmp_path / "output"
    for component in ("transformer", "text_encoder", "vae"):
        (output / component).mkdir(parents=True, exist_ok=True)
    ernie.convert._write_missing_configs(output)
    assert (
        json.loads((output / "transformer" / "config.json").read_text())["_class_name"]
        == "ErnieImageTransformer2DModel"
    )
    assert (
        json.loads((output / "model_index.json").read_text())["_class_name"]
        == "ErnieImagePipeline"
    )


def test_mage_flow_scheduler_matches_static_shift():
    scheduler = mage.scheduler.FlowMatchEulerDiscreteScheduler(
        num_inference_steps=4, shift=6.0
    )
    expected = np.array([1.0, 4.5 / 4.75, 3.0 / 3.5, 1.5 / 2.25, 0.0])
    assert_allclose(scheduler.sigmas, expected, rtol=1e-6)


def test_mage_flow_rope_covers_target_and_references():
    cosine, sine = mage.transformer.image_rope_frequencies(
        [(1, 2, 3), (1, 2, 3)], axes_dim=(2, 2, 4)
    )
    assert cosine.shape == (12, 4)
    assert sine.shape == (12, 4)
    assert not np.allclose(np.array(cosine[:6, 0]), np.array(cosine[6:, 0]))
    assert_allclose(cosine[:6, 1:], cosine[6:, 1:])


def test_mage_flow_quantizes_only_compatible_layers():
    model = _TinyLinear()
    model.incompatible = nn.Linear(63, 32, bias=False)
    config = mage.convert._quantization_parameters("affine", 64, 4)
    mage.convert._quantize_component(model, config)
    assert isinstance(model.proj, nn.QuantizedLinear)
    assert isinstance(model.incompatible, nn.Linear)
    assert model.quantization_config == config


@pytest.mark.parametrize(
    "layer,expected",
    [
        ("transformer_blocks.0.attn.to_q", True),
        ("transformer_blocks.0.img_mod.linear", False),
        ("transformer_blocks.0.txt_mod.linear", False),
        ("proj_out", False),
    ],
)
def test_mage_flow_quantization_skips_sensitive_transformer_layers(layer, expected):
    assert (
        mage.convert._transformer_quantization_predicate(
            layer, nn.Linear(64, 32, bias=False)
        )
        is expected
    )


def test_mage_flow_quantized_weights_require_config():

    quantized = _TinyLinear()
    nn.quantize(quantized, group_size=64, bits=4, mode="affine")
    with pytest.raises(ValueError, match="quantization mode"):
        mage.weights._apply_weights(
            _TinyLinear(), dict(tree_flatten(quantized.parameters())), {}
        )


def test_mage_flow_conversion_rejects_output_inside_source(tmp_path):
    _write_layout(tmp_path, "mage_flow")
    with pytest.raises(ValueError, match="inside its source"):
        mage.convert.convert_mage_flow(tmp_path, tmp_path / "converted")


def test_mage_flow_conversion_rejects_ambiguous_local_variant(tmp_path):
    _write_layout(tmp_path, "mage_flow")
    output = tmp_path.parent / f"{tmp_path.name}-converted"
    with pytest.raises(ValueError, match="--variant"):
        mage.convert.convert_mage_flow(tmp_path, output)


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
            mage.convert, "load_" + name, MagicMock(return_value=result)
        )
    monkeypatch.setattr(mage.convert, "_cast_component", lambda model, dtype: None)
    monkeypatch.setattr(
        mage.convert,
        "_save_component",
        lambda directory, model, quantization: directory.mkdir(
            parents=True, exist_ok=True
        ),
    )

    output = tmp_path.parent / f"{tmp_path.name}-converted"
    mage.convert.convert_mage_flow(
        tmp_path, output, source_id="mage-flow-community/Mage-Flow-Edit-Turbo"
    )
    metadata = json.loads((output / "mlx_mage_flow.json").read_text())
    assert metadata["variant"] == "mage-flow-edit-turbo"


@pytest.mark.parametrize("case", ["expand", "structured", "invalid"])
def test_ideogram_prompt_expansion_policy(monkeypatch, case):
    failure = ideogram.prompting.PromptExpansionCaptionError("bad json")
    expand = MagicMock(
        return_value=ideogram.prompting.PromptExpansionResult(
            text=EXPANDED_CAPTION, raw_text=EXPANDED_CAPTION, model="tiny-text-model"
        ),
        side_effect=failure if case == "invalid" else None,
    )
    monkeypatch.setattr(ideogram.prompting, "generate_prompt_expansion_caption", expand)
    if case == "invalid":
        with pytest.warns(UserWarning, match="falling back"):
            result = ideogram.prompting.prepare_prompt(
                "A red cube.", prompt_expansion_model="bad-model"
            )
        assert result.was_wrapped and not result.prompt_expansion_used
        assert result.prompt_expansion_error == "bad json"
    else:
        prompt = EXPANDED_CAPTION if case == "structured" else "A red cube."
        result = ideogram.prompting.prepare_prompt(
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
