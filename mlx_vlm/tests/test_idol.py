"""Idol inference contracts and optional PyTorch numerical parity checks."""

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.models.idol.birefnet import _DeformableConv2d, _image2patches, resize
from mlx_vlm.models.idol.camera import decode_camera
from mlx_vlm.models.idol.conditioning import DINOv3, Flux2ImageEncoder
from mlx_vlm.models.idol.decoder import (
    GaussianFixedlenDecoder,
    OctreeProbabilityFixedlenDecoder,
    systematic_sample,
)
from mlx_vlm.models.idol.dit import LatentSeqMMFlowModel
from mlx_vlm.models.idol.gaussian import Gaussian
from mlx_vlm.models.idol.pipeline import IdolPipeline
from mlx_vlm.models.idol.preprocess import preprocess_image
from mlx_vlm.models.idol.sampler import sample_flow
from mlx_vlm.models.idol.weights import load_weights

DENOISER_ARGS = dict(
    q_token_length=8,
    in_channels=4,
    cam_channels=5,
    model_channels=24,
    cond_channels=12,
    cond2_channels=8,
    num_blocks=2,
    num_heads=2,
    num_head_channels=12,
)
REPRESENTATION = {
    "num_gaussians": 2,
    "lr": {
        "_xyz": 1.0,
        "_features_dc": 1.0,
        "_opacity": 1.0,
        "_scaling": 1.0,
        "_rotation": 0.1,
    },
    "perturbe_size": 1.5,
    "3d_filter_kernel_size": 0.0009,
    "scaling_bias": 0.004,
    "opacity_bias": 0.1,
}


@pytest.mark.parametrize("offset", [0.0, 0.125, 0.73])
def test_systematic_sampling_matches_searchsorted(offset):
    probs = np.array(
        [[0.25, 0.25, 0.25, 0.25], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]],
        np.float32,
    )
    counts = np.array([4, 11, 0], np.int32)
    actual = np.array(
        systematic_sample(mx.array(probs), mx.array(counts), mx.full((3,), offset))
    )
    expected = []
    for row, count in zip(probs, counts):
        row = row / row.sum() if row.sum() else np.ones(4) / 4
        grid = (offset + np.arange(count)) / max(count, 1)
        bins = np.minimum(np.searchsorted(row.cumsum(), grid, side="left"), 3)
        expected.append(np.bincount(bins, minlength=4))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.sum(-1), counts)


def test_octree_sampling_conserves_points_and_leaf_bounds():
    def model(points, level, cond):
        # Only the high x/y/z child is occupied at every level.
        return mx.broadcast_to(mx.array([-1000.0] * 7 + [0.0]), (*points.shape[:2], 8))

    mx.random.seed(0)
    points = OctreeProbabilityFixedlenDecoder.sample(model, mx.zeros((2, 8, 4)), 37, 4)
    assert points.shape == (2, 37, 3)
    assert bool(mx.all((points >= 15 / 16) & (points < 1)))


@pytest.mark.parametrize("batch_cfg", [False, True])
def test_flow_uses_idol_guidance_and_integrates_to_zero(batch_cfg):
    def model(sample, t, cond):
        value = cond["feature1"][:, :1, :1]
        return {k: mx.broadcast_to(value, v.shape) for k, v in sample.items()}

    noise = {"latent": mx.zeros((1, 2, 3)), "camera": mx.zeros((1, 1, 5))}
    out = sample_flow(
        model,
        noise,
        {"feature1": mx.ones((1, 2, 3))},
        steps=5,
        cfg_strength=3.0,
        rescale_t=3.0,
        batch_cfg=batch_cfg,
    )
    # Idol uses (1+w)*positive - w*negative, so this velocity is 4.
    for value in out.values():
        np.testing.assert_allclose(np.array(value), -4.0, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [{"steps": 0}, {"rescale_t": 0}, {"rescale_t": float("nan")}, {"cfg_strength": -1}],
)
def test_sampler_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        sample_flow(None, {}, {}, **kwargs)


def test_nested_sequential_weight_mapping_and_strictness():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = [nn.Sequential(nn.Sequential(nn.Linear(2, 3)))]

    weights = {
        "stages.0.0.0.weight": mx.ones((3, 2)),
        "stages.0.0.0.bias": mx.zeros((3,)),
    }
    model = load_weights(Nested(), weights)
    np.testing.assert_array_equal(
        np.array(model.stages[0](mx.ones((1, 2)))), [[2, 2, 2]]
    )
    with pytest.raises(ValueError, match="missing"):
        load_weights(Nested(), {})
    with pytest.raises(ValueError, match="unexpected"):
        load_weights(Nested(), {**weights, "unrelated": mx.array(1)})


def test_gaussian_ply_activation_and_coordinate_convention(tmp_path):
    g = Gaussian(
        mx.array([[0.6, 0.7, 0.8]]),
        mx.ones((1, 1, 3)),
        mx.zeros((1, 3)),
        mx.zeros((1, 4)),
        mx.array([[1000.0]]),
        0.004,
        0.1,
        0.0009,
    )
    path = tmp_path / "object.ply"
    g.save_ply(path)
    header, data = path.read_bytes().split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 1" in header
    values = np.frombuffer(data, "<f4").reshape(1, 17)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[0, :3], [0.1, -0.3, 0.2], atol=1e-7)
    np.testing.assert_allclose(values[0, -4:], [2**-0.5, 2**-0.5, 0, 0], atol=1e-7)
    np.testing.assert_allclose(
        np.array(g.scaling), np.sqrt(0.004**2 + 0.0009**2), rtol=1e-6
    )


def test_preprocess_alpha_and_empty_foreground():
    arr = np.zeros((16, 20, 4), dtype=np.uint8)
    arr[4:12, 6:14] = [255, 0, 0, 255]
    out = preprocess_image(Image.fromarray(arr), None, image_size=32)
    assert out.mode == "RGB" and out.size == (32, 32)
    assert np.asarray(out)[16, 16, 0] > 200
    with pytest.raises(ValueError, match="alpha channel is empty"):
        preprocess_image(Image.new("RGBA", (8, 8)), None, image_size=8)
    with pytest.raises(ValueError, match="background removal"):
        preprocess_image(Image.new("RGB", (8, 8)), None, image_size=8)


def test_camera_conventions():
    camera = decode_camera(mx.array([0.0, 1.0, 0.0, 0.5, 1.0]))
    assert camera["yaw"] == 0 and camera["pitch"] == 0 and camera["radius"] == 2
    assert camera["fov"] == pytest.approx(53.1301023542)
    assert decode_camera(mx.zeros((5,)))["radius"] == 1000


def test_seed_is_applied_after_lazy_encoder_initialization(monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            mx.eval(mx.random.normal((17,)))

        def __call__(self, pixels):
            return mx.random.normal((4,))

    monkeypatch.setattr("mlx_vlm.models.idol.conditioning.ImageEncoder", Encoder)
    pipeline = object.__new__(IdolPipeline)
    pipeline.image_encoder = None
    pipeline._dinov3 = pipeline._flux2_vae = "unused"
    pipeline._dinov3_dtype = mx.float32
    pipeline.local_files_only = True
    first = pipeline.encode_image(Image.new("RGB", (16, 16)), seed=42)
    mx.eval(first)
    second = pipeline.encode_image(Image.new("RGB", (16, 16)), seed=42)
    np.testing.assert_array_equal(np.array(first), np.array(second))


def _torch_weights(model):
    return {
        key: (
            mx.array(value.detach().float().numpy())
            if value.is_floating_point()
            else mx.array(value.detach().numpy().astype(np.int32))
        )
        for key, value in model.state_dict().items()
    }


def _assert_torch(actual, expected, atol=1e-5, rtol=3e-4):
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.detach().float().numpy(),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("kernel", [1, 3, 7])
def test_native_deformable_convolution_matches_torchvision(kernel):
    torch = pytest.importorskip("torch")
    ops = pytest.importorskip("torchvision.ops")
    rng = np.random.default_rng(0)
    model = _DeformableConv2d(3, 5, kernel_size=kernel, padding=kernel // 2)
    image = mx.array(rng.normal(size=(2, 7, 9, 3)).astype(np.float32))
    offset = model.offset_conv(image)
    mask = 2 * mx.sigmoid(model.modulator_conv(image))

    def nchw(value):
        return torch.from_numpy(np.array(value.transpose(0, 3, 1, 2)))

    expected = ops.deform_conv2d(
        nchw(image),
        nchw(offset),
        torch.from_numpy(np.array(model.regular_conv.weight.transpose(0, 3, 1, 2))),
        padding=kernel // 2,
        mask=nchw(mask),
    )
    _assert_torch(model(image), expected.permute(0, 2, 3, 1), atol=3e-6)


@pytest.mark.parametrize("size", [(3, 4), (13, 19), (1, 1)])
def test_resize_matches_align_corners(size):
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(0).normal(size=(1, 7, 9, 3)).astype(np.float32)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(x).permute(0, 3, 1, 2),
        size=size,
        mode="bilinear",
        align_corners=True,
    )
    _assert_torch(resize(mx.array(x), size), expected.permute(0, 2, 3, 1))


def test_image_patch_channel_order():
    x = np.arange(1 * 8 * 12 * 3).reshape(1, 8, 12, 3).astype(np.float32)
    expected = (
        x.transpose(0, 3, 1, 2)
        .reshape(1, 3, 2, 4, 3, 4)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(1, 18, 4, 4)
        .transpose(0, 2, 3, 1)
    )
    np.testing.assert_array_equal(
        np.array(_image2patches(mx.array(x), mx.zeros((1, 4, 4, 1)))), expected
    )


@pytest.mark.parametrize("gated", [False, True])
def test_dinov3_matches_transformers(gated):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DINOv3ViTModel"):
        pytest.skip("DINOv3 requires transformers >= 4.56")
    torch.manual_seed(0)
    config = transformers.DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=4,
        num_register_tokens=4,
        use_gated_mlp=gated,
        hidden_act="silu" if gated else "gelu",
    )
    ref = transformers.DINOv3ViTModel(config).eval()
    model = load_weights(DINOv3(config.to_dict()), _torch_weights(ref))
    x = torch.randn(2, 3, 16, 20)
    with torch.no_grad():
        expected = ref(x).last_hidden_state
    _assert_torch(model(mx.array(x.permute(0, 2, 3, 1).numpy())), expected)


def test_flux2_posterior_and_token_packing_matches_diffusers():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        pytest.skip("FLUX.2 requires diffusers >= 0.36")
    torch.manual_seed(0)
    config = dict(
        block_out_channels=(32, 32, 32, 32),
        norm_num_groups=32,
        latent_channels=32,
        layers_per_block=1,
    )
    ref = diffusers.AutoencoderKLFlux2(**config).eval().to(torch.bfloat16)
    weights = {
        k.replace(".to_out.0.", ".to_out."): v
        for k, v in _torch_weights(ref).items()
        if k.startswith(("encoder.", "quant_conv.", "bn."))
    }
    model = load_weights(Flux2ImageEncoder(config), weights, dtype=mx.bfloat16)
    pixels = torch.randn(1, 3, 32, 32).to(torch.bfloat16)
    with torch.no_grad():
        posterior = ref.encode(pixels).latent_dist
        noise = torch.randn_like(posterior.mean)
        latent = posterior.mean + posterior.std * noise
        b, c, h, w = latent.shape
        latent = (
            latent.reshape(b, c, h // 2, 2, w // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(b, c * 4, h // 2, w // 2)
        )
        latent = (latent - ref.bn.running_mean.reshape(1, -1, 1, 1)) / torch.sqrt(
            ref.bn.running_var.reshape(1, -1, 1, 1) + ref.config.batch_norm_eps
        )
        tokens = latent.float().flatten(2).transpose(1, 2)
        expected = torch.cat([torch.zeros(b, 5, c * 4), tokens], 1)

    def nhwc(t):
        return mx.array(t.float().permute(0, 2, 3, 1).numpy()).astype(mx.bfloat16)

    _assert_torch(
        model(nhwc(pixels), noise=nhwc(noise)), expected, atol=0.06, rtol=0.12
    )


@pytest.fixture
def upstream(monkeypatch):
    path = os.environ.get("IDOL_REFERENCE_SDK")
    if not path:
        pytest.skip(
            "Set IDOL_REFERENCE_SDK to an idol-sdk checkout for upstream parity"
        )
    monkeypatch.syspath_prepend(path)
    return SimpleNamespace(
        dit=importlib.import_module("idol.models.dit"),
        decoder=importlib.import_module("idol.models.decoder"),
        layers=importlib.import_module("idol.models.layers"),
    )


def test_denoiser_matches_upstream(upstream):
    torch = pytest.importorskip("torch")
    tensordict = pytest.importorskip("tensordict")
    torch.manual_seed(0)
    ref = upstream.dit.LatentSeqMMFlowModel(**DENOISER_ARGS).float().eval()
    ref.dtype = torch.float32
    model = load_weights(LatentSeqMMFlowModel(**DENOISER_ARGS), _torch_weights(ref))
    model.dtype = mx.float32
    sample = tensordict.TensorDict(
        {"latent": torch.randn(2, 8, 4), "camera": torch.randn(2, 1, 5)}, batch_size=2
    )
    condition = tensordict.TensorDict(
        {"feature1": torch.randn(2, 6, 12), "feature2": torch.randn(2, 6, 8)},
        batch_size=2,
    )
    inputs = {k: mx.array(v.numpy()) for k, v in sample.items()}
    cond = {k: mx.array(v.numpy()) for k, v in condition.items()}
    with torch.no_grad():
        expected = ref(sample, torch.tensor([999.0, 10.0]), condition)
    actual = model(inputs, mx.array([999.0, 10.0]), cond)
    for key in actual:
        _assert_torch(actual[key], expected[key])


@pytest.mark.parametrize("gaussian", [False, True])
def test_decoders_match_upstream(upstream, gaussian):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    args = dict(model_channels=24, cond_channels=4, num_blocks=2, num_heads=2)
    if gaussian:
        args.update(in_channels=3, representation_config=REPRESENTATION)
        ref = upstream.decoder.GaussianFixedlenDecoder(**args).float().eval()
        model = GaussianFixedlenDecoder(**args)
    else:
        ref = upstream.decoder.OctreeProbabilityFixedlenDecoder(**args).float().eval()
        model = OctreeProbabilityFixedlenDecoder(**args)
    load_weights(model, _torch_weights(ref))
    ref.dtype, model.dtype = torch.float32, mx.float32
    p, c = torch.rand(2, 5, 3), torch.randn(2, 8, 4)
    pm, cm = mx.array(p.numpy()), mx.array(c.numpy())
    with torch.no_grad():
        expected = ref(p, c) if gaussian else ref(p, torch.tensor([2, 256]), c)
    actual = model(pm, cm) if gaussian else model(pm, mx.array([2, 256]), cm)
    _assert_torch(actual, expected)
    if gaussian:
        ga = model.to_representation(pm, actual)
        gb = ref.to_representation(p, expected)
        for name in ("xyz", "features_dc", "scaling", "rotation", "opacity"):
            _assert_torch(getattr(ga[0], name), getattr(gb[0], name))


@pytest.fixture
def checkpoint_path():
    path = os.environ.get("IDOL_WEIGHTS")
    if not path:
        pytest.skip(
            "Set IDOL_WEIGHTS to a downloaded Idol snapshot for checkpoint tests"
        )
    return Path(path)


@pytest.mark.parametrize("name", ["denoiser", "decoder", "decoder_gs"])
def test_trained_checkpoint_matches_upstream(upstream, checkpoint_path, name):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    tensordict = pytest.importorskip("tensordict")
    torch.manual_seed(0)
    config = json.loads((checkpoint_path / "config.json").read_text())
    classes = {
        "denoiser": (LatentSeqMMFlowModel, upstream.dit.LatentSeqMMFlowModel),
        "decoder": (
            OctreeProbabilityFixedlenDecoder,
            upstream.decoder.OctreeProbabilityFixedlenDecoder,
        ),
        "decoder_gs": (
            GaussianFixedlenDecoder,
            upstream.decoder.GaussianFixedlenDecoder,
        ),
    }
    mlx_cls, torch_cls = classes[name]
    model = load_weights(
        mlx_cls(**config[name]["args"]),
        checkpoint_path / f"{name}.safetensors",
        mixed_precision=True,
    )
    ref = torch_cls(**config[name]["args"]).eval()
    ref.load_state_dict(
        safetensors.load_file(str(checkpoint_path / f"{name}.safetensors"))
    )
    with torch.no_grad():
        if name == "denoiser":
            ref.pos_pe = ref.pos_pe[:, :32]
            model.pos_pe = model.pos_pe[:, :32]
            sample = tensordict.TensorDict(
                {"latent": torch.randn(1, 32, 16), "camera": torch.randn(1, 1, 5)},
                batch_size=1,
            )
            cond = tensordict.TensorDict(
                {
                    "feature1": torch.randn(1, 17, 1280),
                    "feature2": torch.randn(1, 17, 128),
                },
                batch_size=1,
            )
            actual = model(
                {k: mx.array(v.numpy()) for k, v in sample.items()},
                mx.array([500.0]),
                {k: mx.array(v.numpy()) for k, v in cond.items()},
            )
            expected = ref(sample, torch.tensor([500.0]), cond)
        else:
            points, cond = torch.rand(1, 16, 3), torch.randn(1, 32, 16)
            pm, cm = mx.array(points.numpy()), mx.array(cond.numpy())
            actual = {
                "out": (
                    model(pm, mx.array([32]), cm)
                    if name == "decoder"
                    else model(pm, cm)
                )
            }
            expected = {
                "out": (
                    ref(points, torch.tensor([32]), cond)
                    if name == "decoder"
                    else ref(points, cond)
                )
            }
    for key in actual:
        a = np.array(actual[key])
        b = expected[key].float().numpy()
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-10) < 0.01


def test_checkpoint_sampling_and_export(checkpoint_path, tmp_path):
    pipeline = IdolPipeline.from_pretrained(
        checkpoint_path, local_files_only=True, background_removal=False
    )
    condition = {
        "feature1": mx.zeros((1, 4101, 1280)),
        "feature2": mx.zeros((1, 4101, 128)),
    }
    sampled = pipeline.sample(condition, steps=2, cfg_strength=0)
    gaussian = pipeline.decode(sampled["latent"], 32768)
    assert gaussian.xyz.shape == (32768, 3)
    assert bool(mx.all(mx.isfinite(gaussian.xyz)))
    gaussian.save_ply(tmp_path / "smoke.ply")
    assert (tmp_path / "smoke.ply").stat().st_size > 32768 * 17 * 4


def test_image_conditioned_pipeline(checkpoint_path, tmp_path):
    image_path = os.environ.get("IDOL_TEST_IMAGE")
    if not image_path:
        pytest.skip("Set IDOL_TEST_IMAGE and cache both encoders for image integration")
    pipeline = IdolPipeline.from_pretrained(checkpoint_path, local_files_only=True)
    result = pipeline(image_path, seed=42, steps=2, num_gaussians=32768)
    assert result.preprocessed_image.size == (1024, 1024)
    assert result.gaussian.xyz.shape == (32768, 3)
    for name in ("xyz", "features_dc", "scaling", "rotation", "opacity"):
        assert bool(mx.all(mx.isfinite(getattr(result.gaussian, name))))
    assert all(np.isfinite(value) for value in result.camera.values())
    result.save(tmp_path / "image.ply")
    assert (tmp_path / "image.ply").stat().st_size > 32768 * 17 * 4
    assert (tmp_path / "image_input.png").is_file()
    assert json.loads((tmp_path / "image_camera.json").read_text()) == result.camera
