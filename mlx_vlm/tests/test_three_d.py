"""3D generation contracts, model components, and optional numerical parity checks."""

import base64
import importlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.generate import generate_3d, load_3d_model, three_d
from mlx_vlm.generate.dispatch import parse_arguments
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
from mlx_vlm.models.idol.layers import (
    PcdAbsolutePositionEmbedder,
    PcdAbsolutePositionEmbedderv2,
    SinusoidalEmbedder,
    sanitize_weights,
)
from mlx_vlm.models.idol.pipeline import IdolOutput, IdolPipeline
from mlx_vlm.models.idol.preprocess import preprocess_image
from mlx_vlm.models.idol.sampler import sample_flow, sample_multiview

THREE_D_CASES = json.loads(Path(__file__).with_name("three_d_cases.json").read_text())
THREE_D_FAMILIES = THREE_D_CASES["families"]


def _cli_args(*flags):
    with patch.object(sys, "argv", ["mlx_vlm.generate", "--model", "demo", *flags]):
        return parse_arguments()


def _checkpoint_path(family):
    env = THREE_D_FAMILIES[family]["weights_env"]
    path = os.environ.get(env)
    if not path:
        pytest.skip(f"Set {env} to a downloaded {family} snapshot for checkpoint tests")
    return Path(path)


# Shared API, loader, and CLI contracts.


@pytest.mark.parametrize("save", [False, True])
def test_generate_3d_returns_model_result(tmp_path, save):
    result = SimpleNamespace(save=Mock())
    model = Mock(return_value=result)
    output = tmp_path / "object.ply" if save else None
    actual = generate_3d(model, "object.png", output_path=output, seed=7, steps=2)
    assert actual is result
    model.assert_called_once_with("object.png", seed=7, steps=2)
    if save:
        result.save.assert_called_once_with(output)
    else:
        result.save.assert_not_called()


def test_3d_loader_rejects_unsupported_config(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "qwen2_vl"}')
    download = Mock(side_effect=AssertionError("Unexpected download"))
    monkeypatch.setattr("mlx_vlm.utils.snapshot_download", download)
    with pytest.raises(ValueError, match="Unsupported 3D generation model"):
        load_3d_model(tmp_path)
    download.assert_not_called()


@pytest.mark.parametrize("family", THREE_D_FAMILIES)
def test_image_conditioned_generation(family, tmp_path):
    case = THREE_D_FAMILIES[family]
    weights = _checkpoint_path(family)
    image_path = os.environ.get(case["image_env"])
    if not image_path:
        pytest.skip(f"Set {case['image_env']} and cache encoders for image integration")
    model = load_3d_model(weights, local_files_only=True)
    output = tmp_path / "image.ply"
    result = generate_3d(
        model, image_path, output_path=output, **case["generate_kwargs"]
    )
    assert result.preprocessed_image.size == tuple(case["image_size"])
    assert result.gaussian.xyz.shape == (case["num_gaussians"], 3)
    for name in case["gaussian_attributes"]:
        assert bool(mx.all(mx.isfinite(getattr(result.gaussian, name))))
    assert all(np.isfinite(value) for value in result.camera.values())
    assert result.path == output
    assert output.stat().st_size > case["num_gaussians"] * 17 * 4
    assert (tmp_path / "image_input.png").is_file()
    assert json.loads((tmp_path / "image_camera.json").read_text()) == result.camera


@pytest.mark.parametrize("images", [["castle.png"], ["front.png", "back.png"]])
def test_3d_cli_uses_shared_options_and_saves(tmp_path, images):
    output = tmp_path / "castle.ply"
    args = _cli_args(
        "--output-modality",
        "3d",
        "--image",
        *images,
        "--output",
        str(output),
        "--steps",
        "12",
        "--seed",
        "7",
        "--guidance",
        "2.5",
        "--revision",
        "checkpoint",
        "--gen-kwargs",
        '{"mode": "real", "num_gaussians": 32768}',
    )
    result = SimpleNamespace(path=output, save=Mock())
    model = Mock(return_value=result)
    with patch.object(three_d, "load_3d_model", return_value=model) as load:
        three_d.run_3d_generation_cli(args)
    load.assert_called_once_with("demo", revision="checkpoint", force_download=False)
    model.assert_called_once_with(
        images[0] if len(images) == 1 else images,
        mode="real",
        num_gaussians=32768,
        steps=12,
        seed=7,
        cfg_strength=2.5,
        verbose=False,
    )
    result.save.assert_called_once_with(output)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--image", "one.png", "--output", "out.png"],
        ["--image", "one.png", "--output", "out.glb"],
        ["--image", "one.png", "--output", "out.obj"],
        ["--image", "one.png", "--task", "edit"],
    ],
)
def test_3d_cli_rejects_invalid_inputs_before_loading(flags):
    with patch.object(three_d, "load_3d_model") as load:
        with pytest.raises(ValueError):
            three_d.run_3d_generation_cli(_cli_args("--output-modality", "3d", *flags))
    load.assert_not_called()


# Idol model components and reference comparisons.


@pytest.mark.parametrize(
    "layer_class,input_shape",
    [
        (PcdAbsolutePositionEmbedder, (2, 3)),
        (PcdAbsolutePositionEmbedderv2, (2, 3)),
        (SinusoidalEmbedder, (2,)),
    ],
)
def test_preloaded_embeddings_run_on_worker_thread(layer_class, input_shape):
    layer = layer_class(48)
    mx.eval(layer.parameters())

    def run():
        return np.array(layer(mx.ones(input_shape)))

    # The worker must be the first caller: evaluating on the loading thread
    # beforehand would hide lazy constants tied to that thread's MLX stream.
    with ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(run).result()
    np.testing.assert_array_equal(actual, run())


def _ellipsoid():
    gaussian = object.__new__(Gaussian)
    gaussian.xyz = mx.array([[0.1, 0.2, 0.3]])
    gaussian.scaling = mx.array([[0.12, 0.07, 0.04]])
    gaussian.rotation = mx.array([[1.0, 0.0, 0.0, 0.0]])
    gaussian.opacity = mx.array([[0.9]])
    gaussian.features_dc = mx.array([[[1.0, -1.0, 0.0]]])
    return gaussian


@pytest.fixture
def mesh_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import mlx_vlm.server as server

    monkeypatch.setattr(server.runtime, "mesh_jobs", None)
    monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
    monkeypatch.setattr(server.runtime, "metrics", server.ServerMetricsStore())
    monkeypatch.setenv("MLX_VLM_MESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("MLX_VLM_PRELOAD_MESH_MODEL", raising=False)
    with TestClient(server.app) as client:
        yield client


def _wait_mesh(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/v1/meshes/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.01)
    pytest.fail("Mesh job did not finish")


def test_mesh_server_job_lifecycle(mesh_client, monkeypatch, tmp_path):
    import mlx_vlm.server as server
    from mlx_vlm.utils import load_image

    started, release = Event(), Event()
    calls = []

    def model(image, **kwargs):
        image = load_image(image, mode="RGBA")
        calls.append((image, kwargs))
        started.set()
        assert release.wait(5)
        return IdolOutput(_ellipsoid(), image.convert("RGB"), {"fov": 30.0})

    load = Mock(return_value=(model, None, None))
    monkeypatch.setattr(server, "get_cached_model", load)
    image = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 127)).save(image, format="PNG")
    try:
        response = mesh_client.post(
            "/v1/meshes",
            data={
                "model": "idol",
                "num_inference_steps": "2",
                "guidance_scale": "2.5",
            },
            files={"image": ("../../input.png", image.getvalue(), "image/png")},
        )
        assert response.status_code == 200, response.text
        job = response.json()
        assert job["status"] == "queued"
        assert started.wait(2)
        assert (
            mesh_client.get(f"/v1/meshes/{job['id']}").json()["status"] == "in_progress"
        )
        assert mesh_client.get(f"/v1/meshes/{job['id']}/content").status_code == 404
        assert mesh_client.get("/health").status_code == 200
    finally:
        release.set()
    completed = _wait_mesh(mesh_client, job["id"])
    assert completed["status"] == "completed", completed
    assert completed["format"] == "ply"
    assert completed["representation"] == "gaussian_splat"
    assert completed["camera"] == {"fov": 30.0}
    content = mesh_client.get(completed["url"])
    assert content.status_code == 200
    assert len(content.content) == completed["file_size_bytes"]
    assert content.content.startswith(b"ply\n")
    assert calls[0][0].getpixel((0, 0))[-1] == 127
    assert calls[0][1]["steps"] == 2 and calls[0][1]["cfg_strength"] == 2.5
    load.assert_called_once_with("idol", model_kind="mesh_generation")
    listing = mesh_client.get("/v1/meshes").json()
    assert listing["first_id"] == listing["last_id"] == job["id"]
    assert listing["has_more"] is False
    assert mesh_client.delete(f"/v1/meshes/{job['id']}").json()["status"] == "deleted"
    assert mesh_client.get(f"/v1/meshes/{job['id']}").status_code == 404
    assert mesh_client.get(completed["url"]).status_code == 404
    assert (tmp_path / job["id"] / "output.ply").is_file()
    assert not (tmp_path.parent / "input.png").exists()


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"model": "idol"}, 422),
        ({"model": "idol", "input_images": []}, 422),
        ({"model": "idol", "input_images": ["a.png"] * 11}, 422),
        ({"model": "idol", "input_images": ["a.png", ""]}, 422),
        ({"model": "idol", "input_images": ["a.png", 7]}, 422),
        ({"model": "idol", "input_images": "a.png"}, 422),
        ({"model": "idol", "input_image": "a.png", "input_images": ["b.png"]}, 422),
        ({"input_image": "image.png"}, 400),
        ({"model": "idol", "input_image": "image.png", "output_format": "stl"}, 422),
        ({"model": "idol", "input_image": "image.png", "output_format": "glb"}, 422),
        ({"model": "idol", "input_image": "image.png", "output_format": "obj"}, 422),
        ({"model": "idol", "input_image": "image.png", "num_inference_steps": 0}, 422),
        ({"model": "idol", "input_image": "image.png", "guidance_scale": "nan"}, 422),
        ({"model": "idol", "input_image": "image.png", "prompt": "red castle"}, 422),
        ({"model": "idol", "input_image": "image.png", "num_gaussians": 32769}, 422),
        ([], 422),
    ],
)
def test_mesh_server_rejects_invalid_requests(mesh_client, payload, status):
    response = mesh_client.post("/v1/meshes", json=payload)
    assert response.status_code == status, response.text
    assert mesh_client.get("/v1/meshes").json()["data"] == []


def test_mesh_server_failed_export_is_not_downloadable(
    mesh_client, monkeypatch, tmp_path
):
    import mlx_vlm.server as server
    from mlx_vlm.server import meshes

    monkeypatch.setattr(
        server, "get_cached_model", Mock(return_value=(None, None, None))
    )

    def fail(*args, output_path, **kwargs):
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"partial output")
        raise ValueError("export failed")

    monkeypatch.setattr(meshes, "generate_3d", fail)
    response = mesh_client.post(
        "/v1/meshes", json={"model": "idol", "input_image": "image.png"}
    )
    job = _wait_mesh(mesh_client, response.json()["id"])
    assert job["status"] == "failed" and job["error"]["message"] == "export failed"
    assert mesh_client.get(f"/v1/meshes/{job['id']}/content").status_code == 404


@pytest.mark.parametrize("transport", ["json", "images", "image", "mixed"])
def test_mesh_server_preserves_view_order_and_saves_cameras(
    mesh_client, monkeypatch, tmp_path, transport
):
    import mlx_vlm.server as server
    from mlx_vlm.utils import load_image

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    cameras = [{"yaw": yaw} for yaw in (0.1, 2.0, -1.0)]
    data = []
    for color in colors:
        stream = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(stream, format="PNG")
        data.append(stream.getvalue())
    uris = [
        "data:image/png;base64," + base64.b64encode(value).decode() for value in data
    ]

    def model(images, **kwargs):
        prepared = [load_image(image) for image in images]
        assert [image.getpixel((0, 0)) for image in prepared] == colors
        return IdolOutput(
            _ellipsoid(),
            prepared[0],
            cameras[0],
            preprocessed_images=prepared,
            cameras=cameras,
        )

    monkeypatch.setattr(
        server, "get_cached_model", Mock(return_value=(model, None, None))
    )
    if transport == "json":
        response = mesh_client.post(
            "/v1/meshes", json={"model": "idol", "input_images": uris}
        )
    else:
        files = [
            (transport, (f"{i}.png", value, "image/png"))
            for i, value in enumerate(data)
        ]
        if transport == "mixed":
            files = [
                ("images", ("first.png", data[0], "image/png")),
                ("url", (None, uris[1])),
                ("image[]", ("last.png", data[2], "image/png")),
            ]
        response = mesh_client.post("/v1/meshes", data={"model": "idol"}, files=files)
    assert response.status_code == 200, response.text
    job = _wait_mesh(mesh_client, response.json()["id"])
    assert job["status"] == "completed", job
    assert job["input_count"] == 3
    assert job["camera"] == cameras[0] and job["cameras"] == cameras
    folder = tmp_path / job["id"]
    for index, color in enumerate(colors):
        name = "output_input.png" if index == 0 else f"output_input_{index}.png"
        with Image.open(folder / name) as image:
            assert image.getpixel((0, 0)) == color
    assert json.loads((folder / "output_cameras.json").read_text()) == cameras
    assert mesh_client.get(job["url"]).content == (folder / "output.ply").read_bytes()


def test_mesh_server_bounds_multipart_views_before_loading(mesh_client, monkeypatch):
    import mlx_vlm.server as server

    load = Mock()
    monkeypatch.setattr(server, "get_cached_model", load)
    response = mesh_client.post(
        "/v1/meshes",
        data={"model": "idol"},
        files=[("images", (f"{i}.png", b"unused", "image/png")) for i in range(11)],
    )
    assert response.status_code == 422
    load.assert_not_called()
    assert mesh_client.get("/v1/meshes").json()["data"] == []


def test_mesh_schema_advertises_multiple_views(mesh_client):
    body = mesh_client.get("/openapi.json").json()["paths"]["/v1/meshes"]["post"][
        "requestBody"
    ]["content"]
    images = body["multipart/form-data"]["schema"]["properties"]["images"]
    assert images["type"] == "array" and images["maxItems"] == 10
    assert images["items"]["format"] == "binary"
    assert "input_images" in body["application/json"]["schema"]["properties"]


def test_mesh_cache_reuse_switch_and_unload(monkeypatch):
    import mlx_vlm.server as server

    registry = server.ModelCacheRegistry()
    registry.set("image_generation", {"model": "image model"})
    monkeypatch.setattr(server.runtime, "model_cache", registry)
    monkeypatch.setattr(server.runtime, "mesh_jobs", None)
    load = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(server._app_module, "load_3d_model", load)
    first = server.get_cached_model("idol-a", model_kind="mesh_generation")
    assert server.get_cached_model("idol-a", model_kind="mesh_generation") == first
    second = server.get_cached_model("idol-b", model_kind="mesh_generation")
    assert first[0] is not second[0] and load.call_count == 2
    assert registry.for_kind("image_generation")["model"] == "image model"
    assert server._app_module._unload_model_cache_group("mesh_generation")
    assert registry.for_kind("mesh_generation") == {}


def test_mesh_jobs_serialize_and_survive_record_deletion_and_unload(
    mesh_client, monkeypatch, tmp_path
):
    import mlx_vlm.server as server
    from mlx_vlm.utils import load_image

    started, release = Event(), Event()
    loads = []
    server.runtime.model_cache.set("mesh_generation", {"model_path": "preloaded"})
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    source = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()

    def load(name, **options):
        loads.append(name)

        def model(image, **kwargs):
            if kwargs["seed"] == 0:
                started.set()
                assert release.wait(5)
            return IdolOutput(_ellipsoid(), load_image(image), {"fov": 30.0})

        return model, None, None

    monkeypatch.setattr(server, "get_cached_model", load)
    payload = dict(input_image=source, output_format="ply", steps=2, guidance=2)
    try:
        first = mesh_client.post("/v1/meshes", json={**payload, "seed": 0}).json()
        assert first["model"] == "preloaded" and started.wait(2)
        second = mesh_client.post(
            "/v1/meshes", json={**payload, "model": "other", "seed": 1}
        ).json()
        third = mesh_client.post(
            "/v1/meshes", json={**payload, "model": "other", "seed": 2}
        ).json()
        assert loads == ["preloaded"]
        assert (
            mesh_client.get(f"/v1/meshes/{second['id']}").json()["status"] == "queued"
        )
        page = mesh_client.get("/v1/meshes?limit=1&order=asc").json()
        assert page["last_id"] == first["id"] and page["has_more"]
        next_page = mesh_client.get(
            f"/v1/meshes?limit=1&order=asc&after={first['id']}"
        ).json()
        assert next_page["first_id"] == second["id"]
        assert mesh_client.get("/v1/meshes?limit=0").status_code == 422
        assert mesh_client.get("/v1/meshes?after=unknown").status_code == 404
        assert mesh_client.delete(f"/v1/meshes/{first['id']}").status_code == 200
    finally:
        release.set()
    for job in (second, third):
        assert _wait_mesh(mesh_client, job["id"])["status"] == "completed"
    assert loads == ["preloaded", "other", "other"]
    assert mesh_client.get(f"/v1/meshes/{first['id']}").status_code == 404
    assert (tmp_path / first["id"] / "output.ply").is_file()
    assert mesh_client.post("/unload").status_code == 200
    assert mesh_client.get(f"/v1/meshes/{third['id']}/content").status_code == 200
    resumed = mesh_client.post(
        "/v1/meshes", json={**payload, "model": "other", "seed": 3}
    ).json()
    assert _wait_mesh(mesh_client, resumed["id"])["status"] == "completed"


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/v1/meshes"),
        ("get", "/v1/meshes"),
        ("get", "/v1/meshes/id"),
        ("get", "/v1/meshes/id/content"),
        ("delete", "/v1/meshes/id"),
    ],
)
def test_mesh_endpoints_require_configured_auth(mesh_client, monkeypatch, method, path):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret")
    assert getattr(mesh_client, method)(path).status_code == 401


IDOL_DENOISER_ARGS = dict(
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
IDOL_REPRESENTATION = {
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


def test_idol_denoiser_forward():
    model = LatentSeqMMFlowModel(**(IDOL_DENOISER_ARGS | {"num_blocks": 1}))
    model.dtype = mx.float32
    sample = {"latent": mx.zeros((1, 8, 4)), "camera": mx.zeros((1, 1, 5))}
    cond = {"feature1": mx.zeros((1, 6, 12)), "feature2": mx.zeros((1, 6, 8))}
    output = model(sample, mx.array([500.0]), cond)
    for name, value in output.items():
        assert value.shape == sample[name].shape
        assert bool(mx.all(mx.isfinite(value)))


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


@pytest.mark.parametrize("batch_cfg", [False, True])
@pytest.mark.parametrize("strength", [0.0, 3.0])
def test_multiview_shared_latent_and_independent_cameras(batch_cfg, strength):
    def model(sample, t, cond):
        value = cond["feature1"][:, :1, :1]
        return {
            "latent": 0.1 * sample["latent"] + sample["camera"] + value,
            "camera": 0.5 * sample["camera"] + value,
        }

    noise = {"latent": mx.zeros((1, 2, 1)), "camera": mx.zeros((1, 1, 1))}
    conditions = [{"feature1": mx.full((1, 1, 1), v)} for v in (1.0, 3.0)]
    for views in (conditions, conditions[::-1]):
        result = sample_multiview(
            model,
            noise,
            views,
            steps=2,
            rescale_t=1.0,
            cfg_strength=strength,
            batch_cfg=batch_cfg,
        )
        # Two Euler steps, dt=1/2. All views must see the same old latent.
        np.testing.assert_allclose(
            np.array(result["latent"]), -1.45 * (1 + strength), rtol=1e-6
        )
        for camera, cond in zip(result["cameras"], views):
            expected = -0.875 * (1 + strength) * np.array(cond["feature1"])
            np.testing.assert_allclose(np.array(camera), expected, rtol=1e-6)
        np.testing.assert_array_equal(
            np.array(result["camera"]), np.array(result["cameras"][0])
        )


@pytest.mark.parametrize("batch_cfg", [False, True])
@pytest.mark.parametrize("strength", [0.0, 3.0])
def test_multiview_one_or_duplicate_view_matches_single_sampler(batch_cfg, strength):
    mx.random.seed(7)
    model = LatentSeqMMFlowModel(**IDOL_DENOISER_ARGS)
    model.dtype = mx.float32
    noise = {
        "latent": mx.random.normal((1, 8, 4)),
        "camera": mx.random.normal((1, 1, 5)),
    }
    cond = {
        "feature1": mx.random.normal((1, 6, 12)),
        "feature2": mx.random.normal((1, 6, 8)),
    }
    options = dict(steps=3, cfg_strength=strength, batch_cfg=batch_cfg)
    expected = sample_flow(model, noise, cond, **options)
    for count in (1, 2):
        actual = sample_multiview(model, noise, [cond] * count, **options)
        for key in ("latent", "camera"):
            np.testing.assert_array_equal(
                np.array(actual[key]), np.array(expected[key])
            )


def test_multiview_rejects_empty_or_batched_conditions():
    noise = {"latent": mx.zeros((1, 2, 3)), "camera": mx.zeros((1, 1, 5))}
    with pytest.raises(ValueError, match="At least one"):
        sample_multiview(None, noise, [])
    with pytest.raises(ValueError, match="one image per view"):
        sample_multiview(None, noise, [{"feature1": mx.zeros((2, 2, 3))}])


def test_sanitize_nested_sequential_weights():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Sequential(nn.Sequential(nn.Linear(2, 3)))]

    weights = {
        "layers.0.0.0.weight": mx.ones((3, 2)),
        "layers.0.0.0.bias": mx.zeros((3,)),
    }
    model = Nested()
    model.load_weights(list(sanitize_weights(model, weights).items()))
    np.testing.assert_array_equal(
        np.array(model.layers[0](mx.ones((1, 2)))), [[2, 2, 2]]
    )
    # Preserve unknown keys so standard strict loading can report them.
    with pytest.raises(ValueError, match="not in model"):
        model.load_weights(
            list(sanitize_weights(model, {**weights, "unrelated": mx.array(1)}).items())
        )


def test_denoiser_sanitize_preserves_mixed_precision():
    model = LatentSeqMMFlowModel(**IDOL_DENOISER_ARGS)
    weights = model.sanitize(
        {
            "blocks.0.mlp.mlp.0.weight": mx.ones((96, 24)),
            "noise_refiner.0.mlp.mlp.0.weight": mx.ones((96, 24)),
            "context_refiner.0.mlp.mlp.0.weight": mx.ones((96, 24)),
            "input_layer.weight": mx.ones((24, 4), dtype=mx.float16),
            "repo_layers.0.final_map.weight": mx.ones((6, 24), dtype=mx.float16),
        }
    )
    for prefix in ("blocks", "noise_refiner", "context_refiner"):
        assert weights[f"{prefix}.0.mlp.mlp.layers.0.weight"].dtype == mx.float16
    assert weights["input_layer.weight"].dtype == mx.float32
    assert weights["repo_layers.0.final_map.weight"].dtype == mx.float32


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
    # Metal softplus rounding varies across GPUs at these small scales.
    np.testing.assert_allclose(
        np.array(g.scaling), np.sqrt(0.004**2 + 0.0009**2), rtol=1e-5
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


@pytest.mark.parametrize("source_kind", ["path", "bytes", "data_uri", "pil"])
def test_pipeline_uses_shared_image_loading_with_alpha(tmp_path, source_kind):
    from mlx_vlm.utils import load_image

    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[4:12, 4:12] = [255, 0, 0, 255]
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path = tmp_path / "cutout.png"
    path.write_bytes(buffer.getvalue())
    source = {
        "path": path,
        "bytes": io.BytesIO(buffer.getvalue()),
        "data_uri": "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode(),
        "pil": image,
    }[source_kind]
    pipeline = object.__new__(IdolPipeline)
    pipeline.config = {"image_size": 32}
    pipeline.background_removal = False
    pipeline.background_remover = None
    result = pipeline.preprocess(source)
    assert result.size == (32, 32)
    assert np.asarray(result)[16, 16, 0] > 200
    # Existing VLM callers keep the RGB default.
    assert load_image(path).mode == "RGB"
    assert load_image(path, mode="RGBA").getchannel("A").getextrema() == (0, 255)


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


def test_pipeline_multiview_keeps_primary_rng_and_single_image_outputs(tmp_path):
    pipe = object.__new__(IdolPipeline)
    pipe.config = {"image_size": 16}
    pipe.background_removal = False
    pipe.background_remover = None
    pipe.denoiser = LatentSeqMMFlowModel(**IDOL_DENOISER_ARGS)
    pipe.denoiser.dtype = mx.float32
    pipe.decoder_gs = SimpleNamespace(rep_config={"num_gaussians": 32})
    pipe.image_encoder = lambda pixels: {
        "feature1": mx.random.normal((1, 6, 12)) + pixels.mean(),
        "feature2": mx.random.normal((1, 6, 8)),
    }
    decoded = []

    def decode(latent, count):
        decoded.append((np.array(latent), np.array(mx.random.normal((8,)))))
        return _ellipsoid()

    pipe.decode = decode
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 127))
    outputs = [pipe(source, steps=2) for source in (image, [image], [image, image])]
    for (latent, decoder_noise), output in zip(decoded[1:], outputs[1:]):
        np.testing.assert_array_equal(latent, decoded[0][0])
        np.testing.assert_array_equal(decoder_noise, decoded[0][1])
        assert output.camera == outputs[0].camera
        assert (
            output.preprocessed_image.tobytes()
            == outputs[0].preprocessed_image.tobytes()
        )
    assert len(outputs[2].cameras) == len(outputs[2].preprocessed_images) == 2
    outputs[0].save(tmp_path / "single.ply")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "single.ply",
        "single_camera.json",
        "single_input.png",
    ]
    with pytest.raises(ValueError, match="At least one"):
        pipe([], steps=2)


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
    model = DINOv3(config.to_dict())
    model.load_weights(list(model.sanitize(_torch_weights(ref)).items()))
    model.eval()
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
    model = Flux2ImageEncoder(config)
    model.load_weights(list(model.sanitize(_torch_weights(ref)).items()))
    model.set_dtype(mx.bfloat16)
    model.eval()
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
def idol_upstream(monkeypatch):
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


def test_denoiser_matches_upstream(idol_upstream):
    torch = pytest.importorskip("torch")
    tensordict = pytest.importorskip("tensordict")
    torch.manual_seed(0)
    ref = idol_upstream.dit.LatentSeqMMFlowModel(**IDOL_DENOISER_ARGS).float().eval()
    ref.dtype = torch.float32
    model = LatentSeqMMFlowModel(**IDOL_DENOISER_ARGS)
    model.dtype = mx.float32
    model.load_weights(list(model.sanitize(_torch_weights(ref)).items()))
    model.eval()
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
def test_decoders_match_upstream(idol_upstream, gaussian):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    args = dict(model_channels=24, cond_channels=4, num_blocks=2, num_heads=2)
    if gaussian:
        args.update(in_channels=3, representation_config=IDOL_REPRESENTATION)
        ref = idol_upstream.decoder.GaussianFixedlenDecoder(**args).float().eval()
        model = GaussianFixedlenDecoder(**args)
    else:
        ref = (
            idol_upstream.decoder.OctreeProbabilityFixedlenDecoder(**args)
            .float()
            .eval()
        )
        model = OctreeProbabilityFixedlenDecoder(**args)
    ref.dtype, model.dtype = torch.float32, mx.float32
    model.load_weights(list(model.sanitize(_torch_weights(ref)).items()))
    model.eval()
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
def idol_checkpoint_path():
    return _checkpoint_path("idol")


@pytest.mark.parametrize("name", ["denoiser", "decoder", "decoder_gs"])
def test_trained_checkpoint_matches_upstream(idol_upstream, idol_checkpoint_path, name):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    tensordict = pytest.importorskip("tensordict")
    torch.manual_seed(0)
    config = json.loads((idol_checkpoint_path / "config.json").read_text())
    classes = {
        "denoiser": (LatentSeqMMFlowModel, idol_upstream.dit.LatentSeqMMFlowModel),
        "decoder": (
            OctreeProbabilityFixedlenDecoder,
            idol_upstream.decoder.OctreeProbabilityFixedlenDecoder,
        ),
        "decoder_gs": (
            GaussianFixedlenDecoder,
            idol_upstream.decoder.GaussianFixedlenDecoder,
        ),
    }
    mlx_cls, torch_cls = classes[name]
    model = mlx_cls(**config[name]["args"])
    weights = model.sanitize(mx.load(str(idol_checkpoint_path / f"{name}.safetensors")))
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    ref = torch_cls(**config[name]["args"]).eval()
    ref.load_state_dict(
        safetensors.load_file(str(idol_checkpoint_path / f"{name}.safetensors"))
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


def test_checkpoint_sampling_and_export(idol_checkpoint_path, tmp_path):
    pipeline = IdolPipeline.from_pretrained(
        idol_checkpoint_path, local_files_only=True, background_removal=False
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
