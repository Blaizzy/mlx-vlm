"""Audio-only extraction and safe in-place checkpoint patching."""

import copy
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from dev import patch_muse_spark_audio as patcher
from mlx_vlm.tests.test_muse_spark import _CASE, create_model, hf_checkpoint
from mlx_vlm.utils import load_model


def write_json(path, value):
    path.write_text(json.dumps(value))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_unrelated_fp8_tensor(path):
    """The original NVFP4 shard contains dtypes that MLX cannot parse directly."""
    header, offset = patcher.read_header(path)
    payload = path.read_bytes()[offset:]
    header["model.language_model.unrelated_fp8_scale"] = {
        "dtype": "F8_E4M3",
        "shape": [1],
        "data_offsets": [len(payload), len(payload) + 1],
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + b"\x00")


def make_checkpoints(tmp_path, *, indexed_target=True, native_keys=False):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    config = copy.deepcopy(_CASE["config"])
    model = create_model(vision=True, audio=True)
    model.audio_tower.update(
        tree_map(
            lambda value: value.astype(mx.bfloat16), model.audio_tower.parameters()
        )
    )
    audio = {
        key: value
        for key, value in hf_checkpoint(model).items()
        if key.startswith("model.audio_tower.")
    }
    keys = sorted(audio)
    weight_map = {"model.language_model.unneeded": "never-download.safetensors"}
    for idx, group in enumerate((keys[::2], keys[1::2])):
        filename = f"source-{idx}.safetensors"
        mx.save_safetensors(str(source / filename), {key: audio[key] for key in group})
        add_unrelated_fp8_tensor(source / filename)
        weight_map.update({key: filename for key in group})
    write_json(source / patcher.INDEX_NAME, {"weight_map": weight_map})
    write_json(source / "config.json", config)
    write_json(
        source / "processor_config.json",
        {"feature_extractor": {"sampling_rate": 24000}},
    )

    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized") and model.quant_predicate(path, module)
        ),
    )
    weights = {
        key: value
        for key, value in tree_flatten(model.parameters())
        if not key.startswith("audio_tower.")
    }
    if not native_keys:
        weights = {
            (
                "model.language_model." + key[len("language_model.model.") :]
                if key.startswith("language_model.model.")
                else key[len("language_model.") :]
                if key.startswith("language_model.lm_head.")
                else "model." + key
            ): value
            for key, value in weights.items()
        }
    mx.save_safetensors(str(target / "model.safetensors"), weights)
    config["quantization"] = config["quantization_config"] = {
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
    }
    # Audio config is sometimes removed by conversions that drop its weights.
    config.pop("audio_config")
    write_json(target / "config.json", config)
    write_json(
        target / "processor_config.json",
        {"image_processor": {"max_image_tokens": 16}, "custom": True},
    )
    if indexed_target:
        write_json(
            target / patcher.INDEX_NAME,
            {
                "metadata": {
                    "total_size": sum(value.nbytes for value in weights.values())
                },
                "weight_map": {key: "model.safetensors" for key in weights},
            },
        )
    return source, target, model, config


@pytest.mark.parametrize("indexed_target,native_keys", [(True, False), (False, True)])
def test_patch_preserves_backbone_and_strictly_loads(
    tmp_path, indexed_target, native_keys
):
    source, target, original, config = make_checkpoints(
        tmp_path, indexed_target=indexed_target, native_keys=native_keys
    )
    before = checksum(target / "model.safetensors")
    base_header, _ = patcher.read_header(target / "model.safetensors")
    base_size = sum(
        info["data_offsets"][1] - info["data_offsets"][0]
        for name, info in base_header.items()
        if name != "__metadata__"
    )
    original_metadata = {
        name: (target / name).read_bytes()
        for name in ("config.json", "processor_config.json")
    }
    result = patcher.patch_audio(str(target), str(source))
    assert checksum(target / "model.safetensors") == before
    assert result["audio_tensors"] == len(
        tree_flatten(original.audio_tower.parameters())
    )
    for name, contents in original_metadata.items():
        assert (Path(result["backup"]) / name).read_bytes() == contents
    patched_config = json.loads((target / "config.json").read_text())
    assert patched_config["quantization"] == config["quantization"]
    assert patched_config["quantization_config"] == config["quantization_config"]
    processor = json.loads((target / "processor_config.json").read_text())
    assert processor["image_processor"] == {"max_image_tokens": 16}
    assert processor["custom"] is True
    assert processor["feature_extractor"]["samples_per_token"] == 4
    loaded = load_model(target)
    waveform = mx.random.normal((2, 16))
    assert mx.allclose(
        loaded.encode_audio(waveform), original.encode_audio(waveform), atol=1e-5
    )
    for name, weight in tree_flatten(original.audio_tower.parameters()):
        assert mx.array_equal(
            dict(tree_flatten(loaded.audio_tower.parameters()))[name], weight
        )
    for idx in (1, 2):
        assert loaded.layers[idx].mlp.experts.gate_proj.bits == 4
        assert loaded.layers[idx].mlp.experts.gate_proj.mode == "affine"
    index = json.loads((target / patcher.INDEX_NAME).read_text())
    assert index["metadata"]["total_size"] == base_size + result["audio_bytes"]
    with pytest.raises(ValueError, match="already contains"):
        patcher.patch_audio(str(target), str(source))


def test_remote_source_downloads_only_required_shards_at_pinned_revision(
    tmp_path, monkeypatch
):
    source, target, _, _ = make_checkpoints(tmp_path)
    calls = []
    monkeypatch.setattr(
        patcher,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda repo, revision: SimpleNamespace(sha="pinned-commit")
        ),
    )

    def download(repo, filename, revision):
        assert repo == "meta-models/source"
        assert revision == "pinned-commit"
        calls.append(filename)
        return str(source / filename)

    monkeypatch.setattr(patcher, "hf_hub_download", download)
    patcher.patch_audio(str(target), "meta-models/source", revision="requested-tag")
    assert set(calls) == {
        "config.json",
        "processor_config.json",
        patcher.INDEX_NAME,
        "source-0.safetensors",
        "source-1.safetensors",
    }


@pytest.mark.parametrize(
    "failure", ["missing", "wrong_shape", "wrong_width", "wrong_token"]
)
def test_invalid_source_does_not_modify_target(tmp_path, failure):
    source, target, _, _ = make_checkpoints(tmp_path)
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    if failure == "missing":
        path = source / patcher.INDEX_NAME
        index = json.loads(path.read_text())
        index["weight_map"].pop(
            next(key for key in index["weight_map"] if "audio_tower" in key)
        )
        write_json(path, index)
    elif failure == "wrong_shape":
        # The index still advertises a complete tower, but one tensor is malformed.
        path = source / "source-0.safetensors"
        header, start = patcher.read_header(path)
        key = next(
            key
            for key in header
            if "audio_tower" in key and len(header[key]["shape"]) > 1
        )
        header[key]["shape"] = [1, math.prod(header[key]["shape"])]
        payload = path.read_bytes()[start:]
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    else:
        path = source / "config.json"
        config = json.loads(path.read_text())
        if failure == "wrong_width":
            config["audio_config"]["out_hidden_size"] = 123
        else:
            config["audio_token_id"] = 123
        write_json(path, config)
    with pytest.raises(ValueError):
        patcher.patch_audio(str(target), str(source))
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def test_install_failure_rolls_back_metadata_and_audio(tmp_path, monkeypatch):
    source, target, _, _ = make_checkpoints(tmp_path)
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    replace = patcher.os.replace

    def fail_index(src, dst):
        if Path(dst) == target / patcher.INDEX_NAME:
            raise OSError("simulated index installation failure")
        return replace(src, dst)

    monkeypatch.setattr(patcher.os, "replace", fail_index)
    with pytest.raises(OSError, match="simulated"):
        patcher.patch_audio(str(target), str(source))
    assert {
        path.name: path.read_bytes() for path in target.iterdir() if path.is_file()
    } == before
    assert len(list(target.glob("audio-patch-backup-*"))) == 1
