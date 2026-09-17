import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import scan_cache_dir

from mlx_vlm.model_discovery import discover_models, is_model_directory


def model_directory(path, config=None):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config or {"model_type": "qwen2_vl"}))
    (path / "model.safetensors").write_bytes(b"test weights")
    return path


def test_metadata_without_tokenizer_is_discoverable(tmp_path):
    model = model_directory(tmp_path / "vision-model")
    assert is_model_directory(model)
    (model / "config.json").write_text("not json")
    assert not is_model_directory(model)
    (model / "config.json").write_text("{}")
    assert not is_model_directory(model)


def test_sharded_model_requires_every_indexed_weight_file(tmp_path):
    model = model_directory(tmp_path / "sharded")
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "model.safetensors", "b": "second.safetensors"}}
        )
    )
    assert not is_model_directory(model)
    (model / "second.safetensors").write_bytes(b"second shard")
    assert is_model_directory(model)
    (model / "second.safetensors").write_bytes(b"")
    assert not is_model_directory(model)


@pytest.mark.parametrize(
    "weight_map", [{}, [], {"a": "../outside.safetensors"}, {"a": 1}]
)
def test_bad_indexes_do_not_advertise_partial_models(tmp_path, weight_map):
    model = model_directory(tmp_path / "partial")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    assert not is_model_directory(model)


def test_adapters_and_broken_weight_links_are_not_models(tmp_path):
    model = model_directory(tmp_path / "adapter")
    (model / "model.safetensors").rename(model / "adapter_model.safetensors")
    assert not is_model_directory(model)
    (model / "model.safetensors").symlink_to(model / "missing.safetensors")
    assert not is_model_directory(model)


def test_pipeline_components_use_the_same_weight_checks(tmp_path):
    pipeline = tmp_path / "pipeline"
    model_directory(pipeline / "transformer", {"_class_name": "FluxTransformer2DModel"})
    (pipeline / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"})
    )
    assert is_model_directory(pipeline)
    (pipeline / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}})
    )
    assert not is_model_directory(pipeline)


def test_real_hf_cache_handles_non_main_revisions_and_prefers_main(tmp_path):
    repo = tmp_path / "models--local--vision"
    snapshot = model_directory(repo / "snapshots" / ("a" * 40))
    cache = scan_cache_dir(tmp_path)
    candidates = discover_models(cache)
    assert [m["id"] for m in candidates] == [str(snapshot)]
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    assert [m["id"] for m in discover_models(scan_cache_dir(tmp_path))] == [
        "local/vision"
    ]


def test_custom_roots_direct_paths_and_symlinks_are_deduplicated(tmp_path):
    root = tmp_path / "models"
    model = model_directory(root / "custom")
    alias = root / "alias"
    alias.symlink_to(model, target_is_directory=True)
    (root / "unrelated").mkdir()
    candidates = discover_models(
        SimpleNamespace(repos=[]),
        [
            "~/" + os.path.relpath(root, Path.home()),
            str(model),
            str(alias),
            str(root / "missing"),
        ],
    )
    assert [m["id"] for m in candidates] == [str(model)]
    assert candidates[0]["path"] == model


def test_custom_model_also_in_hf_cache_has_one_entry(tmp_path):
    model = model_directory(
        tmp_path / "models--local--vision" / "snapshots" / ("a" * 40)
    )
    candidates = discover_models(scan_cache_dir(tmp_path), [str(model)])
    assert len(candidates) == 1


def test_incomplete_main_uses_an_available_snapshot_path(tmp_path):
    repo = tmp_path / "models--local--vision"
    incomplete = model_directory(repo / "snapshots" / ("a" * 40))
    (incomplete / "model.safetensors").unlink()
    complete = model_directory(repo / "snapshots" / ("b" * 40))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    assert [m["id"] for m in discover_models(scan_cache_dir(tmp_path))] == [
        str(complete)
    ]


def test_discovery_never_executes_checkpoint_code(tmp_path):
    model = model_directory(
        tmp_path / "custom", {"model_type": "custom", "model_file": "model.py"}
    )
    (model / "model.py").write_text("raise RuntimeError('must not execute')")
    assert is_model_directory(model)


def test_cache_script_shares_custom_path_filter_when_cache_is_missing(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "skills/skills/hf-cache-models/scripts/list_supported_hf_cache_models.py"
    )
    spec = importlib.util.spec_from_file_location("cache_model_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = model_directory(tmp_path / "local")
    found = module.supported_models(
        str(tmp_path / "missing-cache"), model_dirs=[str(model)]
    )
    assert [m["id"] for m in found] == [str(model)]
    (model / "model.safetensors").unlink()
    assert (
        module.supported_models(
            str(tmp_path / "missing-cache"), model_dirs=[str(model)]
        )
        == []
    )
