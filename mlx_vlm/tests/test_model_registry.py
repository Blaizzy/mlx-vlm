import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_vlm.model_registry import (
    MODEL_ALIASES_ENV,
    MODEL_PATHS_ENV,
    AmbiguousModelIdentifier,
    InvalidModelRegistryConfiguration,
    LocalModelNotFound,
    ModelRegistry,
    ModelResolution,
    encode_model_aliases,
    parse_model_alias,
    parse_model_aliases,
    parse_model_paths,
    private_model_id,
    public_model_id,
)


def _model(path: Path, *, nested_weights: bool = False) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type":"test"}')
    weight_root = path / "weights" if nested_weights else path
    weight_root.mkdir(exist_ok=True)
    (weight_root / "model.safetensors").write_bytes(b"weights")
    return path.resolve()


def _hf_repo(repo_id: str, snapshot_path: Path, file_names=None):
    files = file_names or [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
    ]
    return SimpleNamespace(
        repo_id=repo_id,
        repo_type="model",
        last_modified=123.0,
        refs={
            "main": SimpleNamespace(
                snapshot_path=snapshot_path,
                files=[
                    SimpleNamespace(file_path=SimpleNamespace(name=name))
                    for name in files
                ],
            )
        },
    )


def test_parse_model_paths_expands_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    paths = parse_model_paths(
        os.pathsep.join(["~/models", str(tmp_path / "models"), "", "~/other"])
    )

    assert paths == [(tmp_path / "models").resolve(), (tmp_path / "other").resolve()]


def test_alias_environment_is_json_and_preserves_path_characters(tmp_path):
    model_path = tmp_path / "models,with=punctuation" / "demo"
    encoded = encode_model_aliases({"demo": model_path})

    assert parse_model_aliases(encoded) == {"demo": model_path.resolve()}
    assert json.loads(encoded) == {"demo": str(model_path)}


@pytest.mark.parametrize("value", ["[]", '"alias=/model"', "{bad json"])
def test_alias_environment_rejects_invalid_json(value):
    with pytest.raises(InvalidModelRegistryConfiguration):
        parse_model_aliases(value)


def test_alias_environment_rejects_duplicate_normalized_ids(tmp_path):
    value = json.dumps(
        {
            "demo": str(tmp_path / "first"),
            " demo ": str(tmp_path / "second"),
        }
    )

    with pytest.raises(InvalidModelRegistryConfiguration, match="more than once"):
        parse_model_aliases(value)


@pytest.mark.parametrize("model_id", ["", "/private/model", "../model", "a//b", r"a\b"])
def test_aliases_require_safe_public_identifiers(tmp_path, model_id):
    with pytest.raises(InvalidModelRegistryConfiguration) as error:
        parse_model_alias(f"{model_id}={tmp_path}")
    assert str(tmp_path) not in str(error.value)


def test_registry_discovers_root_child_and_nested_models(tmp_path):
    root_model = _model(tmp_path / "root-model")
    models_root = tmp_path / "models"
    child_model = _model(models_root / "child")
    nested_model = _model(models_root / "org" / "nested", nested_weights=True)

    root_entry = ModelRegistry([root_model], include_hf_cache=False).entries()
    nested_entries = ModelRegistry([models_root], include_hf_cache=False).entries()

    assert [(entry.id, entry.path) for entry in root_entry] == [
        ("local/root-model", root_model)
    ]
    assert [(entry.id, entry.path) for entry in nested_entries] == [
        ("local/child", child_model),
        ("local/org/nested", nested_model),
    ]


def test_registry_ignores_hidden_incomplete_and_overly_deep_directories(tmp_path):
    root = tmp_path / "models"
    _model(root / ".hidden")
    incomplete = root / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}")
    _model(root / "one" / "two" / "too-deep")

    assert ModelRegistry([root], include_hf_cache=False).entries() == []


def test_registry_resolves_symlinks_once_and_avoids_cycles(tmp_path):
    root = tmp_path / "models"
    external = _model(tmp_path / "external" / "demo")
    root.mkdir()
    (root / "demo").symlink_to(external, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)

    registry = ModelRegistry([root], include_hf_cache=False)

    assert [(entry.id, entry.path) for entry in registry.entries()] == [
        ("local/demo", external)
    ]
    assert registry.resolve("local/demo").path == external


def test_registry_refreshes_discovery_without_restarting(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    registry = ModelRegistry([root], include_hf_cache=False)

    assert registry.entries() == []
    model_path = _model(root / "new-model")
    assert registry.resolve("local/new-model").path == model_path
    for child in model_path.iterdir():
        child.unlink()
    model_path.rmdir()
    assert registry.entries() == []


def test_registry_rejects_ambiguous_derived_identifiers(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _model(first / "org" / "model")
    _model(second / "org" / "model")

    registry = ModelRegistry([first, second], include_hf_cache=False)

    with pytest.raises(AmbiguousModelIdentifier, match="configure an explicit alias"):
        registry.entries()


def test_explicit_aliases_resolve_discovery_collisions(tmp_path):
    first = _model(tmp_path / "first" / "org" / "model")
    second = _model(tmp_path / "second" / "org" / "model")
    registry = ModelRegistry(
        [tmp_path / "first", tmp_path / "second"],
        aliases={"first-model": first, "second-model": second},
        include_hf_cache=False,
    )

    assert [entry.id for entry in registry.entries()] == [
        "first-model",
        "second-model",
    ]
    assert registry.resolve("first-model").path == first
    assert registry.resolve("second-model").path == second


def test_alias_replaces_discovered_id_as_canonical_identity(tmp_path):
    model_path = _model(tmp_path / "models" / "org" / "model")
    registry = ModelRegistry(
        [tmp_path / "models"],
        aliases={"friendly": model_path},
        include_hf_cache=False,
    )

    assert [entry.id for entry in registry.entries()] == ["friendly"]
    assert registry.resolve(str(model_path)).id == "friendly"
    assert registry.resolve("friendly").path == model_path
    assert registry.resolve("local/org/model").path is None


def test_alias_rejects_missing_or_invalid_model_directories(tmp_path):
    invalid = tmp_path / "invalid"
    invalid.mkdir()

    with pytest.raises(InvalidModelRegistryConfiguration, match="valid model"):
        ModelRegistry(aliases={"invalid": invalid}, include_hf_cache=False).entries()
    with pytest.raises(InvalidModelRegistryConfiguration, match="missing directory"):
        ModelRegistry(
            aliases={"missing": tmp_path / "missing"}, include_hf_cache=False
        ).entries()


def test_hugging_face_models_keep_repository_identity(tmp_path):
    snapshot = _model(tmp_path / "snapshot")
    cache = SimpleNamespace(repos=[_hf_repo("org/model", snapshot)])
    registry = ModelRegistry(cache_scanner=lambda: cache)

    assert [(entry.id, entry.path) for entry in registry.entries()] == [
        ("org/model", snapshot)
    ]
    assert registry.resolve("org/model") == ModelResolution(
        id="org/model", load_target="org/model", path=None
    )


def test_hugging_face_scan_excludes_incomplete_models(tmp_path):
    snapshot = _model(tmp_path / "snapshot")
    cache = SimpleNamespace(
        repos=[
            _hf_repo(
                "missing/config",
                snapshot,
                ["model.safetensors", "tokenizer_config.json"],
            ),
            _hf_repo(
                "missing/weights",
                snapshot,
                ["config.json", "tokenizer_config.json"],
            ),
            _hf_repo(
                "missing/tokenizer",
                snapshot,
                ["config.json", "model.safetensors"],
            ),
        ]
    )

    assert ModelRegistry(cache_scanner=lambda: cache).entries() == []


def test_same_model_in_hf_cache_and_search_path_is_deduplicated(tmp_path):
    snapshot = _model(tmp_path / "snapshot")
    cache = SimpleNamespace(repos=[_hf_repo("org/model", snapshot)])
    registry = ModelRegistry([snapshot], cache_scanner=lambda: cache)

    assert [entry.id for entry in registry.entries()] == ["org/model"]
    assert registry.resolve("local/snapshot").id == "org/model"


def test_exact_unregistered_path_gets_private_stable_identifier(tmp_path):
    model_path = _model(tmp_path / "private" / "model with spaces")
    registry = ModelRegistry(include_hf_cache=False)

    first = registry.resolve(str(model_path))
    second = registry.resolve(str(model_path))

    assert first.id == second.id == private_model_id(model_path)
    assert first.id.startswith("local/model-with-spaces-")
    assert str(tmp_path) not in first.id
    assert first.path == model_path


def test_unknown_repository_id_remains_available_for_hub_download():
    resolution = ModelRegistry(cache_scanner=pytest.fail).resolve("org/remote-model")

    assert resolution.id == "org/remote-model"
    assert resolution.load_target == "org/remote-model"
    assert resolution.path is None


def test_local_hugging_face_namespace_remains_available():
    resolution = ModelRegistry(include_hf_cache=False).resolve("local/remote-model")

    assert resolution.load_target == "local/remote-model"
    assert resolution.path is None


def test_missing_explicit_local_path_does_not_fall_through_to_hub(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(LocalModelNotFound, match="does not exist"):
        ModelRegistry(include_hf_cache=False).resolve(str(missing))


def test_public_model_id_hides_missing_explicit_local_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    model_id = public_model_id("./missing")

    assert model_id.startswith("local/missing-")
    assert str(tmp_path) not in model_id


def test_environment_constructs_registry(tmp_path, monkeypatch):
    root = tmp_path / "models"
    model_path = _model(root / "model")
    monkeypatch.setenv(MODEL_PATHS_ENV, str(root))
    monkeypatch.setenv(MODEL_ALIASES_ENV, encode_model_aliases({"demo": model_path}))

    registry = ModelRegistry.from_environment(include_hf_cache=False)

    assert [entry.id for entry in registry.entries()] == ["demo"]
    assert registry.resolve("demo").path == model_path


def test_get_model_path_resolves_registry_before_hub_download(tmp_path, monkeypatch):
    from mlx_vlm import utils

    root = tmp_path / "models"
    model_path = _model(root / "demo")
    monkeypatch.setenv(MODEL_PATHS_ENV, str(root))
    snapshot_download = pytest.fail
    monkeypatch.setattr(utils, "snapshot_download", snapshot_download)

    assert utils.get_model_path("local/demo") == model_path


def test_get_model_path_preserves_exact_local_path(tmp_path, monkeypatch):
    from mlx_vlm import utils

    model_path = _model(tmp_path / "exact-model")
    monkeypatch.setattr(utils, "snapshot_download", pytest.fail)

    assert utils.get_model_path(str(model_path)) == model_path


def test_get_model_path_preserves_hub_fallback(tmp_path, monkeypatch):
    from mlx_vlm import utils

    downloaded = tmp_path / "downloaded"
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return downloaded

    monkeypatch.delenv(MODEL_PATHS_ENV, raising=False)
    monkeypatch.delenv(MODEL_ALIASES_ENV, raising=False)
    monkeypatch.setattr(utils, "snapshot_download", snapshot_download)

    assert utils.get_model_path("org/remote", revision="v1") == downloaded
    assert calls[0]["repo_id"] == "org/remote"
    assert calls[0]["revision"] == "v1"
