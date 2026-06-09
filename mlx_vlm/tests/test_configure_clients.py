from types import SimpleNamespace

import pytest

from mlx_vlm.configure_clients import (
    opencode_config,
    patch_hermes_config_text,
    pi_config,
    resolve_default_model,
)
from mlx_vlm.model_catalog import local_model_infos


def _repo(repo_id, file_names, *, repo_type="model", refs=None):
    if refs is None:
        refs = {
            "main": SimpleNamespace(
                files=[
                    SimpleNamespace(file_path=SimpleNamespace(name=file_name))
                    for file_name in file_names
                ]
            )
        }
    return SimpleNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        refs=refs,
        last_modified=123.0,
    )


def test_local_model_infos_filters_supported_cache_repos():
    cache = SimpleNamespace(
        repos=[
            _repo(
                "local/single",
                ["config.json", "tokenizer_config.json", "model.safetensors"],
            ),
            _repo(
                "local/sharded",
                [
                    "config.json",
                    "tokenizer_config.json",
                    "model.safetensors.index.json",
                ],
            ),
            _repo("local/no-weights", ["config.json", "tokenizer_config.json"]),
            _repo("dataset/nope", ["config.json"], repo_type="dataset"),
            _repo(
                "local/no-main",
                ["config.json", "tokenizer_config.json", "model.safetensors"],
                refs={},
            ),
        ]
    )

    models = local_model_infos(lambda: cache, sort=True)

    assert [model["id"] for model in models] == ["local/sharded", "local/single"]


def test_pi_config_replaces_only_target_provider():
    existing = {"providers": {"other": {"baseUrl": "https://example.test/v1"}}}

    config = pi_config(
        existing,
        model_ids=["org/model-a"],
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        provider_id="mlx-vlm",
        provider_name="MLX-VLM Local",
        context_window=4096,
        max_tokens=512,
    )

    assert "other" in config["providers"]
    provider = config["providers"]["mlx-vlm"]
    assert provider["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert provider["api"] == "openai-completions"
    assert provider["compat"]["maxTokensField"] == "max_tokens"
    assert provider["models"] == [
        {
            "id": "org/model-a",
            "name": "model a (MLX-VLM Local)",
            "contextWindow": 4096,
            "maxTokens": 512,
        }
    ]


def test_opencode_config_replaces_only_target_provider():
    existing = {"provider": {"other": {"name": "Other"}}}

    config = opencode_config(
        existing,
        model_ids=["org/model-a"],
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        provider_id="mlx-vlm",
        provider_name="MLX-VLM Local",
    )

    assert config["$schema"] == "https://opencode.ai/config.json"
    assert "other" in config["provider"]
    provider = config["provider"]["mlx-vlm"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"] == {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "not-needed",
    }
    assert provider["models"] == {"org/model-a": {"name": "model a"}}


def test_patch_hermes_config_text_updates_model_and_provider_idempotently():
    original = """model:
  default: old/model
  provider: custom
  base_url: http://old.test/v1
agent:
  max_turns: 10
custom_providers:
- name: "Other"
  base_url: "https://other.test/v1"
- name: "MLX-VLM Local"
  base_url: "http://old.test/v1"
  model: "old/model"
"""

    patched = patch_hermes_config_text(
        original,
        model_ids=["org/model-a", "org/model-b"],
        default_model="org/model-b",
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        provider_name="MLX-VLM Local",
        context_window=4096,
    )
    patched_again = patch_hermes_config_text(
        patched,
        model_ids=["org/model-a", "org/model-b"],
        default_model="org/model-b",
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",
        provider_name="MLX-VLM Local",
        context_window=4096,
    )

    assert patched == patched_again
    assert 'default: "org/model-b"' in patched
    assert 'base_url: "http://127.0.0.1:8080/v1"' in patched
    assert 'api_key: "not-needed"' in patched
    assert "agent:\n  max_turns: 10\n" in patched
    assert patched.count('name: "MLX-VLM Local"') == 1
    assert 'name: "Other"' in patched
    assert '    "org/model-a":\n      context_length: 4096\n' in patched


def test_resolve_default_model_preserves_existing_hermes_default(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        'model:\n  default: "Org/Model-A"\n  provider: custom\n',
        encoding="utf-8",
    )

    assert resolve_default_model(["org/model-a", "org/model-b"], None, config) == (
        "org/model-a"
    )


def test_resolve_default_model_rejects_missing_explicit_default(tmp_path):
    config = tmp_path / "config.yaml"

    with pytest.raises(ValueError, match="not available"):
        resolve_default_model(["org/model-a"], "missing/model", config)
