"""Live APC settings: endpoint validation, reloads, and manager configuration."""

import os
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
from mlx_vlm import apc
from mlx_vlm.models.cache import KVCache
from mlx_vlm.server.runtime_config import RuntimeConfig


@pytest.fixture
def settings_client(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("APC_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
    monkeypatch.setattr(server.runtime, "model_cache", server.ModelCacheRegistry())
    monkeypatch.setattr(server.runtime, "response_generator", None)
    monkeypatch.setattr(server.runtime, "apc_manager", None)
    monkeypatch.setattr(
        server._app_module, "is_image_generation_model", lambda _: False
    )

    # Exercise the real cache factory / APC manager while avoiding model downloads.
    generators = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(make_cache=lambda: [KVCache()])
            self.processor = SimpleNamespace()
            self.config = SimpleNamespace(model_type="test")
            self.apc_manager = kwargs["apc_manager"]
            self.stopped = False
            generators.append(self)

        def wait_until_ready(self):
            return self.model, self.processor, self.config

        def stop_and_join(self):
            self.stopped = True
            if self.apc_manager is not None:
                self.apc_manager.close()

    monkeypatch.setattr(server._app_module, "ResponseGenerator", FakeGenerator)
    client = TestClient(server.app)
    yield client, generators
    for generator in generators:
        if not generator.stopped:
            generator.stop_and_join()
    client.close()


def test_apc_patch_reaches_next_model_request(settings_client, tmp_path):
    client, generators = settings_client
    server.get_cached_model("demo")
    assert server.runtime.apc_manager is None
    settings = {
        "apc_enabled": True,
        "apc_disk_enabled": True,
        "apc_disk_path": str(tmp_path / "live"),
        "apc_block_size": 32,
        "apc_num_blocks": 8,
        "apc_disk_max_gb": 0,
        "apc_memory_max_gb": 1,
        "apc_memory_reserve_gb": 0.25,
        "apc_disk_queue_max_gb": 0,
        "apc_disk_shard_max_blocks": 64,
        "apc_checkpoint_entries": 3,
        "apc_checkpoint_interval_tokens": 128,
        "apc_checkpoint_guard_tokens": 2,
    }
    response = client.patch("/v1/settings", json=settings)
    assert response.status_code == 200
    assert response.json()["applied"] == settings
    assert response.json()["rejected"] == []
    assert response.json()["reload_kinds"] == ["text_generation"]
    assert len(generators) == 1  # Configuration changes apply at the next load.

    body = client.get("/v1/settings").json()
    assert settings.keys() <= {knob["name"] for knob in body["schema"]}
    assert {name: body["current"][name] for name in settings} == settings
    server.get_cached_model("demo")
    assert len(generators) == 2 and generators[0].stopped
    manager = server.runtime.apc_manager
    assert manager is generators[-1].apc_manager
    assert (manager.block_size, manager.num_blocks) == (32, 8)
    assert manager.memory_max_bytes == 1 << 30
    assert manager.memory_reserve_bytes == 1 << 28
    assert manager._exact_cache_max == 3
    assert manager.checkpoint_interval_tokens == 128
    assert manager.exact_cache_guard_tokens == 2
    assert manager.disk.dir.parent == tmp_path / "live"
    assert manager.disk.max_bytes is None
    assert manager.disk.queue_max_bytes == 0
    assert manager.disk._shard_max_blocks == 64
    assert client.get("/v1/cache/stats").json()["memory_max_bytes"] == 1 << 30

    response = client.patch("/v1/settings", json=settings)
    assert response.json()["reload_kinds"] == []  # A no-op retains the model/cache.
    server.get_cached_model("demo")
    assert len(generators) == 2

    client.patch("/v1/settings", json={"apc_memory_max_gb": 0})
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.memory_max_bytes == 0
    assert all(not thread.is_alive() for thread in manager.disk._workers)

    client.patch("/v1/settings", json={"apc_enabled": False})
    server.get_cached_model("demo")
    assert server.runtime.apc_manager is None
    assert client.get("/v1/cache/stats").json() == {"enabled": False}


def test_apc_empty_path_and_null_have_distinct_fingerprints(settings_client):
    client, _ = settings_client
    client.patch("/v1/settings", json={"apc_enabled": True})
    original = client.get("/v1/settings").json()["fingerprint"]
    response = client.patch("/v1/settings", json={"apc_disk_path": ""})
    assert response.json()["fingerprint"] != original
    assert response.json()["current"]["apc_disk_path"] == ""
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.disk is None

    response = client.patch("/v1/settings", json={"apc_disk_path": None})
    assert response.json()["fingerprint"] == original
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.disk is not None

    client.patch("/v1/settings", json={"apc_disk_enabled": False})
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.disk is None


def test_apc_zero_null_and_replace_override_environment(
    settings_client, monkeypatch, tmp_path
):
    client, _ = settings_client
    monkeypatch.setattr(apc, "_metal_working_set_bytes", lambda: 40 << 30)
    monkeypatch.setenv("APC_ENABLED", "1")
    monkeypatch.setenv("APC_DISK_ENABLED", "0")
    monkeypatch.setenv("APC_DISK_PATH", str(tmp_path / "environment"))
    for name in (
        "APC_DISK_MAX_GB",
        "APC_MEMORY_MAX_GB",
        "APC_MEMORY_RESERVE_GB",
        "APC_DISK_QUEUE_MAX_GB",
    ):
        monkeypatch.setenv(name, "2")
    monkeypatch.setattr(server.runtime, "config", RuntimeConfig.from_env())
    before_env = dict(os.environ)
    zero_values = {
        "apc_memory_max_gb": 0,
        "apc_memory_reserve_gb": 0,
        "apc_disk_max_gb": 0,
        "apc_disk_queue_max_gb": 0,
    }
    body = client.patch(
        "/v1/settings", json={**zero_values, "apc_disk_enabled": True}
    ).json()
    assert all(body["current"][name] == 0 for name in zero_values)
    server.get_cached_model("demo")
    manager = server.runtime.apc_manager
    assert manager.memory_max_bytes == manager.memory_reserve_bytes == 0
    assert manager.disk.max_bytes is None
    assert manager.disk.queue_max_bytes == 0

    client.patch(
        "/v1/settings",
        json={**{name: None for name in zero_values}, "apc_disk_path": None},
    )
    server.get_cached_model("demo")
    manager = server.runtime.apc_manager
    assert manager.memory_max_bytes == manager.memory_reserve_bytes == 4 << 30
    assert manager.disk.max_bytes == 20 << 30
    assert manager.disk.queue_max_bytes == 1 << 30
    assert manager.disk.dir.parent == apc.default_disk_path()

    body = client.patch("/v1/settings", json={"op": "replace", "values": {}}).json()
    assert body["reload_kinds"] == ["text_generation"]
    server.get_cached_model("demo")
    assert server.runtime.apc_manager.memory_max_bytes == 2 << 30
    assert server.runtime.apc_manager.disk is None
    assert dict(os.environ) == before_env


def test_apc_environment_preserves_zero_and_legacy_checkpoint_values(monkeypatch):
    for name in (
        "APC_DISK_MAX_GB",
        "APC_MEMORY_MAX_GB",
        "APC_MEMORY_RESERVE_GB",
        "APC_DISK_QUEUE_MAX_GB",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("APC_DISK_PATH", "")
    monkeypatch.delenv("APC_CHECKPOINT_ENTRIES", raising=False)
    monkeypatch.delenv("APC_CHECKPOINT_GUARD_TOKENS", raising=False)
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    monkeypatch.setenv("APC_EXACT_PREFIX_GUARD_TOKENS", "2")
    values = RuntimeConfig.from_env().apc_overrides()
    assert (
        values["disk_max_gb"]
        == values["memory_max_gb"]
        == values["memory_reserve_gb"]
        == values["disk_queue_max_gb"]
        == 0
    )
    assert values["disk_path"] == ""
    assert values["checkpoint_entries"] == 0
    assert values["checkpoint_guard_tokens"] == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("apc_block_size", 0),
        ("apc_num_blocks", -1),
        ("apc_checkpoint_entries", -1),
        ("apc_checkpoint_entries", 1.5),
        ("apc_checkpoint_guard_tokens", 0),
        ("apc_checkpoint_interval_tokens", -1),
        ("apc_disk_shard_max_blocks", 0),
        ("apc_memory_max_gb", -1),
        ("apc_memory_max_gb", "NaN"),
        ("apc_memory_reserve_gb", "inf"),
        ("apc_disk_queue_max_gb", 1e308),
        ("apc_disk_max_gb", -1),
        ("apc_num_blocks", True),
        ("apc_disk_path", []),
        ("apc_disk_enabled", None),
    ],
)
def test_invalid_apc_values_are_rejected_without_reload(settings_client, name, value):
    client, _ = settings_client
    client.patch("/v1/settings", json={"apc_enabled": True})
    before = client.get("/v1/settings").json()
    response = client.patch("/v1/settings", json={name: value})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == {} and body["rejected"][0]["name"] == name
    assert body["reload_kinds"] == []
    assert body["current"] == before["current"]
    assert body["fingerprint"] == before["fingerprint"]


def test_disabled_apc_settings_are_staged_until_enabled(settings_client):
    client, _ = settings_client
    cfg = server.runtime.config
    text_before = cfg.fingerprint(kinds={"text_generation"})
    image_before = cfg.fingerprint(kinds={"image_generation"})
    body = client.patch(
        "/v1/settings",
        json={
            "apc_checkpoint_entries": "0",
            "apc_checkpoint_interval_tokens": 0,
            "apc_num_blocks": 0,
            "apc_memory_max_gb": "0.5",
        },
    ).json()
    assert body["rejected"] == [] and body["reload_kinds"] == []
    assert cfg.fingerprint(kinds={"text_generation"}) == text_before
    client.patch("/v1/settings", json={"apc_enabled": True})
    assert cfg.fingerprint(kinds={"image_generation"}) == image_before
    server.get_cached_model("demo")
    manager = server.runtime.apc_manager
    assert manager.num_blocks == 0
    assert manager._exact_cache_max == manager.checkpoint_interval_tokens == 0
    assert manager.memory_max_bytes == 1 << 29


def test_apc_settings_require_management_key(settings_client, monkeypatch):
    client, _ = settings_client
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "secret")
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        assert client.get("/v1/settings", headers=headers).status_code == 401
        assert (
            client.patch(
                "/v1/settings", headers=headers, json={"apc_enabled": True}
            ).status_code
            == 401
        )
    assert server.runtime.config.apc_enabled is False
    headers = {"Authorization": "Bearer secret"}
    assert (
        client.patch(
            "/v1/settings", headers=headers, json={"apc_enabled": True}
        ).status_code
        == 200
    )
    assert (
        client.get("/v1/settings", headers=headers).json()["current"]["apc_enabled"]
        is True
    )


@pytest.mark.parametrize("fail", [False, True])
def test_generation_worker_closes_disk_after_final_store(tmp_path, fail):
    disk = apc.DiskBlockStore(tmp_path)
    manager = apc.APCManager(num_blocks=1, disk=disk)
    generator = server.ResponseGenerator.__new__(server.ResponseGenerator)
    generator.apc_manager = manager

    def finish_request():
        cache = KVCache()
        cache.keys = cache.values = mx.ones((1, 1, 32, 4))
        cache.offset = 32
        manager.store_exact_cache(list(range(32)), [cache])
        if fail:
            raise RuntimeError("worker failed")

    generator._run_impl = MagicMock(side_effect=finish_request)
    errors = []

    def run_worker():
        try:
            generator._run()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run_worker, daemon=True)
    try:
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        if fail:
            assert len(errors) == 1 and str(errors[0]) == "worker failed"
        else:
            assert errors == []
        assert disk.num_exact_indexed == 1
        assert disk.pending_bytes == 0
        assert all(not thread.is_alive() for thread in disk._workers)
    finally:
        if any(thread.is_alive() for thread in disk._workers):
            manager.close()
