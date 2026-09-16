"""TDD: APC_TRACE + layout self-check (observability hygiene)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.apc import (
    APCManager,
    APCSelfCheckResult,
    DiskBlockStore,
    apc_trace,
    apc_trace_enabled,
    classify_layer_for_apc,
    self_check_model_apc,
    validate_prompt_cache_layout,
)
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    KVCache,
    QuantizedKVCache,
)

BLOCK_SIZE = 16
GROUP_SIZE = 64
BITS = 8


@pytest.mark.parametrize("mode", ["block", "exact"])
@pytest.mark.parametrize("tier", ["memory", "disk"])
@pytest.mark.parametrize("restore", ["single", "batch", "failed-batch"])
def test_served_tokens_count_successful_restores(
    mode, tier, restore, tmp_path, monkeypatch
):
    from mlx_vlm.tests.test_apc import _make_exact_row_cache

    def make_manager():
        disk = DiskBlockStore(tmp_path, namespace="stats") if tier == "disk" else None
        return APCManager(num_blocks=8, block_size=BLOCK_SIZE, disk=disk)

    def make_cache():
        return [ArraysCache(2), KVCache()] if mode == "exact" else [KVCache()]

    tokens = list(range(32))
    row = _make_exact_row_cache(len(tokens))
    manager = make_manager()
    try:
        if mode == "exact":
            assert manager.store_exact_cache(tokens, row)
        else:
            blocks = manager.store_kv_blocks(tokens, [row[1].keys], [row[1].values])
            manager.release(blocks)
        stats = manager.stats_snapshot()
        assert stats["served_tokens"] == 0
        assert stats["stored_tokens"] == (32 if mode == "block" else 0)
        if tier == "disk":
            manager.close()
            manager = make_manager()

        coordinator = manager.coordinator(SimpleNamespace(make_cache=make_cache))
        hit = coordinator.lookup(
            tokens + [99],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _: True,
            prefix_has_media=lambda _: False,
        )
        assert hit is not None
        before = manager.stats_snapshot()
        assert before["matched_tokens"] > 0
        assert before["served_tokens"] == 0
        if restore == "single":
            caches = coordinator.materialize_single(hit, min_capacity_tokens=33)
        else:
            if restore == "failed-batch":
                builder = "exact" if mode == "exact" else "kv"
                monkeypatch.setattr(
                    f"mlx_vlm.apc.make_warm_batch_{builder}_cache_multi",
                    lambda *a, **kw: (None, 0),
                )
            caches, _ = coordinator.merge_rows([hit, None], [hit["prefix_len"], 0])
        coordinator.release_hit(hit)
        stats = manager.stats_snapshot()
        assert (caches is None) == (restore == "failed-batch")
        assert stats["served_tokens"] == (0 if caches is None else hit["prefix_len"])
        assert stats["token_hit_rate"] == before["token_hit_rate"]
        if tier == "disk":
            assert stats["disk_hits"] > 0
        manager.reset_stats()
        assert manager.stats_snapshot()["served_tokens"] == 0
        assert manager.stats_snapshot()["stored_tokens"] == 0
    finally:
        manager.close()


@pytest.fixture(autouse=True)
def _clear_apc_trace_env(monkeypatch):
    monkeypatch.delenv("APC_TRACE", raising=False)
    yield
    monkeypatch.delenv("APC_TRACE", raising=False)


class TestApcTrace:
    def test_disabled_by_default(self):
        assert apc_trace_enabled() is False

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("APC_TRACE", "1")
        assert apc_trace_enabled() is True
        monkeypatch.setenv("APC_TRACE", "true")
        assert apc_trace_enabled() is True
        monkeypatch.setenv("APC_TRACE", "0")
        assert apc_trace_enabled() is False

    def test_trace_emits_logger_info_when_enabled(self, monkeypatch, caplog):
        monkeypatch.setenv("APC_TRACE", "1")
        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            apc_trace("store", mode="exact", ok=True, token_len=32)
        assert any("APC_TRACE store" in r.message for r in caplog.records)
        assert any("mode=exact" in r.message for r in caplog.records)
        assert any("token_len=32" in r.message for r in caplog.records)

    def test_trace_silent_when_disabled(self, caplog):
        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            apc_trace("store", mode="exact", ok=True)
        assert not any("APC_TRACE" in r.message for r in caplog.records)

    def test_reject_records_emit_trace(self, monkeypatch, caplog):
        monkeypatch.setenv("APC_TRACE", "1")
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        token_ids = list(range(BLOCK_SIZE))

        class UnclonableCache:
            keys = "not-an-array"
            values = "not-an-array"

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            assert manager.store_exact_cache(token_ids, [UnclonableCache()]) is False
        assert any("APC_TRACE reject" in r.message for r in caplog.records)
        assert any("unclonable" in r.message for r in caplog.records)


class TestClassifyLayer:
    def test_plain_kv_ok(self):
        c = KVCache()
        c.update_and_fetch(
            mx.zeros((1, 2, 4, 8), dtype=mx.float32),
            mx.zeros((1, 2, 4, 8), dtype=mx.float32),
        )
        result = classify_layer_for_apc(c)
        assert result.status == "ok"
        assert result.type_name == "KVCache"

    def test_quantized_with_dequant_ok(self):
        # Last dim must be divisible by group_size for mx.quantize.
        c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        c.update_and_fetch(
            mx.random.normal((1, 2, 8, GROUP_SIZE)),
            mx.random.normal((1, 2, 8, GROUP_SIZE)),
        )
        result = classify_layer_for_apc(c)
        assert result.status == "ok"

    def test_batch_quantized_empty_ok(self):
        c = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        result = classify_layer_for_apc(c)
        assert result.status in ("ok", "empty_ok")

    def test_batch_rotating_empty_ok(self):
        c = BatchRotatingKVCache(32, [0])
        result = classify_layer_for_apc(c)
        assert result.status in ("ok", "empty_ok")

    def test_unsupported_opaque_type(self):
        class Bogus:
            pass

        result = classify_layer_for_apc(Bogus())
        assert result.status == "unsupported"
        assert result.reason


class TestValidateLayout:
    def test_all_supported_layout_ok(self):
        caches = [
            BatchRotatingKVCache(32, [0]),
            BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS),
            BatchKVCache([0]),
        ]
        result = validate_prompt_cache_layout(caches, apc_mode="exact")
        assert isinstance(result, APCSelfCheckResult)
        assert result.ok is True
        assert result.apc_mode == "exact"
        assert result.layer_count == 3
        assert result.unsupported == []

    def test_mixed_unsupported_not_ok(self):
        class Bogus:
            pass

        result = validate_prompt_cache_layout([KVCache(), Bogus()], apc_mode="block")
        assert result.ok is False
        assert len(result.unsupported) == 1
        assert result.unsupported[0].type_name == "Bogus"


class TestSelfCheckModel:
    def test_supported_model_ok(self, caplog):
        class FakeLang:
            def make_cache(self):
                return [
                    BatchRotatingKVCache(32, [0]),
                    BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS),
                    BatchKVCache([0]),
                ]

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            result = self_check_model_apc(FakeLang(), kv_bits=8.0)
        assert result.ok is True
        assert result.apc_mode == "exact"
        assert any("APC self-check ok" in r.message for r in caplog.records)

    def test_unsupported_model_logs_error(self, caplog):
        class FakeLang:
            def make_cache(self):
                class Bogus:
                    pass

                return [Bogus()]

        with caplog.at_level(logging.ERROR, logger="mlx_vlm.apc"):
            result = self_check_model_apc(FakeLang())
        assert result.ok is False
        assert any("APC self-check" in r.message for r in caplog.records)

    def test_no_make_cache_not_ok(self, caplog):
        class NoCache:
            pass

        result = self_check_model_apc(NoCache())
        assert result.ok is False

    def test_unwraps_language_model(self):
        class FakeLang:
            def make_cache(self):
                return [KVCache()]

        class VLM:
            language_model = FakeLang()

        result = self_check_model_apc(VLM())
        assert result.ok is True
        assert result.apc_mode == "block"

    def test_does_not_raise_on_failure(self):
        class FakeLang:
            def make_cache(self):
                raise RuntimeError("boom")

        result = self_check_model_apc(FakeLang())
        assert result.ok is False
        assert result.notes


class TestSelfCheckCapabilityAndSchema:
    def test_layer_capability_is_reported(self):
        from mlx_vlm.models.cache import RotatingKVCache

        assert classify_layer_for_apc(KVCache()).capability == "pageable"
        assert classify_layer_for_apc(RotatingKVCache(max_size=64)).capability == (
            "windowed"
        )

    def test_layout_reports_capabilities_and_schema(self):
        from mlx_vlm.apc_adapters import ADAPTER_SCHEMA_VERSION

        result = validate_prompt_cache_layout([KVCache(), KVCache()], apc_mode="block")
        assert result.capabilities == ["pageable", "pageable"]
        assert result.schema_version == ADAPTER_SCHEMA_VERSION

    def test_self_check_log_includes_schema_and_caps(self, caplog):
        from mlx_vlm.apc_adapters import ADAPTER_SCHEMA_VERSION

        class FakeLang:
            def make_cache(self):
                return [KVCache(), KVCache()]

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            result = self_check_model_apc(FakeLang())
        assert result.schema_version == ADAPTER_SCHEMA_VERSION
        assert result.capabilities == ["pageable", "pageable"]
        assert any(
            ("schema=v%d" % ADAPTER_SCHEMA_VERSION) in r.message
            and "caps=[pageable,pageable]" in r.message
            for r in caplog.records
        )
