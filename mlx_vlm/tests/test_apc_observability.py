"""TDD: APC_TRACE + layout self-check (observability hygiene)."""

from __future__ import annotations

import logging

import mlx.core as mx
import pytest

from mlx_vlm.apc import (
    APCManager,
    APCSelfCheckResult,
    apc_trace,
    apc_trace_enabled,
    classify_layer_for_apc,
    harvest_blocks_from_batch_cache,
    model_apc_mode,
    self_check_model_apc,
    validate_prompt_cache_layout,
)
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    KVCache,
    QuantizedKVCache,
)

BLOCK_SIZE = 16
GROUP_SIZE = 64
BITS = 8


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

    def test_block_store_emits_trace(self, monkeypatch, caplog):
        monkeypatch.setenv("APC_TRACE", "1")
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        keys = [mx.zeros((1, 2, BLOCK_SIZE, 8), dtype=mx.float32)]
        values = [mx.zeros((1, 2, BLOCK_SIZE, 8), dtype=mx.float32)]

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            blocks = manager.store_kv_blocks(
                list(range(BLOCK_SIZE)),
                keys,
                values,
            )

        assert len(blocks) == 1
        assert any(
            "APC_TRACE store mode=block" in r.message and "memory_blocks=1" in r.message
            for r in caplog.records
        )
        manager.release(blocks)

    def test_block_harvest_reject_emits_reason(self, monkeypatch, caplog):
        monkeypatch.setenv("APC_TRACE", "1")
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)

        class NoHarvestAdapter:
            pass

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            blocks = harvest_blocks_from_batch_cache(
                manager,
                [NoHarvestAdapter()],
                0,
                list(range(BLOCK_SIZE)),
            )

        assert blocks == []
        assert manager.stats.rejects_by_reason["no_block_harvest_adapter"] == 1
        assert any(
            "APC_TRACE reject" in r.message
            and "reason=no_block_harvest_adapter" in r.message
            and "mode=block" in r.message
            for r in caplog.records
        )


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

    def test_exact_clone_is_materialized(self, monkeypatch):
        c = KVCache()
        c.update_and_fetch(
            mx.zeros((1, 2, 4, 8), dtype=mx.float32),
            mx.zeros((1, 2, 4, 8), dtype=mx.float32),
        )

        def fail_eval(*args, **kwargs):
            raise RuntimeError("materialization failed")

        monkeypatch.setattr("mlx_vlm.apc.mx.eval", fail_eval)
        result = classify_layer_for_apc(c)
        assert result.status == "unsupported"
        assert result.reason == "clone_error:RuntimeError"


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

    def test_runtime_cache_factory_failure_is_non_fatal(self):
        class FakeLang:
            def make_cache(self):
                return [KVCache()]

        def fail_factory():
            raise RuntimeError("runtime cache construction failed")

        result = self_check_model_apc(
            FakeLang(),
            cache_factory=fail_factory,
            apc_mode="block",
        )
        assert result.ok is False
        assert "runtime cache construction failed" in result.notes[-1]

    @pytest.mark.parametrize(
        ("kv_bits", "kv_quant_scheme", "expected_type"),
        [
            (8, "uniform", "BatchQuantizedKVCache"),
            (3.5, "turboquant", "BatchTurboQuantKVCache"),
        ],
    )
    def test_server_worker_checks_effective_quantized_runtime_layout(
        self,
        kv_bits,
        kv_quant_scheme,
        expected_type,
    ):
        from mlx_vlm.server.generation import ResponseGenerator

        class FakeLang:
            def make_cache(self):
                return [KVCache(), KVCache()]

        worker = ResponseGenerator.__new__(ResponseGenerator)
        worker.model = type("FakeVLM", (), {"language_model": FakeLang()})()
        worker.kv_bits = kv_bits
        worker.kv_group_size = GROUP_SIZE
        worker.kv_quant_scheme = kv_quant_scheme

        result = worker._run_apc_self_check()

        assert result.ok is True
        assert result.apc_mode == "block"
        assert result.layer_types == [expected_type, expected_type]
        assert f"kv_bits={kv_bits}" in result.notes
        assert f"kv_quant_scheme={kv_quant_scheme}" in result.notes

    def test_server_worker_checks_implicit_runtime_cache_layout(self):
        from mlx_vlm.server.generation import ResponseGenerator

        class FakeLang:
            layers = [object(), object()]

        language_model = FakeLang()
        assert model_apc_mode(language_model) == "block"

        worker = ResponseGenerator.__new__(ResponseGenerator)
        worker.model = type("FakeVLM", (), {"language_model": language_model})()
        worker.kv_bits = None
        worker.kv_group_size = GROUP_SIZE
        worker.kv_quant_scheme = "uniform"

        result = worker._run_apc_self_check()

        assert result.ok is True
        assert result.apc_mode == "block"
        assert result.layer_types == ["BatchKVCache", "BatchKVCache"]

    @pytest.mark.parametrize(
        ("kv_bits", "kv_quant_scheme"),
        [(8, "uniform"), (3.5, "turboquant")],
    )
    def test_effective_quantized_runtime_cache_harvests_and_traces(
        self,
        monkeypatch,
        caplog,
        kv_bits,
        kv_quant_scheme,
    ):
        from mlx_vlm.generate.ar import _make_cache

        class FakeLang:
            def make_cache(self):
                return [KVCache(), KVCache()]

        caches = _make_cache(
            FakeLang(),
            [0],
            kv_bits=kv_bits,
            kv_group_size=GROUP_SIZE,
            kv_quant_scheme=kv_quant_scheme,
        )
        for cache in caches:
            cache.update_and_fetch(
                mx.random.normal((1, 2, BLOCK_SIZE, GROUP_SIZE)),
                mx.random.normal((1, 2, BLOCK_SIZE, GROUP_SIZE)),
            )

        monkeypatch.setenv("APC_TRACE", "1")
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            blocks = harvest_blocks_from_batch_cache(
                manager,
                caches,
                0,
                list(range(BLOCK_SIZE)),
            )

        assert len(blocks) == 1
        assert manager.stats.stores == 1
        assert any(
            "APC_TRACE store mode=block" in record.message
            and "memory_blocks=1" in record.message
            for record in caplog.records
        )
        manager.release(blocks)
