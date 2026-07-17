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
