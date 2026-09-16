"""TDD: APC_TRACE + layout self-check (observability hygiene)."""

from __future__ import annotations

import logging

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, classify_layer_for_apc, self_check_model_apc
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
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
    def test_quantized_with_dequant_ok(self):
        # Last dim must be divisible by group_size for mx.quantize.
        c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        c.update_and_fetch(
            mx.random.normal((1, 2, 8, GROUP_SIZE)),
            mx.random.normal((1, 2, 8, GROUP_SIZE)),
        )
        result = classify_layer_for_apc(c)
        assert result.status == "ok"

    def test_unsupported_opaque_type(self):
        class Bogus:
            pass

        result = classify_layer_for_apc(Bogus())
        assert result.status == "unsupported"
        assert result.reason


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

    def test_no_make_cache_not_ok(self, caplog):
        class NoCache:
            pass

        result = self_check_model_apc(NoCache())
        assert result.ok is False

    def test_does_not_raise_on_failure(self):
        class FakeLang:
            def make_cache(self):
                raise RuntimeError("boom")

        result = self_check_model_apc(FakeLang())
        assert result.ok is False
        assert result.notes
