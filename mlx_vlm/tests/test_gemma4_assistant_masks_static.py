"""Static Gemma assistant drafter mask tests that do not import MLX."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


class _FakeArray:
    def __init__(self, values):
        self.values = list(values)


def _fake_minimum(values: _FakeArray, limit: int) -> _FakeArray:
    return _FakeArray([min(value, limit) for value in values.values])


def _load_masks_module(monkeypatch):
    fake_mx = types.SimpleNamespace(
        Dtype=object,
        array=_FakeArray,
        float32=object(),
        minimum=_fake_minimum,
    )
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx
    fake_cache = types.ModuleType("mlx_lm.models.cache")
    fake_cache.dynamic_roll = lambda tensor, *_args, **_kwargs: tensor
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models", types.ModuleType("mlx_lm.models"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)

    path = (
        Path(__file__).resolve().parents[1]
        / "speculative/drafters/gemma4_assistant/masks.py"
    )
    loader = importlib.machinery.SourceFileLoader("gemma4_masks_under_test", str(path))
    spec = importlib.util.spec_from_loader("gemma4_masks_under_test", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_local_window_offset_clamps_absolute_decode_position(monkeypatch):
    module = _load_masks_module(monkeypatch)

    assert module._local_window_offset(128, 8) == 8


def test_local_window_offset_clamps_batched_decode_positions(monkeypatch):
    module = _load_masks_module(monkeypatch)

    result = module._local_window_offset(_FakeArray([5, 128]), 8)

    assert result.values == [5, 8]
