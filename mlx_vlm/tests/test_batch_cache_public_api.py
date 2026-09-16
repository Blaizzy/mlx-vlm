"""Public batch-cache conversion for schedulers outside mlx-vlm.

``make_batch_cache`` converts a model's singleton caches into their
batch-aware counterparts. A downstream continuous-batching scheduler cannot
use ``mlx_lm.generate._make_cache`` for an mlx-vlm model: that converter only
recognises mlx-lm's own cache classes and raises ``ValueError: ... does not
yet support batching`` for every mlx-vlm cache class, ``ArraysCache``
included. These tests pin the public entry point and the conversion rules it
must keep applying.
"""

import importlib

import mlx.core as mx
import pytest

from mlx_vlm.generate import make_batch_cache
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache


class _ModelOwnedCache:
    """Cache that ships its own batch conversion via ``to_batch``."""

    def __init__(self):
        self.conversions = []

    def to_batch(self, left_padding):
        self.conversions.append(list(left_padding))
        return f"batched:{list(left_padding)}"


class _FakeModel:
    def __init__(self, *caches):
        self._caches = list(caches)

    def make_cache(self):
        return self._caches


def test_make_batch_cache_is_public():
    # ``import mlx_vlm.generate as ...`` resolves to the generate() function
    # (mlx_vlm/__init__.py re-exports it), so import the package properly.
    generate_pkg = importlib.import_module("mlx_vlm.generate")
    ar = importlib.import_module("mlx_vlm.generate.ar")

    assert generate_pkg.make_batch_cache is ar.make_batch_cache
    assert "make_batch_cache" in generate_pkg.__all__


def test_private_alias_is_kept():
    assert _make_cache is make_batch_cache


def test_arrays_cache_gets_left_padding():
    model = _FakeModel(ArraysCache(2))

    converted = make_batch_cache(model, [0])

    assert isinstance(converted[0], ArraysCache)
    assert mx.array_equal(converted[0].left_padding, mx.array([0])).item()


def test_kv_layer_becomes_batch_kv_cache():
    model = _FakeModel(KVCache(), ArraysCache(1))

    converted = make_batch_cache(model, [0, 0])

    assert isinstance(converted[0], BatchKVCache)
    assert isinstance(converted[1], ArraysCache)


def test_cache_list_converts_recursively():
    model = _FakeModel(CacheList(ArraysCache(1), KVCache()))

    converted = make_batch_cache(model, [0])

    assert isinstance(converted[0], CacheList)
    hybrid, full_attention = converted[0].caches
    assert isinstance(hybrid, ArraysCache)
    assert mx.array_equal(hybrid.left_padding, mx.array([0])).item()
    assert isinstance(full_attention, BatchKVCache)


def test_model_owned_to_batch_takes_precedence():
    owned = _ModelOwnedCache()

    converted = make_batch_cache(_FakeModel(owned), [3])

    assert owned.conversions == [[3]]
    assert converted == ["batched:[3]"]


def test_model_owned_to_batch_rejects_quantized_batching():
    owned = _ModelOwnedCache()

    with pytest.raises(NotImplementedError, match="quantized continuous batching"):
        make_batch_cache(_FakeModel(owned), [0], kv_bits=8, kv_group_size=64)
