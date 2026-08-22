"""Weight-free APC compatibility audit for every in-tree model package.

The audit deliberately imports architecture code only. It never constructs a
model, downloads a checkpoint, calls ``load_weights``, or evaluates parameters.
APC compatibility is a cache-layout property, so covering every cache factory
used by the model sources is both cheaper and more precise than allocating all
of the model weights.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_vlm.models as model_packages
from mlx_vlm.apc import APCManager
from mlx_vlm.apc_adapters import (
    build_prefix_cache_plan,
    build_prefix_cache_plan_from_caches,
    clone_cache_entry,
)
from mlx_vlm.models.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    PoolingCache,
    RotatingKVCache,
    SimpleKVCache,
    StaticPrefixKVCache,
)
from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache
from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache


def _model_source_root() -> Path:
    return Path(model_packages.__file__).resolve().parent


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _cache_factories_by_package() -> dict[str, set[str]]:
    """Statically discover cache constructors inside every ``make_cache``."""
    found: dict[str, set[str]] = {}
    for path in _model_source_root().rglob("*.py"):
        tree = ast.parse(path.read_text())
        package = path.relative_to(_model_source_root()).parts[0]
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if function.name != "make_cache":
                continue
            names = found.setdefault(package, set())
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    name = _call_name(node)
                    if name is not None and (
                        name.endswith("Cache") or name == "CacheList"
                    ):
                        names.add(name)
    return found


def _cache_samples():
    """One unpopulated instance of every cache family used by model sources."""
    return {
        "ArraysCache": ArraysCache(2),
        "CacheList": CacheList(KVCache(), ArraysCache(1)),
        "ChunkedKVCache": ChunkedKVCache(chunk_size=16),
        "KVCache": KVCache(),
        "MiniMaxM3KVCache": MiniMaxM3KVCache(),
        "PoolingCache": PoolingCache(ratio=2),
        "RingSlidingKVCache": RingSlidingKVCache(window_size=16),
        "RotatingKVCache": RotatingKVCache(max_size=16),
        "SimpleKVCache": SimpleKVCache(),
        "StaticPrefixKVCache": StaticPrefixKVCache(max_size=16),
    }


def _cache_names_in_callable(function) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, IndentationError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None and (name.endswith("Cache") or name == "CacheList"):
                names.add(name)
    return names


def _model_cache_contract(model_cls: type) -> tuple[str, ...]:
    """Resolve a wrapper's cache factory without constructing the wrapper."""
    visited: set[type] = set()

    def visit(cls: type) -> set[str]:
        if cls in visited:
            return set()
        visited.add(cls)

        make_cache = getattr(cls, "make_cache", None)
        if callable(make_cache):
            names = _cache_names_in_callable(make_cache)
            if names:
                return names

        # Wrapper factories and LanguageModel factories often delegate to a
        # class imported as one of these conventional names.
        functions = [getattr(cls, "__init__", None), make_cache]
        for function in functions:
            namespace = getattr(function, "__globals__", {})
            for name in ("LanguageModel", "TextModel", "Model"):
                target = namespace.get(name)
                if isinstance(target, type) and target is not cls:
                    names = visit(target)
                    if names:
                        return names
        return set()

    # No custom factory means generation uses one ordinary KVCache per layer.
    return tuple(sorted(visit(model_cls) or {"KVCache"}))


def _all_generative_model_contracts() -> list[tuple[str, tuple[str, ...]]]:
    contracts = []
    local_factories = _cache_factories_by_package()
    for info in pkgutil.iter_modules(model_packages.__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"mlx_vlm.models.{info.name}")
        except ModuleNotFoundError:
            # Some Omni packages depend on a newer optional companion package
            # than the minimum version in the lockfile. Their cache factories
            # are still fully discoverable from local source, so keep them in
            # the weight-free APC matrix without importing that dependency.
            names = local_factories.get(info.name)
            if names:
                contracts.append((info.name, tuple(sorted(names))))
            continue
        model_cls = getattr(module, "Model", None)
        if model_cls is not None and callable(
            getattr(model_cls, "get_input_embeddings", None)
        ):
            names = local_factories.get(info.name) or set(
                _model_cache_contract(model_cls)
            )
            contracts.append((info.name, tuple(sorted(names))))
    return sorted(contracts)


def _populated_cache(name: str, token_count: int):
    shape = (1, 1, token_count, 4)
    if name == "KVCache":
        cache = KVCache()
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "RotatingKVCache":
        cache = RotatingKVCache(max_size=token_count * 2)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        cache._idx = token_count
        return cache
    if name == "ChunkedKVCache":
        cache = ChunkedKVCache(chunk_size=8)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        cache.start_position = 0
        return cache
    if name == "ArraysCache":
        cache = ArraysCache(2)
        cache.cache = [mx.ones((1, 2, 4)), mx.ones((1, 1, 4)) * 2]
        return cache
    if name == "PoolingCache":
        cache = PoolingCache(ratio=2)
        cache.pooled = mx.ones((1, token_count // 2, 4))
        cache.buf_kv = mx.ones((1, 2, 4))
        cache.buf_gate = mx.ones((1, 2, 1))
        cache.remainder = 1
        return cache
    if name == "SimpleKVCache":
        cache = SimpleKVCache()
        cache.update_and_fetch(mx.ones(shape), mx.ones(shape) * 2)
        return cache
    if name == "StaticPrefixKVCache":
        cache = StaticPrefixKVCache(max_size=token_count)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "RingSlidingKVCache":
        cache = RingSlidingKVCache(window_size=token_count)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "MiniMaxM3KVCache":
        cache = MiniMaxM3KVCache()
        cache.update_and_fetch(mx.ones(shape), mx.ones(shape) * 2)
        cache.update_index_and_fetch(mx.ones(shape))
        return cache
    if name == "CacheList":
        return CacheList(
            _populated_cache("KVCache", token_count),
            _populated_cache("ArraysCache", token_count),
        )
    raise AssertionError(f"No populated APC sample for {name}")


MODEL_CACHE_CONTRACTS = _all_generative_model_contracts()


def test_all_generative_model_packages_discovered_without_weights():
    """Every discovered LM/VLM/Omni contract has local architecture source."""
    # This is intentionally a lower bound: new architectures increase it,
    # while accidentally dropping a large family makes the audit fail loudly.
    assert len(MODEL_CACHE_CONTRACTS) >= 120
    for package, _ in MODEL_CACHE_CONTRACTS:
        assert (_model_source_root() / package).is_dir()


def test_every_model_cache_factory_has_a_restorable_apc_adapter():
    """All cache types referenced by all model factories are APC-compatible."""
    by_package = _cache_factories_by_package()
    discovered = set().union(*by_package.values())
    samples = _cache_samples()
    unknown = discovered - samples.keys()
    assert not unknown, (
        "New model cache types need an APC adapter/sample: " f"{sorted(unknown)}"
    )

    # Exercise planning and cloning, not just name registration. Empty caches
    # are sufficient because the protocol and constructor metadata are what
    # vary across architectures; populated round-trips live in the APC tests.
    for name in sorted(discovered):
        cache = samples[name]
        plan = build_prefix_cache_plan_from_caches([cache])
        assert plan.restorable, f"{name}: {plan.describe()}"
        eval_targets: list[mx.array] = []
        clone = clone_cache_entry(
            cache,
            min_capacity_tokens=None,
            eval_targets=eval_targets,
        )
        assert clone is not None, f"{name} cannot be cloned for APC"

    # Also build one heterogeneous plan per model package. This catches a
    # future combination that is individually registered but cannot be
    # coordinated as one architecture.
    for package, names in by_package.items():
        if not names:
            continue
        plan = build_prefix_cache_plan_from_caches(
            [samples[name] for name in sorted(names)]
        )
        assert plan.restorable, f"{package}: {plan.describe()}"

    assert len(by_package) >= 70


@pytest.mark.parametrize(
    ("model_name", "cache_names"),
    MODEL_CACHE_CONTRACTS,
    ids=[name for name, _ in MODEL_CACHE_CONTRACTS],
)
def test_cache_hit_for_every_model(model_name, cache_names, monkeypatch):
    """A synthetic second request hits APC for every model cache contract."""
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    block_size = 8
    token_count = 2 * block_size
    token_ids = list(range(token_count))
    caches = [_populated_cache(name, token_count) for name in cache_names]
    plan = build_prefix_cache_plan_from_caches(caches)
    assert plan.restorable, f"{model_name}: {plan.describe()}"

    manager = APCManager(num_blocks=8, block_size=block_size)

    class SyntheticModel:
        def make_cache(self):
            return caches

    coordinator = manager.coordinator(SyntheticModel())
    assert coordinator.strategy == plan.strategy, model_name
    try:
        if plan.strategy == "block":
            stored = manager.store_kv_blocks(
                token_ids,
                [cache.keys for cache in caches],
                [cache.values for cache in caches],
            )
            manager.release(stored)
        else:
            assert manager.store_exact_cache(token_ids, caches), model_name

        hit = coordinator.lookup(
            token_ids + [999],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _prefix_len: True,
            prefix_has_media=lambda _prefix_len: False,
        )
        assert hit is not None, model_name
        assert hit["prefix_len"] == token_count, model_name
        stats = manager.stats_snapshot()
        if plan.strategy == "block":
            assert stats["lookups_hit"] == 1, model_name
        else:
            assert hit["warm_cache"] is not None, model_name
            assert stats["exact_hits"] == 1, model_name
        coordinator.release_hit(hit)
    finally:
        manager.close()


def test_dense_models_without_make_cache_use_generation_fallback():
    """VLM language backbones without a custom factory remain pageable."""

    class DenseLanguageModel:
        layers = [object(), object(), object()]

    class VisionLanguageModel:
        language_model = DenseLanguageModel()

    plan = build_prefix_cache_plan(VisionLanguageModel())
    assert plan.restorable
    assert plan.strategy == "block"
    assert len(plan.components) == len(DenseLanguageModel.layers)
    assert len(plan.groups) == 1
