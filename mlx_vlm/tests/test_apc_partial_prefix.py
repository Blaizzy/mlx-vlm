"""Divergent suffix reuse must restore state from before the divergence."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_vlm.tests.test_apc_exact_mode import _make_tiny_gemma4, _make_tiny_qwen35


def _kv(tokens):
    cache = KVCache()
    values = mx.array(tokens, dtype=mx.float32).reshape(1, 1, -1, 1)
    cache.update_and_fetch(values, values + 1)
    return cache


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    managers = []

    def make(tier="memory"):
        disk = (
            None if tier == "memory" else DiskBlockStore(tmp_path, namespace="partial")
        )
        manager = APCManager(num_blocks=8, block_size=16, disk=disk)
        if tier == "disk-only":
            manager._exact_cache_max = 0
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        if manager.disk is not None:
            manager.close()


@pytest.mark.parametrize("tier", ["memory", "disk", "disk-only"])
def test_dense_checkpoint_reuses_only_common_blocks(manager_factory, tier):
    stored = list(range(80))
    divergent = stored[:37] + [999, 998, 997]
    manager = manager_factory(tier)
    assert manager.store_exact_cache(stored, [_kv(stored)], extra_hash=7)
    if manager.disk:
        manager.close()
        manager.disk = None
        manager = manager_factory(tier)

    assert manager.lookup_exact_cache(divergent, extra_hash=8) == (None, 0)
    restored, count = manager.lookup_exact_cache(divergent, extra_hash=7)
    assert count == 32
    assert restored[0].offset == 32
    assert restored[0].state[0].flatten().tolist() == stored[:32]
    assert manager.stats_snapshot()["matched_tokens"] == 32
    if manager.disk:
        assert manager.stats_snapshot()["disk_hits"] == 1

    # Mutating the returned row must not alter either stored checkpoint.
    restored[0].update_and_fetch(mx.full((1, 1, 1, 1), -1), mx.full((1, 1, 1, 1), -2))
    extended, count = manager.lookup_exact_cache(stored + [1000], extra_hash=7)
    assert count == 80
    assert extended[0].state[0].flatten().tolist() == stored
    limited, count = manager.lookup_exact_cache(
        divergent, extra_hash=7, max_prefix_tokens=31
    )
    assert count == limited[0].offset == 16
    assert manager.lookup_exact_cache(
        divergent, extra_hash=7, min_prefix_tokens=32
    ) == (None, 0)


def test_compact_dense_memory_reuses_divergent_suffix(manager_factory):
    manager = manager_factory()
    manager._layer_major_memory_min_tokens = 16
    stored = list(range(80))
    source = _kv(stored)
    blocks = manager.store_kv_blocks(stored, [source.keys], [source.values])
    manager.release(blocks)
    assert not manager.hash_table
    restored, count = manager.lookup_exact_cache(stored[:53] + [999])
    assert count == restored[0].offset == 48
    assert restored[0].state[0].flatten().tolist() == stored[:48]


@pytest.mark.parametrize("tier", ["memory", "disk"])
@pytest.mark.parametrize("kind", ["recurrent", "rotating"])
def test_stateful_final_checkpoint_cannot_be_trimmed(manager_factory, tier, kind):
    tokens = list(range(80))
    if kind == "recurrent":
        state = ArraysCache(size=1)
        state[0] = mx.ones((1, 2, 3))
    else:
        state = RotatingKVCache(max_size=16)
        values = mx.ones((1, 1, 80, 2))
        state.update_and_fetch(values, values)
    manager = manager_factory(tier)
    assert manager.store_exact_cache(tokens, [state, _kv(tokens)])
    if manager.disk:
        manager.close()
        manager.disk = None
        manager = manager_factory(tier)
    assert manager.lookup_exact_cache(tokens[:70] + [999]) == (None, 0)
    assert manager.lookup_exact_cache(tokens + [999])[1] == 80


@pytest.mark.parametrize("doc_len", [30_000, 50_000, 100_000])
@pytest.mark.parametrize("tier", ["memory", "disk-only"])
def test_long_document_has_bounded_reusable_checkpoint(manager_factory, doc_len, tier):
    manager = manager_factory(tier)
    model = SimpleNamespace(make_cache=lambda: [ArraysCache(size=1)])
    coordinator = manager.coordinator(model)
    tokens = list(range(doc_len)) + [100_001] * 20
    boundaries = coordinator.checkpoint_lengths(tokens, set())
    assert len(boundaries) == 2
    assert boundaries[-1] == len(tokens) - 1
    assert doc_len - 2048 < boundaries[0] <= doc_len
    assert boundaries[0] % 16 == 0
    for boundary in boundaries:
        cache = ArraysCache(size=1)
        cache[0] = mx.full((1, 1), boundary)
        assert coordinator.store_checkpoint(tokens[:boundary], [cache])
    if manager.disk:
        manager.close()
        manager.disk = None
        manager = manager_factory(tier)
    restored, count = manager.lookup_exact_cache(tokens[:doc_len] + [100_002] * 30)
    assert count == boundaries[0]
    assert restored[0][0].item() == count


def test_checkpoint_schedule_respects_media_budget_and_opt_out(manager_factory):
    manager = manager_factory()
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(
        SimpleNamespace(make_cache=lambda: [ArraysCache(1)])
    )
    tokens = list(range(75))
    assert coordinator.checkpoint_lengths(tokens, set()) == [64, 74]
    manager._exact_cache_max = 4
    assert coordinator.checkpoint_lengths(tokens, set()) == [32, 48, 64, 74]
    # A media span crossing a nominal boundary must be completely prefetched.
    tokens[45:67] = [999] * 22
    assert coordinator.checkpoint_lengths(tokens, {999}) == [67, 74]
    manager.checkpoint_interval_tokens = 0
    assert coordinator.checkpoint_lengths(tokens, set()) == [74]


@pytest.mark.parametrize("right_padding", [None, [0]])
def test_warm_batch_captures_new_checkpoints_at_absolute_positions(
    manager_factory, right_padding
):
    from mlx_vlm.generate.ar import PromptProcessingBatch

    lm = _make_tiny_qwen35()
    manager = manager_factory()
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    tokens = [i % 50 + 1 for i in range(91)]
    cache = lm.make_cache()
    lm(mx.array([tokens[:64]]), cache=cache)
    assert manager.store_exact_cache(tokens[:64], cache)
    restored, count = manager.lookup_exact_cache(tokens)
    assert count == 64
    boundaries = coordinator.checkpoint_lengths(tokens, set())
    assert boundaries == [80, 90]
    suffix = tokens[count:]
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0],
        input_ids=[suffix],
        max_tokens=[1],
        inputs_embeds=_embeddings(lm, mx.array([suffix])),
        prompt_kwargs={},
        warm_cache=restored,
        prefill_step_size=16,
        apc_manager=manager,
        apc_coordinator=coordinator,
        right_pad_per_row=right_padding,
        apc_meta=[
            {
                "full_input_ids": tokens,
                "prefix_len": count,
                "checkpoint_lengths": boundaries,
            }
        ],
    )
    steps = []
    while batch.needs_processing():
        steps.append(batch.prompt_step())
        assert steps[-1] > 0
    batch.generate(lambda lp: mx.argmax(lp, axis=-1), [lambda _: False])
    assert steps == [16, 10]
    assert (
        sorted(len(entry.token_ids) for entry in manager._exact_cache.values())
        == boundaries
    )
    repeated_cache, repeated_count = manager.lookup_exact_cache(tokens)
    assert repeated_count == 90
    repeated = PromptProcessingBatch(
        model=lm,
        uids=[1],
        input_ids=[tokens[repeated_count:]],
        max_tokens=[1],
        inputs_embeds=_embeddings(lm, mx.array([tokens[repeated_count:]])),
        prompt_kwargs={},
        warm_cache=repeated_cache,
        apc_manager=manager,
        apc_coordinator=coordinator,
        apc_meta=[
            {
                "full_input_ids": tokens,
                "prefix_len": repeated_count,
                "checkpoint_lengths": boundaries,
            }
        ],
    )
    assert not repeated.needs_processing()
    repeated.generate(lambda lp: mx.argmax(lp, axis=-1), [lambda _: False])
    assert (
        sorted(len(entry.token_ids) for entry in manager._exact_cache.values())
        == boundaries
    )
    assert manager.lookup_exact_cache(tokens[:85] + [63])[1] == 80


def _embeddings(lm, tokens):
    return lm.model.embed_tokens(tokens) * getattr(lm.model, "embed_scale", 1)


@pytest.mark.parametrize("tier", ["memory", "disk"])
def test_diffusion_prefills_only_divergent_suffix(manager_factory, tier):
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.models.diffusion_gemma import Model, ModelConfig
    from mlx_vlm.tests.test_diffusion_gemma import (
        FakeProcessor,
        RecordingEncoder,
        tiny_config_dict,
    )

    mx.random.seed(7)
    model = Model(ModelConfig.from_dict(tiny_config_dict()))
    recorder = RecordingEncoder(model.model.encoder)
    model.model.encoder = recorder
    manager = manager_factory(tier)
    manager.block_size = 2
    manager.checkpoint_interval_tokens = 4
    manager.exact_cache_min_tokens = 1
    tokens = list(range(2, 13))
    kwargs = dict(max_tokens=2, max_denoising_steps=1, _apc_semantic_hash=11)
    list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([tokens]),
            _apc_manager=manager,
            **kwargs,
        )
    )
    # An identical replay has no new checkpoint to capture. It must not store
    # a third full-prompt snapshot that evicts the divergent-prefix checkpoint.
    repeated = list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([tokens]),
            _apc_manager=manager,
            **kwargs,
        )
    )
    assert repeated[-1].cached_tokens == 10
    if manager.disk:
        manager.close()
        manager.disk = None
        manager = manager_factory(tier)
        manager.block_size = 2
        manager.checkpoint_interval_tokens = 4
        manager.exact_cache_min_tokens = 1
    recorder.input_lengths.clear()
    warm = list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([tokens[:9] + [13, 14]]),
            _apc_manager=manager,
            **kwargs,
        )
    )
    assert warm[-1].cached_tokens == 8
    assert recorder.input_lengths == [2, 1]


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
@pytest.mark.parametrize("tier", ["memory", "disk"])
@pytest.mark.parametrize("path", ["stream", "batch"])
def test_hybrid_generation_restores_before_divergence(
    manager_factory, model_factory, tier, path
):
    from mlx_vlm.generate.ar import PromptProcessingBatch, generate_step
    from mlx_vlm.models.base import InputEmbeddingsFeatures

    mx.random.seed(13)
    lm = model_factory()
    manager = manager_factory(tier)
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    tokens = [i % 50 + 1 for i in range(70)] + [51, 52, 53, 54, 55]
    boundaries = coordinator.checkpoint_lengths(tokens, set())
    assert boundaries == [64, 74]
    ids = mx.array([tokens])

    if path == "stream":
        wrapper = SimpleNamespace(
            language_model=lm,
            get_input_embeddings=lambda ids, *a, **kw: InputEmbeddingsFeatures(
                inputs_embeds=_embeddings(lm, ids)
            ),
        )
        list(
            generate_step(
                ids,
                wrapper,
                None,
                None,
                max_tokens=1,
                temperature=0,
                prefill_step_size=16,
                prompt_cache=lm.make_cache(),
                prompt_cache_checkpoint_lengths=boundaries,
                prompt_cache_checkpoint=lambda n, caches: coordinator.store_checkpoint(
                    tokens[:n], caches
                ),
            )
        )
    else:
        batch = PromptProcessingBatch(
            model=lm,
            uids=[0],
            input_ids=[tokens],
            max_tokens=[1],
            inputs_embeds=_embeddings(lm, ids),
            prompt_kwargs={},
            warm_cache=lm.make_cache(),
            prefill_step_size=16,
            apc_manager=manager,
            apc_coordinator=coordinator,
            apc_meta=[
                {
                    "full_input_ids": tokens,
                    "prefix_len": 0,
                    "checkpoint_lengths": boundaries,
                }
            ],
        )
        while batch.needs_processing():
            assert batch.prompt_step() > 0
        batch.generate(lambda lp: mx.argmax(lp, axis=-1), [lambda _: False])
        # Final prompt harvest must retain the intermediate state in the LRU.
        assert (
            sorted(len(entry.token_ids) for entry in manager._exact_cache.values())
            == boundaries
        )

    if manager.disk:
        manager.close()
        manager.disk = None
        manager = manager_factory(tier)
    divergent = tokens[:70] + [60, 61, 62, 63]
    restored, count = manager.lookup_exact_cache(divergent)
    assert count == 64
    # Reference computes the shared document in the same chunks, but never
    # processes A's instructions. This detects stale recurrent/window state.
    cold_cache = lm.make_cache()
    for start in range(0, count, 16):
        lm(mx.array([divergent[start : start + 16]]), cache=cold_cache)
    suffix = mx.array([divergent[count:]])
    cold = lm(suffix, cache=cold_cache).logits
    warm = lm(suffix, cache=restored).logits
    mx.eval(cold, warm)
    assert mx.allclose(cold, warm, atol=1e-5, rtol=1e-5).item()
    assert mx.array_equal(mx.argmax(cold, axis=-1), mx.argmax(warm, axis=-1)).item()
