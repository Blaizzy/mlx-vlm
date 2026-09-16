"""Exact and partial APC prefix reuse across dense and hybrid models."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, DiskBlockStore, model_apc_mode
from mlx_vlm.models.cache import ArraysCache, KVCache

# Hybrid-model fixtures and exact-mode detection

# ============================================================================
# Model factories — tiny random-weight instances, no downloads needed
# ============================================================================


def _make_tiny_gemma4():
    """Create a tiny Gemma 4 language model with mixed cache types.

    Six hidden layers provide full sliding_window_pattern coverage.

    sliding_window_pattern=3 → pattern: [sliding, sliding, full] repeated
    With 6 layers: 4 RotatingKVCache + 2 KVCache → triggers exact mode.
    """
    from mlx_vlm.models import gemma4

    text_config = gemma4.TextConfig(
        model_type="gemma4_text",
        hidden_size=32,
        num_hidden_layers=6,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=0,
        sliding_window=32,
        sliding_window_pattern=3,
        final_logit_softcapping=30.0,
    )
    vision_config = gemma4.VisionConfig(
        model_type="gemma4_vision",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        patch_size=16,
        pooling_kernel_size=2,
        default_output_length=4,
        position_embedding_size=64,
        use_clipped_linears=False,
    )
    config = gemma4.ModelConfig(
        text_config=text_config,
        vision_config=vision_config,
        model_type="gemma4",
        vocab_size=64,
        image_token_id=63,
    )
    model = gemma4.Model(config)
    return model.language_model


def _make_tiny_qwen35():
    """Create a tiny Qwen 3.5 language model with mixed cache types.

    Four hidden layers provide full_attention_interval coverage.

    full_attention_interval=4 → 3 out of 4 layers use ArraysCache (linear/SSM),
    1 out of 4 uses KVCache (full attention) → triggers exact mode.
    """
    from mlx_vlm.models import qwen3_5

    text_config = qwen3_5.TextConfig(
        model_type="qwen3_5",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        head_dim=8,
        full_attention_interval=4,
    )
    config = qwen3_5.ModelConfig(
        text_config=text_config,
        vision_config=qwen3_5.VisionConfig(
            model_type="qwen3_5",
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            num_heads=2,
        ),
        model_type="qwen3_5",
    )
    model = qwen3_5.LanguageModel(text_config, config)
    return model


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("model_factory", [_make_tiny_gemma4, _make_tiny_qwen35])
def test_apc_exact_mode_detected_for_hybrid_models(model_factory):
    """Hybrid models must route to exact mode, not block mode."""
    lm = model_factory()
    assert model_apc_mode(lm) == "exact"


# ============================================================================
# Structural invariants — foundations for per-layer hybrid APC
# ============================================================================


# Divergent suffix reuse


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


def _embeddings(lm, tokens):
    return lm.model.embed_tokens(tokens) * getattr(lm.model, "embed_scale", 1)


@pytest.mark.parametrize("prefill_step_size", [None, 4, 16, 32])
def test_lfm_mixed_prefill_keeps_logits_before_right_padding(
    manager_factory, prefill_step_size
):
    from mlx_vlm.apc import make_warm_batch_exact_cache_multi
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.lfm2 import Model, ModelConfig

    mx.random.seed(19)
    lm = Model(
        ModelConfig(
            model_type="lfm2",
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            block_dim=64,
            block_ff_dim=128,
            block_multiple_of=16,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=True,
            layer_types=["conv", "full_attention"],
        )
    ).language_model
    manager = manager_factory()
    manager._exact_cache_max = 8
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    warm_tokens = [i % 50 + 1 for i in range(71)]
    cold_tokens = [i % 30 + 51 for i in range(25)]
    seed_cache = lm.make_cache()
    lm(mx.array([warm_tokens[:64]]), cache=seed_cache)
    assert manager.store_exact_cache(warm_tokens[:64], seed_cache)
    restored, count = manager.lookup_exact_cache(warm_tokens)
    assert count == 64
    caches, _ = make_warm_batch_exact_cache_multi([restored, lm.make_cache()], [64, 0])
    suffixes = [warm_tokens[64:], cold_tokens]
    padded = mx.array([suffixes[0] + [0] * 18, suffixes[1]])
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=suffixes,
        max_tokens=[1, 1],
        inputs_embeds=_embeddings(lm, padded),
        prompt_kwargs={},
        warm_cache=caches,
        prefill_step_size=prefill_step_size,
        right_pad_per_row=[18, 0],
        suffix_lens=[7, 25],
        apc_manager=manager,
        apc_coordinator=coordinator,
        apc_meta=[
            {
                "full_input_ids": tokens,
                "prefix_len": prefix,
                "checkpoint_lengths": coordinator.checkpoint_lengths(tokens, set()),
            }
            for tokens, prefix in [(warm_tokens, 64), (cold_tokens, 0)]
        ],
    )
    while batch.needs_processing():
        assert batch.prompt_step() > 0
    sampled = []

    def sample(logprobs):
        sampled.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    batch.generate(sample, [lambda _: False, lambda _: False])
    for row, tokens in enumerate([warm_tokens, cold_tokens]):
        reference = lm.make_cache()
        for start in range(0, len(tokens) - 1, 4):
            lm(
                mx.array([tokens[start : min(start + 4, len(tokens) - 1)]]),
                cache=reference,
            )
        logits = lm(mx.array([tokens[-1:]]), cache=reference).logits[0, -1]
        logprobs = logits - mx.logsumexp(logits)
        assert mx.allclose(sampled[0][row], logprobs, atol=1e-4, rtol=1e-4).item()
        assert mx.argmax(sampled[0][row]).item() == mx.argmax(logits).item()
        # Right-padding must preserve LFM's final real convolution state too.
        assert mx.allclose(
            caches[0][0][row : row + 1], reference[0][0], atol=1e-4, rtol=1e-4
        ).item()


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
