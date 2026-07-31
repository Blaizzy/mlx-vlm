import mlx.core as mx

from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList
from mlx_vlm.models.inkling.config import TextConfig
from mlx_vlm.models.inkling.language import (
    LanguageModel,
    _restore_cache_state,
    _snapshot_cache_state,
    banded_additive_mask,
)
from mlx_vlm.speculative.drafters.inkling_mtp import InklingMTPDraftModel
from mlx_vlm.speculative.drafters.inkling_mtp.config import InklingMTPConfig


def _tiny_text_config(num_hidden_layers=2):
    return TextConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        sliding_window_size=8,
        layer_types=["hybrid_sliding", "hybrid"][:num_hidden_layers],
        d_rel=4,
        rel_extent=16,
        sconv_kernel_size=4,
        mlp_layer_types=["dense"] * num_hidden_layers,
        intermediate_size=32,
        dense_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        logits_mup_width_multiplier=1.0,
    )


def test_banded_mask_excludes_left_padding():
    rel = mx.zeros((2, 3, 1, 1))
    proj = mx.zeros((1, 4))
    mask = banded_additive_mask(
        rel,
        proj,
        mx.array(0),
        mx.array(3),
        0,
        4,
        left_padding=mx.array([0, 2]),
    )
    mx.eval(mask)

    assert mx.all(mask[1, :, :, :2] < -1e29).item()
    assert mask[1, 0, 2, 2].item() == 0.0


def test_server_batch_matches_independent_rows():
    mx.random.seed(7)
    model = LanguageModel(_tiny_text_config())
    model.eval()
    mx.eval(model.parameters())

    long_prompt = mx.random.normal((1, 3, 32))
    short_prompt = mx.random.normal((1, 1, 32))
    batch_prompt = mx.concatenate(
        [
            long_prompt,
            mx.concatenate([mx.zeros((1, 2, 32)), short_prompt], axis=1),
        ],
        axis=0,
    )

    batch_cache = _make_cache(model, [0, 2])
    long_cache = model.make_cache()
    short_cache = model.make_cache()

    batch_logits = model(inputs_embeds=batch_prompt, cache=batch_cache).logits
    long_logits = model(inputs_embeds=long_prompt, cache=long_cache).logits
    short_logits = model(inputs_embeds=short_prompt, cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, -1], long_logits[0, -1], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, -1], short_logits[0, -1], atol=3e-3).item()

    next_embeddings = mx.random.normal((2, 1, 32))
    batch_logits = model(inputs_embeds=next_embeddings, cache=batch_cache).logits
    long_logits = model(inputs_embeds=next_embeddings[:1], cache=long_cache).logits
    short_logits = model(inputs_embeds=next_embeddings[1:], cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, 0], long_logits[0, 0], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, 0], short_logits[0, 0], atol=3e-3).item()


def test_server_right_padded_prefill_preserves_conv_state():
    mx.random.seed(11)
    model = LanguageModel(_tiny_text_config())
    model.eval()
    mx.eval(model.parameters())

    long_prompt = mx.random.normal((1, 3, 32))
    short_prompt = mx.random.normal((1, 1, 32))
    batch_prompt = mx.concatenate(
        [
            long_prompt,
            mx.concatenate([short_prompt, mx.zeros((1, 2, 32))], axis=1),
        ],
        axis=0,
    )

    batch_cache = _make_cache(model, [0, 0])
    for cache in batch_cache:
        cache.prepare(right_padding=[0, 2], lengths=[3, 1])
    long_cache = model.make_cache()
    short_cache = model.make_cache()

    batch_logits = model(inputs_embeds=batch_prompt, cache=batch_cache).logits
    long_logits = model(inputs_embeds=long_prompt, cache=long_cache).logits
    short_logits = model(inputs_embeds=short_prompt, cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, 2], long_logits[0, 2], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, 0], short_logits[0, 0], atol=3e-3).item()

    for cache in batch_cache:
        cache.finalize()
    next_embeddings = mx.random.normal((2, 1, 32))
    batch_logits = model(inputs_embeds=next_embeddings, cache=batch_cache).logits
    long_logits = model(inputs_embeds=next_embeddings[:1], cache=long_cache).logits
    short_logits = model(inputs_embeds=next_embeddings[1:], cache=short_cache).logits
    mx.eval(batch_logits, long_logits, short_logits)

    assert mx.allclose(batch_logits[0, 0], long_logits[0, 0], atol=3e-3).item()
    assert mx.allclose(batch_logits[1, 0], short_logits[0, 0], atol=3e-3).item()


def test_empty_batch_cache_snapshot_restores_metadata():
    cache = CacheList(BatchKVCache([0, 2]), ArraysCache(4, left_padding=[0, 2]))
    snapshot = _snapshot_cache_state([cache])

    keys = mx.ones((2, 1, 1, 4))
    cache[0].update_and_fetch(keys, keys)
    cache[1][0] = mx.ones((2, 3, 4))
    cache[1].advance(1)
    _restore_cache_state([cache], snapshot)

    assert cache[0].keys is None
    assert cache[0].offset.tolist() == [0, -2]
    assert cache[0].left_padding.tolist() == [0, 2]
    assert cache[0]._idx == 0
    assert cache[1].cache == [None] * 4
    assert cache[1].left_padding.tolist() == [0, 2]


def test_mtp_eval_state_skips_empty_kv_caches():
    config = InklingMTPConfig(
        text_config=_tiny_text_config(1),
        num_mtp_layers=2,
        mtp_local_layer_ids=[0],
    )
    drafter = InklingMTPDraftModel(config)
    drafter._cache = drafter.make_cache()

    state = drafter.draft_eval_state()

    assert len(state) == 4
