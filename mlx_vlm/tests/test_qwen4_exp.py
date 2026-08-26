import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.qwen4_exp import Model, ModelConfig
from mlx_vlm.models.qwen4_exp.qwen4_exp import merge_ngram_embedding_shards

# A miniature of the real checkpoint: interleaved linear/sparse attention, a PLE
# layer, hyper-connections, a sigmoid output gate and an active QSA indexer.
TEXT_CONFIG = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 128,
    "vocab_size": 128,
    "rms_norm_eps": 1e-6,
    "eos_token_id": 3,
    "output_gate_type": "sigmoid",
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "deepseek_sparse_attention_interval": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "hc_count": 4,
    "hc_lowrank": 16,
    "ple_layer_ids": [2],
    "ple_embed_dim": 32,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 101,
    "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 4,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 8,
    "indexer_budget": 8,
    "indexer_compress_ratio": 2,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
    },
}

VISION_CONFIG = {
    "model_type": "qwen4_exp",
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "in_channels": 3,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 64,
    "out_hidden_size": 32,
    "deepstack_visual_indexes": [],
}


def make_config(**text_overrides):
    text = dict(TEXT_CONFIG, **text_overrides)
    return ModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": text,
            "vision_config": dict(VISION_CONFIG),
            "image_token_id": 100,
            "video_token_id": 101,
            "vision_start_token_id": 98,
            "vision_end_token_id": 99,
        }
    )


class _FakeGroup:
    """Stands in for `mx.distributed.Group`: only `size()` is consulted."""

    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n

    def rank(self):
        return 0


def make_model(**text_overrides):
    model = Model(make_config(**text_overrides))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_config_derives_layer_schedule_and_ple_defaults():
    config = make_config()
    text = config.text_config
    assert text.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ]
    assert text.ple_eos_token_id == 3
    assert text.uses_indexer
    # eos of the text config is extended with the Qwen chat eos on the top level
    assert config.eos_token_id[0] == 3


def test_config_rejects_ple_on_a_sparse_attention_layer():
    with pytest.raises(ValueError, match="linear_attention layers"):
        make_config(ple_layer_ids=[4])


def test_config_rejects_undivisible_ple_embed_dim():
    with pytest.raises(ValueError, match="n-gram heads"):
        make_config(ple_embed_dim=30)


def _hf_weights_for(model, num_shards=4):
    """Rebuild HF-layout weights from an mlx model, inverting `sanitize`."""
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if key.startswith("vision_tower."):
            hf_key = key.replace("vision_tower.", "model.visual.", 1)
            if "patch_embed.proj.weight" in hf_key:
                value = value.transpose(0, 4, 1, 2, 3)
            weights[hf_key] = value
            continue
        hf_key = key.replace("language_model.model.", "model.language_model.", 1)
        hf_key = hf_key.replace("language_model.lm_head", "lm_head", 1)
        if "switch_mlp.gate_proj" in hf_key:
            prefix = hf_key.split(".switch_mlp.")[0]
            weights[f"{prefix}.experts.gate_up_proj"] = mx.concatenate(
                [value, _lookup(model, key.replace("gate_proj", "up_proj"))], axis=-2
            )
            continue
        if "switch_mlp.up_proj" in hf_key:
            continue
        if "switch_mlp.down_proj" in hf_key:
            prefix = hf_key.split(".switch_mlp.")[0]
            weights[f"{prefix}.experts.down_proj"] = value
            continue
        if hf_key.endswith("ngram_embedding.weight"):
            prefix = hf_key[: -len(".weight")]
            rows = value.shape[0] // num_shards
            for shard in range(num_shards):
                weights[f"{prefix}.shard_{shard}.weight"] = value[
                    shard * rows : (shard + 1) * rows
                ]
            continue
        if "conv1d.weight" in hf_key:
            value = value.moveaxis(1, 2)
        weights[hf_key] = value
    return weights


def _lookup(model, path):
    node = model
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _ple_tables(model, config):
    from mlx_vlm.models.qwen4_exp.language import (
        build_layer_multipliers,
        build_ngram_head_tables,
    )

    text = config.text_config
    heads = (text.ngram_size - 1) * text.heads_per_ngram
    sizes, offsets = build_ngram_head_tables(heads, 0, text.ngram_vocab_size_base)
    layer = text.ple_layer_ids[0] - 1
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding"
    return {
        f"{prefix}.layer_multipliers": mx.array(
            build_layer_multipliers(text.vocab_size, text.ngram_size, 0, text.seed),
            dtype=mx.int64,
        ),
        f"{prefix}.ngram_heads_vocab_sizes": mx.array(sizes, dtype=mx.int64),
        f"{prefix}.ngram_heads_offsets": mx.array(offsets, dtype=mx.int64),
    }


def test_sanitize_maps_hf_checkpoint_onto_the_model_parameters():
    config = make_config()
    model = Model(config)
    mx.eval(model.parameters())

    weights = _hf_weights_for(model)
    weights.update(_ple_tables(model, config))
    # The MTP draft shard lives in the same checkpoint and must be dropped.
    weights["mtp.layers.0.self_attn.q_proj.weight"] = mx.zeros((8, 8))

    # mirror `load_model`: the model sanitize runs first, then the vision one
    from mlx_vlm.models.qwen4_exp import VisionModel
    from mlx_vlm.utils import sanitize_weights

    sanitized = model.sanitize(dict(weights))
    sanitized = sanitize_weights(VisionModel, sanitized, config.vision_config)
    expected = {k for k, _ in tree_flatten(model.parameters())}
    assert set(sanitized) == expected
    # loading must succeed with the strict shape/name check
    Model(config).load_weights(list(sanitized.items()))


def test_sanitize_merges_the_ngram_embedding_shards_in_index_order():
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    weights = {
        f"{prefix}.shard_{i}.weight": mx.full((2, 3), float(i)) for i in range(12)
    }
    merged = merge_ngram_embedding_shards(weights)
    assert set(merged) == {f"{prefix}.weight"}
    # shard_10 must not sort between shard_1 and shard_2
    assert merged[f"{prefix}.weight"][:, 0].tolist() == [
        float(i) for i in range(12) for _ in range(2)
    ]


def test_sanitize_offsets_only_the_zero_centered_norms():
    config = make_config()
    model = Model(config)
    mx.eval(model.parameters())
    weights = _hf_weights_for(model)
    weights.update(_ple_tables(model, config))

    zero_centered = (
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.layers.1.ple.norm_conv.weight",
    )
    unit_centered = (
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.visual.blocks.0.norm1.weight",
    )
    for key in zero_centered + unit_centered:
        weights[key] = mx.zeros(weights[key].shape)

    sanitized = model.sanitize(dict(weights))
    for key in zero_centered:
        out = sanitized[key.replace("model.language_model", "language_model.model")]
        assert out.min().item() == 1.0, key
    for key in unit_centered:
        out = sanitized[
            key.replace("model.language_model", "language_model.model").replace(
                "model.visual", "vision_tower"
            )
        ]
        assert out.max().item() == 0.0, key


def test_sanitize_rejects_foreign_ngram_hash_tables():
    config = make_config()
    model = Model(config)
    mx.eval(model.parameters())
    weights = _hf_weights_for(model)
    tables = _ple_tables(model, config)
    key = next(k for k in tables if k.endswith("ngram_heads_vocab_sizes"))
    tables[key] = tables[key] + 2
    weights.update(tables)
    with pytest.raises(ValueError, match="n-gram hash constants"):
        model.sanitize(weights)


def _indexer_mask(model, length, offset=0):
    text = model.language_model.model
    hidden = mx.random.normal((1, length, text.args.hidden_size * text.args.hc_count))
    mixed, _, _ = text.layers[3].attn_hyper_connection(hidden)
    position_ids = mx.broadcast_to(
        mx.arange(offset, offset + length)[None, None], (3, 1, length)
    )
    cos, sin = text.rotary_emb(mixed, position_ids)
    return text.layers[3].self_attn.indexer(mixed, cos, sin, None, offset)


def test_indexer_is_a_noop_while_every_block_fits_the_budget():
    model = make_model()
    indexer = model.language_model.model.layers[3].self_attn.indexer
    budget_blocks = indexer.block_topk * indexer.compress_ratio
    assert _indexer_mask(model, budget_blocks + indexer.compress_ratio - 1) is None
    assert _indexer_mask(model, budget_blocks + indexer.compress_ratio) is not None


def test_indexer_mask_is_causal_bounded_and_keeps_the_trailing_block():
    model = make_model()
    indexer = model.language_model.model.layers[3].self_attn.indexer
    length = 4 * indexer.token_budget
    mask = _indexer_mask(model, length)
    assert mask.shape == (1, 1, length, length)

    allowed = mask[0, 0]
    positions = mx.arange(length)
    causal = positions[None, :] <= positions[:, None]
    assert bool(mx.all(allowed <= causal))

    # never more keys than the budget plus the incomplete trailing block
    ratio = indexer.compress_ratio
    complete = (positions + 1) // ratio
    per_query = allowed.sum(axis=-1)
    assert bool(mx.all(per_query <= indexer.token_budget + ratio - 1))
    # the tail after the last complete block is always visible
    tail = causal & (positions[None, :] >= (complete * ratio)[:, None])
    assert bool(mx.all(allowed >= tail))
    # queries that fit in the budget keep full causal visibility
    short = complete <= indexer.block_topk
    assert bool(mx.all(mx.where(short[:, None], allowed == causal, True)))


def _decode_selection(model, prefix_len):
    """Run a prefill through layer 3's indexer, then select for one more token."""
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    text = model.language_model.model
    length = prefix_len + 1
    hidden = mx.random.normal((1, length, text.args.hidden_size * text.args.hc_count))
    mixed, _, _ = text.layers[3].attn_hyper_connection(hidden)
    position_ids = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length))
    cos, sin = text.rotary_emb(mixed, position_ids)

    indexer = text.layers[3].self_attn.indexer
    cache = Qwen4ExpAttentionCache(compress_ratio=indexer.compress_ratio)
    indexer.select(
        mixed[:, :prefix_len], cos[:, :prefix_len], sin[:, :prefix_len], cache, 0
    )
    selection = indexer.select(
        mixed[:, prefix_len:],
        cos[:, prefix_len:],
        sin[:, prefix_len:],
        cache,
        prefix_len,
    )
    return indexer, selection


def test_decode_gather_picks_exactly_the_keys_the_dense_mask_allows():
    model = make_model()
    indexer, selection = _decode_selection(model, 4 * TEXT_CONFIG["indexer_budget"])
    assert selection is not None, "the indexer should be past its budget here"

    mask = indexer.block_mask(selection)
    indices, valid = indexer.gather_indices(selection)
    # The gathered width is fixed so the compacted KV shape never changes.
    assert indices.shape == (1, 1, indexer.decode_width)
    assert indexer.decode_width == indexer.token_budget + indexer.compress_ratio - 1

    allowed = {i for i, ok in enumerate(mask[0, 0, 0].tolist()) if ok}
    gathered = {
        int(i) for i, ok in zip(indices[0, 0].tolist(), valid[0, 0].tolist()) if ok
    }
    assert gathered == allowed
    assert len(allowed) <= indexer.decode_width


def test_decode_gathers_the_cache_instead_of_masking_it(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language as qwen4_exp_language

    model = make_model()
    ids = mx.array([mx.random.randint(4, 90, (20,)).tolist()])

    widths = []
    gather_along_time = qwen4_exp_language._gather_along_time

    def spy(x, indices):
        widths.append(x.shape[2])
        return gather_along_time(x, indices)

    monkeypatch.setattr(qwen4_exp_language, "_gather_along_time", spy)

    def decode():
        language_model = model.language_model
        language_model._position_ids = None
        language_model._rope_deltas = None
        cache = language_model.make_cache()
        language_model(ids[:, :-1], cache=cache)
        return language_model(ids[:, -1:], cache=cache).logits

    gathered = decode()
    # The single sparse layer compacts its keys and its values, and both saw the
    # whole cache going in.
    assert widths == [ids.shape[1]] * 2, widths

    monkeypatch.setattr(qwen4_exp_language, "_is_gatherable", lambda cache: False)
    widths.clear()
    masked = decode()
    assert widths == []

    assert mx.allclose(gathered, masked, atol=2e-4, rtol=1e-3)


def test_block_cache_matches_pooling_the_whole_prefix(monkeypatch):
    """Streaming into the block cache must equal pooling the prefix in one shot."""
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    model = make_model()
    text = model.language_model.model
    indexer = text.layers[3].self_attn.indexer
    ratio = indexer.compress_ratio
    length = 4 * TEXT_CONFIG["indexer_budget"] + ratio - 1

    hidden = mx.random.normal((1, length, text.args.hidden_size * text.args.hc_count))
    mixed, _, _ = text.layers[3].attn_hyper_connection(hidden)
    position_ids = mx.broadcast_to(mx.arange(length)[None, None], (3, 1, length))
    cos, sin = text.rotary_emb(mixed, position_ids)

    one_shot = indexer._ingest(
        indexer.index_qk_proj(mixed)[..., indexer.n_heads * indexer.head_dim :],
        cos,
        sin,
        None,
        0,
    )

    # Same tokens, but fed in ragged chunks that keep splitting blocks apart.
    cache = Qwen4ExpAttentionCache(compress_ratio=ratio)
    sizes = (1, ratio, 3, 2 * ratio + 1, 5)
    streamed, begin, step = None, 0, 0
    while begin < length:
        stop = min(begin + sizes[step % len(sizes)], length)
        keys = indexer.index_qk_proj(mixed[:, begin:stop])[
            ..., indexer.n_heads * indexer.head_dim :
        ]
        streamed = indexer._ingest(
            keys, cos[:, begin:stop], sin[:, begin:stop], cache, begin
        )
        begin, step = stop, step + 1

    assert streamed.shape[1] == length // ratio
    assert one_shot.shape == streamed.shape
    assert mx.allclose(one_shot, streamed, atol=1e-5)
    # One row per completed block, not one per token.
    assert cache[1].blocks.shape[1] == length // ratio
    # One row per completed block, but every token's raw key is still there --
    # that is what makes `trim` exact.
    assert cache[1].offset == length
    assert cache[1].offset % ratio == length % ratio


def test_block_cache_state_round_trips():
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    model = make_model()
    language_model = model.language_model
    ids = mx.array([mx.random.randint(4, 90, (20,)).tolist()])
    cache = language_model.make_cache()
    language_model(ids[:, :-1], cache=cache)

    sparse = cache[3]
    assert isinstance(sparse, Qwen4ExpAttentionCache)
    ratio = sparse[1].ratio
    assert sparse[1].blocks.shape[1] == (ids.shape[1] - 1) // ratio

    restored = Qwen4ExpAttentionCache.from_state(sparse.state, sparse.meta_state)
    assert restored[1].ratio == ratio
    assert restored.offset == sparse.offset
    assert mx.allclose(restored[1].blocks, sparse[1].blocks)
    # `from_state` goes through `__new__`, so the latch must exist as a class default.
    assert restored.indexer_disabled is False

    # Every token's key travels with the rotation of its own position -- M-RoPE
    # positions are not an arange once images are in the prompt, so a block's
    # rotation cannot be re-derived from its index later.
    assert restored[1].offset == sparse[1].offset
    # Compare through `state`, which is the logical view: the raw buffers are
    # step-allocated, so their capacities need not match.
    for got, expected in zip(restored[1].state, sparse[1].state):
        assert (got is None) == (expected is None)
        if got is not None:
            assert mx.allclose(got, expected)

    cache[3] = restored
    stepped = language_model(ids[:, -1:], cache=cache).logits
    assert not bool(mx.any(mx.isnan(stepped)))


def test_batched_offsets_disable_the_indexer_instead_of_poisoning_it(monkeypatch):
    """A left-padded batch has content-relative blocks we cannot express.

    Only this class's own gate is under test, so the inherited attention body --
    which a genuine batched cache would satisfy differently -- is short circuited.
    """
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    model = make_model()
    layer = model.language_model.model.layers[3]
    indexer = layer.self_attn.indexer

    selected = []
    monkeypatch.setattr(indexer, "select", lambda *a, **k: selected.append(a) or None)
    monkeypatch.setattr(Qwen3_5Attention, "__call__", lambda self, x, **k: x)

    hidden = mx.random.normal((2, 4, model.config.text_config.hidden_size * 4))
    position_ids = mx.broadcast_to(mx.arange(4)[None, None], (3, 2, 4))
    embeddings = model.language_model.model.rotary_emb(hidden, position_ids)
    mixed, _, _ = layer.attn_hyper_connection(hidden)

    cache = Qwen4ExpAttentionCache(compress_ratio=indexer.compress_ratio)
    cache.offset = mx.array([0, 2])
    layer.self_attn(mixed, cache=cache, position_embeddings=embeddings)

    assert cache.indexer_disabled
    # Nothing was written, so no half-built block can be picked up later.
    assert not selected
    assert cache[1].blocks is None and cache[1].offset == 0

    # The flag sticks even once the offsets go back to being a plain int.
    cache.offset = 4
    layer.self_attn(mixed, cache=cache, position_embeddings=embeddings)
    assert not selected


def test_cached_decode_matches_a_single_pass():
    model = make_model()
    language_model = model.language_model
    ids = mx.array([[7, 11, 5, 3, 21, 33, 44, 12, 9, 61, 8, 17, 4, 55]])

    single = language_model(ids).logits

    language_model._position_ids = None
    language_model._rope_deltas = None
    cache = language_model.make_cache()
    language_model(ids[:, :-1], cache=cache)
    stepped = language_model(ids[:, -1:], cache=cache).logits

    assert mx.allclose(single[:, -1], stepped[:, 0], atol=2e-4, rtol=1e-3)


def _ple_states_after(lm, ids, *, block=0, accepted=None, gdn_states=None):
    """Feed `ids`, optionally as a speculative block that is then rolled back."""
    lm._position_ids = None
    lm._rope_deltas = None
    cache = lm.make_cache()
    if block:
        lm(ids[:, :-block], cache=cache)
        out = lm(ids[:, -block:], cache=cache, capture_layer_ids=[], return_hidden=True)
        lm.rollback_speculative_cache(cache, out.gdn_states, accepted, block)
    else:
        lm(ids, cache=cache)
    layer = TEXT_CONFIG["ple_layer_ids"][0] - 1
    return cache[layer][2], cache[layer][3]


def test_ple_states_roll_back_to_the_accepted_prefix():
    """A rejected draft must not leave discarded tokens in the PLE states.

    The shared rollback only knows about the gated-delta-net slots, so the PLE's
    short-conv taps and n-gram history are this model's own responsibility.
    """
    lm = make_model().language_model
    ids = mx.array([[7, 11, 5, 3, 21, 33, 44, 12, 9, 61, 8, 17, 4, 55, 30, 41]])
    block, accepted = 4, 1  # 4 drafted, `accepted + 1` = 2 kept

    rolled_conv, rolled_hist = _ple_states_after(
        lm, ids, block=block, accepted=accepted
    )
    # what the states should be: only the kept tokens were ever seen
    kept = ids[:, : ids.shape[1] - block + accepted + 1]
    want_conv, want_hist = _ple_states_after(lm, kept)

    assert rolled_conv.shape == want_conv.shape
    assert rolled_hist.shape == want_hist.shape
    assert rolled_hist.tolist() == want_hist.tolist()
    assert mx.allclose(rolled_conv, want_conv, atol=2e-5)

    # and the test cannot pass by accident: keeping the whole block differs
    whole_conv, whole_hist = _ple_states_after(lm, ids)
    assert whole_hist.tolist() != want_hist.tolist()


def test_ple_states_keep_the_full_window_only_while_verifying():
    lm = make_model().language_model
    layer_idx = TEXT_CONFIG["ple_layer_ids"][0] - 1
    ple = lm.layers[layer_idx].ple
    ids = mx.array([[7, 11, 5, 3, 21, 33, 44, 12]])

    lm._position_ids = None
    lm._rope_deltas = None
    cache = lm.make_cache()
    lm(ids, cache=cache)
    # an ordinary pass stores just the live window
    assert cache[layer_idx][2].shape[1] == ple.short_conv_state_len
    assert cache[layer_idx][3].shape[1] == ple.ple_embedding.context_len

    block = mx.array([[9, 61, 8]])
    out = lm(block, cache=cache, capture_layer_ids=[], return_hidden=True)
    # verifying widens both windows by the block length
    assert cache[layer_idx][2].shape[1] == ple.short_conv_state_len + block.shape[1]
    assert (
        cache[layer_idx][3].shape[1] == ple.ple_embedding.context_len + block.shape[1]
    )

    lm.rollback_speculative_cache(cache, out.gdn_states, 0, block.shape[1])
    # and the rollback puts them back to the live width
    assert cache[layer_idx][2].shape[1] == ple.short_conv_state_len
    assert cache[layer_idx][3].shape[1] == ple.ple_embedding.context_len


def test_a_trim_keeps_the_indexer_and_lands_on_the_untrimmed_state():
    """A trim is exact, so the indexer keeps running -- it does not go dense.

    Blocks are anchored to absolute positions, so the surviving prefix's blocks are
    exactly the first `offset // ratio` of the ones already pooled. That makes the
    trimmed cache indistinguishable from one that only ever saw the shorter prompt.

    `is_trimmable` also has to answer for the attention half: the speculative
    rollback classifies recurrent vs attention caches by exactly this predicate,
    so reporting False here would file this cache alongside the gated-delta-net
    ones and walk off the end of their captured states.
    """
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    lm = make_model().language_model
    ids = mx.array([mx.random.randint(4, 90, (24,)).tolist()])
    drop = 5

    trimmed = _prefill(lm, ids)
    sparse = [i for i, c in enumerate(trimmed) if isinstance(c, Qwen4ExpAttentionCache)]
    assert sparse
    assert all(trimmed[i][1].blocks is not None for i in sparse)
    assert all(trimmed[i].is_trimmable() for i in sparse)

    for i in sparse:
        assert trimmed[i].trim(drop) == drop

    want = _prefill(lm, ids[:, : ids.shape[1] - drop])
    for i in sparse:
        assert not trimmed[i].indexer_disabled  # still sparse
        assert trimmed[i].offset == want[i].offset
        assert trimmed[i][1].offset == want[i][1].offset
        assert trimmed[i][1].blocks.shape == want[i][1].blocks.shape
        assert mx.allclose(trimmed[i][1].blocks, want[i][1].blocks, atol=2e-5)


def test_cache_layout_follows_the_layer_schedule():
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    cache = make_model().language_model.make_cache()
    # layer 1 carries the PLE and therefore two extra recurrent states
    assert [type(c) for c in cache] == [
        ArraysCache,
        ArraysCache,
        ArraysCache,
        Qwen4ExpAttentionCache,
    ]
    assert [len(c.cache) for c in cache[:3]] == [2, 4, 2]
    assert cache[3].compress_ratio == TEXT_CONFIG["indexer_compress_ratio"]
    assert cache[3].offset == 0


def test_vision_tower_produces_one_embedding_per_merged_patch():
    model = make_model()
    grid = mx.array([[1, 4, 4]])
    patch_dim = (
        VISION_CONFIG["in_channels"]
        * VISION_CONFIG["temporal_patch_size"]
        * VISION_CONFIG["patch_size"] ** 2
    )
    pixel_values = mx.random.normal((16, patch_dim))
    hidden, _ = model.vision_tower(pixel_values, grid)
    merged = VISION_CONFIG["spatial_merge_size"] ** 2
    assert hidden.shape == (16 // merged, VISION_CONFIG["out_hidden_size"])


def test_image_tokens_are_replaced_and_positions_stay_shared():
    model = make_model()
    grid = mx.array([[1, 4, 4]])
    patch_dim = (
        VISION_CONFIG["in_channels"]
        * VISION_CONFIG["temporal_patch_size"]
        * VISION_CONFIG["patch_size"] ** 2
    )
    ids = mx.array([[7, 11, 98] + [100] * 4 + [99, 21, 5, 33, 3, 44, 12]])
    out = model(
        ids, pixel_values=mx.random.normal((16, patch_dim)), image_grid_thw=grid
    )
    assert out.logits.shape == (1, ids.shape[1], TEXT_CONFIG["vocab_size"])
    assert not bool(mx.any(mx.isnan(out.logits)))


def test_quantization_reaches_the_ngram_table_and_spares_the_inject_gates():
    """The n-gram table's rows are 160 wide, which 64 does not divide.

    Left to the requested group size it would silently stay in bf16 -- and it is
    the largest tensor in the model by a wide margin.
    """
    from mlx_vlm.models.qwen4_exp import Model
    from mlx_vlm.quant_utils import quantize_model

    # 640 / ((3 - 1) * 2) heads = 160-wide rows, as in the released config
    config = make_config(ple_embed_dim=640, heads_per_ngram=2)
    model = Model(config)
    mx.eval(model.parameters())
    text = model.language_model.model
    ngram = text.layers[1].ple.ple_embedding.ngram_embedding
    assert ngram.weight.shape[-1] == 160

    _, quantized_config = quantize_model(model, {"model_type": "qwen4_exp"}, 64, 4)
    entries = quantized_config["quantization"]

    assert type(text.layers[1].ple.ple_embedding.ngram_embedding).__name__ == (
        "QuantizedEmbedding"
    )
    ngram_path = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    assert entries[ngram_path]["group_size"] == 32
    # the requested bits are inherited, only the group size is pinned
    assert entries[ngram_path]["bits"] == 4

    inject = "language_model.model.layers.0.attn_hyper_connection.block_inject_weight"
    assert entries[inject]["bits"] == 8
    assert entries[inject]["group_size"] == 64

    # and the global entry is untouched
    assert entries["group_size"] == 64 and entries["bits"] == 4


def test_prompt_utils_knows_the_model():
    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat, get_message_json

    assert MODEL_CONFIG["qwen4_exp"] == MessageFormat.LIST_WITH_IMAGE_FIRST
    video = get_message_json("qwen4_exp", "describe", role="user", video=True)
    assert any(part.get("type") == "video" for part in video["content"])


def test_ngram_head_tables_are_distinct_primes():
    from mlx_vlm.models.qwen4_exp.language import build_ngram_head_tables

    sizes, offsets = build_ngram_head_tables(4, 0, 101)
    assert sizes == (101, 103, 107, 109)
    assert offsets == (0, 101, 204, 311)
    # a second PLE layer continues the prime sequence
    next_sizes, _ = build_ngram_head_tables(4, 1, 101)
    assert next_sizes[0] > sizes[-1]


def _prefill(lm, ids):
    lm._position_ids = None
    lm._rope_deltas = None
    cache = lm.make_cache()
    lm(ids, cache=cache)
    return cache


def _speculative_round(lm, ids, *, block, accepted):
    """Prefill all but `block` tokens, verify that block, then roll it back."""
    cache = _prefill(lm, ids[:, :-block])
    _, _, rollback_state = lm.speculative_verify_hidden(ids[:, -block:], cache)
    lm.rollback_speculative_cache(cache, rollback_state, accepted, block)
    return cache


def test_qsa_survives_a_rollback_that_completes_no_block():
    """The short-draft case: fewer drafted tokens than `compress_ratio`.

    No block is committed, so only the trailing partial block moves. The trim still
    has to land on exactly the same state a shorter prompt would have produced.
    """
    lm = make_model(indexer_compress_ratio=4).language_model
    ids = mx.array([mx.random.randint(4, 90, (26,)).tolist()])
    block, accepted = 2, 0  # 2 drafted, 1 kept -- never fills a 4-token window
    kept = ids[:, : ids.shape[1] - block + accepted + 1]

    rolled = _speculative_round(lm, ids, block=block, accepted=accepted)
    sparse = [
        i for i, c in enumerate(rolled) if type(c).__name__ == "Qwen4ExpAttentionCache"
    ]
    assert sparse

    want = _prefill(lm, kept)
    for i in sparse:
        assert not rolled[i].indexer_disabled
        assert rolled[i][1].offset == want[i][1].offset
        assert mx.allclose(rolled[i][1].blocks, want[i][1].blocks, atol=2e-5)
        # The keys of the partly-filled trailing block have to survive too, or the
        # next token would complete it from the wrong tokens.
        for got, expected in zip(rolled[i][1].state, want[i][1].state):
            assert mx.allclose(got, expected, atol=2e-5)


def test_qsa_survives_a_speculative_rollback():
    """The indexer stays live across a rejected draft instead of going dense.

    A block key is a function of `compress_ratio` consecutive tokens *and* the
    absolute position of the first of them. Because that position never moves, the
    cut landing mid-block is fine: keep the blocks the surviving prefix still fully
    covers and rebuild the partial one from the raw keys, which has to land on
    exactly what a plain forward over the kept tokens would leave.
    """
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttentionCache

    lm = make_model().language_model
    ids = mx.array([mx.random.randint(4, 90, (24,)).tolist()])
    block, accepted = 4, 1  # 4 drafted, `accepted + 1` = 2 kept
    kept = ids[:, : ids.shape[1] - block + accepted + 1]

    rolled = _speculative_round(lm, ids, block=block, accepted=accepted)
    sparse = [i for i, c in enumerate(rolled) if isinstance(c, Qwen4ExpAttentionCache)]
    assert sparse, "no sparse-attention layer to exercise"
    # The prompt has to be long enough that the indexer actually engaged, or this
    # test would pass on an all-dense model.
    assert any(rolled[i][1].blocks.shape[1] > rolled[i][1].ratio for i in sparse)

    step = mx.array([[42]])
    rolled_logits = lm(step, cache=rolled).logits

    want = _prefill(lm, kept)
    want_logits = lm(step, cache=want).logits

    for i in sparse:
        got, expected = rolled[i][1], want[i][1]
        assert not rolled[i].indexer_disabled  # still sparse, not fallen back
        assert rolled[i].offset == want[i].offset
        assert got.offset == expected.offset
        assert got.blocks.shape == expected.blocks.shape
        assert mx.allclose(got.blocks, expected.blocks, atol=2e-5)

    assert mx.allclose(rolled_logits, want_logits, atol=2e-4, rtol=1e-3)

    # ...and it cannot pass by accident: accepting the whole block lands elsewhere.
    whole = _speculative_round(lm, ids, block=block, accepted=block - 1)
    assert whole[sparse[0]][1].blocks.shape != want[sparse[0]][1].blocks.shape


def test_qsa_gives_up_when_batch_rows_accept_different_amounts():
    """No single accepted prefix exists, so the indexer must not guess one.

    `trim` cannot be relied on to notice: when one row accepts the whole block the
    uniform trim is zero and never runs.

    The trim is uniform, so unlike the single-row case it cannot express this, and
    a block key cannot be patched per row after the fact.

    Doubles as the guard for `prepare`/`finalize`: the batched rollback probes for
    them, and the inherited `CacheList` versions forward into a `KVCache` that has
    neither.
    """
    lm = make_model().language_model
    ids = mx.array([mx.random.randint(4, 90, (24,)).tolist()] * 2)
    block = 4

    cache = _prefill(lm, ids[:, :-block])
    _, _, rollback_state = lm.speculative_verify_hidden(ids[:, -block:], cache)
    # row 0 accepted the whole block, row 1 only its first token
    lm.rollback_speculative_cache(cache, rollback_state, [block - 1, 0], block)

    sparse = [
        i for i, c in enumerate(cache) if type(c).__name__ == "Qwen4ExpAttentionCache"
    ]
    assert sparse
    for i in sparse:
        assert cache[i].indexer_disabled

    # and it still runs, dense, from there
    out = lm(mx.array([[9], [9]]), cache=cache)
    assert not bool(mx.any(mx.isnan(out.logits)))


def test_block_rotations_stay_aligned_across_ragged_chunks():
    """One rotation per block-leading position, no matter how the tokens arrive.

    Block `b` starts at absolute position `b * ratio`, so a chunk boundary must not
    shift which rows get kept -- and a trim must drop exactly the ones whose block
    no longer exists.
    """
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpBlockCache

    ratio = 4
    cache = Qwen4ExpBlockCache(ratio)
    fed = 0
    for chunk in (1, ratio, 3, 2 * ratio + 1, 5, 1, 1, 7):
        keys = mx.arange(fed, fed + chunk).reshape(1, chunk, 1).astype(mx.float32)
        cache.append(keys, mx.broadcast_to(keys, (1, chunk, 2)))
        fed += chunk
        assert cache.offset == fed
        # `leading_rotation` is the logical view; the buffer behind it is
        # step-allocated, so its capacity is deliberately larger.
        assert cache.leading_rotation.shape[1] == cache.n_leading == -(-fed // ratio)
        # each kept row really is a block-leading position
        assert cache.leading_rotation[0, :, 0].tolist() == [
            float(b * ratio) for b in range(cache.n_leading)
        ]

    cache.trim(6)
    assert cache.offset == fed - 6
    assert cache.leading_rotation.shape[1] == cache.n_leading
    assert cache.leading_rotation[0, :, 0].tolist() == [
        float(b * ratio) for b in range(cache.n_leading)
    ]
    # the buffer was not reallocated by the trim -- that is the point
    assert cache.block_rotation.shape[1] == cache.step


def test_shard_is_a_no_op_for_a_group_of_one():
    lm = make_model().language_model
    before = {k: v for k, v in tree_flatten(lm.parameters())}
    lm.shard(_FakeGroup(1))
    after = {k: v for k, v in tree_flatten(lm.parameters())}
    assert set(before) == set(after)
    for k in before:
        assert before[k].shape == after[k].shape
    assert all(layer.mlp.sharding_group is None for layer in lm.layers)


def test_shard_rejects_an_indivisible_expert_width():
    # 640 is the released width; ask for a group size that does not divide it.
    lm = make_model(moe_intermediate_size=17).language_model
    with pytest.raises(ValueError, match="moe_intermediate_size=17"):
        lm.shard(_FakeGroup(2))


def test_shard_splits_only_the_experts():
    """Attention, GDN, the indexer and the PLE table must keep their full shape."""
    lm = make_model().language_model
    n = 2
    before = {k: v.shape for k, v in tree_flatten(lm.parameters())}
    lm.shard(_FakeGroup(n))
    after = {k: v.shape for k, v in tree_flatten(lm.parameters())}

    changed = {k for k in before if before[k] != after[k]}
    assert changed, "nothing was sharded"
    for k in changed:
        assert any(
            part in k for part in ("switch_mlp", "shared_expert.")
        ), f"{k} was sharded but should have stayed replicated"
    # the two things that must never be touched
    for k in before:
        if "indexer" in k or "ple_embedding" in k or "linear_attn" in k:
            assert before[k] == after[k], f"{k} must stay replicated"


def test_a_two_way_expert_split_sums_back_to_the_unsharded_output():
    """The algebra `shard()` relies on, checked without needing two processes.

    Splitting the expert intermediate dimension makes each rank's `down_proj`
    produce a *partial* sum; adding the parts must reproduce the whole. That is
    what the block's `all_sum` does at runtime, so if this identity does not hold
    the sharded model is silently wrong rather than slow.
    """
    import copy

    model = make_model()
    moe = model.language_model.layers[1].mlp
    x = mx.random.normal((1, 5, model.config.text_config.hidden_size))
    whole = moe(x)

    width = model.config.text_config.moe_intermediate_size
    shared_width = model.config.text_config.shared_expert_intermediate_size
    assert width % 2 == 0 and shared_width % 2 == 0
    halves = []
    for lo, hi, slo, shi in (
        (0, width // 2, 0, shared_width // 2),
        (width // 2, width, shared_width // 2, shared_width),
    ):
        part = copy.deepcopy(moe)
        sm = part.switch_mlp
        sm.gate_proj.weight = sm.gate_proj.weight[:, lo:hi, :]
        sm.up_proj.weight = sm.up_proj.weight[:, lo:hi, :]
        sm.down_proj.weight = sm.down_proj.weight[:, :, lo:hi]
        se = part.shared_expert
        se.gate_proj.weight = se.gate_proj.weight[slo:shi, :]
        se.up_proj.weight = se.up_proj.weight[slo:shi, :]
        se.down_proj.weight = se.down_proj.weight[:, slo:shi]
        halves.append(part(x))

    # each half alone is wrong; their sum is the answer
    assert not mx.allclose(halves[0], whole, atol=1e-3)
    assert mx.allclose(halves[0] + halves[1], whole, atol=2e-5)
