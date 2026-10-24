from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.apc_images import ImagePrefixContext
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.qwen3_5.qwen3_5 import Model
from mlx_vlm.models.qwen4_exp.qwen4_exp import Model as Qwen4ExpModel


def context(tokens, values=None, grids=None, tenant="test", model_type="qwen3_5"):
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type=model_type,
            image_token_index=99,
            video_token_index=98,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )
    return ImagePrefixContext.prepare(
        model, None, tokens, values, {"image_grid_thw": grids}, tenant
    )


def manager(disk=None):
    result = APCManager(
        num_blocks=8, block_size=4, disk=disk, overrides={"memory_max_gb": 0.05}
    )
    result.exact_cache_min_tokens = 1
    return result


def save(store, ctx, length):
    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, 1, length, 2)), mx.ones((1, 1, length, 2)))
    assert store.store_exact_cache(
        ctx.token_ids[:length], [cache], extra_hash=ctx.prefix_hash(length)
    )


def test_appended_image_reuses_only_unchanged_prefix():
    tokens = [1] * 8 + [99] * 4 + [2] * 8
    pixels = mx.arange(16 * 6).reshape(16, 6)
    grid = mx.array([[1, 4, 4]])
    old = context(tokens, pixels, grid)
    new = context(
        tokens + [3] * 4 + [99] * 4 + [4] * 8,
        mx.concatenate([pixels, pixels + 1]),
        mx.concatenate([grid, grid]),
    )
    store = manager()
    save(store, old, 19)
    hit = new.lookup(store)
    assert hit["prefix_len"] == 19
    assert old.prefix_hash(19) == new.prefix_hash(19)
    suffix, suffix_grid = new.suffix_inputs(19)
    assert mx.array_equal(suffix, pixels + 1).item()
    assert mx.array_equal(suffix_grid, grid).item()
    assert new.suffix_inputs(len(new.token_ids)) == (None, None)
    changed = context(
        new.token_ids,
        mx.concatenate([pixels + 2, pixels + 1]),
        mx.concatenate([grid, grid]),
    )
    assert changed.lookup(store) is None


def test_text_prefix_survives_adding_an_image():
    old = context([1] * 24)
    new = context(
        [1] * 24 + [99] * 4 + [2] * 8, mx.zeros((16, 6)), mx.array([[1, 4, 4]])
    )
    store = manager()
    save(store, old, 23)
    assert new.lookup(store)["prefix_len"] == 23
    assert old.prefix_hash(23) == new.prefix_hash(23)


def test_old_image_change_can_still_reuse_checkpoint_before_that_image():
    tokens = [1] * 24 + [99] * 4 + [2] * 8
    first = context(tokens, mx.zeros((16, 6)), mx.array([[1, 4, 4]]))
    second = context(tokens, mx.ones((16, 6)), mx.array([[1, 4, 4]]))
    store = manager()
    save(store, first, 23)
    save(store, first, 35)
    assert second.lookup(store)["prefix_len"] == 23


def test_grid_order_and_tenant_are_part_of_prefix_identity():
    tokens = [1] * 8 + [99] * 4 + [2] * 4 + [99] * 4 + [3] * 8
    pixels = mx.concatenate([mx.zeros((16, 6)), mx.ones((16, 6))])
    grid = mx.array([[1, 4, 4], [1, 4, 4]])
    old = context(tokens, pixels, grid)
    store = manager()
    save(store, old, 27)
    assert context(tokens, pixels[::-1], grid).lookup(store) is None
    assert (
        context(tokens, pixels, mx.array([[1, 2, 8], [1, 4, 4]])).lookup(store) is None
    )
    assert context(tokens, pixels, grid, tenant="other").lookup(store) is None
    assert (
        context(tokens, mx.array(pixels), mx.array(grid)).lookup(store)["prefix_len"]
        == 27
    )


def test_image_boundary_and_checkpoint_scheduling():
    ctx = context(
        [1] * 8 + [99] * 4 + [2] * 8, mx.zeros((16, 6)), mx.array([[1, 4, 4]])
    )
    assert ctx.prefix_hash(8) == ctx.hashes[0]
    assert ctx.prefix_hash(12) == ctx.hashes[1]
    with pytest.raises(ValueError, match="inside"):
        ctx.suffix_inputs(10)
    coordinator = SimpleNamespace(checkpoint_lengths=lambda tokens, ids: [4, 10, 19])
    assert ctx.checkpoint_lengths(coordinator) == [4, 12, 19]
    at_zero = context([99] * 4 + [2] * 8, mx.zeros((16, 6)), mx.array([[1, 4, 4]]))
    store = Mock()
    store.lookup_exact_cache.return_value = (None, 0)
    at_zero.lookup(store)
    assert all(
        c.kwargs["max_prefix_tokens"] > 0
        for c in store.lookup_exact_cache.call_args_list
    )


@pytest.mark.parametrize(
    "pixels,grid",
    [
        (mx.zeros((15, 6)), mx.array([[1, 4, 4]])),
        (mx.zeros((16, 6)), mx.array([[1, 3, 5]])),
        (mx.zeros((16, 6)), None),
    ],
)
def test_unmatched_media_layout_falls_back(pixels, grid):
    assert context([1, 99, 99, 99, 99, 2], pixels, grid) is None
    assert context([1, 98, 2]) is None


def test_prefix_identity_survives_disk_reopen(tmp_path):
    old = context([1] * 24)
    new = context(
        [1] * 24 + [99] * 4 + [2] * 8, mx.zeros((16, 6)), mx.array([[1, 4, 4]])
    )
    first = manager(DiskBlockStore(tmp_path, "images", max_bytes=1024**2))
    try:
        save(first, old, 23)
    finally:
        first.close()
    second = manager(DiskBlockStore(tmp_path, "images", max_bytes=1024**2))
    try:
        assert new.lookup(second)["prefix_len"] == 23
    finally:
        second.close()


def test_qwen4_exp_uses_the_same_identity_in_its_own_namespace():
    tokens = [1] * 8 + [99] * 4 + [2] * 8
    pixels = mx.zeros((16, 6))
    grid = mx.array([[1, 4, 4]])
    flash = context(tokens, pixels, grid, model_type="qwen4_exp")
    appended = context(
        tokens + [3] * 4 + [99] * 4 + [4] * 8,
        mx.concatenate([pixels, pixels + 1]),
        mx.concatenate([grid, grid]),
        model_type="qwen4_exp",
    )
    store = manager()
    save(store, flash, 19)
    assert appended.lookup(store)["prefix_len"] == 19
    assert flash.prefix_hash(19) != context(tokens, pixels, grid).prefix_hash(19)
    assert context(tokens, pixels, grid).lookup(store) is None


def test_other_model_types_are_not_supported():
    assert context([1, 99, 99, 99, 99, 2], model_type="qwen3_vl") is None


def test_qwen4_exp_inherits_the_position_preserving_embedding():
    assert Qwen4ExpModel.get_input_embeddings is Model.get_input_embeddings


def test_qwen_image_embedding_preserves_full_prompt_positions():
    positions = mx.arange(20)[None, :]
    deltas = mx.array([[4]])
    rope = Mock(side_effect=AssertionError("must not recompute suffix-local positions"))
    tower = Mock(return_value=(mx.ones((4, 2)), None))
    tower.patch_embed.proj.weight.dtype = mx.float32
    model = SimpleNamespace(
        config=SimpleNamespace(image_token_index=99, video_token_index=98),
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=lambda ids: mx.zeros((*ids.shape, 2))),
            get_rope_index=rope,
        ),
        vision_tower=tower,
        merge_input_ids_with_image_features=Model.merge_input_ids_with_image_features,
    )
    result = Model.get_input_embeddings(
        model,
        mx.array([[99] * 4]),
        mx.zeros((16, 6)),
        image_grid_thw=mx.array([[1, 4, 4]]),
        position_ids=positions,
        rope_deltas=deltas,
    )
    assert result.position_ids is positions and result.rope_deltas is deltas
    assert result.inputs_embeds.shape == (1, 4, 2)
    rope.assert_not_called()


@pytest.mark.parametrize(
    "key",
    [
        "position_ids",
        "rope_deltas",
        "cached_image_features",
        "inputs_embeds",
        "prompt_cache",
        "draft_model",
        "max_kv_size",
        "kv_bits",
    ],
)
def test_opaque_overrides_disable_image_prefix_apc(key):
    assert not ImagePrefixContext.supports_overrides(
        {key: object()}, mx.array([[1, 2]]), None
    )


def test_nontrivial_masks_disable_image_prefix_apc():
    ids = mx.array([[1, 2]])
    assert ImagePrefixContext.supports_overrides({}, ids, None)
    assert ImagePrefixContext.supports_overrides({}, ids, mx.ones((1, 2)))
    assert not ImagePrefixContext.supports_overrides({}, ids, mx.array([[0, 1]]))
    assert not ImagePrefixContext.supports_overrides({}, ids, mx.ones((2, 2)))
