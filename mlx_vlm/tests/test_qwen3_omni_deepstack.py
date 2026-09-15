"""Tests for Qwen3-Omni deepstack prefill and batching."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate import ar
from mlx_vlm.models.cache import BatchKVCache, make_prompt_cache
from mlx_vlm.models.qwen3_omni_moe.language import (
    LanguageModel,
    expand_deepstack_visual_embeds,
)
from mlx_vlm.tests.test_qwen3_omni_moe import (
    IMAGE_TOKEN,
    VISION_END,
    VISION_START,
    _image_inputs,
    _tiny_vision_model,
)


def _host(a):
    return np.array(a.astype(mx.float32))


def test_vision_without_deepstack_returns_no_residuals():
    model = _tiny_vision_model()
    model.thinker.vision_tower.deepstack_visual_indexes = []
    model.thinker.vision_tower.deepstack_merger_list = []
    ids, pixels, grid = _image_inputs()
    features = model.get_input_embeddings(ids, pixel_values=pixels, image_grid_thw=grid)
    assert features.deepstack_visual_embeds is None
    logits = model.language_model(ids, **features.to_dict()).logits
    assert bool(mx.all(mx.isfinite(logits)).item())


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_deepstack_expansion_preserves_visual_order_and_zeros_text(expanded, dtype):
    mask = np.array(
        [[0, 1, 1, 0, 1, 0, 0, 0, 0], [0] * 9, [1, 0, 1, 1, 1, 1, 0, 1, 0]],
        dtype=bool,
    )
    # Label every visual position with a unique ID; layers have distinct IDs.
    visual_ids = np.arange(mask.size).reshape(mask.shape)[mask]
    features = [mx.array(visual_ids[:, None] + layer * 100) for layer in range(3)]
    residuals = expand_deepstack_visual_embeds(
        mx.array(mask[..., None] if expanded else mask), features, dtype
    )
    assert residuals.shape == (*mask.shape, 3, 1)
    assert residuals.dtype == dtype
    for layer in range(3):
        expected = np.zeros(mask.shape)
        expected[mask] = visual_ids + layer * 100
        np.testing.assert_array_equal(_host(residuals[:, :, layer, 0]), expected)


@pytest.mark.parametrize("split", [3, 4, 5, 6, 7])
def test_omni_chunked_prefill_matches_full_prompt(split):
    mx.random.seed(17)
    model = _tiny_vision_model()
    ids, pixels, grid = _image_inputs()
    kw = model.get_input_embeddings(
        ids, pixel_values=pixels, image_grid_thw=grid
    ).to_dict()
    embeds = kw.pop("inputs_embeds")
    lm = model.language_model
    full = lm(ids, inputs_embeds=embeds, cache=make_prompt_cache(lm), **kw).logits
    cache = make_prompt_cache(lm)
    chunks = [
        lm(ids[:, :split], inputs_embeds=embeds[:, :split], cache=cache, **kw).logits,
        lm(ids[:, split:], inputs_embeds=embeds[:, split:], cache=cache, **kw).logits,
    ]
    np.testing.assert_allclose(
        _host(mx.concatenate(chunks, axis=1)), _host(full), rtol=1e-4, atol=1e-4
    )
    # Reusing the full visual kwargs during text decode must inject no features.
    with patch.object(
        lm.model, "_deepstack_process", wraps=lm.model._deepstack_process
    ) as inject:
        lm(mx.array([[5]]), cache=cache, **kw)
        inject.assert_not_called()


def _padded_image_batch(model):
    rows = [
        [1] * prefix + [VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END, 2, 3, 4]
        for prefix in [8, 3]
    ]
    lengths = [len(row) for row in rows]
    padding = [max(lengths) - length for length in lengths]
    ids = mx.array([[0] * p + row for p, row in zip(padding, rows)])
    mask = mx.array([[0] * p + [1] * len(row) for p, row in zip(padding, rows)])
    kw = model.get_input_embeddings(
        ids,
        pixel_values=mx.random.normal((32, 24)),
        image_grid_thw=mx.array([[1, 4, 4], [1, 4, 4]]),
        mask=mask,
    ).to_dict()
    return ids, mask, padding, kw


@pytest.mark.parametrize("expanded", [False, True])
def test_omni_full_masks_use_physical_padded_cache_columns(expanded):
    mx.random.seed(17)
    model = _tiny_vision_model()
    ids, _, padding, kw = _padded_image_batch(model)
    embeds = kw.pop("inputs_embeds")
    positions = kw.pop("position_ids")
    if expanded:
        kw["visual_pos_masks"] = kw["visual_pos_masks"][..., None]
    lm = model.language_model

    def caches():
        return [BatchKVCache(padding) for _ in lm.layers]

    full = lm(
        ids, inputs_embeds=embeds, position_ids=positions, cache=caches(), **kw
    ).logits
    cache = caches()
    chunks = [
        lm(
            ids[:, start:stop],
            inputs_embeds=embeds[:, start:stop],
            position_ids=positions[..., start:stop],
            cache=cache,
            **kw,
        ).logits
        for start, stop in [(0, 11), (11, ids.shape[1])]
    ]
    got = mx.concatenate(chunks, axis=1)
    for row, pad in enumerate(padding):
        np.testing.assert_allclose(
            _host(got[row, pad:]), _host(full[row, pad:]), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("batched", [False, True])
def test_omni_prompt_scheduler_keeps_mask_and_features_paired(batched):
    mx.random.seed(17)
    model = _tiny_vision_model()
    lm = model.language_model
    if batched:
        ids, attention_mask, padding, kw = _padded_image_batch(model)
    else:
        ids, pixels, grid = _image_inputs()
        kw = model.get_input_embeddings(
            ids, pixel_values=pixels, image_grid_thw=grid
        ).to_dict()
        attention_mask = mx.ones_like(ids)
        padding = [0]
    full = lm(ids, cache=[BatchKVCache(padding) for _ in lm.layers], **kw).logits
    batch_size = ids.shape[0]
    rows = ar._split_prompt_kwargs_per_row(kw, batch_size)
    prompts, rows = ar._unpad_batch_prompts(ids, attention_mask, rows)
    embeds, prompt_kwargs = ar._merge_prefill_prompt_kwargs(rows, prompts)
    batch = ar.PromptProcessingBatch(
        model=lm,
        uids=list(range(batch_size)),
        input_ids=prompts,
        max_tokens=[1] * batch_size,
        inputs_embeds=embeds,
        prompt_kwargs=prompt_kwargs,
        prefill_step_size=5,
    )
    outputs = []
    original = LanguageModel.__call__

    def recording(self, *args, **kwargs):
        # Each call receives the residuals for its input columns, with zeros
        # at text and padding positions.
        start = sum(o.shape[1] for o in outputs)
        stop = start + args[0].shape[1]
        np.testing.assert_array_equal(
            _host(kwargs["deepstack_visual_embeds"]),
            _host(kw["deepstack_visual_embeds"][:, start:stop]),
        )
        result = original(self, *args, **kwargs)
        outputs.append(result.logits)
        return result

    with patch.object(LanguageModel, "__call__", recording):
        while batch.needs_processing():
            batch.prompt_step()
        batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    got = mx.concatenate(outputs, axis=1)
    for row, pad in enumerate(padding):
        # Compare multi-token logits directly; use cosine similarity for the
        # final token to allow rounding differences between attention kernels.
        np.testing.assert_allclose(
            _host(got[row, pad:-1]), _host(full[row, pad:-1]), rtol=1e-4, atol=1e-4
        )
        a, b = _host(got[row, -1]), _host(full[row, -1])
        assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.999


def test_cold_batch_merge_keeps_features_when_text_row_is_first():
    model = _tiny_vision_model()
    ids, pixels, grid = _image_inputs()
    text = model.get_input_embeddings(mx.array([[1, 2]])).to_dict()
    image = model.get_input_embeddings(
        ids, pixel_values=pixels, image_grid_thw=grid
    ).to_dict()
    assert text["deepstack_visual_embeds"].shape == (1, 2, 2, 16)
    assert not mx.any(text["deepstack_visual_embeds"]).item()
    embeds, kw = ar._merge_prefill_prompt_kwargs(
        [text, image], [[1, 2], ids[0].tolist()]
    )
    assert kw["deepstack_visual_embeds"].shape == (2, ids.shape[1], 2, 16)
    assert not mx.any(kw["deepstack_visual_embeds"][0]).item()
    np.testing.assert_array_equal(
        _host(kw["deepstack_visual_embeds"][1:]),
        _host(image["deepstack_visual_embeds"]),
    )
    lm = model.language_model
    reference = lm(ids, cache=make_prompt_cache(lm), **image).logits[:, -1]
    batch = ar.PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=[[1, 2], ids[0].tolist()],
        max_tokens=[1, 1],
        inputs_embeds=embeds,
        prompt_kwargs=kw,
        prefill_step_size=5,
    )
    while batch.needs_processing():
        batch.prompt_step()
    result = batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    assert result._next_tokens[1:].tolist() == mx.argmax(reference, axis=-1).tolist()


def _mixed_batch(model, rows, prompts, prefix, warm_cache=None):
    bg = object.__new__(ar.BatchGenerator)
    bg.model = model
    bg.apc_manager = object()
    bg.apc = SimpleNamespace(
        merge_rows=lambda *a, **k: (
            ar._apc.make_warm_batch_exact_cache_multi(
                [warm_cache, make_prompt_cache(model)], [prefix, 0]
            )
            if warm_cache is not None
            else ([], prefix)
        ),
        checkpoint_len=lambda *a: 0,
        checkpoint_lengths=lambda *a: [],
    )
    bg.apc_mode = "exact"
    bg.prefill_step_size = 2
    bg.kv_bits = None
    bg.kv_group_size = 64
    bg.kv_quant_scheme = "affine"
    bg._wire_stack = None
    sequences = [
        (i, ids, 1, kw, [], None) for i, (ids, kw) in enumerate(zip(prompts, rows))
    ]
    with patch.object(
        bg, "_apc_pick_for", side_effect=[{"prefix_len": prefix, "extra_hash": 0}, None]
    ):
        batch = bg._build_mixed_prompt_batch(sequences)
    # Limit this fixture to cache restoration and prompt processing.
    batch._apc_meta = []
    batch._apc_manager = None
    return batch


def test_mixed_cached_prefix_slices_visual_features_before_merging():
    # The warm row's cached prefix contains one visual token; the cold row
    # contains two visual tokens.
    calls = []

    def model(ids, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(logits=mx.zeros((*ids.shape, 4)))

    rows = [
        {
            "inputs_embeds": mx.zeros((1, 6, 4)),
            "deepstack_visual_embeds": mx.array([0, 10, 0, 11, 12, 0]).reshape(
                1, 6, 1, 1
            ),
        },
        {
            "inputs_embeds": mx.zeros((1, 4, 4)),
            "deepstack_visual_embeds": mx.array([20, 0, 21, 0]).reshape(1, 4, 1, 1),
        },
    ]
    batch = _mixed_batch(model, rows, [list(range(6)), list(range(4))], 3)
    assert batch._prompt_kwargs["deepstack_visual_embeds"][:, :, 0, 0].tolist() == [
        [11, 12, 0, 0],
        [20, 0, 21, 0],
    ]
    while batch.needs_processing():
        batch.prompt_step()
    batch.generate(lambda x: mx.argmax(x, axis=-1), lambda _: False)
    assert [c["deepstack_visual_embeds"][:, :, 0, 0].tolist() for c in calls] == [
        [[11, 12], [20, 0]],
        [[0], [21]],
        [[0], [0]],
    ]


@pytest.mark.parametrize("prefix", [2, 11, 14])
def test_omni_mixed_warm_and_cold_cache_matches_full_prompt(prefix):
    mx.random.seed(17)
    model = _tiny_vision_model()
    ids, attention_mask, _, kw = _padded_image_batch(model)
    lm = model.language_model
    rows = ar._split_prompt_kwargs_per_row(kw, 2)
    prompts, rows = ar._unpad_batch_prompts(ids, attention_mask, rows)
    reference = mx.concatenate(
        [
            lm(mx.array([p]), cache=make_prompt_cache(lm), **r).logits[:, -1]
            for p, r in zip(prompts, rows)
        ]
    )
    warm_cache = make_prompt_cache(lm)
    warm_kwargs = dict(rows[0])
    warm_kwargs["inputs_embeds"] = warm_kwargs["inputs_embeds"][:, :prefix]
    lm(mx.array([prompts[0][:prefix]]), cache=warm_cache, **warm_kwargs)
    batch = _mixed_batch(lm, rows, prompts, prefix, warm_cache)
    while batch.needs_processing():
        batch.prompt_step()
    sampled_logprobs = []

    def sample(logprobs):
        sampled_logprobs.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    result = batch.generate(sample, lambda _: False)
    assert result._next_tokens.tolist() == mx.argmax(reference, axis=-1).tolist()
    expected = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        _host(sampled_logprobs[0]), _host(expected), atol=0.005, rtol=0.001
    )
