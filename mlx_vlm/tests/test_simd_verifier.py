"""Numerical, dispatch, and rollback contracts for opt-in Metal verification."""

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models import simd_mma as mma
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen3_5 import gated_delta as gd
from mlx_vlm.models.qwen3_5.speculative_verifier import (
    Qwen3_5SpeculativeVerifier,
    replay_recurrent_state,
)
from mlx_vlm.models.small_row_attention import small_row_attention
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.tests.test_speculative import dflash_config, generated
from mlx_vlm.tests.test_speculative import language as make_language
from mlx_vlm.utils import get_model_and_args

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


def language(*args, **kwargs):
    model, config = make_language(*args, **kwargs)
    model.use_simd_verifier = True
    return model, config


def test_simd_verifier_requires_explicit_opt_in(monkeypatch):
    model, _ = make_language("qwen", hidden_size=128, intermediate_size=256)
    monkeypatch.setattr(
        Qwen3_5SpeculativeVerifier, "supports_model", lambda self, model: True
    )
    assert not model._get_speculative_verifier().uses_simd
    model.use_simd_verifier = True
    assert model._get_speculative_verifier().uses_simd
    model.use_simd_verifier = False
    assert not model._get_speculative_verifier().uses_simd


def require_matrix():
    if not mma.supports_simd_mma():
        pytest.skip("Device not selected for direct matrix fragments")


def linear(width, output=48, group_size=64, bias=False, bits=4):
    result = nn.QuantizedLinear(
        width, output, group_size=group_size, bits=bits, bias=bias
    )
    result.set_dtype(mx.bfloat16)
    return result


def float_array(x):
    return np.asarray(x.astype(mx.float32))


@pytest.mark.parametrize("batch,rows,width", [(1, 2, 512), (1, 5, 5120), (2, 8, 17408)])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_matrix_against_independent_unpacked_reference(batch, rows, width, group):
    require_matrix()
    mx.random.seed(width + rows)
    layer = linear(width, group_size=group, bias=True)
    x = mx.random.normal((batch, rows * 2, width)).astype(mx.bfloat16)[:, ::2]
    actual = mma.affine_mma(layer, x)
    codes = (
        (np.asarray(layer.weight)[..., None] >> (4 * np.arange(8, dtype=np.uint32)))
        & 15
    ).reshape(48, width)
    weights = codes.astype(np.float32) * np.repeat(
        float_array(layer.scales), group, axis=1
    ) + np.repeat(float_array(layer.biases), group, axis=1)
    reference = float_array(x) @ weights.T
    # The kernel rounds the projection, then adds the ordinary BF16 bias.
    reference = mx.array(reference).astype(mx.bfloat16) + layer.bias
    a, b = float_array(actual), float_array(reference)
    assert np.isfinite(a).all()
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 0.003
    assert np.max(np.abs(a - b)) < 0.02


@pytest.mark.parametrize("outputs", [(48, 80), (48, 16, 80), (16, 48, 80, 64)])
def test_fused_matrix_preserves_projection_boundaries(outputs):
    require_matrix()
    x = mx.random.normal((2, 5, 512)).astype(mx.bfloat16)
    layers = tuple(linear(512, n, bias=True) for n in outputs)
    fused = mma.affine_mmas(layers, x)
    for layer, actual in zip(layers, fused):
        assert mx.array_equal(actual, mma.affine_mma(layer, x)).item()


@pytest.mark.parametrize(
    "rows,dtype,bits,output",
    [
        (9, mx.bfloat16, 4, 48),
        (8, mx.float16, 4, 48),
        (8, mx.bfloat16, 8, 48),
        (8, mx.bfloat16, 4, 17),
    ],
)
def test_matrix_unsupported_shapes_use_fallback(rows, dtype, bits, output):
    layer = linear(512, output, bits=bits)
    x = mx.ones((1, rows, 512), dtype)
    assert mma.affine_mma(layer, x) is None


def test_matrix_disabled_device_fallback(monkeypatch):
    monkeypatch.setattr(mma, "supports_simd_mma", lambda: False)
    assert mma.affine_mma(linear(512), mx.ones((1, 8, 512), mx.bfloat16)) is None


def test_adapter_preserves_parameters_and_shared_module():
    require_matrix()
    shared = linear(512)
    model = nn.Sequential(shared)
    before = dict(tree_flatten(model.parameters()))
    mma.enable_simd_linears(model)
    assert type(shared) is nn.QuantizedLinear
    assert isinstance(model.layers[0], mma.SIMDQuantizedLinear)
    after = dict(tree_flatten(model.parameters()))
    assert before.keys() == after.keys()
    assert all(before[key] is after[key] for key in before)
    adapted = model.layers[0]
    mma.enable_simd_linears(model)
    assert model.layers[0] is adapted


@pytest.mark.parametrize(
    "rows,length,causal", [(5, 64, True), (8, 257, True), (8, 64, False), (7, 64, True)]
)
def test_attention_against_float32_reference(rows, length, causal):
    mx.random.seed(length + rows)
    q = mx.random.normal((2, 4, rows, 32)).astype(mx.bfloat16)
    k = mx.random.normal((2, 2, length * 2, 32)).astype(mx.bfloat16)[:, :, ::2]
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    a = float_array(
        small_row_attention(q, k, v, scale=32**-0.5, mask="causal" if causal else None)
    )
    keys = np.repeat(float_array(k), 2, axis=1)
    vals = np.repeat(float_array(v), 2, axis=1)
    scores = float_array(q) @ keys.swapaxes(-1, -2) * 32**-0.5
    if causal:
        allowed = (
            np.arange(length)[None, :] <= (length - rows + np.arange(rows))[:, None]
        )
        scores = np.where(allowed, scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = probs @ vals
    np.testing.assert_allclose(a, expected, rtol=0.015, atol=0.006)
    if causal:
        changed = v.at[:, :, -1].add(100)
        b = small_row_attention(q, k, changed, scale=32**-0.5, mask="causal")
        assert mx.array_equal(
            mx.array(a[:, :, :-1]), b[:, :, :-1].astype(mx.float32)
        ).item()


def test_attention_array_mask_fallback():
    q = mx.random.normal((1, 2, 8, 32)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, 64, 32)).astype(mx.bfloat16)
    mask = mx.arange(64) < 35
    actual = small_row_attention(q, k, k, scale=0.125, mask=mask)
    expected = mx.fast.scaled_dot_product_attention(q, k, k, scale=0.125, mask=mask)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("lengths", [(0, 0), (3, 3), (8, 8), (0, 8), (3, 7)])
@pytest.mark.parametrize("masked", [False, True])
def test_replay_matches_existing_recurrent_state(lengths, masked):
    mx.random.seed(52)
    q = mx.random.normal((2, 8, 2, 128)).astype(mx.bfloat16) / 16
    k = mx.random.normal(q.shape).astype(mx.bfloat16) / 16
    v = mx.random.normal((2, 8, 4, 16)).astype(mx.bfloat16)
    g = mx.sigmoid(mx.random.normal((2, 8, 4)))
    beta = mx.sigmoid(mx.random.normal(g.shape)).astype(mx.bfloat16)
    initial = mx.random.normal((2, 4, 16, 128))
    mask = (
        mx.array([[True, False, True, True, False, True, True, True]] * 2)
        if masked
        else None
    )
    actual = replay_recurrent_state(k, v, g, beta, initial, lengths, mask)
    keep = mx.arange(8)[None, :] < mx.array(lengths)[:, None]
    if masked:
        keep = keep & mask
    _, expected = gd.gated_delta_kernel(q, k, v, g, beta, initial, keep)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("lengths", [(0, 0), (8, 8), (3, 7), (0, 8)])
def test_cache_replay_commit_paths(lengths):
    cache = ArraysCache(1)
    initial = mx.zeros((2, 3))
    cache[0] = initial
    cache.lengths = mx.array([20, 30])
    calls = []
    transaction = start_speculative_cache([cache], 8)
    final = initial + 8
    cache[0] = final

    def replay(accepted):
        calls.append(tuple(accepted))
        return initial + mx.array(accepted)[:, None]

    cache.record_speculative_replay(0, 8, final, replay)
    transaction.commit(list(lengths))
    assert cache[0].tolist() == [[float(n)] * 3 for n in lengths]
    assert cache.lengths.tolist() == [20 - lengths[0], 30 - lengths[1]]
    assert calls == ([] if lengths in ((0, 0), (8, 8)) else [lengths])
    if lengths == (0, 0):
        assert cache[0] is initial
    if lengths == (8, 8):
        assert cache[0] is final


def test_cache_replay_validation_abort_and_stale():
    cache = ArraysCache(1)
    initial = mx.zeros((1, 2))
    cache[0] = initial
    stale = start_speculative_cache([cache], 8)
    current = start_speculative_cache([cache], 8)
    with pytest.raises(ValueError, match="complete"):
        cache.record_speculative_replay(0, 3, initial, lambda lengths: initial)
    with pytest.raises(TypeError, match="callable"):
        cache.record_speculative_replay(0, 8, initial, None)

    def unexpected(_):
        raise AssertionError("Abort must not replay")

    cache[0] = initial + 8
    cache.record_speculative_replay(0, 8, cache[0], unexpected)
    with pytest.raises(ValueError, match="complete"):
        cache.record_speculative_replay(0, 8, cache[0], unexpected)
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit([3])
    current.abort()
    assert cache[0] is initial and not cache.is_speculating


@pytest.mark.parametrize("accepted", [0, 3, 8])
def test_simd_model_commit_matches_explicit_state_history(accepted, monkeypatch):
    require_matrix()
    model, _ = language(
        "qwen",
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
    )
    model.set_dtype(mx.bfloat16)
    model.eval()
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    other = deepcopy(model)
    caches, reference = model.make_cache(), other.make_cache()
    prefix = mx.array([[1, 3, 5, 7]])
    mx.eval(model(prefix, cache=caches).logits, other(prefix, cache=reference).logits)
    ids = mx.array([[9, 11, 13, 15, 17, 19, 21, 23]])
    output = model(ids, cache=caches, speculative_verify=True)
    with monkeypatch.context() as context:
        context.setattr(
            Qwen3_5SpeculativeVerifier,
            "_gated_delta_update",
            staticmethod(gd.gated_delta_update),
        )
        expected = other(ids, cache=reference, speculative_verify=True)
    assert mx.array_equal(output.logits, expected.logits).item()
    output.gdn_states.commit([accepted])
    expected.gdn_states.commit([accepted])
    for a, b in zip(caches, reference):
        if isinstance(a, ArraysCache):
            assert all(mx.array_equal(x, y).item() for x, y in zip(a.state, b.state))
    next_ids = mx.array([[25]])
    assert mx.array_equal(
        model(next_ids, cache=caches).logits, other(next_ids, cache=reference).logits
    ).item()


@pytest.mark.parametrize(
    "bits,dtype,device,expected",
    [
        (4, mx.bfloat16, True, True),
        (None, mx.bfloat16, True, True),
        (8, mx.bfloat16, True, False),
        (4, mx.float16, True, False),
        (4, mx.bfloat16, False, False),
    ],
)
def test_opt_in_verifier_selection(bits, dtype, device, expected, monkeypatch):
    from mlx_vlm.models.qwen3_5 import speculative_verifier as verifier

    model, _ = language("qwen", hidden_size=128, intermediate_size=256)
    model.set_dtype(dtype)
    model.eval()
    if bits is not None:
        nn.quantize(
            model,
            group_size=32,
            bits=bits,
            class_predicate=lambda _, m: isinstance(m, nn.Linear)
            and m.weight.shape[-1] % 32 == 0,
        )
    monkeypatch.setattr(verifier, "supports_simd_mma", lambda: device)
    assert model._get_speculative_verifier().uses_simd is expected
    model.train()
    assert not model._get_speculative_verifier().uses_simd


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("width,group", [(512, 32), (5120, 64), (17408, 128)])
def test_matrix_rows_match_singletons_in_every_column(rows, width, group):
    require_matrix()
    mx.random.seed(7 + width + rows)
    projection = linear(width, group_size=group, bias=True)
    hidden = mx.random.normal((2, rows * 2, width)).astype(mx.bfloat16)[:, ::2]
    grouped = mma.affine_mma(projection, hidden)
    expected = mx.concatenate(
        [mma.affine_mma(projection, hidden[:, i : i + 1]) for i in range(rows)], 1
    )
    assert mx.array_equal(grouped, expected).item()


@pytest.mark.parametrize(
    "length",
    [31, 1023, 1024, 1025, 1031, 8192, 16383, 16384, 16385, 65535, 65536, 65537],
)
@pytest.mark.parametrize("rows", [2, 5, 8])
def test_causal_attention_crosses_dispatch_boundaries_exactly(length, rows):
    mx.random.seed(length + rows)
    q = mx.random.normal((1, rows, 6, 256)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    k = mx.random.normal((1, 1, length + 3, 256)).astype(mx.bfloat16)[:, :, :length]
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    output = small_row_attention(q, k, v, scale=1 / 16, mask="causal", exact=True)
    expected = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1],
                k[:, :, : length - rows + i + 1],
                v[:, :, : length - rows + i + 1],
                scale=1 / 16,
            )
            for i in range(rows)
        ],
        axis=2,
    )
    assert mx.array_equal(output, expected).item()


@pytest.mark.parametrize("kind", ["mask", "sinks", "ordinary"])
def test_exact_attention_handles_other_dispatches(kind):
    q = mx.random.normal((2, 4, 8, 64)).astype(mx.bfloat16)
    k = mx.random.normal((2, 2, 1040, 64)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    mask = (
        mx.arange(1040)[None, :] <= (1032 + mx.arange(8))[:, None]
        if kind == "mask"
        else None
    )
    sinks = mx.zeros((4,), dtype=mx.bfloat16) if kind == "sinks" else None
    result = small_row_attention(
        q, k, v, scale=1 / 8, mask=mask, sinks=sinks, exact=True
    )
    expected = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1],
                k,
                v,
                scale=1 / 8,
                mask=mask[i : i + 1] if mask is not None else None,
                sinks=sinks,
            )
            for i in range(8)
        ],
        axis=2,
    )
    assert mx.array_equal(result, expected).item()


@pytest.mark.parametrize(
    "prefix_length,rows,accepted",
    [(4, 2, 0), (4, 5, 3), (1020, 5, 2), (1020, 8, 8), (2044, 8, 3), (4, 16, 11)],
)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("quantized", [False, True])
def test_simd_verification_and_rollback_match_own_serial(
    prefix_length, rows, accepted, batch, quantized
):
    require_matrix()
    mx.random.seed(913)
    model, _ = language(
        "qwen",
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        tie_word_embeddings=False,
    )
    model.set_dtype(mx.bfloat16)
    model.eval()
    if quantized:
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
    cache = model.make_cache()
    prefix = (mx.arange(prefix_length)[None] + mx.arange(batch)[:, None]) % 24
    mx.eval(model(prefix, cache=cache).logits)
    reference = deepcopy(cache)
    ids = (mx.arange(rows)[None] * 2 + 1 + mx.arange(batch)[:, None]) % 24
    options = dict(return_hidden=True, capture_layer_ids=[0, 1, 2, 3])
    grouped = model(ids, cache=cache, speculative_verify=True, **options)
    mx.eval(grouped.logits)
    steps, retained = [], deepcopy(reference)
    for i in range(rows):
        step = model(ids[:, i : i + 1], cache=reference, **options)
        mx.eval(step.logits)
        steps.append(step)
        if i + 1 == accepted:
            retained = deepcopy(reference)
    assert mx.array_equal(
        grouped.logits, mx.concatenate([s.logits for s in steps], 1)
    ).item()
    for layer_index, hidden in enumerate(grouped.hidden_states):
        assert mx.array_equal(
            hidden, mx.concatenate([s.hidden_states[layer_index] for s in steps], 1)
        ).item()
    grouped.gdn_states.commit([accepted] * batch)
    for actual, expected in zip(cache, retained):
        if isinstance(actual, ArraysCache):
            assert all(
                mx.array_equal(a, b).item()
                for a, b in zip(actual.state, expected.state)
            )
    probe = mx.full((batch, 1), 25, dtype=mx.int32)
    assert mx.array_equal(
        model(probe, cache=cache).logits, model(probe, cache=retained).logits
    ).item()


@pytest.mark.parametrize("temperature,seed", [(0, 41), (0.7, 41), (1.0, 99)])
@pytest.mark.parametrize("quantized", [False, True])
def test_simd_draft_and_serial_generation_share_tokens(temperature, seed, quantized):
    require_matrix()
    mx.random.seed(37)
    model, _ = language(
        "qwen",
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        tie_word_embeddings=False,
    )
    model.set_dtype(mx.bfloat16)
    model.eval()
    if quantized:
        nn.quantize(
            model,
            group_size=32,
            bits=4,
            class_predicate=lambda _, m: isinstance(m, nn.Linear)
            and m.weight.shape[-1] % 32 == 0,
        )

    def embeddings(input_ids, pixel_values=None, mask=None, **kwargs):
        positions, deltas = model.get_rope_index(input_ids, attention_mask=mask)
        return InputEmbeddingsFeatures(
            inputs_embeds=model.model.embed_tokens(input_ids),
            position_ids=positions,
            rope_deltas=deltas,
        )

    target = SimpleNamespace(language_model=model, get_input_embeddings=embeddings)
    settings = dflash_config("dflash2")
    settings.update(hidden_size=128, intermediate_size=256)
    arch, _ = get_model_and_args(settings)
    draft = arch.Model(arch.ModelConfig.from_dict(settings))
    draft.set_dtype(mx.bfloat16)
    draft.eval()
    if quantized:
        nn.quantize(
            draft,
            group_size=32,
            bits=4,
            class_predicate=lambda _, m: isinstance(m, nn.Linear)
            and m.weight.shape[-1] % 32 == 0,
        )
    expected = generated(target, temperature=temperature, seed=seed)
    assert generated(target, draft, temperature=temperature, seed=seed) == expected


def test_simd_keeps_embedding_prefill_on_original_path(monkeypatch):
    require_matrix()
    model, _ = language(
        "qwen",
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        tie_word_embeddings=False,
    )
    model.set_dtype(mx.bfloat16)
    model.eval()
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda _, m: isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 32 == 0,
    )
    regular = deepcopy(model)
    cache, reference = model.make_cache(), regular.make_cache()
    prefix = mx.array([[1, 3, 5]])
    mx.eval(model(prefix, cache=cache).logits, regular(prefix, cache=reference).logits)
    token = mx.array([[7]])
    embeddings = model.model.embed_tokens(token)

    def unexpected(*args, **kwargs):
        raise AssertionError("Embedding prefill must not enter the SIMD verifier")

    monkeypatch.setattr(Qwen3_5SpeculativeVerifier, "__call__", unexpected)
    actual = model(token, cache=cache, inputs_embeds=embeddings, return_hidden=True)
    expected = regular(
        token, cache=reference, inputs_embeds=embeddings, return_hidden=True
    )
    assert mx.array_equal(actual.logits, expected.logits).item()
    assert mx.array_equal(actual.hidden_states[-1], expected.hidden_states[-1]).item()


@pytest.mark.parametrize(
    "batch,rows,width", [(1, 1, 136), (2, 5, 512), (1, 8, 5120), (1, 16, 17408)]
)
@pytest.mark.parametrize("sizes", [(48,), (48, 80, 16)])
def test_dense_matrix_accuracy_fusion_and_row_invariance(batch, rows, width, sizes):
    require_matrix()
    mx.random.seed(width + rows)
    layers = []
    for n in sizes:
        layer = nn.Linear(width, n, bias=True)
        layer.set_dtype(mx.bfloat16)
        # Strided weights and inputs exercise the Metal call's contiguity handling.
        layer.weight = mx.random.normal((width, n)).astype(mx.bfloat16).T * 0.01
        layers.append(layer)
    x = mx.random.normal((batch, rows * 2, width)).astype(mx.bfloat16)[:, ::2]
    outputs = mma.simd_mmas(layers, x, chunked=True)
    for layer, output in zip(layers, outputs):
        reference = (
            mx.array(
                float_array(x).astype(np.float64)
                @ float_array(layer.weight).astype(np.float64).T
            ).astype(mx.bfloat16)
            + layer.bias
        )
        assert output.dtype == mx.bfloat16
        # BF16 rounding dominates the relative error; compare to a float64 oracle.
        error = np.linalg.norm(float_array(output) - float_array(reference))
        assert error / max(np.linalg.norm(float_array(reference)), 1e-12) < 0.003
        single = mx.concatenate(
            [mma.dense_mmas((layer,), x[:, i : i + 1])[0] for i in range(rows)], axis=1
        )
        assert mx.array_equal(output, single).item()
        assert mx.array_equal(
            output, mma.dense_mmas((layer,), x, chunked=True)[0]
        ).item()


@pytest.mark.parametrize(
    "dtype,width,output",
    [
        (mx.float16, 128, 48),
        (mx.float32, 128, 48),
        (mx.bfloat16, 127, 48),
        (mx.bfloat16, 128, 17),
    ],
)
def test_dense_matrix_unsupported_shape_fallback(dtype, width, output):
    layer = nn.Linear(width, output, bias=False)
    layer.set_dtype(dtype)
    assert mma.simd_mmas((layer,), mx.ones((1, 5, width), dtype)) is None


def test_dense_matrix_preserves_bfloat16_exponent_range():
    require_matrix()
    layer = nn.Linear(128, 16, bias=False)
    layer.weight = mx.full((16, 128), 2.0**80, mx.bfloat16)
    x = mx.full((2, 5, 128), 2.0**-80, mx.bfloat16)
    output = mma.dense_mmas((layer,), x)[0]
    assert mx.array_equal(output, mx.full((2, 5, 16), 128, mx.bfloat16)).item()


def test_dense_adapter_preserves_shared_parameters_and_fallback(monkeypatch):
    require_matrix()
    original = nn.Linear(128, 48, bias=True)
    original.set_dtype(mx.bfloat16)
    model = nn.Sequential(original)
    before = dict(tree_flatten(model.parameters()))
    mma.enable_simd_linears(model)
    assert type(original) is nn.Linear
    assert isinstance(model.layers[0], mma.SIMDLinear)
    after = dict(tree_flatten(model.parameters()))
    assert before.keys() == after.keys()
    assert all(before[key] is after[key] for key in before)
    x = mx.ones((1, 2, 128), mx.bfloat16)
    monkeypatch.setattr(mma, "supports_simd_mma", lambda: False)
    assert mx.array_equal(model.layers[0](x), original(x)).item()


@pytest.mark.parametrize("length", [31, 1023, 1024, 1025, 16383, 16384, 16385])
@pytest.mark.parametrize("rows", [2, 5, 8])
@pytest.mark.parametrize("causal", [False, True])
def test_minicpm_attention_qualification_and_long_context_fallback(
    length, rows, causal
):
    mx.random.seed(length + rows)
    q = mx.random.normal((1, 8, rows, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, length, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    output = small_row_attention(
        q, k, v, scale=1 / 16, mask="causal" if causal else None, exact=True
    )
    expected = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1],
                k[:, :, : length - rows + i + 1] if causal else k,
                v[:, :, : length - rows + i + 1] if causal else v,
                scale=1 / 16,
            )
            for i in range(rows)
        ],
        axis=2,
    )
    assert mx.array_equal(output, expected).item()
