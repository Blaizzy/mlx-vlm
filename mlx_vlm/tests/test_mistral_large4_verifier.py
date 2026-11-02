from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.mistral_large4 import speculative_verifier as verifier
from mlx_vlm.models.mistral_large4 import verify_kernel, verify_ops
from mlx_vlm.models.mistral_large4.block_mla_kernel import block_sparse_mla
from mlx_vlm.models.mistral_large4.config import TextConfig
from mlx_vlm.models.mistral_large4.language import MoE
from mlx_vlm.models.mistral_large4.mla_kernel import sparse_mla
from mlx_vlm.models.mistral_large4.moe_kernel import masked_qmv
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear
from mlx_vlm.tests.test_speculative import mistral_dspark_case

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="The verifier kernels require Metal"
)


def exact(a, b):
    np.testing.assert_array_equal(
        np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("context", [127, 129, 257, 2051])
@pytest.mark.parametrize("rope", [0, 64])
def test_sparse_block_queries(dtype, context, rope):
    mx.random.seed(71)
    b, h, length, d = 3, 4, 4, 512
    q = mx.random.normal((b, h, length, d)).astype(dtype)
    qr = mx.random.normal((b, h, length, rope)).astype(dtype)
    kv = mx.random.normal((b, 1, context, d)).astype(dtype)
    kr = mx.random.normal((b, 1, context, rope)).astype(dtype)
    counts = []
    selected = []
    masks = []
    expected = []
    for t in range(length):
        end = context - length + t + 1
        k = min(end, 2048)
        indices = mx.sort(
            mx.argsort(mx.random.uniform(shape=(b, end)), axis=-1)[:, :k], axis=-1
        )
        # Unequal row lengths plus one fully masked row.
        lower = mx.array([0, 5, context])[:, None, None, None]
        upper = mx.array([end, end - 2, end])[:, None, None, None]
        mask = (mx.arange(context)[None, None, None, :] >= lower) & (
            mx.arange(context)[None, None, None, :] < upper
        )
        expected.append(
            sparse_mla(
                q[:, :, t : t + 1],
                qr[:, :, t : t + 1],
                kv[:, :, :end],
                kr[:, :, :end],
                indices,
                mask[..., :end],
                0.125,
                chunk=128,
            )
        )
        masks.append(mask)
        selected.append(indices)
        counts.append(k)
    width = max(counts)
    indices = mx.stack(
        [mx.pad(i, [(0, 0), (0, width - i.shape[-1])]) for i in selected], axis=1
    )
    actual = block_sparse_mla(
        q, qr, kv, kr, indices, mx.concatenate(masks, axis=2), 0.125, counts, chunk=128
    )
    exact(mx.concatenate(expected, axis=2), actual)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "batch,length", [(2, 1), (3, 4), (5, 3), (8, 4), (13, 2), (15, 4)]
)
def test_dense_rows(dtype, batch, length):
    mx.random.seed(79)
    w = mx.random.normal((6144, 512)).astype(dtype)
    x = mx.random.normal((batch, length, 512)).astype(dtype)
    actual = verify_kernel.native_wide_linear(w, x)
    if actual is None:
        pytest.skip("Wide dense verification requires Apple GPU generation 15+")
    reference = mx.concatenate(
        [mx.contiguous(x[:, t : t + 1]) @ w.T for t in range(length)], axis=1
    )
    exact(reference, actual)
    parts = [
        verify_kernel.native_wide_linear(w[:3072], x),
        verify_kernel.native_wide_linear(w[3072:], x),
    ]
    exact(reference, mx.concatenate(parts, axis=-1))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch,length", [(1, 1), (2, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize("start", [0, 4])
def test_compiled_local_experts(dtype, batch, length, start):
    mx.random.seed(79)
    ffn = MoE(
        TextConfig(
            hidden_size=512,
            moe_intermediate_size=512,
            n_routed_experts=8,
            expert_start=start,
            expert_end=start + 4,
            n_shared_experts=0,
            nvfp4=True,
        )
    )
    for name in ("w1", "w3", "w2"):
        setattr(
            ffn.experts,
            name,
            ScaledQuantizedSwitchLinear(
                getattr(ffn.experts, name),
                mx.array([0.25, 0.5, 1.0, 0.75]),
                "weight_scale_2",
            ),
        )
    ffn.eval()
    ffn.gate.weight = ffn.gate.weight.astype(dtype)
    x = mx.random.normal((batch, length, 512)).astype(dtype)
    indices, weights = verifier.tokenwise(ffn.route, x)
    local = mx.where(
        (indices >= start) & (indices < start + 4), indices.astype(mx.int32) - start, -1
    )
    order = mx.argsort(local.reshape(-1))
    inverse = mx.argsort(order)
    rx = x.reshape(-1, 512)[order // 4, None]
    ri = local.reshape(-1)[order, None]
    gate = masked_qmv(rx, ffn.experts.w1, ri)
    up = masked_qmv(rx, ffn.experts.w3, ri)
    reference = masked_qmv(nn.silu(gate) * up, ffn.experts.w2, ri)[inverse].reshape(
        batch, length, 4, 512
    )
    modules = [ffn.experts.w1, ffn.experts.w3, ffn.experts.w2]
    output, actual_weights = verify_ops.local_experts(
        x,
        ffn.gate.weight,
        [m.weight for m in modules],
        [m.scales for m in modules],
        [verify_ops.expert_factor(m) for m in modules],
        start,
        1.0,
    )
    exact(weights, actual_weights)
    exact(reference, output)


@pytest.mark.parametrize(
    "prefix,topk", [(3, 2048), (127, 2048), (255, 2048), (127, 128)]
)
@pytest.mark.parametrize("batch,length", [(2, 1), (2, 4), (8, 2), (8, 4)])
@pytest.mark.parametrize("ragged", [False, True])
def test_cache_updates(prefix, topk, batch, length, ragged):
    mx.random.seed(79)
    model, _ = mistral_dspark_case()
    model.set_dtype(mx.bfloat16)
    for layer in model.layers:
        if layer.attention.indexer is not None:
            layer.attention.indexer.topk = topk
    initial = [c.to_batch([i % 3 for i in range(batch)]) for c in model.make_cache()]
    ids = mx.random.randint(1, 60, (batch, prefix))
    mx.eval(model(ids, cache=initial).logits)
    if ragged:
        for c in initial:
            c._ragged = True
            c.offset -= mx.arange(batch) % 3
    ids = mx.random.randint(1, 60, (batch, length))
    reference_cache, actual_cache = deepcopy(initial), deepcopy(initial)
    ordinary = [
        model(ids[:, t : t + 1], cache=reference_cache, capture_layer_ids=[0, 1, 2, 3])
        for t in range(length)
    ]
    expected_hidden = [
        mx.concatenate([r.hidden_states[i] for r in ordinary], axis=1) for i in range(4)
    ]
    expected_logits = mx.concatenate([r.logits for r in ordinary], axis=1)
    hidden, norm, transaction = verifier.verify_hidden(
        model, ids, actual_cache, [0, 1, 2, 3]
    )
    logits = verifier.verify_logits(model, norm)
    for actual, expected in zip(hidden, expected_hidden):
        exact(actual, expected)
    exact(logits, expected_logits)
    expected_state = [
        v
        for _, v in tree_flatten([c.state for c in reference_cache])
        if isinstance(v, mx.array)
    ]
    actual_state = [
        v
        for _, v in tree_flatten([c.state for c in actual_cache])
        if isinstance(v, mx.array)
    ]
    for actual, expected in zip(actual_state, expected_state):
        exact(actual, expected)
    transaction.abort()


def test_expert_scales_refresh_without_serializing_runtime_state():
    model, _ = mistral_dspark_case()
    ffn = MoE(
        TextConfig(
            hidden_size=512,
            moe_intermediate_size=512,
            n_routed_experts=8,
            expert_start=0,
            expert_end=4,
            n_shared_experts=0,
            nvfp4=True,
        )
    )
    for name in ("w1", "w3", "w2"):
        setattr(
            ffn.experts,
            name,
            ScaledQuantizedSwitchLinear(
                getattr(ffn.experts, name),
                mx.array([0.25, 0.5, 1.0, 0.75]),
                "weight_scale_2",
            ),
        )
    ffn.eval()
    ffn.group = object()
    model.layers[0].feed_forward = ffn
    first = verifier._expert_parameters(model, mx.bfloat16)
    second = verifier._expert_parameters(model, mx.bfloat16)
    assert all(a is b for a, b in zip(first[id(ffn)][2], second[id(ffn)][2]))
    assert "_verification_scales" not in model.parameters()
    ffn.experts.w1.weight_scale_2 = mx.array([0.5, 1.0, 2.0, 1.5])
    third = verifier._expert_parameters(model, mx.bfloat16)
    exact(third[id(ffn)][2][0], first[id(ffn)][2][0] * 2)
    assert third[id(ffn)][2][0] is not first[id(ffn)][2][0]
    assert third[id(ffn)][2][1] is first[id(ffn)][2][1]
    ffn.experts.w1.weight = mx.zeros_like(ffn.experts.w1.weight)
    fourth = verifier._expert_parameters(model, mx.bfloat16)
    assert fourth[id(ffn)][0][0] is ffn.experts.w1.weight
    assert fourth[id(ffn)][2][0] is third[id(ffn)][2][0]


def test_cpu_verification_uses_portable_fallback():
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        model, _ = mistral_dspark_case()
        cache = [c.to_batch([0, 0]) for c in model.make_cache()]
        mx.eval(model(mx.array([[1, 2, 3], [3, 2, 1]]), cache=cache).logits)
        expected_cache = deepcopy(cache)
        ids = mx.array([[4, 5], [5, 4]])
        expected = mx.concatenate(
            [model(ids[:, t : t + 1], cache=expected_cache).logits for t in range(2)],
            axis=1,
        )
        result = model(ids, cache=cache, speculative_verify=True)
        np.testing.assert_allclose(
            np.array(result.logits), np.array(expected), atol=2e-5, rtol=2e-5
        )
        result.gdn_states.abort()
    finally:
        mx.set_default_device(previous)
