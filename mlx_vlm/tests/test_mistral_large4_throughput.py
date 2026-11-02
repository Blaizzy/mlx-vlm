from copy import deepcopy

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.mistral_large4 import speculative_prototype as baseline
from mlx_vlm.models.mistral_large4 import speculative_verifier as verifier
from mlx_vlm.models.mistral_large4 import throughput_prototype as candidate
from mlx_vlm.models.mistral_large4 import verify_kernel
from mlx_vlm.models.mistral_large4.config import TextConfig
from mlx_vlm.models.mistral_large4.language import MoE
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear
from mlx_vlm.tests.test_speculative import mistral_dspark_case

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="The verifier prototypes require Metal"
)


def exact(a, b):
    np.testing.assert_array_equal(
        np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "batch,length", [(2, 1), (3, 4), (5, 3), (8, 4), (13, 2), (15, 4)]
)
def test_dense_rows(dtype, batch, length):
    mx.random.seed(79)
    w = mx.random.normal((6144, 512)).astype(dtype)
    x = mx.random.normal((batch, length, 512)).astype(dtype)
    reference = verify_kernel.native_wide_linear(w, x)
    if reference is None:
        pytest.skip("Wide dense verification requires Apple GPU generation 15+")
    exact(reference, candidate._tuned_linear(w, x))
    parts = [candidate._tuned_linear(w[:3072], x), candidate._tuned_linear(w[3072:], x)]
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
    with baseline.enabled(block=True, compiled=True):
        reference = baseline.expert_mlp(ffn.experts, rx, ri)[inverse].reshape(
            batch, length, 4, 512
        )
    modules = [ffn.experts.w1, ffn.experts.w3, ffn.experts.w2]
    output, actual_weights = candidate._local_experts(
        x,
        ffn.gate.weight,
        [m.weight for m in modules],
        [m.scales for m in modules],
        [candidate._factor(m) for m in modules],
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
    results = []
    for optimized in (False, True):
        cache = deepcopy(initial)
        context = (
            candidate.enabled(model=model, compile_experts=True)
            if optimized
            else baseline.enabled(block=True, compiled=True)
        )
        with context:
            hidden, norm, transaction = verifier.verify_hidden(
                model, ids, cache, [0, 1, 2, 3]
            )
            logits = verifier.verify_logits(model, norm)
            state = [
                v
                for _, v in tree_flatten([c.state for c in cache])
                if isinstance(v, mx.array)
            ]
            mx.eval(hidden, logits, state)
        results.append((hidden, logits, state))
        transaction.abort()
    for a, b in zip(results[0][0], results[1][0]):
        exact(a, b)
    exact(results[0][1], results[1][1])
    for a, b in zip(results[0][2], results[1][2]):
        exact(a, b)
