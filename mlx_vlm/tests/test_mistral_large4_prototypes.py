from copy import deepcopy

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.mistral_large4 import speculative_prototype as prototype
from mlx_vlm.models.mistral_large4 import speculative_verifier as verifier
from mlx_vlm.models.mistral_large4.block_mla_kernel import block_sparse_mla
from mlx_vlm.models.mistral_large4.grouped_moe_kernel import grouped_qmv
from mlx_vlm.models.mistral_large4.mla_kernel import sparse_mla
from mlx_vlm.models.mistral_large4.moe_kernel import masked_qmv
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear
from mlx_vlm.tests.test_speculative import mistral_dspark_case

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="The verifier prototypes require Metal"
)


def exact(a, b):
    np.testing.assert_array_equal(
        np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "routes",
    [
        [-1] * 9,
        [-1, -1, 0, 0, 0, 1, 3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
)
def test_grouped_routes(dtype, routes):
    mx.random.seed(71)
    module = QuantizedSwitchLinear(
        512, 512, 4, bias=False, group_size=16, bits=4, mode="nvfp4"
    )
    module = ScaledQuantizedSwitchLinear(
        module, mx.array([0.25, 0.5, 0.75, 1]), "weight_scale_2"
    )
    x = mx.random.normal((len(routes), 1, 512)).astype(dtype)
    indices = mx.array(routes, mx.int32)[:, None]
    expected = masked_qmv(x, module, indices)
    for vectors in (1, 2, 4):
        for rows in (1, 2, 4):
            exact(expected, grouped_qmv(x, module, indices, vectors, rows))


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


@pytest.mark.parametrize("prefix", [3, 127, 255])
@pytest.mark.parametrize("batch,length", [(2, 1), (2, 4), (8, 2), (8, 4)])
@pytest.mark.parametrize("ragged", [False, True])
def test_block_model_and_cache(prefix, batch, length, ragged):
    mx.random.seed(71)
    model, _ = mistral_dspark_case()
    model.set_dtype(mx.bfloat16)
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
        with prototype.enabled(block=optimized, compiled=optimized):
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
