import mlx.core as mx
import pytest

from mlx_vlm.models.sparse_attention import indexed_sparse_attention


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal kernels")
def test_indexed_attention_is_batch_invariant_with_physical_capacity():
    mx.random.seed(925)
    query = mx.random.normal((1, 2, 1, 32)).astype(mx.bfloat16)
    backing = mx.random.normal((1, 1, 8, 32)).astype(mx.bfloat16)
    indices = mx.array([[[0, 1, 2, 3, 4, 5]]], dtype=mx.int32)

    singleton = indexed_sparse_attention(
        query, backing, backing, indices, 32**-0.5, key_length=6
    )
    batched = indexed_sparse_attention(
        mx.repeat(query, 4, axis=0),
        mx.repeat(backing, 4, axis=0),
        mx.repeat(backing, 4, axis=0),
        mx.repeat(indices, 4, axis=0),
        32**-0.5,
        key_length=6,
    )
    compact = indexed_sparse_attention(
        query,
        mx.contiguous(backing[:, :, :6]),
        mx.contiguous(backing[:, :, :6]),
        indices,
        32**-0.5,
    )
    assert singleton is not None
    assert batched is not None
    assert compact is not None
    mx.eval(singleton, batched, compact)

    assert mx.array_equal(singleton, batched[:1]).item()
    assert mx.array_equal(singleton, compact).item()
