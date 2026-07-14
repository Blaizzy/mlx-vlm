import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.quantization.one_bit import (
    OneBitEmbedding,
    OneBitLinear,
    dequantize_one_bit,
    one_bit_quantized_matmul,
    replace_one_bit_modules,
)


def _pack_bits(bits: np.ndarray) -> mx.array:
    packed = np.zeros((*bits.shape[:-1], bits.shape[-1] // 32), dtype=np.uint32)
    for shift in range(32):
        packed |= bits[..., shift::32].astype(np.uint32) << shift
    return mx.array(packed)


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_one_bit_quantized_matmul_matches_dense(group_size):
    rng = np.random.default_rng(7 + group_size)
    input_dims = 128
    output_dims = 7
    bits = rng.integers(0, 2, size=(output_dims, input_dims), dtype=np.uint32)
    weight = _pack_bits(bits)
    scales = mx.array(
        rng.normal(size=(output_dims, input_dims // group_size)).astype(np.float32)
    )
    biases = mx.array(
        rng.normal(size=(output_dims, input_dims // group_size)).astype(np.float32)
    )
    x = mx.array(rng.normal(size=(2, 3, input_dims)).astype(np.float32))

    out = one_bit_quantized_matmul(x, weight, scales, biases, group_size=group_size)
    dense = dequantize_one_bit(weight, scales, biases, group_size)
    reference = x @ dense.T
    mx.eval(out, reference)

    assert out.shape == (2, 3, output_dims)
    assert mx.allclose(out, reference, rtol=1e-5, atol=1e-4).item()


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_one_bit_prompt_matmul_matches_dense(group_size):
    rng = np.random.default_rng(71 + group_size)
    input_dims = 512
    output_dims = 7
    bits = rng.integers(0, 2, size=(output_dims, input_dims), dtype=np.uint32)
    weight = _pack_bits(bits)
    scales = mx.array(
        rng.normal(size=(output_dims, input_dims // group_size)).astype(np.float32)
    )
    biases = mx.array(
        rng.normal(size=(output_dims, input_dims // group_size)).astype(np.float32)
    )
    x = mx.array(rng.normal(size=(2, 9, input_dims)).astype(np.float32))

    out = one_bit_quantized_matmul(x, weight, scales, biases, group_size=group_size)
    dense = dequantize_one_bit(weight, scales, biases, group_size)
    reference = x @ dense.T
    mx.eval(out, reference)

    assert out.shape == (2, 9, output_dims)
    assert mx.allclose(out, reference, rtol=1e-5, atol=1e-4).item()


def test_one_bit_linear_applies_output_bias():
    layer = OneBitLinear(64, 3, bias=True, group_size=64)
    layer.weight = _pack_bits(np.ones((3, 64), dtype=np.uint32))
    layer.scales = mx.ones((3, 1))
    layer.biases = mx.zeros((3, 1))
    layer.bias = mx.array([1.0, 2.0, 3.0])
    x = mx.ones((2, 64))

    out = layer(x)
    mx.eval(out)

    assert mx.array_equal(out, mx.array([[65.0, 66.0, 67.0]] * 2)).item()


def test_one_bit_embedding_lookup_and_linear_projection():
    embedding = OneBitEmbedding(3, 64, group_size=64)
    codes = np.stack([np.zeros(64, dtype=np.uint32), np.ones(64, dtype=np.uint32)] * 2)[
        :3
    ]
    embedding.weight = _pack_bits(codes)
    embedding.scales = mx.ones((3, 1)) * 2
    embedding.biases = mx.ones((3, 1)) * -1

    lookup = embedding(mx.array([0, 1]))
    projection = embedding.as_linear(mx.ones((1, 64)))
    mx.eval(lookup, projection)

    assert mx.array_equal(lookup[0], mx.full((64,), -1.0)).item()
    assert mx.array_equal(lookup[1], mx.full((64,), 1.0)).item()
    assert mx.array_equal(projection, mx.array([[-64.0, 64.0, -64.0]])).item()


def test_replace_one_bit_checkpoint_modules_only():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 8, bias=False)
            self.unquantized = nn.Linear(64, 8, bias=False)
            self.embedding = nn.Embedding(16, 64)

    model = Model()
    weights = {
        "proj.scales": mx.zeros((8, 1)),
        "embedding.scales": mx.zeros((16, 1)),
    }
    replace_one_bit_modules(
        model,
        {"group_size": 64, "bits": 1, "mode": "affine"},
        weights,
    )

    assert isinstance(model.proj, OneBitLinear)
    assert isinstance(model.embedding, OneBitEmbedding)
    assert isinstance(model.unquantized, nn.Linear)
