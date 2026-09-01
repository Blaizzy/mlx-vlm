import math

import mlx.core as mx
import mlx.nn as nn


def max_absorbed_queries(
    kv_lora_rank: int, qk_nope_head_dim: int, v_head_dim: int
) -> int:
    """Query count below which folding into the query beats materializing K/V.

    Absorbed cost scales with the number of new queries; materializing scales
    with the cached length, so the crossover does not depend on context size.
    """
    denom = 2 * kv_lora_rank - qk_nope_head_dim - v_head_dim
    if denom <= 0:
        return 1
    return max(1, int(kv_lora_rank * (qk_nope_head_dim + v_head_dim) / denom))


class MultiLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )

    def __call__(self, x, transpose=True):
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        else:
            return x @ self.weight

    def to_quantized(
        self,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ):
        num_heads, output_dims, input_dims = self.weight.shape
        ql = QuantizedMultiLinear(
            input_dims, output_dims, num_heads, group_size, bits, mode
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
        return ql


class QuantizedMultiLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_heads: int,
        group_size: int,
        bits: int,
        mode: str,
    ):
        super().__init__()

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        self.freeze()

    def __call__(self, x, transpose=True):
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
