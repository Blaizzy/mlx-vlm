"""Python-hosted 1-bit affine inference kernels."""

from functools import lru_cache
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

SUPPORTED_GROUP_SIZES = (32, 64, 128)


_ONE_BIT_QMV_SOURCE = r"""
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint input_row = threadgroup_position_in_grid.y;
    uint input_dims = x_shape[1];
    uint output_dims = weight_shape[0];
    uint groups = scales_shape[1];
    uint output_start = threadgroup_position_in_grid.x * 8 + simd_group * 4;

    float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr uint VALUES_PER_THREAD = 16;
    constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;

    // Derived from PrismML's qmv_fast SIMD layout. Each lane owns one packed
    // 16-bit half-word and 16 contiguous activations, lowering register
    // pressure while four output rows reuse the activation block.
    for (uint block_start = lane * VALUES_PER_THREAD;
         block_start < input_dims;
         block_start += BLOCK_SIZE) {
        float x_thread[VALUES_PER_THREAD];
        float total_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
            float value = static_cast<float>(
                x[input_row * input_dims + block_start + i]);
            x_thread[i] = value;
            total_sum += value;
        }

        uint group = block_start / GROUP_SIZE;
        uint packed_column = block_start >> 5;
        uint packed_shift = block_start & 31;
        for (uint row = 0; row < 4; ++row) {
            uint output_row = output_start + row;
            if (ROW_VALID) {
                ushort packed = ushort(
                    weight[output_row * weight_shape[1] + packed_column] >>
                    packed_shift);
                float selected_sum = 0.0f;
                #pragma clang loop unroll(full)
                for (uint i = 0; i < VALUES_PER_THREAD; ++i) {
                    selected_sum +=
                        (packed & (ushort(1) << i)) ? x_thread[i] : 0.0f;
                }

                uint parameter_index = output_row * groups + group;
                accumulators[row] += selected_sum *
                    static_cast<float>(scales[parameter_index]);
                accumulators[row] += total_sum *
                    static_cast<float>(biases[parameter_index]);
            }
        }
    }

    for (uint row = 0; row < 4; ++row) {
        accumulators[row] = simd_sum(accumulators[row]);
        uint output_row = output_start + row;
        if (lane == 0 && output_row < output_dims) {
            out[input_row * output_dims + output_row] =
                static_cast<T>(accumulators[row]);
        }
    }
"""


def _validate_group_size(group_size: int) -> None:
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"1-bit affine inference requires group_size in "
            f"{SUPPORTED_GROUP_SIZES}, got {group_size}."
        )


@lru_cache(maxsize=None)
def _one_bit_qmv_kernel(group_size: int, output_aligned: bool):
    _validate_group_size(group_size)
    if not hasattr(mx, "metal") or not mx.metal.is_available():
        return None
    source = _ONE_BIT_QMV_SOURCE.replace("GROUP_SIZE", str(group_size)).replace(
        "ROW_VALID", "true" if output_aligned else "output_row < output_dims"
    )
    return mx.fast.metal_kernel(
        name=f"mlx_vlm_affine_1bit_qmv_gs_{group_size}_aligned_{int(output_aligned)}",
        input_names=["x", "weight", "scales", "biases"],
        output_names=["out"],
        source=source,
    )


def dequantize_one_bit(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int,
) -> mx.array:
    """Dequantize packed affine 1-bit rows using ordinary MLX operations."""
    _validate_group_size(group_size)
    if weight.dtype != mx.uint32:
        raise ValueError(f"Packed 1-bit weights must be uint32, got {weight.dtype}.")
    if weight.shape[:-1] != scales.shape[:-1] or scales.shape != biases.shape:
        raise ValueError("1-bit weight, scale, and bias leading dimensions must match.")

    input_dims = weight.shape[-1] * 32
    if scales.shape[-1] * group_size != input_dims:
        raise ValueError(
            "Packed weight width does not match scales and group_size: "
            f"{input_dims} != {scales.shape[-1]} * {group_size}."
        )

    shifts = mx.arange(32, dtype=mx.uint32)
    bits = ((weight[..., None] >> shifts) & 1).reshape(*weight.shape[:-1], input_dims)
    expanded_shape = (*scales.shape, group_size)
    dense_shape = (*scales.shape[:-1], input_dims)
    expanded_scales = mx.broadcast_to(scales[..., None], expanded_shape).reshape(
        dense_shape
    )
    expanded_biases = mx.broadcast_to(biases[..., None], expanded_shape).reshape(
        dense_shape
    )
    return bits.astype(scales.dtype) * expanded_scales + expanded_biases


def one_bit_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
) -> mx.array:
    """Multiply by a transposed, packed affine 1-bit weight matrix."""
    _validate_group_size(group_size)
    if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        raise ValueError("1-bit matmul currently requires 2D weight/scales/biases.")
    if weight.dtype != mx.uint32:
        raise ValueError(f"Packed 1-bit weights must be uint32, got {weight.dtype}.")
    if scales.shape != biases.shape or scales.shape[0] != weight.shape[0]:
        raise ValueError("1-bit weight, scale, and bias row counts must match.")

    input_dims = x.shape[-1]
    if weight.shape[1] * 32 != input_dims:
        raise ValueError(
            f"Input has {input_dims} features but packed weight has "
            f"{weight.shape[1] * 32}."
        )
    if scales.shape[1] * group_size != input_dims:
        raise ValueError(
            f"Input has {input_dims} features but scales describe "
            f"{scales.shape[1] * group_size}."
        )

    output_dims = weight.shape[0]
    output_shape = (*x.shape[:-1], output_dims)
    x_2d = x.reshape(-1, input_dims)
    kernel = _one_bit_qmv_kernel(group_size, output_dims % 8 == 0)
    if kernel is None:
        dense_weight = dequantize_one_bit(weight, scales, biases, group_size).astype(
            x.dtype
        )
        return (x_2d @ dense_weight.T).reshape(output_shape)

    out = kernel(
        inputs=[x_2d, weight, scales, biases],
        template=[("T", x.dtype)],
        grid=(((output_dims + 7) // 8) * 64, x_2d.shape[0], 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(x_2d.shape[0] * output_dims,)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(output_shape)


class OneBitLinear(nn.Module):
    """Inference-only affine 1-bit equivalent of ``nn.QuantizedLinear``."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
    ):
        super().__init__()
        _validate_group_size(group_size)
        if input_dims % group_size != 0:
            raise ValueError(
                f"input_dims ({input_dims}) must be divisible by group_size "
                f"({group_size})."
            )

        self.weight = mx.zeros((output_dims, input_dims // 32), dtype=mx.uint32)
        self.scales = mx.zeros(
            (output_dims, input_dims // group_size), dtype=mx.float32
        )
        self.biases = mx.zeros_like(self.scales)
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.group_size = group_size
        self.bits = 1
        self.mode = "affine"
        self.freeze()

    @property
    def input_dims(self) -> int:
        return self.weight.shape[1] * 32

    @property
    def output_dims(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        x = one_bit_quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            self["biases"],
            group_size=self.group_size,
        )
        if "bias" in self:
            x = x + self["bias"]
        return x

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}, group_size={self.group_size}, bits=1, "
            "mode=affine"
        )


class OneBitEmbedding(nn.Module):
    """Inference-only affine 1-bit equivalent of ``nn.QuantizedEmbedding``."""

    def __init__(self, num_embeddings: int, dims: int, group_size: int = 64):
        super().__init__()
        _validate_group_size(group_size)
        if dims % group_size != 0:
            raise ValueError(
                f"Embedding dims ({dims}) must be divisible by group_size "
                f"({group_size})."
            )
        self.weight = mx.zeros((num_embeddings, dims // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_embeddings, dims // group_size), dtype=mx.float32)
        self.biases = mx.zeros_like(self.scales)
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.bits = 1
        self.mode = "affine"
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        return dequantize_one_bit(
            self["weight"][x],
            self["scales"][x],
            self["biases"][x],
            self.group_size,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        return one_bit_quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            self["biases"],
            group_size=self.group_size,
        )


def _quantization_for_path(quantization: dict, path: str) -> dict:
    base = {
        key: quantization[key]
        for key in ("group_size", "bits", "mode")
        if key in quantization
    }
    per_layer = quantization.get(path)
    if isinstance(per_layer, dict):
        base.update(per_layer)
    return base


def replace_one_bit_modules(
    model: nn.Module,
    quantization: dict,
    weights: Optional[Dict[str, mx.array]] = None,
) -> nn.Module:
    """Replace checkpoint-backed 1-bit Linear/Embedding leaves in ``model``."""

    def replace(path: str, module: nn.Module) -> nn.Module:
        params = _quantization_for_path(quantization, path)
        if params.get("bits") != 1:
            return module
        if weights is not None and f"{path}.scales" not in weights:
            return module
        mode = params.get("mode", "affine")
        if mode != "affine":
            raise ValueError(
                f"1-bit module {path!r} must use affine mode, got {mode!r}."
            )
        group_size = params.get("group_size", 64)

        if isinstance(module, nn.Linear):
            output_dims, input_dims = module.weight.shape
            replacement = OneBitLinear(
                input_dims,
                output_dims,
                bias="bias" in module,
                group_size=group_size,
            )
            if "bias" in module:
                replacement.bias = module.bias
            return replacement
        if isinstance(module, nn.Embedding):
            num_embeddings, dims = module.weight.shape
            return OneBitEmbedding(num_embeddings, dims, group_size=group_size)
        if hasattr(module, "to_quantized"):
            raise ValueError(
                f"1-bit inference does not yet support module {path!r} "
                f"of type {type(module).__name__}."
            )
        return module

    leaves = tree_map_with_path(
        replace, model.leaf_modules(), is_leaf=nn.Module.is_module
    )
    model.update_modules(leaves)
    return model
