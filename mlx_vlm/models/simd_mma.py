"""Small-row affine and BF16 matmuls using Metal SIMD-group matrix operations.

This path accumulates in float32 and rounds its output to BF16. One
row and a block use the same eight-column tile and reduction order. Its
arithmetic differs from the default verifier.
"""

from copy import copy
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

_PREPARE_SOURCE = r"""

    uint lane=thread_index_in_simdgroup;
    uint i=thread_position_in_grid.y;
    uint token=i%8, group=(i/8)%(K/GS), batch=i/(8*(K/GS));
    float sum=0.0f;
    for (uint j=lane;j<GS;j+=32) {
        uint k=group*GS+j;
        float value=token<M ? float(x[(batch*M+token)*K+k]) : 0.0f;
        sum+=value;
        uint rr=k%8;
        uint owner=((rr&3)<<1)|((rr&4)<<2)|((token&2)>>1)|((token&4)<<1);
        packed[((batch*(K/8)+k/8)*32+owner)*2+(token&1)]=value;
    }
    sum=simd_sum(sum);
    if (lane==0) sx[(batch*(K/GS)+group)*8+token]=sum;

"""

_MATRIX_SOURCE = r"""

    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tile = threadgroup_position_in_grid.y;
    uint batch = threadgroup_position_in_grid.z;
    uint part = sg;
    uint rr = ((lane & 6) >> 1) + ((lane & 16) >> 2);
    uint cc = ((lane & 1) << 1) + ((lane & 8) >> 1);
    uint first_n = tile * (8 * NF) + rr;
__SELECT_WEIGHTS__
    simdgroup_float8x8 sums[NF];
    for (int j=0; j<NF; ++j) {
        sums[j].thread_elements()[0] = 0.0f;
        sums[j].thread_elements()[1] = 0.0f;
    }
    for (uint base=part*(K/SPLIT); base<(part+1)*(K/SPLIT); base+=GS) {
        simdgroup_float8x8 block[NF];
        for (int j=0;j<NF;++j) {block[j].thread_elements()[0]=0.0f;block[j].thread_elements()[1]=0.0f;}
        float ss[NF], bb[NF];
        #pragma clang loop unroll(full)
        for (int j=0; j<NF; ++j) {
            uint n=local_n+j*8;
            ss[j]=float(scales[n*(K/GS)+base/GS]);
            bb[j]=float(biases[n*(K/GS)+base/GS]);
        }
        #pragma clang loop unroll(full)
        for (uint offset=0; offset<GS; offset+=8) {
            uint k=base+offset;
            simdgroup_matrix<P,8,8> inputs;
            float2 pair = ((const device float2*)x)[(batch*(K/8)+k/8)*32+lane];
            inputs.thread_elements()[0] = pair.x;
            inputs.thread_elements()[1] = pair.y;
            #pragma clang loop unroll(full)
            for (int j=0; j<NF; ++j) {
                uint n=local_n+j*8;
                uint packed=w[n*(K/8)+k/8];
                simdgroup_matrix<P,8,8> weights;
                weights.thread_elements()[0] = P(float((packed>>(cc*4))&15));
                weights.thread_elements()[1] = P(float((packed>>((cc+1)*4))&15));
                simdgroup_multiply_accumulate(block[j], weights, inputs, block[j]);
            }
        }
        #pragma clang loop unroll(full)
        for (int j=0;j<NF;++j) {
            float2 sumx=((const device float2*)sx)[(batch*(K/GS)+base/GS)*4+cc/2];
            sums[j].thread_elements()[0] += block[j].thread_elements()[0]*ss[j]+sumx.x*bb[j];
            sums[j].thread_elements()[1] += block[j].thread_elements()[1]*ss[j]+sumx.y*bb[j];
        }
    }

    threadgroup float scratch[SPLIT * NF * 64];
    #pragma clang loop unroll(full)
    for (int j=0;j<NF;++j) {
        scratch[(part*NF+j)*64+lane*2]=sums[j].thread_elements()[0];
        scratch[(part*NF+j)*64+lane*2+1]=sums[j].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part==0) {
        #pragma clang loop unroll(full)
        for (int j=0;j<NF;++j) {
            float2 value=0.0f;
            #pragma clang loop unroll(full)
            for (int p=0;p<SPLIT;++p) {
                value+=((threadgroup float2*)scratch)[(p*NF+j)*32+lane];
            }
            uint n=first_n+j*8;
            if (cc<M) y[(batch*M+cc)*N+n]=T(value.x);
            if (cc+1<M) y[(batch*M+cc+1)*N+n]=T(value.y);
        }
    }

"""


@lru_cache(maxsize=1)
def _matrix_layout_supported():
    if mx.device_info().get("architecture") != "applegpu_g15d":
        return False
    # Matrix element ownership is implementation-defined. Verify it before
    # using registers directly; unsupported layouts retain the existing path.
    kernel = mx.fast.metal_kernel(
        name="affine_mma_layout",
        input_names=["x"],
        output_names=["y"],
        header="#include <metal_simdgroup_matrix>\nusing namespace metal;",
        source="""
            simdgroup_float8x8 value;
            simdgroup_load(value, x, 8);
            uint lane = thread_index_in_simdgroup;
            y[2 * lane] = value.thread_elements()[0];
            y[2 * lane + 1] = value.thread_elements()[1];
        """,
    )
    observed = kernel(
        inputs=[mx.arange(64, dtype=mx.float32)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32, 2)],
        output_dtypes=[mx.float32],
    )[0]
    expected = []
    for lane in range(32):
        row = ((lane & 6) >> 1) + ((lane & 16) >> 2)
        col = ((lane & 1) << 1) + ((lane & 8) >> 1)
        expected.append([row * 8 + col, row * 8 + col + 1])
    return mx.array_equal(observed, mx.array(expected, mx.float32)).item()


def supports_simd_mma():
    return (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and _matrix_layout_supported()
    )


@lru_cache(maxsize=1)
def _prepare_kernel():
    return mx.fast.metal_kernel(
        name="affine_mma_prepare",
        input_names=["x"],
        output_names=["packed", "sx"],
        source=_PREPARE_SOURCE,
    )


@lru_cache(maxsize=128)
def _matrix_kernel(sizes):
    inputs = ["x", "sx"]
    selection = [
        "    uint local_n = first_n;",
        "    const device uint* w = w0;",
        "    const device T* scales = scales0;",
        "    const device T* biases = biases0;",
    ]
    offset = 0
    for index, size in enumerate(sizes):
        inputs.extend([f"w{index}", f"scales{index}", f"biases{index}"])
        if index:
            selection.append(
                f"    if (first_n >= {offset}) {{ local_n = first_n - {offset}; "
                f"w = w{index}; scales = scales{index}; biases = biases{index}; }}"
            )
        offset += size
    return mx.fast.metal_kernel(
        name="affine_mma_" + "_".join(map(str, sizes)),
        input_names=inputs,
        output_names=["y"],
        header="#include <metal_simdgroup_matrix>\nusing namespace metal;",
        source=_MATRIX_SOURCE.replace("__SELECT_WEIGHTS__", "\n".join(selection)),
    )


def supports_affine_linear(linear):
    """Whether weights fit the BF16 affine matrix kernel, independent of device."""
    return (
        isinstance(linear, nn.QuantizedLinear)
        and linear.mode == "affine"
        and linear.bits == 4
        and linear.group_size in (32, 64, 128)
        and linear.scales.dtype == mx.bfloat16
        and linear.biases.dtype == mx.bfloat16
        and linear.weight.shape[0] % 16 == 0
        and linear.weight.shape[1] * 8 >= 128
        and linear.weight.shape[1] * 8 % linear.group_size == 0
    )


def affine_mmas(linears, x, *, chunked=False):
    """Return fused projections, or None when the regular path should be used."""
    if (
        not 1 <= len(linears) <= 4
        or x.ndim != 3
        or x.shape[1] < 1
        or (x.shape[1] > 8 and not chunked)
        or x.dtype != mx.bfloat16
        or x.shape[-1] < 128
        or not all(
            supports_affine_linear(linear) and linear.weight.shape[1] * 8 == x.shape[-1]
            for linear in linears
        )
        or any(linear.group_size != linears[0].group_size for linear in linears)
        or not supports_simd_mma()
    ):
        return None
    if x.shape[1] > 8:
        chunks = [
            affine_mmas(linears, x[:, start : start + 8])
            for start in range(0, x.shape[1], 8)
        ]
        return tuple(mx.concatenate(values, axis=1) for values in zip(*chunks))
    batch, rows, width = x.shape
    group_size = linears[0].group_size
    sizes = tuple(linear.weight.shape[0] for linear in linears)
    # Each SIMD group computes a different reduction interval. Keep partial
    # sums in threadgroup memory and write the rounded result only once.
    split = 16 if width > 8192 else 8
    while width % (split * group_size):
        split //= 2
    packed, sums = _prepare_kernel()(
        inputs=[x],
        template=[("M", rows), ("K", width), ("GS", group_size)],
        grid=(32, batch * (width // group_size) * 8, 1),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, width // 8, 32, 2), (batch, width // group_size, 8)],
        output_dtypes=[mx.float32, mx.float32],
    )
    inputs = [packed, sums]
    for linear in linears:
        inputs.extend([linear.weight, linear.scales, linear.biases])
    output = _matrix_kernel(sizes)(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("P", mx.float32),
            ("M", rows),
            ("K", width),
            ("N", sum(sizes)),
            ("NF", 2),
            ("GS", group_size),
            ("SPLIT", split),
        ],
        grid=(32, sum(sizes) // 16 * split, batch),
        threadgroup=(32, split, 1),
        output_shapes=[(batch, rows, sum(sizes))],
        output_dtypes=[x.dtype],
    )[0]
    cuts, offset = [], 0
    for size in sizes[:-1]:
        offset += size
        cuts.append(offset)
    outputs = mx.split(output, cuts, axis=-1) if cuts else (output,)
    return tuple(
        value + linear["bias"] if "bias" in linear else value
        for linear, value in zip(linears, outputs)
    )


def affine_mma(linear, x):
    outputs = affine_mmas((linear,), x)
    return None if outputs is None else outputs[0]


class SIMDQuantizedLinear(nn.QuantizedLinear):
    """Runtime adapter preserving parameter names and the original fallback."""

    def __call__(self, x):
        output = affine_mma(self, x)
        return super().__call__(x) if output is None else output


_DENSE_PREPARE_SOURCE = r"""
uint lane = thread_index_in_simdgroup;
uint tile = thread_position_in_grid.y;
uint batch = thread_position_in_grid.z;
if (tile >= K / 8) return;
uint row = ((lane & 6) >> 1) + ((lane & 16) >> 2);
uint col = ((lane & 1) << 1) + ((lane & 8) >> 1);
uint at = ((batch * (K / 8) + tile) * 32 + lane) * 2;
packed[at] = col < M ? float(x[(batch * M + col) * K + tile * 8 + row]) : 0.0f;
packed[at + 1] = col + 1 < M ? float(x[(batch * M + col + 1) * K + tile * 8 + row]) : 0.0f;
"""

_DENSE_MATRIX_SOURCE = r"""
uint lane = thread_index_in_simdgroup;
uint part = simdgroup_index_in_threadgroup;
uint tile = threadgroup_position_in_grid.y;
uint batch = threadgroup_position_in_grid.z;
uint row = ((lane & 6) >> 1) + ((lane & 16) >> 2);
uint col = ((lane & 1) << 1) + ((lane & 8) >> 1);
uint first_n = tile * (8 * NF) + row;
__SELECT_WEIGHTS__
simdgroup_float8x8 sums[NF];
for (int j = 0; j < NF; ++j) {
    sums[j].thread_elements()[0] = 0.0f;
    sums[j].thread_elements()[1] = 0.0f;
}
for (uint k = part * (K / SPLIT); k < (part + 1) * (K / SPLIT); k += 8) {
    simdgroup_float8x8 inputs;
    float2 pair = ((const device float2*)x)[(batch * (K / 8) + k / 8) * 32 + lane];
    inputs.thread_elements()[0] = pair.x;
    inputs.thread_elements()[1] = pair.y;
    #pragma clang loop unroll(full)
    for (int j = 0; j < NF; ++j) {
        uint n = local_n + j * 8;
        simdgroup_float8x8 weights;
        weights.thread_elements()[0] = float(w[n * K + k + col]);
        weights.thread_elements()[1] = float(w[n * K + k + col + 1]);
        simdgroup_multiply_accumulate(sums[j], weights, inputs, sums[j]);
    }
}
threadgroup float scratch[SPLIT * NF * 64];
#pragma clang loop unroll(full)
for (int j = 0; j < NF; ++j) {
    scratch[(part * NF + j) * 64 + lane * 2] = sums[j].thread_elements()[0];
    scratch[(part * NF + j) * 64 + lane * 2 + 1] = sums[j].thread_elements()[1];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (part == 0) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < NF; ++j) {
        float2 value = 0.0f;
        #pragma clang loop unroll(full)
        for (int p = 0; p < SPLIT; ++p) value += ((threadgroup float2*)scratch)[(p * NF + j) * 32 + lane];
        uint n = first_n + j * 8;
        if (col < M) y[(batch * M + col) * N + n] = T(value.x);
        if (col + 1 < M) y[(batch * M + col + 1) * N + n] = T(value.y);
    }
}
"""


@lru_cache(maxsize=1)
def _dense_prepare_kernel():
    return mx.fast.metal_kernel(
        name="dense_mma_prepare",
        input_names=["x"],
        output_names=["packed"],
        source=_DENSE_PREPARE_SOURCE,
    )


@lru_cache(maxsize=128)
def _dense_matrix_kernel(sizes):
    inputs = ["x"]
    selection = ["uint local_n = first_n;", "const device T* w = w0;"]
    offset = 0
    for index, size in enumerate(sizes):
        inputs.append(f"w{index}")
        if index:
            selection.append(
                f"if (first_n >= {offset}) {{ local_n = first_n - {offset}; w = w{index}; }}"
            )
        offset += size
    return mx.fast.metal_kernel(
        name="dense_mma_" + "_".join(map(str, sizes)),
        input_names=inputs,
        output_names=["y"],
        header="#include <metal_simdgroup_matrix>\nusing namespace metal;",
        source=_DENSE_MATRIX_SOURCE.replace("__SELECT_WEIGHTS__", "\n".join(selection)),
    )


def supports_dense_linear(linear):
    """BF16 dense weights supported by the fixed-reduction matrix kernel."""
    return (
        isinstance(linear, nn.Linear)
        and linear.weight.dtype == mx.bfloat16
        and linear.weight.ndim == 2
        and linear.weight.shape[0] % 16 == 0
        and linear.weight.shape[1] >= 128
        and linear.weight.shape[1] % 8 == 0
        and ("bias" not in linear or linear.bias.dtype == mx.bfloat16)
    )


def supports_simd_linear(linear):
    return supports_affine_linear(linear) or supports_dense_linear(linear)


def dense_mmas(linears, x, *, chunked=False):
    """BF16 projections with identical arithmetic for serial and block decoding."""
    if (
        not 1 <= len(linears) <= 4
        or x.ndim != 3
        or x.shape[1] < 1
        or (x.shape[1] > 8 and not chunked)
        or x.dtype != mx.bfloat16
        or not all(
            supports_dense_linear(linear) and linear.weight.shape[1] == x.shape[-1]
            for linear in linears
        )
        or not supports_simd_mma()
    ):
        return None
    if x.shape[1] > 8:
        chunks = [
            dense_mmas(linears, x[:, start : start + 8])
            for start in range(0, x.shape[1], 8)
        ]
        return tuple(mx.concatenate(values, axis=1) for values in zip(*chunks))
    batch, rows, width = x.shape
    sizes = tuple(linear.weight.shape[0] for linear in linears)
    # Partitioning depends on K, never on the number of verification tokens.
    split = 16 if width > 8192 else 8
    while width % (8 * split):
        split //= 2
    packed = _dense_prepare_kernel()(
        inputs=[x],
        template=[("M", rows), ("K", width)],
        grid=(32, width // 8, batch),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, width // 8, 32, 2)],
        output_dtypes=[mx.float32],
    )[0]
    output = _dense_matrix_kernel(sizes)(
        inputs=[packed] + [linear.weight for linear in linears],
        template=[
            ("T", x.dtype),
            ("M", rows),
            ("K", width),
            ("N", sum(sizes)),
            ("NF", 2),
            ("SPLIT", split),
        ],
        grid=(32, sum(sizes) // 16 * split, batch),
        threadgroup=(32, split, 1),
        output_shapes=[(batch, rows, sum(sizes))],
        output_dtypes=[x.dtype],
    )[0]
    cuts, offset = [], 0
    for size in sizes[:-1]:
        offset += size
        cuts.append(offset)
    outputs = mx.split(output, cuts, axis=-1) if cuts else (output,)
    return tuple(
        value + linear.bias if "bias" in linear else value
        for linear, value in zip(linears, outputs)
    )


def simd_mmas(linears, x, *, chunked=False):
    """Dispatch supported affine or dense weights without changing their format."""
    if not linears:
        return None
    project = affine_mmas if isinstance(linears[0], nn.QuantizedLinear) else dense_mmas
    return project(linears, x, chunked=chunked)


class SIMDLinear(nn.Linear):
    """Dense projection adapter preserving the original module's parameters."""

    def __call__(self, x):
        outputs = dense_mmas((self,), x)
        return super().__call__(x) if outputs is None else outputs[0]


def is_simd_linear(linear):
    return isinstance(linear, (SIMDLinear, SIMDQuantizedLinear))


def enable_simd_linears(model):
    """Adapt compatible owned projections while preserving shared parameters."""
    if not supports_simd_mma():
        return model
    replacements = []
    for name, module in model.named_modules():
        if supports_simd_linear(module) and not is_simd_linear(module):
            replacement = copy(module)
            replacement.__class__ = (
                SIMDQuantizedLinear
                if isinstance(module, nn.QuantizedLinear)
                else SIMDLinear
            )
            replacements.append((name, replacement))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return model
