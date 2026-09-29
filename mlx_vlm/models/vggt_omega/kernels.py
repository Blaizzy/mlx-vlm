"""Fused Metal kernels for the transformer blocks.

- ``prepare_qk``: the aggregator applies LayerNorm over each head (Q/K
  norm) and 2D RoPE to q and k. As separate MLX ops on the strided q/k views
  of the qkv output, this costs more than the attention itself. The kernel
  reads q and k from the qkv output and writes them normed, rotated, cast
  and in the (B, H, N, D) layout of ``scaled_dot_product_attention``.
- ``add_layer_norm``: the float32 residual update ``x + h * gamma`` and the
  next LayerNorm, written as the matmul dtype, in one pass over the row.
"""

from functools import lru_cache
from typing import Optional, Tuple

import mlx.core as mx

_HAS_METAL = mx.metal.is_available()
# Stands in for the inputs of a disabled kernel stage.
_UNUSED = mx.zeros((1,))

# Two (q or k, token, head) rows per simdgroup, 16 lanes each; lane l holds
# the element pairs (2l + 32j, 2l + 32j + 1), so each RoPE pair (i, i + D/2)
# is in the same thread, E/2 pairs apart. Two elements per load keep enough
# reads in flight to saturate memory bandwidth; one row per simdgroup with
# one element per load reached about 40% of it.
_SOURCE = """
    constexpr int E = D / 32;
    uint lane = thread_index_in_simdgroup % 16;
    uint row = thread_position_in_grid.y * 2 + thread_index_in_simdgroup / 16;
    int B = qkv_shape[0];
    int N = qkv_shape[1];
    int H = qkv_shape[3];
    int h = row % H;
    int n = (row / H) % N;
    int b = (row / (H * N)) % B;
    int which = row / (H * N * B);

    size_t src = ((((size_t)b * N + n) * 3 + which) * H + h) * D + 2 * lane;
    float2 v[E];
    for (int j = 0; j < E; j++) {
        v[j] = float2(qkv[src + 32 * j], qkv[src + 32 * j + 1]);
    }

    if (HAS_NORM) {
        // The xor shuffles stay within the 16 lanes of the row.
        float s = 0.0f;
        for (int j = 0; j < E; j++) {
            s += v[j].x + v[j].y;
        }
        for (int o = 8; o > 0; o /= 2) {
            s += simd_shuffle_xor(s, o);
        }
        float mean = s / D;
        float s2 = 0.0f;
        for (int j = 0; j < E; j++) {
            v[j] -= mean;
            s2 += dot(v[j], v[j]);
        }
        for (int o = 8; o > 0; o /= 2) {
            s2 += simd_shuffle_xor(s2, o);
        }
        float inv = metal::precise::rsqrt(s2 / D + EPS);
        for (int j = 0; j < E; j++) {
            int i = 2 * lane + 32 * j;
            float2 w = which == 0 ? float2(q_weight[i], q_weight[i + 1])
                                  : float2(k_weight[i], k_weight[i + 1]);
            float2 c = which == 0 ? float2(q_bias[i], q_bias[i + 1])
                                  : float2(k_bias[i], k_bias[i + 1]);
            v[j] = v[j] * inv * w + c;
        }
    }

    if (HAS_ROPE) {
        for (int j = 0; j < E / 2; j++) {
            int i = 2 * lane + 32 * j;
            size_t ci = (size_t)n * (D / 2) + i;
            // Row 1 of the packed (N, 2, D/2) table holds +sin.
            size_t si = (size_t)n * D + D / 2 + i;
            float2 c = float2(cos[ci], cos[ci + 1]);
            float2 s = float2(sin[si], sin[si + 1]);
            float2 x1 = v[j];
            float2 x2 = v[j + E / 2];
            v[j] = x1 * c - x2 * s;
            v[j + E / 2] = x2 * c + x1 * s;
        }
    }

    device OutT* out = which == 0 ? q : k;
    size_t dst = (((size_t)b * H + h) * N + n) * D + 2 * lane;
    for (int j = 0; j < E; j++) {
        out[dst + 32 * j] = static_cast<OutT>(v[j].x);
        out[dst + 32 * j + 1] = static_cast<OutT>(v[j].y);
    }
"""


@lru_cache(maxsize=None)
def _kernel(eps: float):
    return mx.fast.metal_kernel(
        name="vggt_qk_prepare",
        input_names=["qkv", "q_weight", "q_bias", "k_weight", "k_bias", "sin", "cos"],
        output_names=["q", "k"],
        source=_SOURCE.replace("EPS", repr(float(eps)) + "f"),
    )


def can_prepare_qk(head_dim: int) -> bool:
    return _HAS_METAL and head_dim % 64 == 0


def prepare_qk(
    qkv: mx.array,
    norms: Optional[Tuple],
    rope: Optional[Tuple[mx.array, mx.array]],
    dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    """qkv: (B, N, 3, H, D) projection output -> q, k (B, H, N, D) in ``dtype``.

    ``norms`` is ``(q_norm, k_norm)`` LayerNorms or None; ``rope`` holds the
    packed Sapiens2 tables (sin (N, 2, D/2) as ``[-sin, sin]``, cos
    (N, 1, D/2)) or None. The kernel reads them as is (row ``1`` of sin).
    """
    B, N, _, H, D = qkv.shape
    if norms is not None:
        q_norm, k_norm = norms
        params = [q_norm.weight, q_norm.bias, k_norm.weight, k_norm.bias]
        eps = q_norm.eps
    else:
        params = [_UNUSED] * 4
        eps = 0.0
    sin, cos = rope if rope is not None else (_UNUSED, _UNUSED)
    # Separate outputs: an integer index (``out[0]``) of one stacked output
    # would be a gather, i.e. another copy of q and k.
    q, k = _kernel(eps)(
        inputs=[qkv, *params, sin, cos],
        template=[
            ("D", D),
            ("HAS_NORM", norms is not None),
            ("HAS_ROPE", rope is not None),
            ("OutT", dtype),
        ],
        grid=(32, B * N * H, 1),  # 2 * B * N * H rows, two per simdgroup
        threadgroup=(32, 8, 1),
        output_shapes=[(B, H, N, D)] * 2,
        output_dtypes=[dtype] * 2,
    )
    return q, k


# One threadgroup of TG threads per row of C channels.
_ADD_NORM_SOURCE = """
    constexpr int E = C / TG;
    constexpr int SIMDS = TG / 32;
    threadgroup float partial[SIMDS];
    uint tid = thread_position_in_threadgroup.x;
    uint simd = tid / 32;
    size_t base = (size_t)thread_position_in_grid.y * C;

    float v[E];
    for (int j = 0; j < E; j++) {
        int i = tid + TG * j;
        float value = float(x[base + i]);
        RESIDUAL
        v[j] = value;
    }

    float s = 0.0f;
    for (int j = 0; j < E; j++) {
        s += v[j];
    }
    s = simd_sum(s);
    if (thread_index_in_simdgroup == 0) {
        partial[simd] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.0f;
    for (int i = 0; i < SIMDS; i++) {
        mean += partial[i];
    }
    mean /= C;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s2 = 0.0f;
    for (int j = 0; j < E; j++) {
        float d = v[j] - mean;
        s2 += d * d;
    }
    s2 = simd_sum(s2);
    if (thread_index_in_simdgroup == 0) {
        partial[simd] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (int i = 0; i < SIMDS; i++) {
        var += partial[i];
    }
    float inv = metal::precise::rsqrt(var / C + EPS);

    for (int j = 0; j < E; j++) {
        int i = tid + TG * j;
        float y = (v[j] - mean) * inv * float(weight[i]) + float(bias[i]);
        out[base + i] = static_cast<OutT>(y);
    }
"""

_ADD_NORM_THREADS = 256

_RESIDUAL = """value += float(h[base + i]) * float(gamma[i]);
        x_out[base + i] = value;"""


@lru_cache(maxsize=None)
def _add_norm_kernel(eps: float, residual: bool):
    inputs = ["x", "h", "gamma"] if residual else ["x"]
    return mx.fast.metal_kernel(
        name=f"vggt_add_layer_norm_{int(residual)}",
        input_names=inputs + ["weight", "bias"],
        output_names=(["x_out"] if residual else []) + ["out"],
        source=_ADD_NORM_SOURCE.replace("EPS", repr(float(eps)) + "f").replace(
            "RESIDUAL", _RESIDUAL if residual else ""
        ),
    )


def can_add_layer_norm(x: mx.array) -> bool:
    return _HAS_METAL and x.dtype == mx.float32 and x.shape[-1] % _ADD_NORM_THREADS == 0


def add_layer_norm(
    x: mx.array,
    norm,
    dtype: mx.Dtype,
    h: Optional[mx.array] = None,
    gamma: Optional[mx.array] = None,
):
    """LayerNorm of float32 ``x`` (after ``x += h * gamma`` when ``h`` is
    given) in float32, written as ``dtype``.

    Returns the normed rows, or ``(x, normed)`` with a residual update.
    """
    C = x.shape[-1]
    residual = h is not None
    inputs = [x, h, gamma] if residual else [x]
    outputs = _add_norm_kernel(norm.eps, residual)(
        inputs=inputs + [norm.weight, norm.bias],
        template=[
            ("C", C),
            ("TG", _ADD_NORM_THREADS),
            ("OutT", dtype),
        ],
        grid=(_ADD_NORM_THREADS, x.size // C, 1),
        threadgroup=(_ADD_NORM_THREADS, 1, 1),
        output_shapes=[x.shape] * (2 if residual else 1),
        output_dtypes=([mx.float32] if residual else []) + [dtype],
    )
    return tuple(outputs) if residual else outputs[0]
