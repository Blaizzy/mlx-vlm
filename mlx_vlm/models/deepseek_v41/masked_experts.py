"""Device-only affine expert matvecs that skip routes owned by another rank."""

from functools import lru_cache

import mlx.core as mx

from ..fast_ops import _affine_exact_header
from ..switch_layers import QuantizedSwitchLinear

_SOURCE = r"""
uint route = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup;
int out_row = int(threadgroup_position_in_grid.y) * ROWS_PER_TG +
    int(simdgroup_index_in_threadgroup) * RESULTS_PER_SIMDGROUP;

// The predicate is uniform across the threadgroup. Inactive routes never
// load an expert weight or input activation, and always initialize their output.
if (!active[route]) {
  if (lane == 0) {
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      y[int(route) * N_SIZE + out_row + row] = T(0);
    }
  }
  return;
}

int expert = int(indices[route]);
constexpr int W_ROW_BYTES = K_SIZE / 2;
constexpr int GROUPS = K_SIZE / GROUP_SIZE;
const device uint8_t* ws = (const device uint8_t*)w +
    size_t(expert) * N_SIZE * W_ROW_BYTES + out_row * W_ROW_BYTES +
    int(lane) * PACKS_PER_THREAD * BYTES_PER_PACK;
const device T* sc = scales + size_t(expert) * N_SIZE * GROUPS +
    out_row * GROUPS + int(lane) / SCALE_STEP_PER_THREAD;
const device T* bs = biases + size_t(expert) * N_SIZE * GROUPS +
    out_row * GROUPS + int(lane) / SCALE_STEP_PER_THREAD;
const device T* xk = x + (int(route) / INPUT_REPEATS) * K_SIZE +
    int(lane) * VALUES_PER_THREAD;
float result[RESULTS_PER_SIMDGROUP] = {0.0f};
float x_thread[VALUES_PER_THREAD];
for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
  float sum = load_affine4_vector_exact<T>(xk, x_thread);
  for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    result[row] += affine4_qdot_exact(
        ws + row * W_ROW_BYTES, x_thread,
        float(sc[row * GROUPS]), float(bs[row * GROUPS]), sum);
  }
  ws += BLOCK_SIZE / 2;
  sc += BLOCK_SIZE / GROUP_SIZE;
  bs += BLOCK_SIZE / GROUP_SIZE;
  xk += BLOCK_SIZE;
}
for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
  float value = simd_sum(result[row]);
  if (lane == 0) {
    y[int(route) * N_SIZE + out_row + row] = T(value);
  }
}
"""


@lru_cache(None)
def _kernel(packs):
    return mx.fast.metal_kernel(
        name=f"deepseek_v41_masked_affine4_p{packs}",
        input_names=["x", "indices", "active", "w", "scales", "biases"],
        output_names=["y"],
        header=_affine_exact_header(4, 64, 4, 2, packs),
        source=_SOURCE,
    )


def supports_masked_experts(switch, x, indices):
    """Keep unsupported quantization, dtype, and prefill shapes on the fallback."""
    return (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.ndim == 3
        and x.shape[1] == 1
        and x.dtype in (mx.bfloat16, mx.float16)
        and indices.ndim == 3
        and indices.shape[:2] == x.shape[:2]
        and indices.size < 64
        and all(
            isinstance(p, QuantizedSwitchLinear)
            and p.mode == "affine"
            and p.bits == 4
            and p.group_size == 64
            and "bias" not in p
            and p.biases is not None
            and p.scales.dtype == x.dtype
            and p.biases.dtype == x.dtype
            and p.input_dims % 256 == 0
            and p.output_dims % 8 == 0
            for p in (switch.up_proj, switch.gate_proj, switch.down_proj)
        )
    )


def masked_affine4_linear(linear, x, indices, active, input_repeats=1):
    k, n = linear.input_dims, linear.output_dims
    # Match gather_qmm's fast and ordinary matvec reduction orders.
    packs = 2 if k % 512 == 0 else 1
    return _kernel(packs)(
        inputs=[
            mx.contiguous(x),
            mx.contiguous(indices),
            mx.contiguous(active),
            linear.weight,
            linear.scales,
            linear.biases,
        ],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("GROUP_SIZE", 64),
            ("INPUT_REPEATS", input_repeats),
        ],
        grid=(32, 2 * (n // 8), indices.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


def masked_experts(switch, x, indices, scores, active):
    """Preserve routing-before-rounding and the stock FP32 route reduction."""
    up = masked_affine4_linear(switch.up_proj, x, indices, active, indices.shape[-1])
    gate = masked_affine4_linear(
        switch.gate_proj, x, indices, active, indices.shape[-1]
    )
    activated = switch.activation(up, gate, scores[..., None])
    return masked_affine4_linear(switch.down_proj, activated, indices, active)
