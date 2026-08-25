import mlx.core as mx

_EXACT_SPECULATIVE_VERIFY_GEMV = (
    mx.fast.metal_kernel(
        name="exact_speculative_verify_gemv",
        input_names=["x", "weight"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=r"""
        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) {
            return;
        }

        int out_row = int(out_block * TM);
        if (out_row >= O) {
            return;
        }

        const device T* in_vec = x + row * K;
        const device T* mat = weight + out_row * K;

        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = static_cast<float>(in_vec[col + tn]);
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    result[tm] +=
                        static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                }
            }

            col += blockN;
        }

        if (leftover > 0) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] =
                    (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    T m =
                        (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                    result[tm] += static_cast<float>(m) * v[tn];
                }
            }
        }

        for (int tm = 0; tm < TM; ++tm) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }

        if (lane == 0) {
            for (int tm = 0; tm < TM; ++tm) {
                out[row * O + out_row + tm] = static_cast<T>(result[tm]);
            }
        }
    """,
    )
    if mx.metal.is_available()
    else None
)

_EXACT_SPECULATIVE_VERIFY_SWITCH_GEMV = (
    mx.fast.metal_kernel(
        name="exact_speculative_verify_switch_gemv",
        input_names=["x", "weight", "expert_indices"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=r"""
        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) {
            return;
        }

        int out_row = int(out_block * TM);
        if (out_row >= O) {
            return;
        }

        uint input_row = BROADCAST_EXPERTS ? row / S : row;
        uint expert = expert_indices[row];
        const device T* in_vec = x + input_row * K;
        const device T* mat = weight + expert * O * K + out_row * K;

        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = static_cast<float>(in_vec[col + tn]);
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    result[tm] +=
                        static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                }
            }

            col += blockN;
        }

        if (leftover > 0) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] =
                    (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    T m =
                        (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                    result[tm] += static_cast<float>(m) * v[tn];
                }
            }
        }

        for (int tm = 0; tm < TM; ++tm) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }

        if (lane == 0) {
            for (int tm = 0; tm < TM; ++tm) {
                out[row * O + out_row + tm] = static_cast<T>(result[tm]);
            }
        }
    """,
    )
    if mx.metal.is_available()
    else None
)


def exact_speculative_verify_dense_available() -> bool:
    return _EXACT_SPECULATIVE_VERIFY_GEMV is not None


def exact_speculative_verify_weight(weight: mx.array, x: mx.array) -> mx.array | None:
    """Multiply a small verification block by a dense BF16/FP16 weight."""
    if _EXACT_SPECULATIVE_VERIFY_GEMV is None or x.ndim != 3:
        return None

    batch, length, dimensions = x.shape
    outputs = weight.shape[0]
    if (
        outputs < 4
        or outputs % 4 != 0
        or dimensions >= 16 * outputs
        or weight.dtype != x.dtype
    ):
        return None

    rows = batch * length
    padded_rows = ((rows + 7) // 8) * 8
    out = _EXACT_SPECULATIVE_VERIFY_GEMV(
        inputs=[x.reshape(rows, dimensions), weight],
        template=[
            ("T", x.dtype),
            ("K", dimensions),
            ("O", outputs),
            ("R", rows),
        ],
        grid=(32, outputs // 4, padded_rows),
        threadgroup=(32, 1, 8),
        output_shapes=[(rows, outputs)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(batch, length, outputs)


def exact_speculative_verify_switch_weight(
    weight: mx.array,
    x: mx.array,
    expert_indices: mx.array,
) -> mx.array | None:
    """Apply selected expert weights with singleton-equivalent accumulation."""
    if _EXACT_SPECULATIVE_VERIFY_SWITCH_GEMV is None or x.ndim < 3:
        return None

    broadcast_experts = x.shape[:-1] == expert_indices.shape[:-1]
    aligned_experts = x.shape[:-1] == expert_indices.shape
    if not broadcast_experts and not aligned_experts:
        return None

    dimensions = x.shape[-1]
    outputs = weight.shape[-2]
    if (
        weight.ndim != 3
        or weight.shape[-1] != dimensions
        or outputs < 4
        or outputs % 4 != 0
        or dimensions >= 16 * outputs
        or weight.dtype != x.dtype
    ):
        return None

    selected_shape = expert_indices.shape
    rows = expert_indices.size
    padded_rows = ((rows + 7) // 8) * 8
    out = _EXACT_SPECULATIVE_VERIFY_SWITCH_GEMV(
        inputs=[
            x.reshape(-1, dimensions),
            weight,
            expert_indices.astype(mx.uint32).reshape(-1),
        ],
        template=[
            ("T", x.dtype),
            ("K", dimensions),
            ("O", outputs),
            ("R", rows),
            ("S", selected_shape[-1]),
            ("BROADCAST_EXPERTS", broadcast_experts),
        ],
        grid=(32, outputs // 4, padded_rows),
        threadgroup=(32, 1, 8),
        output_shapes=[(rows, outputs)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*selected_shape, outputs)


__all__ = [
    "exact_speculative_verify_dense_available",
    "exact_speculative_verify_switch_weight",
    "exact_speculative_verify_weight",
]
