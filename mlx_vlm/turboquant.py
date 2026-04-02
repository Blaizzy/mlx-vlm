from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple, Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import _BaseCache, create_attention_mask

DEFAULT_TURBOQUANT_SEED = 0
_EPS = 1e-6
_POLAR_MAX_LEVELS = 4


def _metal_available() -> bool:
    return hasattr(mx, "metal") and mx.metal.is_available()


@lru_cache(maxsize=None)
def _mse_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = packed_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= packed_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(q_ptr[d]) * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_mse_score",
        input_names=["q_rot", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _pack_lowbit_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) {
            return;
        }

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) {
                packed_word |= value << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= value >> (Bits - spill);
                }
            }
        }

        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="turboquant_pack_lowbit",
        input_names=["values"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _unpack_lowbit_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) {
            return;
        }

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) {
            value |= packed_ptr[word_idx + 1] << (Bits - spill);
        }
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """
    return mx.fast.metal_kernel(
        name="turboquant_unpack_lowbit",
        input_names=["packed"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _qjl_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_proj_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = signs + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int word_idx = d / 32;
            int offset = d % 32;
            uint bit = (packed_ptr[word_idx] >> offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            acc += static_cast<float>(q_ptr[d]) * sign;
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc
                * static_cast<float>(norms[idx])
                * static_cast<float>(residual_norms[idx])
                * scale[0];
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_qjl_score",
        input_names=["q_proj", "norms", "residual_norms", "signs", "scale"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _prod_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto q_proj_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        float mse_acc = 0.0f;
        float qjl_acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) {
                value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            }
            value &= ((1u << MseBits) - 1u);
            mse_acc += static_cast<float>(q_rot_ptr[d]) * codebook[value];

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_acc += static_cast<float>(q_proj_ptr[d]) * sign;
        }

        mse_acc = simd_sum(mse_acc);
        qjl_acc = simd_sum(qjl_acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                static_cast<float>(norms[idx]) * (
                    mse_acc
                    + scale[0] * static_cast<float>(residual_norms[idx]) * qjl_acc
                );
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_prod_score",
        input_names=[
            "q_rot",
            "q_proj",
            "norms",
            "residual_norms",
            "mse_packed",
            "signs",
            "codebook",
            "scale",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _prod_score_multi_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_base = q_rot + ((b * kv_heads + h) * RepeatCount) * Dim;
        auto q_proj_base = q_proj + ((b * kv_heads + h) * RepeatCount) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        float mse_acc[RepeatCount];
        float qjl_acc[RepeatCount];
        for (int r = 0; r < RepeatCount; ++r) {
            mse_acc[r] = 0.0f;
            qjl_acc[r] = 0.0f;
        }

        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) {
                value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            }
            value &= ((1u << MseBits) - 1u);
            float code = codebook[value];

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;

            for (int r = 0; r < RepeatCount; ++r) {
                mse_acc[r] += static_cast<float>(q_rot_base[r * Dim + d]) * code;
                qjl_acc[r] += static_cast<float>(q_proj_base[r * Dim + d]) * sign;
            }
        }

        for (int r = 0; r < RepeatCount; ++r) {
            mse_acc[r] = simd_sum(mse_acc[r]);
            qjl_acc[r] = simd_sum(qjl_acc[r]);
        }

        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            float norm = static_cast<float>(norms[idx]);
            float residual_norm = static_cast<float>(residual_norms[idx]);
            for (int r = 0; r < RepeatCount; ++r) {
                out[((b * kv_heads + h) * RepeatCount + r) * token_count + t] =
                    norm * (mse_acc[r] + scale[0] * residual_norm * qjl_acc[r]);
            }
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_prod_score_multi",
        input_names=[
            "q_rot",
            "q_proj",
            "norms",
            "residual_norms",
            "mse_packed",
            "signs",
            "codebook",
            "scale",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _mse_weighted_rot_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = weights_shape[2];
        auto b = n / (kv_heads * repeat_count);
        auto rem = n % (kv_heads * repeat_count);
        auto h = rem / repeat_count;
        auto repeat_idx = rem % repeat_count;

        auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc = 0.0f;
        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            int bit_offset = dim_idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(weights_ptr[t])
                * static_cast<float>(norms_ptr[t])
                * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_mse_weighted_rot",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _mse_weighted_rot_multi_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto b = n / kv_heads;
        auto h = n % kv_heads;

        auto weights_base =
            weights + ((b * kv_heads + h) * RepeatCount) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc[RepeatCount];
        for (int r = 0; r < RepeatCount; ++r) {
            acc[r] = 0.0f;
        }

        int bit_offset = dim_idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;

        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            float code = codebook[value];
            float norm = static_cast<float>(norms_ptr[t]);
            for (int r = 0; r < RepeatCount; ++r) {
                acc[r] += static_cast<float>(weights_base[r * token_count + t]) * norm * code;
            }
        }

        for (int r = 0; r < RepeatCount; ++r) {
            acc[r] = simd_sum(acc[r]);
        }

        if (thread_index_in_simdgroup == 0) {
            for (int r = 0; r < RepeatCount; ++r) {
                out[((b * kv_heads + h) * RepeatCount + r) * Dim + dim_idx] =
                    acc[r];
            }
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_mse_weighted_rot_multi",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _prod_score_repeat_kernel(repeat_count: int):
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = q_rot_shape[2];",
        "",
        "        auto b = n / (kv_heads * token_count);",
        "        auto rem = n % (kv_heads * token_count);",
        "        auto h = rem / token_count;",
        "        auto t = rem % token_count;",
        "",
        "        auto q_rot_base = q_rot + ((b * kv_heads + h) * repeat_count) * Dim;",
        "        auto q_proj_base = q_proj + ((b * kv_heads + h) * repeat_count) * Dim;",
        "        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;",
        "        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;",
        "",
        "        auto idx = (b * kv_heads + h) * token_count + t;",
        "        float norm = static_cast<float>(norms[idx]);",
        "        float residual_norm = static_cast<float>(residual_norms[idx]);",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float mse_acc_{r} = 0.0f;")
        lines.append(f"        float qjl_acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int d = lane; d < Dim; d += 32) {",
        "            int bit_offset = d * MseBits;",
        "            int word_idx = bit_offset / 32;",
        "            int offset = bit_offset % 32;",
        "            uint value = mse_ptr[word_idx] >> offset;",
        "            int spill = offset + MseBits - 32;",
        "            if (spill > 0) {",
        "                value |= mse_ptr[word_idx + 1] << (MseBits - spill);",
        "            }",
        "            value &= ((1u << MseBits) - 1u);",
        "            float code = codebook[value];",
        "",
        "            int sign_word = d / 32;",
        "            int sign_offset = d % 32;",
        "            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;",
        "            float sign = bit ? 1.0f : -1.0f;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            mse_acc_{r} += static_cast<float>(q_rot_base[{r} * Dim + d]) * code;"
        )
        lines.append(
            f"            qjl_acc_{r} += static_cast<float>(q_proj_base[{r} * Dim + d]) * sign;"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float mse_sum_{r} = simd_sum(mse_acc_{r});")
        lines.append(f"        float qjl_sum_{r} = simd_sum(qjl_acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * token_count + t] ="
        )
        lines.append(
            f"                norm * (mse_sum_{r} + scale[0] * residual_norm * qjl_sum_{r});"
        )
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_prod_score_repeat_{repeat_count}",
        input_names=[
            "q_rot",
            "q_proj",
            "norms",
            "residual_norms",
            "mse_packed",
            "signs",
            "codebook",
            "scale",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _polar_prod_score_kernel(level_bits: tuple[int, ...]):
    if not _metal_available() or len(level_bits) == 0:
        return None

    input_names = ["q_rot", "norms", "radii"]
    for level in range(len(level_bits)):
        input_names.append(f"angles_{level + 1}")
    for level in range(len(level_bits)):
        input_names.append(f"cos_{level + 1}")
        input_names.append(f"sin_{level + 1}")

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto repeat_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = q_rot_shape[2];",
        "        if (repeat_idx >= repeat_count) {",
        "            return;",
        "        }",
        "",
        "        auto b = n / (kv_heads * token_count);",
        "        auto rem = n % (kv_heads * token_count);",
        "        auto h = rem / token_count;",
        "        auto t = rem % token_count;",
        "",
        "        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;",
        "        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;",
        "",
        "        float acc = 0.0f;",
        "        for (int d = lane; d < Dim; d += 32) {",
        "            int block_idx = d >> Levels;",
        "            float coeff = static_cast<float>(radii_ptr[block_idx]);",
        "",
    ]

    for level, bits in enumerate(level_bits, start=1):
        mask = (1 << bits) - 1
        lines += [
            f"            auto angle_ptr_{level} = angles_{level} + ((b * kv_heads + h) * token_count + t) * PackedWidth{level};",
            f"            int angle_idx_{level} = d >> {level};",
            f"            int bit_offset_{level} = angle_idx_{level} * {bits};",
            f"            int word_idx_{level} = bit_offset_{level} / 32;",
            f"            int offset_{level} = bit_offset_{level} % 32;",
            f"            uint value_{level} = angle_ptr_{level}[word_idx_{level}] >> offset_{level};",
            f"            int spill_{level} = offset_{level} + {bits} - 32;",
            f"            if (spill_{level} > 0) {{",
            f"                value_{level} |= angle_ptr_{level}[word_idx_{level} + 1] << ({bits} - spill_{level});",
            "            }",
            f"            value_{level} &= {mask}u;",
            f"            bool use_sin_{level} = ((d >> {level - 1}) & 1) != 0;",
            f"            coeff *= use_sin_{level} ? static_cast<float>(sin_{level}[value_{level}]) : static_cast<float>(cos_{level}[value_{level}]);",
            "",
        ]

    lines += [
        "            acc += static_cast<float>(q_ptr[d]) * coeff;",
        "        }",
        "",
        "        acc = simd_sum(acc);",
        "        if (thread_index_in_simdgroup == 0) {",
        "            auto idx = (b * kv_heads + h) * token_count + t;",
        "            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =",
        "                acc * static_cast<float>(norms[idx]);",
        "        }",
    ]

    return mx.fast.metal_kernel(
        name="turboquant_polar_prod_score_" + "_".join(str(bit) for bit in level_bits),
        input_names=input_names,
        output_names=["out"],
        source="\n".join(lines),
    )


@lru_cache(maxsize=None)
def _polar_turbo_score_repeat_kernel(
    level_bits: tuple[int, ...], repeat_count: int
):
    if (
        not _metal_available()
        or len(level_bits) != 4
        or repeat_count <= 0
    ):
        return None

    bits1, bits2, bits3, bits4 = level_bits
    mask1 = (1 << bits1) - 1
    mask2 = (1 << bits2) - 1
    mask3 = (1 << bits3) - 1
    mask4 = (1 << bits4) - 1

    input_names = ["q_rot", "q_proj", "norms", "radii"]
    for level in range(4):
        input_names.append(f"angles_{level + 1}")
    input_names.extend(["residual_norms", "signs", "scale"])
    for level in range(4):
        input_names.append(f"cos_{level + 1}")
        input_names.append(f"sin_{level + 1}")

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "",
        "        auto b = n / (kv_heads * token_count);",
        "        auto rem = n % (kv_heads * token_count);",
        "        auto h = rem / token_count;",
        "        auto t = rem % token_count;",
        "",
        "        auto q_rot_base = q_rot + ((b * kv_heads + h) * RepeatCount) * Dim;",
        "        auto q_proj_base = q_proj + ((b * kv_heads + h) * RepeatCount) * Dim;",
        "        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;",
        "        auto angle1_ptr = angles_1 + ((b * kv_heads + h) * token_count + t) * PackedWidth1;",
        "        auto angle2_ptr = angles_2 + ((b * kv_heads + h) * token_count + t) * PackedWidth2;",
        "        auto angle3_ptr = angles_3 + ((b * kv_heads + h) * token_count + t) * PackedWidth3;",
        "        auto angle4_ptr = angles_4 + ((b * kv_heads + h) * token_count + t) * PackedWidth4;",
        "        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;",
        "",
        "        auto idx = (b * kv_heads + h) * token_count + t;",
        "        float norm = static_cast<float>(norms[idx]);",
        "        float residual_norm = static_cast<float>(residual_norms[idx]);",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        threadgroup float level1_{r}[Dim / 2];")
        lines.append(f"        threadgroup float level2_{r}[Dim / 4];")
        lines.append(f"        threadgroup float level3_{r}[Dim / 8];")
        lines.append(f"        threadgroup float level4_{r}[BlockCount];")
    lines += [""]
    for r in range(repeat_count):
        lines.append(f"        float qjl_acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int pair_idx = lane; pair_idx < Dim / 2; pair_idx += 32) {",
        f"            int bit_offset_1 = pair_idx * {bits1};",
        "            int word_idx_1 = bit_offset_1 / 32;",
        "            int offset_1 = bit_offset_1 % 32;",
        "            uint value_1 = angle1_ptr[word_idx_1] >> offset_1;",
        f"            int spill_1 = offset_1 + {bits1} - 32;",
        "            if (spill_1 > 0) {",
        f"                value_1 |= angle1_ptr[word_idx_1 + 1] << ({bits1} - spill_1);",
        "            }",
        f"            value_1 &= {mask1}u;",
        "            float cos_1_val = static_cast<float>(cos_1[value_1]);",
        "            float sin_1_val = static_cast<float>(sin_1[value_1]);",
        "            int d0 = pair_idx << 1;",
        "            int d1 = d0 + 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level1_{r}[pair_idx] = static_cast<float>(q_rot_base[{r} * Dim + d0]) * cos_1_val + static_cast<float>(q_rot_base[{r} * Dim + d1]) * sin_1_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int pair_idx = lane; pair_idx < Dim / 4; pair_idx += 32) {",
        f"            int bit_offset_2 = pair_idx * {bits2};",
        "            int word_idx_2 = bit_offset_2 / 32;",
        "            int offset_2 = bit_offset_2 % 32;",
        "            uint value_2 = angle2_ptr[word_idx_2] >> offset_2;",
        f"            int spill_2 = offset_2 + {bits2} - 32;",
        "            if (spill_2 > 0) {",
        f"                value_2 |= angle2_ptr[word_idx_2 + 1] << ({bits2} - spill_2);",
        "            }",
        f"            value_2 &= {mask2}u;",
        "            float cos_2_val = static_cast<float>(cos_2[value_2]);",
        "            float sin_2_val = static_cast<float>(sin_2[value_2]);",
        "            int child = pair_idx << 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level2_{r}[pair_idx] = level1_{r}[child] * cos_2_val + level1_{r}[child + 1] * sin_2_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int pair_idx = lane; pair_idx < Dim / 8; pair_idx += 32) {",
        f"            int bit_offset_3 = pair_idx * {bits3};",
        "            int word_idx_3 = bit_offset_3 / 32;",
        "            int offset_3 = bit_offset_3 % 32;",
        "            uint value_3 = angle3_ptr[word_idx_3] >> offset_3;",
        f"            int spill_3 = offset_3 + {bits3} - 32;",
        "            if (spill_3 > 0) {",
        f"                value_3 |= angle3_ptr[word_idx_3 + 1] << ({bits3} - spill_3);",
        "            }",
        f"            value_3 &= {mask3}u;",
        "            float cos_3_val = static_cast<float>(cos_3[value_3]);",
        "            float sin_3_val = static_cast<float>(sin_3[value_3]);",
        "            int child = pair_idx << 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level3_{r}[pair_idx] = level2_{r}[child] * cos_3_val + level2_{r}[child + 1] * sin_3_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int pair_idx = lane; pair_idx < BlockCount; pair_idx += 32) {",
        f"            int bit_offset_4 = pair_idx * {bits4};",
        "            int word_idx_4 = bit_offset_4 / 32;",
        "            int offset_4 = bit_offset_4 % 32;",
        "            uint value_4 = angle4_ptr[word_idx_4] >> offset_4;",
        f"            int spill_4 = offset_4 + {bits4} - 32;",
        "            if (spill_4 > 0) {",
        f"                value_4 |= angle4_ptr[word_idx_4 + 1] << ({bits4} - spill_4);",
        "            }",
        f"            value_4 &= {mask4}u;",
        "            float cos_4_val = static_cast<float>(cos_4[value_4]);",
        "            float sin_4_val = static_cast<float>(sin_4[value_4]);",
        "            int child = pair_idx << 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level4_{r}[pair_idx] = level3_{r}[child] * cos_4_val + level3_{r}[child + 1] * sin_4_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int d = lane; d < Dim; d += 32) {",
        "            int sign_word = d / 32;",
        "            int sign_offset = d % 32;",
        "            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;",
        "            float sign = bit ? 1.0f : -1.0f;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            qjl_acc_{r} += static_cast<float>(q_proj_base[{r} * Dim + d]) * sign;"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"        float polar_acc_{r} = lane < BlockCount ? static_cast<float>(radii_ptr[lane]) * level4_{r}[lane] : 0.0f;"
        )
        lines.append(f"        float polar_sum_{r} = simd_sum(polar_acc_{r});")
        lines.append(f"        float qjl_sum_{r} = simd_sum(qjl_acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * RepeatCount + {r}) * token_count + t] ="
        )
        lines.append(
            f"                norm * (polar_sum_{r} + scale[0] * residual_norm * qjl_sum_{r});"
        )
    lines += [
        "        }",
    ]

    return mx.fast.metal_kernel(
        name="turboquant_polar_turbo_score_"
        + "_".join(str(bit) for bit in level_bits)
        + f"_repeat_{repeat_count}",
        input_names=input_names,
        output_names=["out"],
        source="\n".join(lines),
    )


@lru_cache(maxsize=None)
def _mse_weighted_rot_repeat_kernel(repeat_count: int):
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto dim_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        if (dim_idx >= Dim) {",
        "            return;",
        "        }",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = weights_shape[2];",
        "        auto b = n / kv_heads;",
        "        auto h = n % kv_heads;",
        "",
        "        auto weights_base = weights + ((b * kv_heads + h) * repeat_count) * token_count;",
        "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
        "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
        "            auto token_ptr = packed_ptr + t * PackedWidth;",
        "            uint value = token_ptr[word_idx] >> offset;",
        "            int spill = offset + Bits - 32;",
        "            if (spill > 0) {",
        "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
        "            }",
        "            value &= ((1u << Bits) - 1u);",
        "            float code = codebook[value];",
        "            float norm = static_cast<float>(norms_ptr[t]);",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            acc_{r} += static_cast<float>(weights_base[{r} * token_count + t]) * norm * code;"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_sum_{r} = simd_sum(acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] = acc_sum_{r};"
        )
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_mse_weighted_rot_repeat_{repeat_count}",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _mse_scores_weighted_rot_repeat_kernel(repeat_count: int):
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto dim_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        if (dim_idx >= Dim) {",
        "            return;",
        "        }",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = scores_shape[2];",
        "        auto b = n / kv_heads;",
        "        auto h = n % kv_heads;",
        "",
        "        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;",
        "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
        "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float max_{r} = -INFINITY;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            max_{r} = max(max_{r}, static_cast<float>(scores_base[{r} * token_count + t]));"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float max_score_{r} = simd_max(max_{r});")
    lines += [""]
    for r in range(repeat_count):
        lines.append(f"        float acc_{r} = 0.0f;")
        lines.append(f"        float denom_{r} = 0.0f;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
        "            auto token_ptr = packed_ptr + t * PackedWidth;",
        "            uint value = token_ptr[word_idx] >> offset;",
        "            int spill = offset + Bits - 32;",
        "            if (spill > 0) {",
        "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
        "            }",
        "            value &= ((1u << Bits) - 1u);",
        "            float code = codebook[value];",
        "            float norm = static_cast<float>(norms_ptr[t]);",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            float weight_{r} = exp(static_cast<float>(scores_base[{r} * token_count + t]) - max_score_{r});"
        )
        lines.append(f"            acc_{r} += weight_{r} * norm * code;")
        lines.append(f"            denom_{r} += weight_{r};")
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_sum_{r} = simd_sum(acc_{r});")
        lines.append(f"        float denom_sum_{r} = simd_sum(denom_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] ="
        )
        lines.append(f"                acc_sum_{r} / max(denom_sum_{r}, 1e-6f);")
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_mse_scores_weighted_rot_repeat_{repeat_count}",
        input_names=["scores", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _mse_scores_weighted_rot_sum_repeat_kernel(repeat_count: int):
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto dim_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        if (dim_idx >= Dim) {",
        "            return;",
        "        }",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = scores_shape[2];",
        "        auto b = n / kv_heads;",
        "        auto h = n % kv_heads;",
        "",
        "        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;",
        "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
        "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float max_{r} = -INFINITY;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            max_{r} = max(max_{r}, static_cast<float>(scores_base[{r} * token_count + t]));"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float max_score_{r} = simd_max(max_{r});")
        lines.append(f"        float acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
        "            auto token_ptr = packed_ptr + t * PackedWidth;",
        "            uint value = token_ptr[word_idx] >> offset;",
        "            int spill = offset + Bits - 32;",
        "            if (spill > 0) {",
        "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
        "            }",
        "            value &= ((1u << Bits) - 1u);",
        "            float code = codebook[value];",
        "            float norm = static_cast<float>(norms_ptr[t]);",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            float weight_{r} = exp(static_cast<float>(scores_base[{r} * token_count + t]) - max_score_{r});"
        )
        lines.append(f"            acc_{r} += weight_{r} * norm * code;")
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_sum_{r} = simd_sum(acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] = acc_sum_{r};"
        )
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_mse_scores_weighted_rot_sum_repeat_{repeat_count}",
        input_names=["scores", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


def _metal_mse_score(
    q_rot: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or q_rot.ndim != 4
        or state.norms.shape[2] == 0
    ):
        return None

    kernel = _mse_score_kernel()
    if kernel is None:
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    scores = kernel(
        inputs=[
            q_rot,
            state.norms,
            state.indices.astype(mx.uint32),
            codebook,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)


def _metal_qjl_score(
    q_proj: mx.array,
    state: TurboQuantProdState,
    scale: mx.array,
) -> Optional[mx.array]:
    if not _metal_available() or q_proj.ndim != 4 or state.norms.shape[2] == 0:
        return None

    kernel = _qjl_score_kernel()
    if kernel is None:
        return None

    B, H, R, D = q_proj.shape
    T = state.norms.shape[2]
    scores = kernel(
        inputs=[
            q_proj,
            state.norms,
            state.residual_norms,
            state.qjl_signs.astype(mx.uint32),
            scale,
        ],
        template=[
            ("Dim", D),
            ("PackedWidth", state.qjl_signs.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)


def _metal_prod_score(
    q_rot: mx.array,
    q_proj: mx.array,
    state: TurboQuantProdState,
    mse_bits: int,
    codebook: mx.array,
    scale: mx.array,
) -> Optional[mx.array]:
    if (
        mse_bits <= 0
        or not _metal_available()
        or q_rot.ndim != 4
        or q_proj.ndim != 4
        or state.norms.shape[2] == 0
    ):
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    if R > 1:
        kernel = _prod_score_repeat_kernel(R)
        if kernel is not None:
            scores = kernel(
                inputs=[
                    q_rot,
                    q_proj,
                    state.norms,
                    state.residual_norms,
                    state.mse_indices.astype(mx.uint32),
                    state.qjl_signs.astype(mx.uint32),
                    codebook,
                    scale,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("MseBits", mse_bits),
                    ("MsePackedWidth", state.mse_indices.shape[-1]),
                    ("SignPackedWidth", state.qjl_signs.shape[-1]),
                ],
                grid=(32, 1, B * H * T),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, H, R, T)],
                output_dtypes=[mx.float32],
            )[0]
            return mx.expand_dims(scores, axis=3)

    kernel = _prod_score_kernel()
    if kernel is None:
        return None

    scores = kernel(
        inputs=[
            q_rot,
            q_proj,
            state.norms,
            state.residual_norms,
            state.mse_indices.astype(mx.uint32),
            state.qjl_signs.astype(mx.uint32),
            codebook,
            scale,
        ],
        template=[
            ("Dim", D),
            ("MseBits", mse_bits),
            ("MsePackedWidth", state.mse_indices.shape[-1]),
            ("SignPackedWidth", state.qjl_signs.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)


def _metal_polar_prod_score(
    q_rot: mx.array,
    state: TurboQuantPolarProdState,
    level_bits: tuple[int, ...],
    cos_tables: tuple[mx.array, ...],
    sin_tables: tuple[mx.array, ...],
) -> Optional[mx.array]:
    if (
        not _metal_available()
        or q_rot.ndim != 4
        or state.norms.shape[2] == 0
        or len(level_bits) == 0
    ):
        return None

    kernel = _polar_prod_score_kernel(level_bits)
    if kernel is None:
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    levels = len(level_bits)
    inputs = [q_rot, state.norms, state.polar_state.radii]
    inputs.extend(level.astype(mx.uint32) for level in state.polar_state.level_indices)
    for cos_table, sin_table in zip(cos_tables, sin_tables):
        inputs.extend([cos_table, sin_table])

    template = [("Dim", D), ("Levels", levels), ("BlockCount", state.polar_state.radii.shape[-1])]
    for level_idx, level in enumerate(state.polar_state.level_indices, start=1):
        template.append((f"PackedWidth{level_idx}", level.shape[-1]))

    scores = kernel(
        inputs=inputs,
        template=template,
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)


def _metal_polar_turbo_score(
    q_rot: mx.array,
    q_proj: mx.array,
    state: TurboQuantPolarProdState,
    level_bits: tuple[int, ...],
    cos_tables: tuple[mx.array, ...],
    sin_tables: tuple[mx.array, ...],
    scale: mx.array,
) -> Optional[mx.array]:
    if (
        not _metal_available()
        or q_rot.ndim != 4
        or q_proj.ndim != 4
        or q_rot.shape != q_proj.shape
        or state.norms.shape[2] == 0
        or len(level_bits) == 0
    ):
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    levels = len(level_bits)
    kernel = _polar_turbo_score_repeat_kernel(level_bits, R)
    if kernel is None:
        return None

    inputs = [q_rot, q_proj, state.norms, state.polar_state.radii]
    inputs.extend(level.astype(mx.uint32) for level in state.polar_state.level_indices)
    inputs.extend(
        [
            state.residual_norms,
            state.qjl_signs.astype(mx.uint32),
            scale,
        ]
    )
    for cos_table, sin_table in zip(cos_tables, sin_tables):
        inputs.extend([cos_table, sin_table])

    template = [
        ("Dim", D),
        ("Levels", levels),
        ("RepeatCount", R),
        ("BlockCount", state.polar_state.radii.shape[-1]),
        ("SignPackedWidth", state.qjl_signs.shape[-1]),
    ]
    for level_idx, level in enumerate(state.polar_state.level_indices, start=1):
        template.append((f"PackedWidth{level_idx}", level.shape[-1]))

    scores = kernel(
        inputs=inputs,
        template=template,
        grid=(32, 1, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)


def _metal_mse_weighted_sum(
    weights: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
    rotation: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or weights.ndim != 5
        or weights.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    weights_2d = weights.reshape(
        weights.shape[0],
        weights.shape[1],
        weights.shape[2],
        weights.shape[-1],
    )
    B, H, R, T = weights_2d.shape
    D = rotation.shape[0]
    if R > 1:
        kernel = _mse_weighted_rot_repeat_kernel(R)
        if kernel is not None:
            weighted_rot = kernel(
                inputs=[
                    weights_2d,
                    state.norms,
                    state.indices.astype(mx.uint32),
                    codebook,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("Bits", bits),
                    ("PackedWidth", state.indices.shape[-1]),
                ],
                grid=(32, D, B * H),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, H, R, D)],
                output_dtypes=[mx.float32],
            )[0]
            output = mx.matmul(weighted_rot, rotation)
            return mx.expand_dims(output, axis=3)

    kernel = _mse_weighted_rot_kernel()
    if kernel is None:
        return None

    weighted_rot = kernel(
        inputs=[
            weights_2d,
            state.norms,
            state.indices.astype(mx.uint32),
            codebook,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H * R),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)


def _metal_mse_weighted_sum_from_scores(
    scores: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
    rotation: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or scores.ndim != 5
        or scores.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    scores_2d = scores.reshape(
        scores.shape[0],
        scores.shape[1],
        scores.shape[2],
        scores.shape[-1],
    )
    B, H, R, T = scores_2d.shape
    if R <= 1:
        return None

    kernel = _mse_scores_weighted_rot_repeat_kernel(R)
    if kernel is None:
        return None

    D = rotation.shape[0]
    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices.astype(mx.uint32),
            codebook,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)


def _metal_mse_weighted_sum_sum_from_scores(
    scores: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
    rotation: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or scores.ndim != 5
        or scores.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    scores_2d = scores.reshape(
        scores.shape[0],
        scores.shape[1],
        scores.shape[2],
        scores.shape[-1],
    )
    B, H, R, T = scores_2d.shape
    if R <= 1:
        return None

    kernel = _mse_scores_weighted_rot_sum_repeat_kernel(R)
    if kernel is None:
        return None

    D = rotation.shape[0]
    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices.astype(mx.uint32),
            codebook,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)


@lru_cache(maxsize=None)
def _compiled_integer_decode_kernel(bits: int):
    mse_bits = max(bits - 1, 0)

    @mx.compile
    def _decode(
        grouped_queries: mx.array,
        key_norms: mx.array,
        key_mse_indices: mx.array,
        key_residual_norms: mx.array,
        key_qjl_signs: mx.array,
        value_norms: mx.array,
        value_indices: mx.array,
        key_query_transform_t: mx.array,
        key_codebook: mx.array,
        key_scale: mx.array,
        value_codebook: mx.array,
        value_rotation: mx.array,
    ) -> mx.array:
        query_transformed = mx.matmul(grouped_queries, key_query_transform_t)
        dim = grouped_queries.shape[-1]
        q_rot = query_transformed[..., :dim]
        q_proj = query_transformed[..., dim:]
        scores = _metal_prod_score(
            q_rot.reshape(q_rot.shape[0], q_rot.shape[1], q_rot.shape[2], q_rot.shape[-1]),
            q_proj.reshape(
                q_proj.shape[0], q_proj.shape[1], q_proj.shape[2], q_proj.shape[-1]
            ),
            TurboQuantProdState(
                key_norms,
                key_mse_indices,
                key_residual_norms,
                key_qjl_signs,
            ),
            mse_bits,
            key_codebook,
            key_scale,
        )
        return _metal_mse_weighted_sum_from_scores(
            scores,
            TurboQuantMSEState(value_norms, value_indices),
            bits,
            value_codebook,
            value_rotation,
        )

    return _decode


class TurboQuantMSEState(NamedTuple):
    norms: mx.array
    indices: mx.array


class TurboQuantProdState(NamedTuple):
    norms: mx.array
    mse_indices: mx.array
    residual_norms: mx.array
    qjl_signs: mx.array


class TurboQuantPolarState(NamedTuple):
    radii: mx.array
    level_indices: tuple[mx.array, ...]


class TurboQuantPolarProdState(NamedTuple):
    norms: mx.array
    polar_state: TurboQuantPolarState
    residual_norms: mx.array
    qjl_signs: mx.array


class TurboQuantSplitState(NamedTuple):
    low: object
    high: object


def _validate_bits(bits: float) -> float:
    bits = float(bits)
    if bits < 1:
        raise ValueError("TurboQuant requires kv_bits >= 1.")
    rounded = round(bits * 2) / 2
    if not math.isclose(bits, rounded, abs_tol=1e-6):
        raise ValueError(
            f"TurboQuant currently supports integer and .5 bit-widths, got {bits}."
        )
    return rounded


def turboquant_enabled(bits: Optional[float], scheme: Optional[str] = None) -> bool:
    if bits is None:
        return False
    if scheme == "turboquant":
        return True
    bits = float(bits)
    return not math.isclose(bits, round(bits), abs_tol=1e-6)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _polar_levels(dim: int) -> int:
    if dim <= 1:
        return 0
    return min(_POLAR_MAX_LEVELS, int(math.log2(dim)))


def _polar_level_bits(dim: int, bits: int) -> tuple[int, ...]:
    if bits != 4:
        raise ValueError(f"PolarQuant key codec currently expects 4 bits, got {bits}.")
    levels = _polar_levels(dim)
    if levels == 0:
        return ()
    return (4,) + (2,) * (levels - 1)


@lru_cache(maxsize=None)
def _rotation_matrix(dim: int, seed: int) -> mx.array:
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    if dim == 1:
        return mx.ones((1, 1), dtype=mx.float32)

    rng = np.random.default_rng(seed + dim * 7919)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    return mx.array(q.astype(np.float32))


@lru_cache(maxsize=None)
def _projection_matrix(dim: int, seed: int) -> mx.array:
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    rng = np.random.default_rng(seed + dim * 2971 + 17)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    return mx.array(matrix.astype(np.float32))


def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 1:
        pdf = np.ones_like(grid)
    else:
        coeff = math.gamma(dim / 2) / (
            math.sqrt(math.pi) * math.gamma((dim - 1) / 2)
        )
        pdf = coeff * np.power(np.clip(1.0 - grid**2, 0.0, None), (dim - 3) / 2)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _codebook(dim: int, bits: int) -> mx.array:
    if bits <= 0:
        return mx.zeros((0,), dtype=mx.float32)
    levels = 1 << bits
    if dim <= 1:
        centroids = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
        return mx.array(centroids)

    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return mx.array(centroids.astype(np.float32))


def _polar_angle_pdf(grid: np.ndarray, level: int) -> np.ndarray:
    if level <= 1:
        pdf = np.ones_like(grid)
    else:
        exponent = (1 << (level - 1)) - 1
        pdf = np.power(np.clip(np.sin(2.0 * grid), 0.0, None), exponent)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _polar_angle_codebook(level: int, bits: int) -> mx.array:
    if bits <= 0:
        return mx.zeros((0,), dtype=mx.float32)

    level_count = 1 << bits
    if level <= 1:
        step = (2.0 * math.pi) / level_count
        centroids = np.arange(level_count, dtype=np.float32) * step + step / 2.0
        return mx.array(centroids.astype(np.float32))

    grid = np.linspace(1e-6, math.pi / 2 - 1e-6, 32768, dtype=np.float32)
    weights = _polar_angle_pdf(grid, level)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(level_count, dtype=np.float32) + 0.5) / level_count
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(level_count + 1, dtype=np.float32)
        boundaries[0] = 0.0
        boundaries[-1] = math.pi / 2
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(level_count):
            if i == level_count - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return mx.array(centroids.astype(np.float32))


def _packed_width(length: int, bits: int) -> int:
    if length == 0 or bits == 0:
        return 0
    return (length * bits + 31) // 32


def _pack_lowbit(values: mx.array, bits: int) -> mx.array:
    if bits == 0:
        return mx.zeros((*values.shape[:-1], 0), dtype=mx.uint32)

    values = values.astype(mx.uint32)
    length = values.shape[-1]
    packed_width = _packed_width(length, bits)
    flat = values.reshape((-1, length))

    kernel = _pack_lowbit_kernel()
    if kernel is not None:
        packed = kernel(
            inputs=[flat],
            template=[
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", packed_width),
            ],
            grid=(packed_width, flat.shape[0], 1),
            threadgroup=(min(32, packed_width), 1, 1),
            output_shapes=[(flat.shape[0], packed_width)],
            output_dtypes=[mx.uint32],
        )[0]
        return packed.reshape((*values.shape[:-1], packed_width))

    packed = mx.zeros((flat.shape[0], packed_width), dtype=mx.uint32)

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        packed[:, word_idx] |= flat[:, idx] << offset
        spill = offset + bits - 32
        if spill > 0:
            packed[:, word_idx + 1] |= flat[:, idx] >> (bits - spill)

    return packed.reshape((*values.shape[:-1], packed_width))


def _unpack_lowbit(packed: mx.array, bits: int, length: int) -> mx.array:
    if bits == 0:
        return mx.zeros((*packed.shape[:-1], 0), dtype=mx.uint32)

    packed = packed.astype(mx.uint32)
    flat = packed.reshape((-1, packed.shape[-1]))

    kernel = _unpack_lowbit_kernel()
    if kernel is not None:
        unpacked = kernel(
            inputs=[flat],
            template=[
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", flat.shape[-1]),
            ],
            grid=(length, flat.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(flat.shape[0], length)],
            output_dtypes=[mx.uint32],
        )[0]
        return unpacked.reshape((*packed.shape[:-1], length))

    unpacked = mx.zeros((flat.shape[0], length), dtype=mx.uint32)
    mask = (1 << bits) - 1

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        value = flat[:, word_idx] >> offset
        spill = offset + bits - 32
        if spill > 0:
            value |= flat[:, word_idx + 1] << (bits - spill)
        unpacked[:, idx] = value & mask

    return unpacked.reshape((*packed.shape[:-1], length))


def _concat_state(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if isinstance(lhs, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.indices, rhs.indices], axis=2),
        )
    if isinstance(lhs, TurboQuantProdState):
        return TurboQuantProdState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.mse_indices, rhs.mse_indices], axis=2),
            mx.concatenate([lhs.residual_norms, rhs.residual_norms], axis=2),
            mx.concatenate([lhs.qjl_signs, rhs.qjl_signs], axis=2),
        )
    if isinstance(lhs, TurboQuantPolarState):
        return TurboQuantPolarState(
            mx.concatenate([lhs.radii, rhs.radii], axis=2),
            tuple(
                mx.concatenate([lhs_idx, rhs_idx], axis=2)
                for lhs_idx, rhs_idx in zip(lhs.level_indices, rhs.level_indices)
            ),
        )
    if isinstance(lhs, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            _concat_state(lhs.polar_state, rhs.polar_state),
            mx.concatenate([lhs.residual_norms, rhs.residual_norms], axis=2),
            mx.concatenate([lhs.qjl_signs, rhs.qjl_signs], axis=2),
        )
    if isinstance(lhs, TurboQuantSplitState):
        return TurboQuantSplitState(
            _concat_state(lhs.low, rhs.low),
            _concat_state(lhs.high, rhs.high),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(lhs)!r}")


def _slice_state(state, end: int):
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(state.norms[..., :end], state.indices[..., :end, :])
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[..., :end],
            state.mse_indices[..., :end, :],
            state.residual_norms[..., :end],
            state.qjl_signs[..., :end, :],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[..., :end, :],
            tuple(level[..., :end, :] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[..., :end],
            _slice_state(state.polar_state, end),
            state.residual_norms[..., :end],
            state.qjl_signs[..., :end, :],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _slice_state(state.low, end),
            _slice_state(state.high, end),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _slice_state_range(state, start: int, end: int):
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            state.norms[..., start:end],
            state.indices[..., start:end, :],
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[..., start:end],
            state.mse_indices[..., start:end, :],
            state.residual_norms[..., start:end],
            state.qjl_signs[..., start:end, :],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[..., start:end, :],
            tuple(level[..., start:end, :] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[..., start:end],
            _slice_state_range(state.polar_state, start, end),
            state.residual_norms[..., start:end],
            state.qjl_signs[..., start:end, :],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _slice_state_range(state.low, start, end),
            _slice_state_range(state.high, start, end),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _state_nbytes(state) -> int:
    if state is None:
        return 0
    if isinstance(state, TurboQuantSplitState):
        return _state_nbytes(state.low) + _state_nbytes(state.high)
    if isinstance(state, tuple):
        return sum(_state_nbytes(part) for part in state)
    if isinstance(state, mx.array):
        return state.nbytes
    return 0


def _state_length(state) -> int:
    if state is None:
        return 0
    if isinstance(state, TurboQuantSplitState):
        return _state_length(state.low)
    if isinstance(state, TurboQuantMSEState):
        return state.norms.shape[2]
    if isinstance(state, TurboQuantProdState):
        return state.norms.shape[2]
    if isinstance(state, TurboQuantPolarState):
        return state.radii.shape[2]
    if isinstance(state, TurboQuantPolarProdState):
        return state.norms.shape[2]
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _allocate_state_like(state, length: int):
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.zeros((*state.norms.shape[:2], length), dtype=state.norms.dtype),
            mx.zeros(
                (*state.indices.shape[:2], length, state.indices.shape[-1]),
                dtype=state.indices.dtype,
            ),
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            mx.zeros((*state.norms.shape[:2], length), dtype=state.norms.dtype),
            mx.zeros(
                (*state.mse_indices.shape[:2], length, state.mse_indices.shape[-1]),
                dtype=state.mse_indices.dtype,
            ),
            mx.zeros(
                (*state.residual_norms.shape[:2], length),
                dtype=state.residual_norms.dtype,
            ),
            mx.zeros(
                (*state.qjl_signs.shape[:2], length, state.qjl_signs.shape[-1]),
                dtype=state.qjl_signs.dtype,
            ),
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            mx.zeros(
                (*state.radii.shape[:2], length, state.radii.shape[-1]),
                dtype=state.radii.dtype,
            ),
            tuple(
                mx.zeros(
                    (*level.shape[:2], length, level.shape[-1]),
                    dtype=level.dtype,
                )
                for level in state.level_indices
            ),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            mx.zeros((*state.norms.shape[:2], length), dtype=state.norms.dtype),
            _allocate_state_like(state.polar_state, length),
            mx.zeros(
                (*state.residual_norms.shape[:2], length),
                dtype=state.residual_norms.dtype,
            ),
            mx.zeros(
                (*state.qjl_signs.shape[:2], length, state.qjl_signs.shape[-1]),
                dtype=state.qjl_signs.dtype,
            ),
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _allocate_state_like(state.low, length),
            _allocate_state_like(state.high, length),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _write_state(dst, src, start: int):
    if src is None:
        return
    end = start + _state_length(src)
    if isinstance(dst, TurboQuantMSEState):
        dst.norms[..., start:end] = src.norms
        dst.indices[..., start:end, :] = src.indices
        return
    if isinstance(dst, TurboQuantProdState):
        dst.norms[..., start:end] = src.norms
        dst.mse_indices[..., start:end, :] = src.mse_indices
        dst.residual_norms[..., start:end] = src.residual_norms
        dst.qjl_signs[..., start:end, :] = src.qjl_signs
        return
    if isinstance(dst, TurboQuantPolarState):
        dst.radii[..., start:end, :] = src.radii
        for dst_level, src_level in zip(dst.level_indices, src.level_indices):
            dst_level[..., start:end, :] = src_level
        return
    if isinstance(dst, TurboQuantPolarProdState):
        dst.norms[..., start:end] = src.norms
        _write_state(dst.polar_state, src.polar_state, start)
        dst.residual_norms[..., start:end] = src.residual_norms
        dst.qjl_signs[..., start:end, :] = src.qjl_signs
        return
    if isinstance(dst, TurboQuantSplitState):
        _write_state(dst.low, src.low, start)
        _write_state(dst.high, src.high, start)
        return
    raise TypeError(f"Unsupported TurboQuant state type: {type(dst)!r}")


def _reserve_state_capacity(state, used: int, needed: int, step: int):
    if state is None:
        return None
    capacity = _state_length(state)
    if capacity >= needed:
        return state
    # Round up to next step boundary — avoids 2x growth spikes
    new_capacity = ((needed + step - 1) // step) * step
    grown = _allocate_state_like(state, new_capacity)
    if used > 0:
        _write_state(grown, _slice_state(state, used), 0)
    return grown


class _TurboQuantMSECodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.rotation = _rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.transpose() if dim > 0 else self.rotation
        self.codebook = _codebook(dim, bits)

    def _quantize_unit_with_estimate(
        self, unit_vectors: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.bits == 0:
            return (
                mx.zeros((*unit_vectors.shape[:-1], 0), dtype=mx.uint32),
                mx.zeros(unit_vectors.shape, dtype=mx.float32),
            )

        rotated = mx.matmul(unit_vectors, self.rotation_t)
        distances = mx.abs(rotated[..., None] - self.codebook)
        indices = mx.argmin(distances, axis=-1).astype(mx.uint32)
        packed = _pack_lowbit(indices, self.bits)
        estimated_rotated = mx.take(self.codebook, indices, axis=0)
        return packed, mx.matmul(estimated_rotated, self.rotation)

    def _quantize_unit(self, unit_vectors: mx.array) -> mx.array:
        packed, _ = self._quantize_unit_with_estimate(unit_vectors)
        return packed

    def _dequantize_unit(self, packed_indices: mx.array) -> mx.array:
        if self.bits == 0:
            return mx.zeros((*packed_indices.shape[:-1], self.dim), dtype=mx.float32)

        indices = _unpack_lowbit(packed_indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        return mx.matmul(rotated, self.rotation)

    def quantize(self, vectors: mx.array) -> TurboQuantMSEState:
        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        # safe_norms >= _EPS, so division never produces inf/nan
        unit_vectors = vectors_f32 / mx.maximum(norms[..., None], _EPS)
        return TurboQuantMSEState(
            norms.astype(mx.float16),
            self._quantize_unit(unit_vectors),
        )

    def dequantize(self, state: TurboQuantMSEState) -> mx.array:
        unit_vectors = self._dequantize_unit(state.indices)
        return state.norms[..., None].astype(unit_vectors.dtype) * unit_vectors

    def prepare_queries(self, queries: mx.array) -> mx.array:
        return mx.matmul(queries, self.rotation_t)

    def score_prepared(
        self, prepared_queries: mx.array, state: TurboQuantMSEState
    ) -> mx.array:
        if prepared_queries.shape[-2] == 1:
            fast_scores = _metal_mse_score(
                prepared_queries.reshape(
                    prepared_queries.shape[0],
                    prepared_queries.shape[1],
                    prepared_queries.shape[2],
                    prepared_queries.shape[-1],
                ),
                state,
                self.bits,
                self.codebook,
            )
            if fast_scores is not None:
                return fast_scores

        indices = _unpack_lowbit(state.indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        dots = mx.einsum("bhmld,bhtd->bhmlt", prepared_queries, rotated)
        return dots * state.norms.astype(mx.float32)[:, :, None, None, :]

    def score(self, queries: mx.array, state: TurboQuantMSEState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)

    def weighted_sum(self, weights: mx.array, state: TurboQuantMSEState) -> mx.array:
        if weights.shape[-2] == 1:
            fast_output = _metal_mse_weighted_sum(
                weights,
                state,
                self.bits,
                self.codebook,
                self.rotation,
            )
            if fast_output is not None:
                return fast_output

        indices = _unpack_lowbit(state.indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        weighted_rot = mx.einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights,
            state.norms.astype(mx.float32),
            rotated,
        )
        return mx.matmul(weighted_rot, self.rotation)

    def weighted_sum_from_scores(
        self, scores: mx.array, state: TurboQuantMSEState
    ) -> mx.array:
        fast_output = _metal_mse_weighted_sum_from_scores(
            scores,
            state,
            self.bits,
            self.codebook,
            self.rotation,
        )
        if fast_output is not None:
            return fast_output
        return self.weighted_sum(mx.softmax(scores, axis=-1), state)

    def weighted_sum_stats_from_scores(
        self, scores: mx.array, state: TurboQuantMSEState
    ) -> tuple[mx.array, mx.array, mx.array]:
        max_scores = mx.max(scores, axis=-1)
        fast_output = _metal_mse_weighted_sum_sum_from_scores(
            scores,
            state,
            self.bits,
            self.codebook,
            self.rotation,
        )
        if fast_output is not None:
            denom = mx.sum(mx.exp(scores - max_scores[..., None]), axis=-1)
            return fast_output, denom, max_scores

        weights = mx.exp(scores - max_scores[..., None])
        output = self.weighted_sum(weights, state)
        denom = mx.sum(weights, axis=-1)
        return output, denom, max_scores


class _PolarQuantUnitCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        if not _is_power_of_two(dim):
            raise ValueError(f"PolarQuant requires a power-of-two dimension, got {dim}.")
        self.dim = dim
        self.bits = bits
        self.level_bits = _polar_level_bits(dim, bits)
        self.levels = len(self.level_bits)
        self.rotation = _rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.transpose() if dim > 0 else self.rotation
        self.angle_codebooks = tuple(
            _polar_angle_codebook(level, level_bits)
            for level, level_bits in enumerate(self.level_bits, start=1)
        )
        self.cos_tables = tuple(mx.cos(codebook) for codebook in self.angle_codebooks)
        self.sin_tables = tuple(mx.sin(codebook) for codebook in self.angle_codebooks)

    def _quantize_level(self, angles: mx.array, level: int) -> mx.array:
        codebook = self.angle_codebooks[level - 1]
        diffs = mx.abs(angles[..., None] - codebook)
        if level == 1:
            diffs = mx.minimum(diffs, (2.0 * math.pi) - diffs)
        return mx.argmin(diffs, axis=-1).astype(mx.uint32)

    def _dequantize_preconditioned(self, state: TurboQuantPolarState) -> mx.array:
        radii = state.radii.astype(mx.float32)
        for bits, indices_packed, cos_table, sin_table in zip(
            reversed(self.level_bits),
            reversed(state.level_indices),
            reversed(self.cos_tables),
            reversed(self.sin_tables),
        ):
            angle_count = radii.shape[-1]
            indices = _unpack_lowbit(indices_packed, bits, angle_count).astype(mx.int32)
            cosines = mx.take(cos_table, indices, axis=0)
            sines = mx.take(sin_table, indices, axis=0)
            radii = mx.stack([radii * cosines, radii * sines], axis=-1).reshape(
                (*radii.shape[:-1], radii.shape[-1] * 2)
            )
        return radii

    def quantize_unit_with_estimate(
        self, unit_vectors: mx.array, storage_dtype
    ) -> tuple[TurboQuantPolarState, mx.array]:
        preconditioned = mx.matmul(unit_vectors, self.rotation_t)
        radii = preconditioned
        packed_levels = []
        for level, bits in enumerate(self.level_bits, start=1):
            pairs = radii.reshape((*radii.shape[:-1], radii.shape[-1] // 2, 2))
            angles = mx.arctan2(pairs[..., 1], pairs[..., 0])
            if level == 1:
                angles = mx.where(angles < 0, angles + 2.0 * math.pi, angles)
            indices = self._quantize_level(angles, level)
            packed_levels.append(_pack_lowbit(indices, bits))
            radii = mx.linalg.norm(pairs, axis=-1)

        state = TurboQuantPolarState(
            radii.astype(storage_dtype),
            tuple(packed_levels),
        )
        approx_preconditioned = self._dequantize_preconditioned(state)
        approx_unit = mx.matmul(approx_preconditioned, self.rotation)
        return state, approx_unit

    def dequantize_unit(self, state: TurboQuantPolarState) -> mx.array:
        return mx.matmul(self._dequantize_preconditioned(state), self.rotation)

    def score_prepared(
        self, prepared_queries: mx.array, state: TurboQuantPolarState, norms: mx.array
    ) -> mx.array:
        if prepared_queries.shape[-2] == 1:
            fast_scores = _metal_polar_prod_score(
                prepared_queries.reshape(
                    prepared_queries.shape[0],
                    prepared_queries.shape[1],
                    prepared_queries.shape[2],
                    prepared_queries.shape[-1],
                ),
                TurboQuantPolarProdState(
                    norms,
                    state,
                    mx.zeros_like(norms),
                    mx.zeros((*norms.shape, 0), dtype=mx.uint32),
                ),
                self.level_bits,
                self.cos_tables,
                self.sin_tables,
            )
            if fast_scores is not None:
                return fast_scores

        approx_preconditioned = self._dequantize_preconditioned(state)
        dots = mx.einsum("bhmld,bhtd->bhmlt", prepared_queries, approx_preconditioned)
        return dots * norms.astype(mx.float32)[:, :, None, None, :]


class _TurboQuantPolarProdCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.polar_codec = _PolarQuantUnitCodec(dim, bits, seed)
        self.projection = _projection_matrix(dim, seed + 1)
        self.projection_t = (
            self.projection.transpose() if dim > 0 else self.projection
        )
        self.query_transform_t = (
            mx.concatenate([self.polar_codec.rotation_t, self.projection_t], axis=-1)
            if dim > 0
            else mx.zeros((0, 0), dtype=mx.float32)
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
        self.scale_array = mx.array([self.scale], dtype=mx.float32)

    def quantize(self, vectors: mx.array) -> TurboQuantPolarProdState:
        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        unit_vectors = vectors_f32 / mx.maximum(norms[..., None], _EPS)

        polar_state, approx_unit = self.polar_codec.quantize_unit_with_estimate(
            unit_vectors,
            storage_dtype=vectors.dtype,
        )
        residual = unit_vectors - approx_unit
        residual_norms = mx.linalg.norm(residual, axis=-1)
        projected = mx.matmul(residual, self.projection_t)
        signs = mx.where(projected >= 0, 1, 0).astype(mx.uint32)

        return TurboQuantPolarProdState(
            norms.astype(mx.float16),
            polar_state,
            residual_norms.astype(mx.float16),
            _pack_lowbit(signs, 1),
        )

    def dequantize(self, state: TurboQuantPolarProdState) -> mx.array:
        polar_unit = self.polar_codec.dequantize_unit(state.polar_state)
        sign_bits = _unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = self.scale * state.residual_norms[..., None].astype(
            mx.float32
        ) * mx.matmul(signs, self.projection)
        return state.norms[..., None].astype(mx.float32) * (polar_unit + qjl_unit)

    def prepare_queries(self, queries: mx.array) -> tuple[mx.array, mx.array]:
        transformed = mx.matmul(queries, self.query_transform_t)
        return transformed[..., : self.dim], transformed[..., self.dim :]

    def score_prepared(
        self,
        prepared_queries: tuple[mx.array, mx.array],
        state: TurboQuantPolarProdState,
    ) -> mx.array:
        polar_queries, proj_queries = prepared_queries
        if proj_queries.shape[-2] == 1:
            fast_scores = _metal_polar_turbo_score(
                polar_queries.reshape(
                    polar_queries.shape[0],
                    polar_queries.shape[1],
                    polar_queries.shape[2],
                    polar_queries.shape[-1],
                ),
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.polar_codec.level_bits,
                self.polar_codec.cos_tables,
                self.polar_codec.sin_tables,
                self.scale_array,
            )
            if fast_scores is not None:
                return fast_scores

        polar_score = self.polar_codec.score_prepared(
            polar_queries,
            state.polar_state,
            state.norms,
        )

        if proj_queries.shape[-2] == 1:
            fast_qjl = _metal_qjl_score(
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.scale_array,
            )
            if fast_qjl is not None:
                return polar_score + fast_qjl

        sign_bits = _unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_score = self.scale * state.residual_norms.astype(mx.float32)[
            :, :, None, None, :
        ] * mx.einsum(
            "bhmld,bhtd->bhmlt",
            proj_queries,
            signs,
        )

        norms = state.norms.astype(mx.float32)[:, :, None, None, :]
        return polar_score + norms * qjl_score

    def score(self, queries: mx.array, state: TurboQuantPolarProdState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)


class _TurboQuantProdCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.mse_codec = _TurboQuantMSECodec(dim, max(bits - 1, 0), seed)
        self.projection = _projection_matrix(dim, seed + 1)
        self.projection_t = (
            self.projection.transpose() if dim > 0 else self.projection
        )
        self.query_transform_t = (
            mx.concatenate([self.mse_codec.rotation_t, self.projection_t], axis=-1)
            if dim > 0
            else mx.zeros((0, 0), dtype=mx.float32)
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
        self.scale_array = mx.array([self.scale], dtype=mx.float32)

    def quantize(self, vectors: mx.array) -> TurboQuantProdState:
        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        unit_vectors = vectors_f32 / mx.maximum(norms[..., None], _EPS)

        mse_indices, mse_unit = self.mse_codec._quantize_unit_with_estimate(
            unit_vectors
        )
        residual = unit_vectors - mse_unit
        residual_norms = mx.linalg.norm(residual, axis=-1)
        projected = mx.matmul(residual, self.projection_t)
        signs = mx.where(projected >= 0, 1, 0).astype(mx.uint32)

        return TurboQuantProdState(
            norms.astype(mx.float16),
            mse_indices,
            residual_norms.astype(mx.float16),
            _pack_lowbit(signs, 1),
        )

    def dequantize(self, state: TurboQuantProdState) -> mx.array:
        mse_unit = self.mse_codec._dequantize_unit(state.mse_indices)
        sign_bits = _unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = self.scale * state.residual_norms[..., None].astype(
            mx.float32
        ) * mx.matmul(signs, self.projection)
        return state.norms[..., None].astype(mx.float32) * (mse_unit + qjl_unit)

    def prepare_queries(self, queries: mx.array) -> tuple[mx.array, mx.array]:
        transformed = mx.matmul(queries, self.query_transform_t)
        return transformed[..., : self.dim], transformed[..., self.dim :]

    def score_prepared(
        self,
        prepared_queries: tuple[mx.array, mx.array],
        state: TurboQuantProdState,
    ) -> mx.array:
        mse_queries, proj_queries = prepared_queries
        if proj_queries.shape[-2] == 1:
            fast_scores = _metal_prod_score(
                mse_queries.reshape(
                    mse_queries.shape[0],
                    mse_queries.shape[1],
                    mse_queries.shape[2],
                    mse_queries.shape[-1],
                ),
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.mse_codec.bits,
                self.mse_codec.codebook,
                self.scale_array,
            )
            if fast_scores is not None:
                return fast_scores

        if self.mse_codec.bits > 0:
            mse_score = self.mse_codec.score_prepared(
                mse_queries,
                TurboQuantMSEState(state.norms, state.mse_indices),
            )
        else:
            mse_score = mx.zeros(
                (
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[3],
                    state.norms.shape[2],
                ),
                dtype=mx.float32,
            )

        if proj_queries.shape[-2] == 1:
            fast_qjl = _metal_qjl_score(
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.scale_array,
            )
            if fast_qjl is not None:
                return mse_score + fast_qjl

        sign_bits = _unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_score = self.scale * state.residual_norms.astype(mx.float32)[
            :, :, None, None, :
        ] * mx.einsum(
            "bhmld,bhtd->bhmlt",
            proj_queries,
            signs,
        )

        norms = state.norms.astype(mx.float32)[:, :, None, None, :]
        return mse_score + norms * qjl_score

    def score(self, queries: mx.array, state: TurboQuantProdState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)


def _select_outlier_indices(tensor: mx.array, avg_bits: float) -> tuple[np.ndarray, np.ndarray]:
    lower_bits = math.floor(avg_bits)
    upper_bits = math.ceil(avg_bits)
    if lower_bits == upper_bits:
        raise ValueError("Mixed-precision selection requires a fractional bit-width.")

    dim = tensor.shape[-1]
    high_count = int(round((avg_bits - lower_bits) * dim / (upper_bits - lower_bits)))
    high_count = max(1, min(dim - 1, high_count))

    scores = mx.mean(mx.abs(tensor.astype(mx.float32)), axis=(0, 1, 2))
    order = np.argsort(np.asarray(scores))
    high_idx = np.sort(order[-high_count:].astype(np.int32))
    low_mask = np.ones(dim, dtype=bool)
    low_mask[high_idx] = False
    low_idx = np.nonzero(low_mask)[0].astype(np.int32)
    return low_idx, high_idx


class _SplitCodec:
    def __init__(self, tensor: mx.array, bits: float, mode: str, seed: int):
        self.bits = bits
        self.mode = mode
        self.dim = tensor.shape[-1]
        self.lower_bits = math.floor(bits)
        self.upper_bits = math.ceil(bits)
        low_idx, high_idx = _select_outlier_indices(tensor, bits)
        self.low_idx = mx.array(low_idx, dtype=mx.int32)
        self.high_idx = mx.array(high_idx, dtype=mx.int32)

        concat_order = np.concatenate([low_idx, high_idx])
        self.restore_order = mx.array(np.argsort(concat_order), dtype=mx.int32)

        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        self.low_codec = codec_cls(len(low_idx), self.lower_bits, seed)
        self.high_codec = codec_cls(len(high_idx), self.upper_bits, seed + 97)

    def quantize(self, tensor: mx.array) -> TurboQuantSplitState:
        low_tensor = mx.take(tensor, self.low_idx, axis=-1)
        high_tensor = mx.take(tensor, self.high_idx, axis=-1)
        return TurboQuantSplitState(
            self.low_codec.quantize(low_tensor),
            self.high_codec.quantize(high_tensor),
        )

    def dequantize(self, state: TurboQuantSplitState) -> mx.array:
        low_tensor = self.low_codec.dequantize(state.low)
        high_tensor = self.high_codec.dequantize(state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def prepare_queries(self, queries: mx.array):
        low_tensor = mx.take(queries, self.low_idx, axis=-1)
        high_tensor = mx.take(queries, self.high_idx, axis=-1)
        return (
            self.low_codec.prepare_queries(low_tensor),
            self.high_codec.prepare_queries(high_tensor),
        )

    def score_prepared(self, prepared_queries, state: TurboQuantSplitState) -> mx.array:
        low_queries, high_queries = prepared_queries
        return self.low_codec.score_prepared(
            low_queries,
            state.low,
        ) + self.high_codec.score_prepared(
            high_queries,
            state.high,
        )

    def score(self, queries: mx.array, state: TurboQuantSplitState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)

    def weighted_sum(self, weights: mx.array, state: TurboQuantSplitState) -> mx.array:
        low_tensor = self.low_codec.weighted_sum(weights, state.low)
        high_tensor = self.high_codec.weighted_sum(weights, state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def weighted_sum_from_scores(
        self, scores: mx.array, state: TurboQuantSplitState
    ) -> mx.array:
        low_tensor = self.low_codec.weighted_sum_from_scores(scores, state.low)
        high_tensor = self.high_codec.weighted_sum_from_scores(scores, state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def weighted_sum_stats_from_scores(
        self, scores: mx.array, state: TurboQuantSplitState
    ) -> tuple[mx.array, mx.array, mx.array]:
        low_tensor, denom, max_scores = self.low_codec.weighted_sum_stats_from_scores(
            scores, state.low
        )
        high_tensor, _, _ = self.high_codec.weighted_sum_stats_from_scores(
            scores, state.high
        )
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1), denom, max_scores


def _build_codec(tensor: mx.array, bits: float, mode: str, seed: int):
    bits = _validate_bits(bits)
    if math.isclose(bits, round(bits), abs_tol=1e-6):
        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        return codec_cls(tensor.shape[-1], int(round(bits)), seed)
    return _SplitCodec(tensor, bits, mode, seed)


class TurboQuantKVCache(_BaseCache):
    decode_key_chunk_size = 65536
    prefill_key_chunk_size = 512
    prefill_query_block_size = 16
    cache_step = 256

    def __init__(self, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED):
        self.bits = _validate_bits(bits)
        self.seed = seed
        self.offset = 0
        self.keys = None
        self.values = None
        self.key_codec = None
        self.value_codec = None
        self._cached_state = None
        self._cached_state_offset = -1

    @classmethod
    def from_cache(
        cls, cache, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED
    ) -> "TurboQuantKVCache":
        turbo_cache = cls(bits=bits, seed=seed)
        keys, values = cache.state
        if keys is not None:
            turbo_cache.update_and_fetch(keys, values)
        return turbo_cache

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self.key_codec is None:
            self.key_codec = _build_codec(keys, self.bits, mode="prod", seed=self.seed)
        if self.value_codec is None:
            self.value_codec = _build_codec(
                values, self.bits, mode="mse", seed=self.seed + 1
            )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._ensure_codecs(keys, values)
        new_keys = self.key_codec.quantize(keys)
        new_values = self.value_codec.quantize(values)

        new_end = self.offset + keys.shape[2]
        if self.keys is None:
            self.keys = _allocate_state_like(new_keys, new_end)
            self.values = _allocate_state_like(new_values, new_end)
        else:
            self.keys = _reserve_state_capacity(
                self.keys, self.offset, new_end, self.cache_step
            )
            self.values = _reserve_state_capacity(
                self.values, self.offset, new_end, self.cache_step
            )

        _write_state(self.keys, new_keys, self.offset)
        _write_state(self.values, new_values, self.offset)
        n_new = keys.shape[2]
        self.offset = new_end
        self._cached_state = None
        self._cached_state_offset = -1
        # Only eval during prefill (multiple tokens) to prevent graph buildup.
        # Single-token decode steps have a tiny graph — eval would stall the pipeline.
        if n_new > 1:
            mx.eval(self.keys, self.values)
        return self.state

    def dequantize(self, keys_state=None, values_state=None):
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys = self.key_codec.dequantize(keys_state).astype(mx.float32)
        values = self.value_codec.dequantize(values_state).astype(mx.float32)
        return keys, values

    def _apply_attention_mask(
        self,
        scores: mx.array,
        mask: Optional[mx.array],
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        total_queries: int,
        total_tokens: int,
    ) -> mx.array:
        if mask is None:
            return scores
        if isinstance(mask, str):
            if mask == "causal":
                past_tokens = total_tokens - total_queries
                q_idx = mx.arange(past_tokens + q_start, past_tokens + q_end)
                k_idx = mx.arange(k_start, k_end)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                causal_mask = causal_mask[None, None, None, :, :]
                return mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
            raise ValueError(f"Unsupported TurboQuant attention mask: {mask}")

        mask_chunk = mask[..., q_start:q_end, k_start:k_end]
        if mask_chunk.ndim == scores.ndim - 1:
            mask_chunk = mx.expand_dims(mask_chunk, axis=2)

        if mask_chunk.dtype == mx.bool_:
            return mx.where(mask_chunk, scores, mx.finfo(scores.dtype).min)
        return scores + mask_chunk

    def quantized_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state

        B, n_q_heads, L, D = queries.shape
        n_kv_heads = keys_state.low.norms.shape[1] if isinstance(
            keys_state, TurboQuantSplitState
        ) else keys_state.norms.shape[1]
        n_repeats = n_q_heads // n_kv_heads

        grouped_queries = (queries * scale).reshape(
            B,
            n_kv_heads,
            n_repeats,
            L,
            D,
        )

        value_dim = self.value_codec.dim
        total_tokens = _state_length(keys_state)
        key_chunk_size = (
            self.decode_key_chunk_size if L == 1 else self.prefill_key_chunk_size
        )
        query_block_size = 1 if L == 1 else self.prefill_query_block_size

        outputs = []
        for q_start in range(0, L, query_block_size):
            q_end = min(L, q_start + query_block_size)
            q_block = grouped_queries[..., q_start:q_end, :]
            prepared_queries = self.key_codec.prepare_queries(q_block)

            output = mx.zeros(
                (B, n_kv_heads, n_repeats, q_end - q_start, value_dim),
                dtype=mx.float32,
            )
            normalizer = mx.zeros(
                (B, n_kv_heads, n_repeats, q_end - q_start),
                dtype=mx.float32,
            )
            max_score = mx.full(
                (B, n_kv_heads, n_repeats, q_end - q_start),
                -float("inf"),
                dtype=mx.float32,
            )

            for k_start in range(0, total_tokens, key_chunk_size):
                k_end = min(total_tokens, k_start + key_chunk_size)
                key_chunk = _slice_state_range(keys_state, k_start, k_end)
                value_chunk = _slice_state_range(values_state, k_start, k_end)

                scores = self.key_codec.score_prepared(prepared_queries, key_chunk)
                scores = self._apply_attention_mask(
                    scores,
                    mask,
                    q_start,
                    q_end,
                    k_start,
                    k_end,
                    L,
                    total_tokens,
                )

                chunk_output, chunk_denom, chunk_max = (
                    self.value_codec.weighted_sum_stats_from_scores(scores, value_chunk)
                )
                new_max = mx.maximum(max_score, chunk_max)
                prev_scale = mx.exp(max_score - new_max)
                chunk_scale = mx.exp(chunk_max - new_max)

                output = (
                    output * prev_scale[..., None]
                    + chunk_output * chunk_scale[..., None]
                )
                normalizer = normalizer * prev_scale + chunk_denom * chunk_scale
                max_score = new_max
                mx.eval(output, normalizer, max_score)

            outputs.append(output / mx.maximum(normalizer[..., None], _EPS))
            mx.eval(outputs[-1])

        output = mx.concatenate(outputs, axis=3)
        output = output.reshape(B, n_q_heads, L, value_dim)
        return output.astype(queries.dtype)

    def _compiled_integer_decode_attention(
        self,
        grouped_queries: mx.array,
        keys_state,
        values_state,
    ) -> Optional[mx.array]:
        if not (
            _metal_available()
            and isinstance(self.key_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec, _TurboQuantMSECodec)
            and self.key_codec.bits == self.value_codec.bits
            and self.key_codec.mse_codec.bits > 0
            and isinstance(keys_state, TurboQuantProdState)
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        bits = int(self.value_codec.bits)
        if bits != self.value_codec.bits:
            return None

        decode = _compiled_integer_decode_kernel(bits)
        return decode(
            grouped_queries,
            keys_state.norms,
            keys_state.mse_indices,
            keys_state.residual_norms,
            keys_state.qjl_signs,
            values_state.norms,
            values_state.indices,
            self.key_codec.query_transform_t,
            self.key_codec.mse_codec.codebook,
            self.key_codec.scale_array,
            self.value_codec.codebook,
            self.value_codec.rotation,
        )

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state

        if queries.shape[-2] != 1:
            raise ValueError("TurboQuant decode attention expects a single query token.")

        B, n_q_heads, L, D = queries.shape
        n_kv_heads = keys_state.low.norms.shape[1] if isinstance(
            keys_state, TurboQuantSplitState
        ) else keys_state.norms.shape[1]
        n_repeats = n_q_heads // n_kv_heads

        grouped_queries = (queries * scale).reshape(
            B,
            n_kv_heads,
            n_repeats,
            L,
            D,
        )

        value_dim = self.value_codec.dim
        total_tokens = _state_length(keys_state)
        if total_tokens <= self.decode_key_chunk_size and mask in (None, "causal"):
            fast_output = self._compiled_integer_decode_attention(
                grouped_queries,
                keys_state,
                values_state,
            )
            if fast_output is not None:
                output = fast_output.reshape(B, n_q_heads, L, value_dim)
                return output.astype(queries.dtype)

            prepared_queries = self.key_codec.prepare_queries(grouped_queries)
            scores = self.key_codec.score_prepared(prepared_queries, keys_state)
            output = self.value_codec.weighted_sum_from_scores(scores, values_state)
            output = output.reshape(B, n_q_heads, L, value_dim)
            return output.astype(queries.dtype)

        prepared_queries = self.key_codec.prepare_queries(grouped_queries)

        output = mx.zeros((B, n_kv_heads, n_repeats, L, value_dim), dtype=mx.float32)
        normalizer = mx.zeros((B, n_kv_heads, n_repeats, L), dtype=mx.float32)
        max_score = mx.full(
            (B, n_kv_heads, n_repeats, L),
            -float("inf"),
            dtype=mx.float32,
        )

        for k_start in range(0, total_tokens, self.decode_key_chunk_size):
            k_end = min(total_tokens, k_start + self.decode_key_chunk_size)
            key_chunk = _slice_state_range(keys_state, k_start, k_end)
            value_chunk = _slice_state_range(values_state, k_start, k_end)

            scores = self.key_codec.score_prepared(prepared_queries, key_chunk)
            scores = self._apply_attention_mask(
                scores,
                mask,
                0,
                L,
                k_start,
                k_end,
                L,
                total_tokens,
            )

            chunk_output, chunk_denom, chunk_max = (
                self.value_codec.weighted_sum_stats_from_scores(scores, value_chunk)
            )
            new_max = mx.maximum(max_score, chunk_max)
            prev_scale = mx.exp(max_score - new_max)
            chunk_scale = mx.exp(chunk_max - new_max)

            output = (
                output * prev_scale[..., None]
                + chunk_output * chunk_scale[..., None]
            )
            normalizer = normalizer * prev_scale + chunk_denom * chunk_scale
            max_score = new_max
            mx.eval(output, normalizer, max_score)

        output = output / mx.maximum(normalizer[..., None], _EPS)
        output = output.reshape(B, n_q_heads, L, value_dim)
        return output.astype(queries.dtype)

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.keys is None:
            return None, None
        if self._cached_state_offset == self.offset:
            return self._cached_state
        sliced = _slice_state(self.keys, self.offset), _slice_state(self.values, self.offset)
        self._cached_state = sliced
        self._cached_state_offset = self.offset
        return sliced

    @state.setter
    def state(self, value):
        self._cached_state = None
        self._cached_state_offset = -1
        if value is None:
            self.keys, self.values = None, None
            self.offset = 0
            return
        self.keys, self.values = value
        self.offset = _state_length(self.keys)

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.seed)))

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])
        self.bits = float(value[1])
        self.seed = int(value[2])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._cached_state = None
        self._cached_state_offset = -1
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return _state_nbytes(self.state)
