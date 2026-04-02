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
    """Single-pass fused softmax + weighted sum kernel.

    Takes precomputed max_scores to avoid a separate token-dimension pass.
    """
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
        "        auto max_base = max_scores + (b * kv_heads + h) * repeat_count;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float max_score_{r} = static_cast<float>(max_base[{r}]);")
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
        input_names=["scores", "norms", "packed", "codebook", "max_scores"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _mse_scores_weighted_rot_sum_repeat_kernel(repeat_count: int):
    """Single-pass kernel for unnormalized weighted sum (used in chunked attention).

    Takes precomputed max_scores to avoid a separate token-dimension pass.
    """
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
        "        auto max_base = max_scores + (b * kv_heads + h) * repeat_count;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float max_score_{r} = static_cast<float>(max_base[{r}]);")
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
        input_names=["scores", "norms", "packed", "codebook", "max_scores"],
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
            state.indices,
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
            state.qjl_signs,
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
                    state.mse_indices,
                    state.qjl_signs,
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
            state.mse_indices,
            state.qjl_signs,
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
    inputs.extend(level for level in state.polar_state.level_indices)
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
    inputs.extend(level for level in state.polar_state.level_indices)
    inputs.extend(
        [
            state.residual_norms,
            state.qjl_signs,
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
                    state.indices,
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
            state.indices,
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

    # Precompute max scores on the host to avoid a second pass in the kernel
    max_scores = mx.max(scores_2d, axis=-1)  # (B, H, R)

    D = rotation.shape[0]
    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices,
            codebook,
            max_scores,
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
    max_scores: mx.array,
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

    # max_scores shape: (B, H, R) — already precomputed by caller
    D = rotation.shape[0]
    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices,
            codebook,
            max_scores,
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


@lru_cache(maxsize=None)
def _fused_integer_decode_kernel(bits: int, repeat_count: int, key_mse_bits: int = -1):
    """Single Metal kernel: score + online-softmax + weighted-sum for integer-bit
    ProdCodec keys + MSECodec values. Eliminates intermediate scores tensor.

    Grid: (32, num_val_tiles, B*H*num_tok_tiles) with threadgroup=(32,1,1).
    32 SIMD lanes cooperate on scoring (simd_sum across dims) and each lane
    accumulates one value dim with online softmax.
    """
    if not _metal_available() or repeat_count < 1:
        return None

    mse_bits = key_mse_bits if key_mse_bits >= 0 else max(bits - 1, 0)
    mse_mask = (1 << mse_bits) - 1
    val_mask = (1 << bits) - 1

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto val_tile = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        int val_dim = val_tile * 32 + lane;",
        "",
        "        auto token_count = key_norms_shape[2];",
        "        auto kv_heads = key_norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto k_norms = key_norms + base * token_count;",
        "        auto k_mse = key_mse + base * token_count * KMsePackedWidth;",
        "        auto k_res = key_res_norms + base * token_count;",
        "        auto k_signs = key_signs + base * token_count * KSignPackedWidth;",
        "        auto v_norms = val_norms + base * token_count;",
        "        auto v_packed = val_packed + base * token_count * VPackedWidth;",
        "",
        "        bool v_valid = val_dim < Dim;",
        "        int v_bo = val_dim * ValBits;",
        "        int v_word = v_bo / 32;",
        "        int v_off = v_bo % 32;",
        f"        bool v_spills = (v_off + ValBits > 32);",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qr_{r} = q_rot + (base * RepeatCount + {r}) * Dim;",
            f"        auto qp_{r} = q_proj + (base * RepeatCount + {r}) * Dim;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY;",
            f"        float lsum_{r} = 0.0f;",
            f"        float lacc_{r} = 0.0f;",
        ]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto mse_t = k_mse + t * KMsePackedWidth;",
        "            auto sign_t = k_signs + t * KSignPackedWidth;",
        "            float kn = static_cast<float>(k_norms[t]);",
        "            float ksr = kn * key_scale[0] * static_cast<float>(k_res[t]);",
    ]

    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    lines += [
        f"            for (int d = lane; d < Dim; d += 32) {{",
        f"                int bo = d * {mse_bits};",
        f"                uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                if (((bo & 31) + {mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({mse_bits} - ((bo & 31) + {mse_bits} - 32));",
        f"                idx &= {mse_mask}u;",
        f"                float code = key_codebook[idx];",
        f"                uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [f"                ps_{r} += kn * static_cast<float>(qr_{r}[d]) * code + ksr * (sb ? static_cast<float>(qp_{r}[d]) : -static_cast<float>(qp_{r}[d]));"]
    lines += [
        "            }",
    ]
    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Value decode + online softmax
    lines += [
        "",
        "            float v_code = 0.0f;",
        "            if (v_valid) {",
        "                auto vt = v_packed + t * VPackedWidth;",
        "                uint vv = (vt[v_word] >> v_off);",
        f"                if (v_spills) vv |= vt[v_word + 1] << (ValBits - (v_off + ValBits - 32));",
        f"                v_code = val_codebook[vv & {val_mask}u] * static_cast<float>(v_norms[t]);",
        "            }",
    ]

    for r in range(repeat_count):
        lines += [
            f"            float om_{r} = lmax_{r};",
            f"            lmax_{r} = max(lmax_{r}, s_{r});",
            f"            float rs_{r} = exp(om_{r} - lmax_{r});",
            f"            float w_{r} = exp(s_{r} - lmax_{r});",
            f"            lsum_{r} = lsum_{r} * rs_{r} + w_{r};",
            f"            lacc_{r} = lacc_{r} * rs_{r} + w_{r} * v_code;",
        ]

    lines += ["        }", ""]

    lines += ["        int out_stride = Dim;"]
    for r in range(repeat_count):
        lines += [
            f"        if (v_valid) {{",
            f"            out_acc[((bh * num_tok_tiles + tok_tile) * RepeatCount + {r}) * out_stride + val_dim] = lacc_{r};",
            f"        }}",
            f"        if (val_dim == 0) {{",
            f"            int sm_base = (bh * num_tok_tiles + tok_tile) * RepeatCount + {r};",
            f"            out_sum[sm_base] = lsum_{r};",
            f"            out_max[sm_base] = lmax_{r};",
            f"        }}",
        ]

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_integer_decode_{bits}_r{repeat_count}",
        input_names=[
            "q_rot", "q_proj",
            "key_norms", "key_mse", "key_res_norms", "key_signs",
            "val_norms", "val_packed",
            "key_codebook", "key_scale", "val_codebook",
        ],
        output_names=["out_acc", "out_sum", "out_max"],
        source="\n".join(lines),
    )


@lru_cache(maxsize=None)
def _multi_query_prod_score_kernel(key_mse_bits: int, repeat_count: int, num_queries: int, dims_per_lane: int):
    """Multi-query score kernel: unpack key data ONCE per token, loop over L queries.
    Avoids R*L repeat explosion that causes register spill."""
    if not _metal_available() or repeat_count < 1 or num_queries < 1:
        return None

    mse_mask = (1 << key_mse_bits) - 1

    return mx.fast.metal_kernel(
        name=f"mq_prod_score_{key_mse_bits}_r{repeat_count}_l{num_queries}",
        input_names=["q_rot", "q_proj", "key_norms", "key_mse", "key_res_norms",
                     "key_signs", "key_codebook", "key_scale"],
        output_names=["out"],
        source=f"""
            auto lane = thread_position_in_grid.x;
            auto ri = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;
            auto tc = key_norms_shape[2];
            auto kh = key_norms_shape[1];
            auto b = n / (kh * tc);
            auto rem = n % (kh * tc);
            auto h = rem / tc;
            auto t = rem % tc;
            if (ri >= {repeat_count}) return;

            auto mt = key_mse + ((b*kh+h)*tc+t) * KMsePackedWidth;
            auto st = key_signs + ((b*kh+h)*tc+t) * KSignPackedWidth;
            float kn = static_cast<float>(key_norms[(b*kh+h)*tc+t]);
            float ksr = kn * key_scale[0] * static_cast<float>(key_res_norms[(b*kh+h)*tc+t]);

            // Unpack key ONCE
            float kc[{dims_per_lane}], ksf[{dims_per_lane}];
            for (int i=0, d=lane; d < Dim; i++, d+=32) {{
                int bo = d * {key_mse_bits};
                uint idx = (mt[bo>>5] >> (bo&31));
                if (((bo&31)+{key_mse_bits}) > 32) idx |= mt[(bo>>5)+1] << ({key_mse_bits} - ((bo&31)+{key_mse_bits}-32));
                idx &= {mse_mask}u;
                kc[i] = key_codebook[idx];
                uint sb = (st[d>>5] >> (d&31)) & 1u;
                ksf[i] = sb ? 1.0f : -1.0f;
            }}

            // Loop over L queries, reusing unpacked key
            auto bq = (b*kh+h) * {repeat_count} + ri;
            for (int l = 0; l < {num_queries}; l++) {{
                float ps = 0.0f;
                for (int i=0, d=lane; d < Dim; i++, d+=32) {{
                    ps += kn * static_cast<float>(q_rot[(bq*{num_queries}+l)*Dim+d]) * kc[i]
                        + ksr * static_cast<float>(q_proj[(bq*{num_queries}+l)*Dim+d]) * ksf[i];
                }}
                float s = simd_sum(ps);
                if (lane == 0)
                    out[((b*kh+h)*{repeat_count}+ri)*{num_queries}*tc + l*tc + t] = s;
            }}
        """,
    )


@lru_cache(maxsize=None)
def _single_tile_value_weighted_sum_kernel(bits: int, repeat_count: int, dims_per_lane: int):
    """Single-tile value weighted sum with precomputed softmax weights.
    TG=Dim: one thread per value dim, no exp() calls in inner loop.
    2x faster than online-softmax variant."""
    if not _metal_available() or repeat_count < 1:
        return None

    val_mask = (1 << bits) - 1

    lines = [
        "        auto dim = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto wt = weights + bh * RepeatCount * token_count;",
        "        auto nm = norms + bh * token_count;",
        "        auto pk = packed + bh * token_count * PackedWidth;",
        "",
        f"        int bo = dim * {bits};",
        f"        int v_word = bo / 32;",
        f"        int v_shift = bo % 32;",
        f"        bool v_spill = (bo % 32 + {bits}) > 32;",
        "",
    ]

    for r in range(repeat_count):
        lines += [f"        float acc_{r} = 0.0f;"]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto pt = pk + t * PackedWidth;",
        "            uint vv = (pt[v_word] >> v_shift);",
        f"            if (v_spill) vv |= pt[v_word+1] << ({bits} - (v_shift+{bits}-32));",
        f"            float val = codebook[vv & {val_mask}u] * static_cast<float>(nm[t]);",
        "",
    ]

    for r in range(repeat_count):
        lines += [f"            acc_{r} += wt[{r}*token_count+t] * val;"]

    lines += ["        }", ""]

    for r in range(repeat_count):
        lines += [
            f"        if (dim < Dim) out[((bh*num_tok_tiles+tok_tile)*RepeatCount+{r})*Dim+dim] = acc_{r};",
        ]

    return mx.fast.metal_kernel(
        name=f"turboquant_single_tile_value_{bits}_r{repeat_count}",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source="\n".join(lines),
    )


@lru_cache(maxsize=None)
def _fused_integer_decode_single_tile_kernel(bits: int, repeat_count: int, dims_per_lane: int, key_mse_bits: int = -1):
    """Single-tile fused kernel — each lane handles multiple value dims.
    Zero key read redundancy: keys are read once per token, not once per val_tile.
    Faster than multi-tile at long contexts (256k+) where memory bandwidth dominates.
    """
    if not _metal_available() or repeat_count < 1:
        return None

    mse_bits = key_mse_bits if key_mse_bits >= 0 else max(bits - 1, 0)
    mse_mask = (1 << mse_bits) - 1
    val_mask = (1 << bits) - 1

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "        auto token_count = key_norms_shape[2];",
        "        auto kv_heads = key_norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto k_norms = key_norms + base * token_count;",
        "        auto k_mse = key_mse + base * token_count * KMsePackedWidth;",
        "        auto k_res = key_res_norms + base * token_count;",
        "        auto k_signs = key_signs + base * token_count * KSignPackedWidth;",
        "        auto v_norms = val_norms + base * token_count;",
        "        auto v_packed = val_packed + base * token_count * VPackedWidth;",
        "",
        "        // Precompute value bit offsets for all dims this lane handles",
        "        int v_words[DimsPerLane], v_offs[DimsPerLane];",
        "        bool v_spills[DimsPerLane], v_valids[DimsPerLane];",
        "        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {",
        f"            v_valids[i] = vd < Dim;",
        f"            int vbo = vd * {bits};",
        f"            v_words[i] = vbo / 32;",
        f"            v_offs[i] = vbo % 32;",
        f"            v_spills[i] = (vbo % 32 + {bits}) > 32;",
        "        }",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qr_{r} = q_rot + (base * RepeatCount + {r}) * Dim;",
            f"        auto qp_{r} = q_proj + (base * RepeatCount + {r}) * Dim;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY, lsum_{r} = 0.0f;",
            f"        float lacc_{r}[DimsPerLane] = {{}};",
        ]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto mse_t = k_mse + t * KMsePackedWidth;",
        "            auto sign_t = k_signs + t * KSignPackedWidth;",
        "            float kn = static_cast<float>(k_norms[t]);",
        "            float ksr = kn * key_scale[0] * static_cast<float>(k_res[t]);",
        "",
    ]

    # Score
    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    lines += [
        f"            for (int d = lane; d < Dim; d += 32) {{",
        f"                int bo = d * {mse_bits};",
        f"                uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                if (((bo & 31) + {mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({mse_bits} - ((bo & 31) + {mse_bits} - 32));",
        f"                idx &= {mse_mask}u;",
        f"                float code = key_codebook[idx];",
        f"                uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [f"                ps_{r} += kn * static_cast<float>(qr_{r}[d]) * code + ksr * (sb ? static_cast<float>(qp_{r}[d]) : -static_cast<float>(qp_{r}[d]));"]
    lines += ["            }"]

    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Online softmax + multi-dim value accumulation
    lines += [
        "",
        "            auto vt = v_packed + t * VPackedWidth;",
        "            float vnorm = static_cast<float>(v_norms[t]);",
    ]

    for r in range(repeat_count):
        lines += [
            f"            {{ float om = lmax_{r};",
            f"              lmax_{r} = max(lmax_{r}, s_{r});",
            f"              float rs = exp(om - lmax_{r});",
            f"              float w = exp(s_{r} - lmax_{r});",
            f"              lsum_{r} = lsum_{r} * rs + w;",
            f"              for (int i = 0; i < DimsPerLane; i++) {{",
            f"                  lacc_{r}[i] *= rs;",
            f"                  if (v_valids[i]) {{",
            f"                      uint vv = (vt[v_words[i]] >> v_offs[i]);",
            f"                      if (v_spills[i]) vv |= vt[v_words[i]+1] << ({bits} - (v_offs[i]+{bits}-32));",
            f"                      lacc_{r}[i] += w * val_codebook[vv & {val_mask}u] * vnorm;",
            f"                  }}",
            f"              }}",
            f"            }}",
        ]

    lines += ["        }", ""]

    # Write unnormalized acc + scalar sum/max for cross-tile reduction
    for r in range(repeat_count):
        lines += [
            f"        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {{",
            f"            if (vd < Dim) out_acc[((bh*num_tok_tiles+tok_tile)*RepeatCount+{r})*Dim+vd] = lacc_{r}[i];",
            f"        }}",
            f"        if (lane == 0) {{",
            f"            int sm_base = (bh*num_tok_tiles+tok_tile)*RepeatCount+{r};",
            f"            out_sum[sm_base] = lsum_{r};",
            f"            out_max[sm_base] = lmax_{r};",
            f"        }}",
        ]

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_integer_single_tile_{bits}_r{repeat_count}",
        input_names=[
            "q_rot", "q_proj",
            "key_norms", "key_mse", "key_res_norms", "key_signs",
            "val_norms", "val_packed",
            "key_codebook", "key_scale", "val_codebook",
        ],
        output_names=["out_acc", "out_sum", "out_max"],
        source="\n".join(lines),
    )


@lru_cache(maxsize=None)
def _fused_split_decode_kernel(low_bits: int, high_bits: int, repeat_count: int):
    """Single Metal kernel: score + online-softmax + weighted-sum for SplitCodec.

    Grid: (32, ValDim, B*H) — for each token: 32 lanes cooperate on key-dim
    reduction (simd_sum) to produce a scalar score, then each lane unpacks+
    accumulates its value dim weighted by the softmax weight. Uses online
    softmax across tokens with a cross-lane reduction at the end.
    """
    if not _metal_available() or repeat_count < 1:
        return None

    low_mse_bits = max(low_bits - 1, 0)
    high_mse_bits = max(high_bits - 1, 0)
    low_mse_mask = (1 << low_mse_bits) - 1
    high_mse_mask = (1 << high_mse_bits) - 1

    # Architecture: 32 SIMD lanes cooperate on BOTH key scoring (simd_sum across
    # key dims) AND value accumulation (each lane handles its own value dim).
    #
    # Grid: (32, NumTiles, B*H) — NumTiles = ceil(ValDim/32).
    # Each tile's 32 lanes handle 32 value dims. The score is computed ONCE
    # per token per tile (not once per value dim), eliminating 32x redundancy.
    #
    # Per token:
    #   1. 32 lanes split key dims → partial scores → simd_sum → scalar score
    #   2. Each lane unpacks its value dim, applies online softmax weight
    #
    # Cross-lane score is free (simd_sum broadcasts to all lanes).

    val_dim_total = "DimLow + DimHigh"

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto val_tile = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        f"        int val_dim = val_tile * 32 + lane;",
        "",
        "        auto token_count = key_low_norms_shape[2];",
        "        auto kv_heads = key_low_norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto kl_norms = key_low_norms + base * token_count;",
        "        auto kl_mse = key_low_mse + base * token_count * KLMsePackedWidth;",
        "        auto kl_res = key_low_res_norms + base * token_count;",
        "        auto kl_signs = key_low_signs + base * token_count * KLSignPackedWidth;",
        "        auto kh_norms = key_high_norms + base * token_count;",
        "        auto kh_mse = key_high_mse + base * token_count * KHMsePackedWidth;",
        "        auto kh_res = key_high_res_norms + base * token_count;",
        "        auto kh_signs = key_high_signs + base * token_count * KHSignPackedWidth;",
        "",
        "        auto vl_norms = val_low_norms + base * token_count;",
        "        auto vl_packed = val_low_packed + base * token_count * VLPackedWidth;",
        "        auto vh_norms = val_high_norms + base * token_count;",
        "        auto vh_packed = val_high_packed + base * token_count * VHPackedWidth;",
        "",
        "        // Value bit offset for this lane's dim",
        f"        bool is_low_val = val_dim < DimLow;",
        f"        int vd_local = is_low_val ? val_dim : (val_dim - DimLow);",
        f"        int v_bits = is_low_val ? VLBits : VHBits;",
        f"        int v_bo = vd_local * v_bits;",
        f"        int v_word = v_bo / 32;",
        f"        int v_off = v_bo % 32;",
        f"        uint v_mask = (1u << v_bits) - 1u;",
        f"        bool v_spills = (v_off + v_bits > 32);",
        f"        bool v_valid = val_dim < ({val_dim_total});",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qrl_{r} = q_rot_low + (base * RepeatCount + {r}) * DimLow;",
            f"        auto qpl_{r} = q_proj_low + (base * RepeatCount + {r}) * DimLow;",
            f"        auto qrh_{r} = q_rot_high + (base * RepeatCount + {r}) * DimHigh;",
            f"        auto qph_{r} = q_proj_high + (base * RepeatCount + {r}) * DimHigh;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY;",
            f"        float lsum_{r} = 0.0f;",
            f"        float lacc_{r} = 0.0f;",
        ]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
    ]

    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    # Low half scoring — hoisted ksr, conditional negate
    lines += [
        "            {",
        "                auto mse_t = kl_mse + t * KLMsePackedWidth;",
        "                auto sign_t = kl_signs + t * KLSignPackedWidth;",
        "                float kn = static_cast<float>(kl_norms[t]);",
        "                float ksr = kn * key_low_scale[0] * static_cast<float>(kl_res[t]);",
        f"                for (int d = lane; d < DimLow; d += 32) {{",
        f"                    int bo = d * {low_mse_bits};",
        f"                    uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                    if (((bo & 31) + {low_mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({low_mse_bits} - ((bo & 31) + {low_mse_bits} - 32));",
        f"                    idx &= {low_mse_mask}u;",
        f"                    float code = key_low_codebook[idx];",
        f"                    uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [f"                    ps_{r} += kn * static_cast<float>(qrl_{r}[d]) * code + ksr * (sb ? static_cast<float>(qpl_{r}[d]) : -static_cast<float>(qpl_{r}[d]));"]
    lines += [
        "                }",
        "            }",
    ]
    # High half scoring
    lines += [
        "            {",
        "                auto mse_t = kh_mse + t * KHMsePackedWidth;",
        "                auto sign_t = kh_signs + t * KHSignPackedWidth;",
        "                float kn = static_cast<float>(kh_norms[t]);",
        "                float ksr = kn * key_high_scale[0] * static_cast<float>(kh_res[t]);",
        f"                for (int d = lane; d < DimHigh; d += 32) {{",
        f"                    int bo = d * {high_mse_bits};",
        f"                    uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                    if (((bo & 31) + {high_mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({high_mse_bits} - ((bo & 31) + {high_mse_bits} - 32));",
        f"                    idx &= {high_mse_mask}u;",
        f"                    float code = key_high_codebook[idx];",
        f"                    uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [f"                    ps_{r} += kn * static_cast<float>(qrh_{r}[d]) * code + ksr * (sb ? static_cast<float>(qph_{r}[d]) : -static_cast<float>(qph_{r}[d]));"]
    lines += [
        "                }",
        "            }",
    ]

    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Value decode + online softmax accumulation
    lines += [
        "",
        "            float v_code = 0.0f;",
        "            if (v_valid) {",
        "                if (is_low_val) {",
        "                    auto vt = vl_packed + t * VLPackedWidth;",
        "                    uint vv = (vt[v_word] >> v_off);",
        "                    if (v_spills) vv |= vt[v_word + 1] << (v_bits - (v_off + v_bits - 32));",
        "                    v_code = val_low_codebook[vv & v_mask] * static_cast<float>(vl_norms[t]);",
        "                } else {",
        "                    auto vt = vh_packed + t * VHPackedWidth;",
        "                    uint vv = (vt[v_word] >> v_off);",
        "                    if (v_spills) vv |= vt[v_word + 1] << (v_bits - (v_off + v_bits - 32));",
        "                    v_code = val_high_codebook[vv & v_mask] * static_cast<float>(vh_norms[t]);",
        "                }",
        "            }",
    ]

    for r in range(repeat_count):
        lines += [
            f"            float om_{r} = lmax_{r};",
            f"            lmax_{r} = max(lmax_{r}, s_{r});",
            f"            float rs_{r} = exp(om_{r} - lmax_{r});",
            f"            float w_{r} = exp(s_{r} - lmax_{r});",
            f"            lsum_{r} = lsum_{r} * rs_{r} + w_{r};",
            f"            lacc_{r} = lacc_{r} * rs_{r} + w_{r} * v_code;",
        ]

    lines += ["        }", ""]

    # Write acc per val_dim, but sum/max are identical across lanes —
    # only write once per (bh, tok_tile, repeat) to avoid redundant writes.
    lines += [f"        int out_stride = ({val_dim_total});"]
    for r in range(repeat_count):
        lines += [
            f"        if (v_valid) {{",
            f"            out_acc[((bh * num_tok_tiles + tok_tile) * RepeatCount + {r}) * out_stride + val_dim] = lacc_{r};",
            f"        }}",
            f"        if (val_dim == 0) {{",
            f"            int sm_base = (bh * num_tok_tiles + tok_tile) * RepeatCount + {r};",
            f"            out_sum[sm_base] = lsum_{r};",
            f"            out_max[sm_base] = lmax_{r};",
            f"        }}",
        ]

    source = "\n".join(lines)

    input_names = [
        "q_rot_low", "q_proj_low", "q_rot_high", "q_proj_high",
        "key_low_norms", "key_low_mse", "key_low_res_norms", "key_low_signs",
        "key_high_norms", "key_high_mse", "key_high_res_norms", "key_high_signs",
        "val_low_norms", "val_low_packed",
        "val_high_norms", "val_high_packed",
        "key_low_codebook", "key_high_codebook",
        "key_low_scale", "key_high_scale",
        "val_low_codebook", "val_high_codebook",
    ]

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_split_decode_{low_bits}_{high_bits}_r{repeat_count}",
        input_names=input_names,
        output_names=["out_acc", "out_sum", "out_max"],
        source=source,
    )


@lru_cache(maxsize=None)
def _compiled_split_decode_kernel(low_bits: int, high_bits: int):
    """Fused decode kernel for SplitCodec — handles both halves in a single
    compiled function, avoiding double dispatch and intermediate allocations."""
    low_mse_bits = max(low_bits - 1, 0)
    high_mse_bits = max(high_bits - 1, 0)

    @mx.compile
    def _decode(
        grouped_queries: mx.array,
        # Low key state
        key_low_norms: mx.array,
        key_low_mse_indices: mx.array,
        key_low_residual_norms: mx.array,
        key_low_qjl_signs: mx.array,
        # High key state
        key_high_norms: mx.array,
        key_high_mse_indices: mx.array,
        key_high_residual_norms: mx.array,
        key_high_qjl_signs: mx.array,
        # Low value state
        value_low_norms: mx.array,
        value_low_indices: mx.array,
        # High value state
        value_high_norms: mx.array,
        value_high_indices: mx.array,
        # Codec params
        key_low_transform_t: mx.array,
        key_high_transform_t: mx.array,
        key_low_codebook: mx.array,
        key_high_codebook: mx.array,
        key_low_scale: mx.array,
        key_high_scale: mx.array,
        value_low_codebook: mx.array,
        value_high_codebook: mx.array,
        value_low_rotation: mx.array,
        value_high_rotation: mx.array,
        # Split indices
        low_idx: mx.array,
        high_idx: mx.array,
        restore_order: mx.array,
    ) -> mx.array:
        dim = grouped_queries.shape[-1]
        # Split queries by dimension
        q_low = mx.take(grouped_queries, low_idx, axis=-1)
        q_high = mx.take(grouped_queries, high_idx, axis=-1)

        # Score low half
        qt_low = mx.matmul(q_low, key_low_transform_t)
        d_low = q_low.shape[-1]
        scores_low = _metal_prod_score(
            qt_low[..., :d_low].reshape(qt_low.shape[0], qt_low.shape[1], qt_low.shape[2], d_low),
            qt_low[..., d_low:].reshape(qt_low.shape[0], qt_low.shape[1], qt_low.shape[2], d_low),
            TurboQuantProdState(key_low_norms, key_low_mse_indices, key_low_residual_norms, key_low_qjl_signs),
            low_mse_bits, key_low_codebook, key_low_scale,
        )

        # Score high half
        qt_high = mx.matmul(q_high, key_high_transform_t)
        d_high = q_high.shape[-1]
        scores_high = _metal_prod_score(
            qt_high[..., :d_high].reshape(qt_high.shape[0], qt_high.shape[1], qt_high.shape[2], d_high),
            qt_high[..., d_high:].reshape(qt_high.shape[0], qt_high.shape[1], qt_high.shape[2], d_high),
            TurboQuantProdState(key_high_norms, key_high_mse_indices, key_high_residual_norms, key_high_qjl_signs),
            high_mse_bits, key_high_codebook, key_high_scale,
        )

        # Combined scores
        scores = scores_low + scores_high

        # Weighted sum of low values
        out_low = _metal_mse_weighted_sum_from_scores(
            scores, TurboQuantMSEState(value_low_norms, value_low_indices),
            low_bits, value_low_codebook, value_low_rotation,
        )

        # Weighted sum of high values
        out_high = _metal_mse_weighted_sum_from_scores(
            scores, TurboQuantMSEState(value_high_norms, value_high_indices),
            high_bits, value_high_codebook, value_high_rotation,
        )

        # Merge and reorder
        merged = mx.concatenate([out_low, out_high], axis=-1)
        return mx.take(merged, restore_order, axis=-1)

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
        # Precompute midpoints for fast comparison-based quantization
        if bits > 0 and self.codebook.shape[0] > 1:
            self._midpoints = (self.codebook[:-1] + self.codebook[1:]) / 2
        else:
            self._midpoints = mx.zeros((0,), dtype=mx.float32)

    def _quantize_unit_with_estimate(
        self, unit_vectors: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.bits == 0:
            return (
                mx.zeros((*unit_vectors.shape[:-1], 0), dtype=mx.uint32),
                mx.zeros(unit_vectors.shape, dtype=mx.float32),
            )

        rotated = mx.matmul(unit_vectors, self.rotation_t)
        # Use comparison-based quantization: O(T*D*bits) instead of
        # O(T*D*2^bits) broadcast argmin. 11-28x faster.
        indices = mx.zeros(rotated.shape, dtype=mx.uint32)
        for m in range(self._midpoints.shape[0]):
            indices = indices + (rotated > self._midpoints[m]).astype(mx.uint32)
        packed = _pack_lowbit(indices, self.bits)
        estimated_rotated = mx.take(self.codebook, indices.astype(mx.int32), axis=0)
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
        # Metal kernel fast path: only for single-query decode (L=1)
        if scores.ndim == 5 and scores.shape[-2] == 1:
            max_scores_2d = max_scores.reshape(
                max_scores.shape[0], max_scores.shape[1], max_scores.shape[2],
            )
            fast_output = _metal_mse_weighted_sum_sum_from_scores(
                scores,
                state,
                self.bits,
                self.codebook,
                self.rotation,
                max_scores_2d,
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

        # Pre-build combined query transform for fused decode:
        # single (D, 2*dim_low + 2*dim_high) matrix replaces 2 takes + 2 matmuls
        if mode == "prod" and isinstance(self.low_codec, _TurboQuantProdCodec):
            dim = tensor.shape[-1]
            dl = len(low_idx)
            dh = len(high_idx)
            combined = mx.zeros((dim, 2 * dl + 2 * dh), dtype=mx.float32)
            combined[self.low_idx, :dl] = self.low_codec.query_transform_t[:, :dl]
            combined[self.low_idx, dl:2*dl] = self.low_codec.query_transform_t[:, dl:]
            combined[self.high_idx, 2*dl:2*dl+dh] = self.high_codec.query_transform_t[:, :dh]
            combined[self.high_idx, 2*dl+dh:] = self.high_codec.query_transform_t[:, dh:]
            self.combined_query_transform_t = combined
        else:
            self.combined_query_transform_t = None

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
        # Launch both sub-codec scores before any sync — enables GPU overlap
        low_scores = self.low_codec.score_prepared(low_queries, state.low)
        high_scores = self.high_codec.score_prepared(high_queries, state.high)
        return low_scores + high_scores

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
        # Launch both before concat to enable overlap
        low_tensor = self.low_codec.weighted_sum_from_scores(scores, state.low)
        high_tensor = self.high_codec.weighted_sum_from_scores(scores, state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def weighted_sum_stats_from_scores(
        self, scores: mx.array, state: TurboQuantSplitState
    ) -> tuple[mx.array, mx.array, mx.array]:
        # Launch both before concat to enable overlap
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


class _QuantizedStateProxy:
    """Wraps a quantized state tuple, providing .shape for model compatibility.

    Some models access keys.shape[-2] after cache.update_and_fetch() to slice
    masks. This proxy makes that work without dequantization.
    """
    __slots__ = ("_state", "shape")

    def __init__(self, state, n_tokens: int, n_heads: int):
        self._state = state
        # Mimic (B, H, T, D) shape — only T is needed by downstream code
        self.shape = (1, n_heads, n_tokens, 0)

    def __getattr__(self, name):
        return getattr(self._state, name)

    def __iter__(self):
        return iter(self._state)


class TurboQuantKVCache(_BaseCache):
    # Process all tokens in one pass during decode — chunking adds significant
    # overhead from the online-softmax recombination loop. Memory is already
    # bounded by the quantized cache itself.
    decode_key_chunk_size = 1 << 30  # ~1B tokens, effectively no chunking
    prefill_key_chunk_size = 2048  # match DEFAULT_PREFILL_STEP_SIZE from generate.py
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
            # For fractional bits (e.g. 3.5), use lower bits for keys and higher
            # for values instead of SplitCodec. Both stay as fast integer codecs
            # with single-tile kernel support. Values benefit more from extra bits.
            key_bits = math.floor(self.bits) if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6) else self.bits
            self.key_codec = _build_codec(keys, key_bits, mode="prod", seed=self.seed)
        if self.value_codec is None:
            val_bits = math.ceil(self.bits) if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6) else self.bits
            self.value_codec = _build_codec(
                values, val_bits, mode="mse", seed=self.seed + 1
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

        B, n_heads = keys.shape[0], keys.shape[1]
        self.offset = new_end
        self._cached_state = None
        self._cached_state_offset = -1
        n_new = keys.shape[2]
        # Only eval during prefill (multiple tokens) to prevent graph buildup.
        if n_new > 1:
            mx.eval(self.keys, self.values)
        # Return proxied states so model code can access .shape[-2] for mask slicing
        ks, vs = self.state
        return (
            _QuantizedStateProxy(ks, self.offset, n_heads),
            _QuantizedStateProxy(vs, self.offset, n_heads),
        )

    @staticmethod
    def _unwrap(state):
        return state._state if isinstance(state, _QuantizedStateProxy) else state

    def dequantize(self, keys_state=None, values_state=None):
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)
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
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)

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

    def prefill_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Fast prefill: fold L queries into R dimension, reuse decode kernels.
        Avoids the expensive O(T×D²) dequantize rotation matmul."""
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)

        if not (
            isinstance(self.key_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec, _TurboQuantMSECodec)
            and isinstance(keys_state, TurboQuantProdState)
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        B, n_q_heads, L, D = queries.shape
        n_kv_heads = keys_state.norms.shape[1]
        n_repeats = n_q_heads // n_kv_heads
        T = keys_state.norms.shape[2]

        if T == 0:
            return None  # empty cache, let fallback handle it

        val_bits = int(self.value_codec.bits)
        if val_bits != self.value_codec.bits:
            return None
        dims_per_lane = (D + 31) // 32

        val_kernel = _single_tile_value_weighted_sum_kernel(val_bits, n_repeats * L, dims_per_lane)
        if val_kernel is None:
            return None

        # Multi-query score: unpack key ONCE per token, loop over L queries
        mq_score = _multi_query_prod_score_kernel(
            self.key_codec.mse_codec.bits, n_repeats, L, dims_per_lane,
        )
        if mq_score is None:
            return None

        grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
        qt = mx.matmul(grouped, self.key_codec.query_transform_t)
        q_rot = qt[..., :D].reshape(B * n_kv_heads * n_repeats, L, D)
        q_proj = qt[..., D:].reshape(B * n_kv_heads * n_repeats, L, D)

        scores = mq_score(
            inputs=[
                q_rot, q_proj,
                keys_state.norms, keys_state.mse_indices,
                keys_state.residual_norms, keys_state.qjl_signs,
                self.key_codec.mse_codec.codebook, self.key_codec.scale_array,
            ],
            template=[
                ("Dim", D),
                ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
            ],
            grid=(32, n_repeats, B * n_kv_heads * T),
            threadgroup=(32, 1, 1),
            output_shapes=[(B * n_kv_heads * n_repeats, L, T)],
            output_dtypes=[mx.float32],
        )[0]
        # Reshape: (B*H*R, L, T) → (B, H, R, L, T)
        scores = scores.reshape(B, n_kv_heads, n_repeats, L, T)

        # Apply mask (causal or explicit)
        if mask is not None:
            if isinstance(mask, mx.array):
                if mask.ndim == scores.ndim - 1:
                    mask = mx.expand_dims(mask, axis=2)
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores = scores + mask
            # string masks not supported here, fall back
            elif isinstance(mask, str):
                return None

        # Softmax + reshape for value kernel: (B*H, R*L, T)
        weights = mx.softmax(scores, axis=-1).reshape(B * n_kv_heads, n_repeats * L, T)

        # Value weighted sum using TG=D kernel
        tok_tile_size = 1024
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
        out_shape = (B * n_kv_heads * num_tok_tiles, n_repeats * L, D)

        out_tiled = val_kernel(
            inputs=[
                weights,
                values_state.norms,
                values_state.indices,
                self.value_codec.codebook,
            ],
            template=[
                ("Dim", D),
                ("RepeatCount", n_repeats * L),
                ("TokTileSize", tok_tile_size),
                ("DimsPerLane", dims_per_lane),
                ("PackedWidth", values_state.indices.shape[-1]),
            ],
            grid=(D, 1, B * n_kv_heads * num_tok_tiles),
            threadgroup=(D, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[mx.float32],
        )[0]

        # Reduce across tiles
        out_tiled = out_tiled.reshape(B * n_kv_heads, num_tok_tiles, n_repeats * L, D)
        if num_tok_tiles > 1:
            out_rotated = mx.sum(out_tiled, axis=1)
        else:
            out_rotated = out_tiled.squeeze(1)
        out_rotated = out_rotated.reshape(B, n_kv_heads, n_repeats, L, D)

        # Rotate values back
        output = mx.matmul(out_rotated, self.value_codec.rotation)
        return output.reshape(B, n_q_heads, L, D).astype(queries.dtype)

    def _separate_score_value_decode(
        self,
        grouped_queries: mx.array,
        keys_state,
        values_state,
    ) -> Optional[mx.array]:
        """Separate-kernel decode: fast key scoring + single-tile value weighted sum.
        2-3x faster than the fused kernel at large D because each kernel only
        iterates its own dimensions once."""
        if not (
            _metal_available()
            and isinstance(self.key_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec, _TurboQuantMSECodec)
            and isinstance(keys_state, TurboQuantProdState)
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        B = grouped_queries.shape[0]
        H = grouped_queries.shape[1]
        R = grouped_queries.shape[2]
        D = grouped_queries.shape[-1]
        T = keys_state.norms.shape[2]

        val_bits = int(self.value_codec.bits)
        if val_bits != self.value_codec.bits:
            return None
        dims_per_lane = (D + 31) // 32

        val_kernel = _single_tile_value_weighted_sum_kernel(val_bits, R, dims_per_lane)
        if val_kernel is None:
            return None

        # Step 1: Key scoring — uses existing optimized Metal kernel
        prepared_queries = self.key_codec.prepare_queries(grouped_queries)
        scores = self.key_codec.score_prepared(prepared_queries, keys_state)
        # scores: (B, H, R, 1, T) → (B*H, R, T)
        scores_2d = scores.reshape(B * H, R, T)

        # Step 2: Precompute softmax weights (avoids exp() in value kernel)
        weights = mx.softmax(scores_2d, axis=-1)  # (B*H, R, T)

        # Step 3: Single-tile value weighted sum with precomputed weights
        tok_tile_size = 1024
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
        out_shape = (B * H * num_tok_tiles, R, D)

        out_tiled = val_kernel(
            inputs=[
                weights,
                values_state.norms,
                values_state.indices,
                self.value_codec.codebook,
            ],
            template=[
                ("Dim", D),
                ("RepeatCount", R),
                ("TokTileSize", tok_tile_size),
                ("DimsPerLane", dims_per_lane),
                ("PackedWidth", values_state.indices.shape[-1]),
            ],
            grid=(D, 1, B * H * num_tok_tiles),
            threadgroup=(D, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[mx.float32],
        )[0]

        # Cross-tile reduction (simple sum since weights are pre-normalized)
        out_tiled = out_tiled.reshape(B * H, num_tok_tiles, R, D)
        if num_tok_tiles > 1:
            out_rotated = mx.sum(out_tiled, axis=1)
        else:
            out_rotated = out_tiled.squeeze(1)
        out_rotated = out_rotated.reshape(B, H, R, D)

        # Rotate values back to original space
        output = mx.matmul(out_rotated, self.value_codec.rotation)
        return mx.expand_dims(output, axis=3)

    def _compiled_split_decode_attention(
        self,
        grouped_queries: mx.array,
        keys_state,
        values_state,
    ) -> Optional[mx.array]:
        """Fused decode for SplitCodec — single Metal kernel for score + softmax
        + weighted_sum across both low/high halves."""
        if not (
            _metal_available()
            and isinstance(self.key_codec, _SplitCodec)
            and isinstance(self.value_codec, _SplitCodec)
            and isinstance(keys_state, TurboQuantSplitState)
            and isinstance(values_state, TurboQuantSplitState)
            and isinstance(self.key_codec.low_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec.low_codec, _TurboQuantMSECodec)
        ):
            return None

        kc = self.key_codec
        vc = self.value_codec
        low_bits = kc.lower_bits
        high_bits = kc.upper_bits

        if kc.low_codec.mse_codec.bits <= 0 or kc.high_codec.mse_codec.bits <= 0:
            return None

        B = grouped_queries.shape[0]
        H = grouped_queries.shape[1]
        R = grouped_queries.shape[2]
        dim_low = kc.low_codec.dim
        dim_high = kc.high_codec.dim
        T = keys_state.low.norms.shape[2]

        kernel = _fused_split_decode_kernel(low_bits, high_bits, R)
        if kernel is None:
            return None

        # Single combined query transform: 1 matmul replaces 2 takes + 2 matmuls
        if kc.combined_query_transform_t is not None:
            qt = mx.matmul(grouped_queries, kc.combined_query_transform_t)
            dl2 = dim_low * 2
            q_rot_low = qt[..., :dim_low].reshape(B, H, R, dim_low)
            q_proj_low = qt[..., dim_low:dl2].reshape(B, H, R, dim_low)
            q_rot_high = qt[..., dl2:dl2+dim_high].reshape(B, H, R, dim_high)
            q_proj_high = qt[..., dl2+dim_high:].reshape(B, H, R, dim_high)
        else:
            q_low = mx.take(grouped_queries, kc.low_idx, axis=-1)
            q_high = mx.take(grouped_queries, kc.high_idx, axis=-1)
            qt_low = mx.matmul(q_low, kc.low_codec.query_transform_t)
            qt_high = mx.matmul(q_high, kc.high_codec.query_transform_t)
            q_rot_low = qt_low[..., :dim_low].reshape(B, H, R, dim_low)
            q_proj_low = qt_low[..., dim_low:].reshape(B, H, R, dim_low)
            q_rot_high = qt_high[..., :dim_high].reshape(B, H, R, dim_high)
            q_proj_high = qt_high[..., dim_high:].reshape(B, H, R, dim_high)

        low_mse_bits = max(low_bits - 1, 0)
        high_mse_bits = max(high_bits - 1, 0)
        val_dim = dim_low + dim_high

        tok_tile_size = 1024
        num_val_tiles = (val_dim + 31) // 32
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size

        acc_shape = (B * H * num_tok_tiles, R, val_dim)
        sm_shape = (B * H * num_tok_tiles * R,)  # scalar per (bh, tile, repeat)
        out_acc, out_sum, out_max = kernel(
            inputs=[
                q_rot_low, q_proj_low, q_rot_high, q_proj_high,
                keys_state.low.norms, keys_state.low.mse_indices,
                keys_state.low.residual_norms, keys_state.low.qjl_signs,
                keys_state.high.norms, keys_state.high.mse_indices,
                keys_state.high.residual_norms, keys_state.high.qjl_signs,
                values_state.low.norms, values_state.low.indices,
                values_state.high.norms, values_state.high.indices,
                kc.low_codec.mse_codec.codebook, kc.high_codec.mse_codec.codebook,
                kc.low_codec.scale_array, kc.high_codec.scale_array,
                vc.low_codec.codebook, vc.high_codec.codebook,
            ],
            template=[
                ("DimLow", dim_low),
                ("DimHigh", dim_high),
                ("RepeatCount", R),
                ("TokTileSize", tok_tile_size),
                ("KLMsePackedWidth", keys_state.low.mse_indices.shape[-1]),
                ("KLSignPackedWidth", keys_state.low.qjl_signs.shape[-1]),
                ("KHMsePackedWidth", keys_state.high.mse_indices.shape[-1]),
                ("KHSignPackedWidth", keys_state.high.qjl_signs.shape[-1]),
                ("VLPackedWidth", values_state.low.indices.shape[-1]),
                ("VHPackedWidth", values_state.high.indices.shape[-1]),
                ("VLBits", vc.low_codec.bits),
                ("VHBits", vc.high_codec.bits),
            ],
            grid=(32, num_val_tiles, B * H * num_tok_tiles),
            threadgroup=(32, 1, 1),
            output_shapes=[acc_shape, sm_shape, sm_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

        # Cross-tile reduction: sum/max are (BH*tiles*R,) scalars, acc is (BH*tiles, R, D)
        out_acc = out_acc.reshape(B * H, num_tok_tiles, R, val_dim)
        out_sum = out_sum.reshape(B * H, num_tok_tiles, R)
        out_max = out_max.reshape(B * H, num_tok_tiles, R)
        global_max = mx.max(out_max, axis=1, keepdims=True)  # (BH, 1, R)
        scale_factors = mx.exp(out_max - global_max)  # (BH, tiles, R)
        scaled_acc = mx.sum(out_acc * scale_factors[..., None], axis=1)  # (BH, R, D)
        denom = mx.sum(out_sum * scale_factors, axis=1)  # (BH, R)
        out_rotated = (scaled_acc / mx.maximum(denom[..., None], _EPS)).reshape(B, H, R, val_dim)

        out_low = mx.matmul(out_rotated[..., :dim_low], vc.low_codec.rotation)
        out_high = mx.matmul(out_rotated[..., dim_low:], vc.high_codec.rotation)
        merged = mx.concatenate([out_low, out_high], axis=-1)
        output = mx.take(merged, vc.restore_order, axis=-1)
        return mx.expand_dims(output, axis=3)

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
            and self.key_codec.mse_codec.bits > 0
            and isinstance(keys_state, TurboQuantProdState)
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        bits = int(self.value_codec.bits)
        if bits != self.value_codec.bits:
            return None

        B, H, R = grouped_queries.shape[0], grouped_queries.shape[1], grouped_queries.shape[2]
        D = grouped_queries.shape[-1]
        T = keys_state.norms.shape[2]

        # Try fused kernel — choose single-tile (zero key redundancy) for
        # long contexts where memory bandwidth dominates, multi-tile for
        # shorter contexts where parallelism matters more.
        qt = mx.matmul(grouped_queries, self.key_codec.query_transform_t)
        q_rot = qt[..., :D].reshape(B, H, R, D)
        q_proj = qt[..., D:].reshape(B, H, R, D)
        tok_tile_size = 1024
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
        dims_per_lane = (D + 31) // 32

        # Key MSE bits may differ from value bits (e.g. 3-bit keys + 4-bit values)
        key_mse_bits = self.key_codec.mse_codec.bits

        # Single-tile path: each lane handles all its value dims.
        # Zero key read redundancy — faster at 256k+ where bandwidth dominates.
        single_kernel = _fused_integer_decode_single_tile_kernel(bits, R, dims_per_lane, key_mse_bits)
        # Multi-tile path: one val_dim per lane, multiple tiles read keys redundantly.
        # Better parallelism at shorter contexts.
        multi_kernel = _fused_integer_decode_kernel(bits, R, key_mse_bits)

        # Single-tile wins when val_tile redundancy outweighs parallelism benefit.
        # More val_tiles (larger D) → lower crossover. With 8 val_tiles (D=256),
        # single-tile wins even at 128k. With 5 tiles (D=160), crossover ~256k.
        num_val_tiles = (D + 31) // 32
        min_threadgroups = 64
        use_single = (
            single_kernel is not None
            and num_tok_tiles * B * H >= min_threadgroups
            and (num_val_tiles >= 8 or T >= 262144)
        )

        if use_single:
            acc_shape = (B * H * num_tok_tiles, R, D)
            sm_shape = (B * H * num_tok_tiles * R,)

            out_acc, out_sum, out_max = single_kernel(
                inputs=[
                    q_rot, q_proj,
                    keys_state.norms, keys_state.mse_indices,
                    keys_state.residual_norms, keys_state.qjl_signs,
                    values_state.norms, values_state.indices,
                    self.key_codec.mse_codec.codebook,
                    self.key_codec.scale_array,
                    self.value_codec.codebook,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("TokTileSize", tok_tile_size),
                    ("DimsPerLane", dims_per_lane),
                    ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                    ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
                    ("VPackedWidth", values_state.indices.shape[-1]),
                ],
                grid=(32, 1, B * H * num_tok_tiles),
                threadgroup=(32, 1, 1),
                output_shapes=[acc_shape, sm_shape, sm_shape],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )

            # Same cross-tile reduction as multi-tile
            out_acc = out_acc.reshape(B * H, num_tok_tiles, R, D)
            out_sum = out_sum.reshape(B * H, num_tok_tiles, R)
            out_max = out_max.reshape(B * H, num_tok_tiles, R)
            global_max = mx.max(out_max, axis=1, keepdims=True)
            scale_factors = mx.exp(out_max - global_max)
            scaled_acc = mx.sum(out_acc * scale_factors[..., None], axis=1)
            denom = mx.sum(out_sum * scale_factors, axis=1)
            out_rotated = (scaled_acc / mx.maximum(denom[..., None], _EPS)).reshape(B, H, R, D)

            output = mx.matmul(out_rotated, self.value_codec.rotation)
            return mx.expand_dims(output, axis=3)

        elif multi_kernel is not None:
            num_val_tiles = (D + 31) // 32
            acc_shape = (B * H * num_tok_tiles, R, D)
            sm_shape = (B * H * num_tok_tiles * R,)

            out_acc, out_sum, out_max = multi_kernel(
                inputs=[
                    q_rot, q_proj,
                    keys_state.norms, keys_state.mse_indices,
                    keys_state.residual_norms, keys_state.qjl_signs,
                    values_state.norms, values_state.indices,
                    self.key_codec.mse_codec.codebook,
                    self.key_codec.scale_array,
                    self.value_codec.codebook,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("TokTileSize", tok_tile_size),
                    ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                    ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
                    ("VPackedWidth", values_state.indices.shape[-1]),
                    ("ValBits", bits),
                ],
                grid=(32, num_val_tiles, B * H * num_tok_tiles),
                threadgroup=(32, 1, 1),
                output_shapes=[acc_shape, sm_shape, sm_shape],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )

            # Cross-tile reduction with scalar sum/max
            out_acc = out_acc.reshape(B * H, num_tok_tiles, R, D)
            out_sum = out_sum.reshape(B * H, num_tok_tiles, R)
            out_max = out_max.reshape(B * H, num_tok_tiles, R)
            global_max = mx.max(out_max, axis=1, keepdims=True)
            scale_factors = mx.exp(out_max - global_max)
            scaled_acc = mx.sum(out_acc * scale_factors[..., None], axis=1)
            denom = mx.sum(out_sum * scale_factors, axis=1)
            out_rotated = (scaled_acc / mx.maximum(denom[..., None], _EPS)).reshape(B, H, R, D)

            output = mx.matmul(out_rotated, self.value_codec.rotation)
            return mx.expand_dims(output, axis=3)

        # Fallback: compiled two-dispatch path
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
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)

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

        if total_tokens <= self.decode_key_chunk_size and (mask is None or (isinstance(mask, str) and mask == "causal")):
            # Separate-kernel path: score keys (fast) → single-tile value weighted sum
            # 2-3x faster than fused kernel at large D because each kernel
            # only iterates its own dims once, avoiding the fused kernel's
            # double-iteration (score all dims + accumulate all dims per token).
            sep_output = self._separate_score_value_decode(
                grouped_queries, keys_state, values_state,
            )
            if sep_output is not None:
                output = sep_output.reshape(B, n_q_heads, L, value_dim)
                return output.astype(queries.dtype)

            # Fallback: fused kernel paths
            fast_output = self._compiled_integer_decode_attention(
                grouped_queries,
                keys_state,
                values_state,
            )
            if fast_output is not None:
                output = fast_output.reshape(B, n_q_heads, L, value_dim)
                return output.astype(queries.dtype)

            fast_output = self._compiled_split_decode_attention(
                grouped_queries, keys_state, values_state,
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
