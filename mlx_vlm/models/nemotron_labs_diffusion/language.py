import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from ..diffusion_visualizer import DiffusionUnmaskingVisualizer
from ..rope_utils import initialize_rope
from .config import ModelConfig

_HAS_METAL = mx.metal.is_available()


def _load_mlx_steel_gemm_header() -> Optional[str]:
    include_root = Path(mx.__file__).parent / "include"
    if not include_root.exists():
        return None

    seen: set[Path] = set()

    def expand(include_path: str) -> str:
        path = include_root / include_path
        if path in seen:
            return ""
        seen.add(path)
        lines = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "mlx/') and stripped.endswith('"'):
                lines.append(expand(stripped[len('#include "') : -1]))
            elif stripped != "#pragma once":
                lines.append(line)
        return "\n".join(lines)

    try:
        return expand("mlx/backend/metal/kernels/steel/gemm/gemm.h")
    except OSError:
        return None


def _make_bm32_linear_kernel():
    header = _load_mlx_steel_gemm_header()
    if header is None:
        return None

    return mx.fast.metal_kernel(
        name="nemotron_bm32_steel_linear_nt",
        input_names=["x", "weight"],
        output_names=["out"],
        header=header + "\nusing namespace metal;\nusing namespace mlx::steel;\n",
        source=r"""
            constexpr short BM = 32;
            constexpr short BN = 64;
            constexpr short BK = 16;
            constexpr short WM = 2;
            constexpr short WN = 2;

            using gemm_kernel = GEMMKernel<
                T, T, BM, BN, BK, WM, WN,
                false, true, true, true, float>;
            using loader_a_t = typename gemm_kernel::loader_a_t;
            using loader_b_t = typename gemm_kernel::loader_b_t;
            using mma_t = typename gemm_kernel::mma_t;

            const uint tid_x = threadgroup_position_in_grid.x;
            const uint tid_y = threadgroup_position_in_grid.y;
            const int c_row = int(tid_y) * BM;
            const int c_col = int(tid_x) * BN;

            const device T* A = x + c_row * K;
            const device T* B = weight + c_col * K;
            device T* D = out + c_row * O + c_col;

            threadgroup T As[gemm_kernel::tgp_mem_size_a];
            threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
            threadgroup_barrier(mem_flags::mem_none);

            thread mma_t mma_op(
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            thread loader_a_t loader_a(
                A, K, As, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
            thread loader_b_t loader_b(
                B, K, Bs, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);

            for (int kk = 0; kk < K / BK; ++kk) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                loader_a.load_unsafe();
                loader_b.load_unsafe();

                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma_op.mma(As, Bs);

                loader_a.next();
                loader_b.next();
            }

            threadgroup_barrier(mem_flags::mem_none);
            mma_op.store_result(D, O);
        """,
    )


_BM32_LINEAR_KERNEL = _make_bm32_linear_kernel() if _HAS_METAL else None


def _topk(x: mx.array, k: int, axis: int = -1) -> Tuple[mx.array, mx.array]:
    indices = mx.argpartition(-x, kth=k - 1, axis=axis)[..., :k]
    values = mx.take_along_axis(x, indices, axis=axis)
    order = mx.argsort(-values, axis=axis)
    return mx.take_along_axis(values, order, axis=axis), mx.take_along_axis(
        indices, order, axis=axis
    )


def _first_token_index(tokens: mx.array, token_ids: set[int]) -> Optional[int]:
    values = tokens.tolist()
    return next(
        (index for index, token_id in enumerate(values) if token_id in token_ids),
        None,
    )


def _make_bidirectional_mask(
    attention_mask: Optional[mx.array], x: mx.array
) -> Optional[mx.array]:
    zero = mx.array(0, dtype=x.dtype)
    neg_large = mx.array(mx.finfo(x.dtype).min, dtype=x.dtype)
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        if attention_mask.dtype == mx.bool_:
            return attention_mask
        return mx.where(attention_mask.astype(mx.bool_), zero, neg_large)
    if attention_mask.ndim != 2:
        return attention_mask

    if attention_mask.shape[-1] == 0 or bool(mx.all(attention_mask).item()):
        return None
    mask = attention_mask[:, None, None, :].astype(mx.bool_)
    return mx.where(mask, zero, neg_large)


def _llama4_attention_scale(
    config: ModelConfig, length: int, offset: Any, dtype: mx.Dtype
) -> mx.array:
    beta = config.rope_parameters.get("llama_4_scaling_beta")
    original_max = config.rope_parameters.get("original_max_position_embeddings")
    if beta is None or original_max is None:
        return mx.array(1.0, dtype=dtype)
    positions = mx.arange(length, dtype=mx.float32) + offset
    scale = 1.0 + float(beta) * mx.log1p(mx.floor(positions / float(original_max)))
    return scale.astype(dtype)[None, None, :, None]


def _transformers_eager_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array | str],
) -> mx.array:
    if keys.shape[1] != queries.shape[1]:
        repeats = queries.shape[1] // keys.shape[1]
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)

    scores = (
        mx.matmul(
            queries.astype(mx.float32),
            keys.astype(mx.float32).transpose(0, 1, 3, 2),
        )
        * scale
    )
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"Unsupported attention mask {mask!r}.")
        query_length = queries.shape[-2]
        key_length = keys.shape[-2]
        query_positions = mx.arange(query_length)[:, None] + (key_length - query_length)
        key_positions = mx.arange(key_length)[None, :]
        causal = key_positions <= query_positions
        neg_large = mx.array(mx.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = mx.where(causal[None, None, :, :], scores, neg_large)
    elif mask is not None:
        scores = scores + mask.astype(scores.dtype)

    weights = mx.softmax(scores, axis=-1).astype(queries.dtype)
    return mx.matmul(weights, values)


_SMALL_ROW_GEMV_KERNEL = (
    mx.fast.metal_kernel(
        name="nemotron_small_row_gemv",
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
                        result[tm] += static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                    }
                }

                col += blockN;
            }

            if (leftover > 0) {
                float v[TN];
                for (int tn = 0; tn < TN; ++tn) {
                    v[tn] = (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
                }

                for (int tm = 0; tm < TM; ++tm) {
                    for (int tn = 0; tn < TN; ++tn) {
                        T m = (col + tn < K) ? mat[tm * K + col + tn] : T(0);
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
    if _HAS_METAL
    else None
)


def _small_row_gemv_weight(
    weight: mx.array, x: mx.array, max_sequence_length: int
) -> Optional[mx.array]:
    if (
        _SMALL_ROW_GEMV_KERNEL is None
        or x.ndim != 3
        or x.dtype != weight.dtype
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not (2 <= x.shape[1] <= max_sequence_length)
    ):
        return None

    batch, length, in_dim = x.shape
    out_dim, weight_in_dim = weight.shape
    if in_dim != weight_in_dim or out_dim < 4 or out_dim % 4 != 0:
        return None

    rows = batch * length
    rows8 = ((rows + 7) // 8) * 8
    out = _SMALL_ROW_GEMV_KERNEL(
        inputs=[x.reshape(rows, in_dim), weight],
        template=[
            ("T", x.dtype),
            ("K", in_dim),
            ("O", out_dim),
            ("R", rows),
        ],
        grid=(32, out_dim // 4, rows8),
        threadgroup=(32, 1, 8),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(batch, length, out_dim)


def _small_row_linear(
    linear: nn.Linear, x: mx.array, max_sequence_length: int
) -> Optional[mx.array]:
    if not isinstance(linear, nn.Linear):
        return None
    out = _small_row_gemv_weight(linear.weight, x, max_sequence_length)
    if out is None:
        return None
    bias = getattr(linear, "bias", None)
    if bias is not None:
        out = out + bias.astype(out.dtype)
    return out


def _bm32_linear(linear: nn.Linear, x: mx.array) -> Optional[mx.array]:
    if (
        _BM32_LINEAR_KERNEL is None
        or not isinstance(linear, nn.Linear)
        or x.ndim != 3
        or x.dtype != linear.weight.dtype
        or x.dtype != mx.bfloat16
        or x.shape[-2] != 32
    ):
        return None

    batch, length, in_dim = x.shape
    rows = batch * length
    out_dim, weight_in_dim = linear.weight.shape
    if (
        in_dim != weight_in_dim
        or rows % 32 != 0
        or out_dim % 64 != 0
        or in_dim % 16 != 0
    ):
        return None

    out = _BM32_LINEAR_KERNEL(
        inputs=[x.reshape(rows, in_dim), linear.weight],
        template=[
            ("T", x.dtype),
            ("K", in_dim),
            ("O", out_dim),
        ],
        grid=(128 * (out_dim // 64), rows // 32, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x.dtype],
    )[0].reshape(batch, length, out_dim)
    bias = getattr(linear, "bias", None)
    if bias is not None:
        out = out + bias.astype(out.dtype)
    return out


def _chunked_greedy_score_weight(
    weight: mx.array,
    x: mx.array,
    chunks: int,
    return_prob: bool,
) -> Optional[mx.array | Tuple[mx.array, mx.array]]:
    if (
        chunks <= 1
        or x.ndim != 3
        or x.dtype != weight.dtype
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return None

    _, _, in_dim = x.shape
    out_dim, weight_in_dim = weight.shape
    if in_dim != weight_in_dim or out_dim % chunks != 0:
        return None

    best_token = None
    best_logit = None
    normalizer_max = None
    normalizer_sum = None
    offset = 0
    for weight_chunk in mx.split(weight, chunks, axis=0):
        logits = mx.matmul(x, weight_chunk.T)
        chunk_token = mx.argmax(logits, axis=-1).astype(mx.int32)
        chunk_logit = mx.take_along_axis(logits, chunk_token[..., None], axis=-1)[
            ..., 0
        ]
        chunk_token = chunk_token + offset

        if best_logit is None:
            best_logit = chunk_logit
            best_token = chunk_token
        else:
            take_chunk = chunk_logit > best_logit
            best_logit = mx.where(take_chunk, chunk_logit, best_logit)
            best_token = mx.where(take_chunk, chunk_token, best_token)

        if return_prob:
            logits = logits.astype(mx.float32)
            chunk_max = mx.max(logits, axis=-1)
            chunk_sum = mx.sum(mx.exp(logits - chunk_max[..., None]), axis=-1)
            if normalizer_max is None:
                normalizer_max = chunk_max
                normalizer_sum = chunk_sum
            else:
                new_max = mx.maximum(normalizer_max, chunk_max)
                normalizer_sum = normalizer_sum * mx.exp(
                    normalizer_max - new_max
                ) + chunk_sum * mx.exp(chunk_max - new_max)
                normalizer_max = new_max

        offset += weight_chunk.shape[0]

    if not return_prob:
        return best_token

    log_prob = best_logit.astype(mx.float32) - (normalizer_max + mx.log(normalizer_sum))
    return best_token, mx.exp(log_prob).astype(x.dtype)


_SMALL_ROW_SWIGLU_KERNEL = (
    mx.fast.metal_kernel(
        name="nemotron_small_row_swiglu",
        input_names=["x", "gate_weight", "up_weight"],
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
            const device T* gate_mat = gate_weight + out_row * K;
            const device T* up_mat = up_weight + out_row * K;

            float gate_result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
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
                        float value = v[tn];
                        gate_result[tm] +=
                            static_cast<float>(gate_mat[tm * K + col + tn]) * value;
                        up_result[tm] +=
                            static_cast<float>(up_mat[tm * K + col + tn]) * value;
                    }
                }

                col += blockN;
            }

            if (leftover > 0) {
                float v[TN];
                for (int tn = 0; tn < TN; ++tn) {
                    v[tn] = (col + tn < K) ? static_cast<float>(in_vec[col + tn]) : 0.0f;
                }

                for (int tm = 0; tm < TM; ++tm) {
                    for (int tn = 0; tn < TN; ++tn) {
                        float value = v[tn];
                        T gate_weight_value =
                            (col + tn < K) ? gate_mat[tm * K + col + tn] : T(0);
                        T up_weight_value =
                            (col + tn < K) ? up_mat[tm * K + col + tn] : T(0);
                        gate_result[tm] += static_cast<float>(gate_weight_value) * value;
                        up_result[tm] += static_cast<float>(up_weight_value) * value;
                    }
                }
            }

            for (int tm = 0; tm < TM; ++tm) {
                for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                    gate_result[tm] += simd_shuffle_down(gate_result[tm], sn);
                    up_result[tm] += simd_shuffle_down(up_result[tm], sn);
                }
            }

            if (lane == 0) {
                for (int tm = 0; tm < TM; ++tm) {
                    float gate = static_cast<float>(static_cast<T>(gate_result[tm]));
                    float up = static_cast<float>(static_cast<T>(up_result[tm]));
                    float activated = gate / (1.0f + exp(-gate));
                    out[row * O + out_row + tm] = static_cast<T>(activated * up);
                }
            }
        """,
    )
    if _HAS_METAL
    else None
)


def _small_row_swiglu(
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    x: mx.array,
    max_sequence_length: int,
) -> Optional[mx.array]:
    if (
        _SMALL_ROW_SWIGLU_KERNEL is None
        or not isinstance(gate_proj, nn.Linear)
        or not isinstance(up_proj, nn.Linear)
        or getattr(gate_proj, "bias", None) is not None
        or getattr(up_proj, "bias", None) is not None
        or x.ndim != 3
        or x.dtype != gate_proj.weight.dtype
        or x.dtype != up_proj.weight.dtype
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not (2 <= x.shape[1] <= max_sequence_length)
    ):
        return None

    batch, length, in_dim = x.shape
    out_dim, weight_in_dim = gate_proj.weight.shape
    up_out_dim, up_weight_in_dim = up_proj.weight.shape
    if (
        in_dim != weight_in_dim
        or in_dim != up_weight_in_dim
        or out_dim != up_out_dim
        or out_dim < 4
        or out_dim % 4 != 0
    ):
        return None

    rows = batch * length
    rows8 = ((rows + 7) // 8) * 8
    out = _SMALL_ROW_SWIGLU_KERNEL(
        inputs=[x.reshape(rows, in_dim), gate_proj.weight, up_proj.weight],
        template=[
            ("T", x.dtype),
            ("K", in_dim),
            ("O", out_dim),
            ("R", rows),
        ],
        grid=(32, out_dim // 4, rows8),
        threadgroup=(32, 1, 8),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(batch, length, out_dim)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_bm32 = True
        self.small_sequence_chunks = 4 if config.intermediate_size % 4 == 0 else 1
        self.tiny_sequence_chunks = 28 if config.intermediate_size % 28 == 0 else 1
        self.medium_sequence_chunks = 4 if config.intermediate_size % 4 == 0 else 1
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    @staticmethod
    def _chunked_linear(linear: nn.Linear, x: mx.array, chunks: int) -> mx.array:
        if not isinstance(linear, nn.Linear):
            return linear(x)
        weight_chunks = mx.split(linear.weight, chunks, axis=0)
        outputs = [mx.matmul(x, weight.T) for weight in weight_chunks]
        bias = getattr(linear, "bias", None)
        if bias is not None:
            bias_chunks = mx.split(bias, chunks, axis=0)
            outputs = [output + bias for output, bias in zip(outputs, bias_chunks)]
        return mx.concatenate(outputs, axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        sequence_length = x.shape[-2]
        if self.use_bm32 and sequence_length == 32:
            gate = _bm32_linear(self.gate_proj, x)
            up = _bm32_linear(self.up_proj, x)
            if gate is not None and up is not None:
                return self.down_proj(swiglu(gate, up))

        if 2 <= sequence_length <= 8:
            hidden = _small_row_swiglu(
                self.gate_proj,
                self.up_proj,
                x,
                max_sequence_length=8,
            )
            if hidden is not None:
                down = _small_row_linear(self.down_proj, hidden, max_sequence_length=8)
                if down is not None:
                    return down
                return self.down_proj(hidden)
            gate = _small_row_linear(self.gate_proj, x, max_sequence_length=8)
            up = _small_row_linear(self.up_proj, x, max_sequence_length=8)
            if gate is not None and up is not None:
                return self.down_proj(swiglu(gate, up))

        chunks = 1
        if 2 <= sequence_length <= 8:
            chunks = self.tiny_sequence_chunks
        elif sequence_length <= 16:
            chunks = self.small_sequence_chunks
        elif sequence_length <= 32:
            chunks = self.medium_sequence_chunks

        if chunks > 1:
            gate = self._chunked_linear(self.gate_proj, x, chunks)
            up = self._chunked_linear(self.up_proj, x, chunks)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        return self.down_proj(swiglu(gate, up))


class DraftLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.linear = linear
        self.scale = scale
        out_dim, packed_or_in_dim = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            in_dim = (packed_or_in_dim * 32) // linear.bits
            self.lora_dtype = linear.scales.dtype
        else:
            in_dim = packed_or_in_dim
            self.lora_dtype = linear.weight.dtype
        self.lora_a = mx.zeros((in_dim, rank), dtype=self.lora_dtype)
        self.lora_b = mx.zeros((rank, out_dim), dtype=self.lora_dtype)
        self.enabled = False

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if not self.enabled:
            return y
        z = (x @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)
        return y + (self.scale * z).astype(y.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_transformers_eager_attention = False
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        use_cache: bool = True,
        attention_scale: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = _small_row_linear(self.q_proj, x, max_sequence_length=8)
        if queries is None:
            queries = self.q_proj(x)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if attention_scale is None:
            attention_scale = _llama4_attention_scale(
                self.config, L, offset, queries.dtype
            )
        queries = queries * attention_scale

        if cache is not None:
            if use_cache:
                keys, values = cache.update_and_fetch(keys, values)
            elif cache.keys is not None:
                keys = mx.concatenate(
                    [cache.keys[..., : cache.offset, :], keys], axis=2
                )
                values = mx.concatenate(
                    [cache.values[..., : cache.offset, :], values], axis=2
                )

        if self.use_transformers_eager_attention:
            output = _transformers_eager_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        projected = _small_row_linear(self.o_proj, output, max_sequence_length=8)
        if projected is not None:
            return projected
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        use_cache: bool = True,
        attention_scale: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
            use_cache=use_cache,
            attention_scale=attention_scale,
        )
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class NemotronLabsDiffusionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        use_cache: bool = True,
        use_causal_mask: bool = False,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if use_causal_mask:
            layer_mask = create_attention_mask(h, cache[0])
        else:
            layer_mask = _make_bidirectional_mask(
                mask if mask is not None else attention_mask, h
            )
        first_cache = cache[0] if cache else None
        offset = first_cache.offset if first_cache is not None else 0
        attention_scale = _llama4_attention_scale(
            self.config, h.shape[1], offset, h.dtype
        )
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(
                h,
                mask=layer_mask,
                cache=layer_cache,
                use_cache=use_cache,
                attention_scale=attention_scale,
            )
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dlm_paradigm not in ("bidirectional", "autoregressive"):
            raise ValueError(
                f"Unsupported Nemotron Labs Diffusion paradigm: {config.dlm_paradigm}"
            )
        self.config = config
        self.model_type = config.model_type
        self.model = NemotronLabsDiffusionEncoder(config)
        if not config.tie_word_embeddings:
            self.diffusion_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        self.small_sequence_head_chunks = (
            32 if config.vocab_size >= 4096 and config.vocab_size % 32 == 0 else 1
        )
        self.greedy_score_chunks = (
            4 if config.vocab_size >= 4096 and config.vocab_size % 4 == 0 else 1
        )
        self._linear_spec_lora_loaded = False

    def _set_transformers_parity_runtime(self, enabled: bool) -> None:
        for layer in self.model.layers:
            layer.self_attn.use_transformers_eager_attention = enabled
            layer.mlp.use_bm32 = not enabled

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            attention_mask=kwargs.get("attention_mask"),
            cache=cache,
            use_cache=kwargs.get("use_cache", True),
            use_causal_mask=kwargs.get("use_causal_mask", True),
        )
        return LanguageModelOutput(logits=self._project_hidden(out))

    @staticmethod
    def _top_k_logits(logits: mx.array, k: Optional[int]) -> mx.array:
        if k is None or k <= 0:
            return logits
        values = _topk(logits, k=k, axis=-1)[0]
        neg_large = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
        return mx.where(logits < values[..., -1:], neg_large, logits)

    @staticmethod
    def _top_p_logits(logits: mx.array, p: Optional[float]) -> mx.array:
        if p is None or p >= 1.0:
            return logits
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(
            mx.softmax(sorted_logits, axis=-1, precise=True), axis=-1
        )
        sorted_mask = cumulative_probs > p
        sorted_mask = mx.concatenate(
            [mx.zeros_like(sorted_mask[..., :1]), sorted_mask[..., :-1]], axis=-1
        )
        inverse_indices = mx.argsort(sorted_indices, axis=-1)
        mask = mx.take_along_axis(sorted_mask, inverse_indices, axis=-1)
        neg_large = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
        return mx.where(mask, neg_large, logits)

    def _sample_with_temperature_topk_topp(
        self,
        logits: mx.array,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        if temperature == 0.0:
            token = mx.argmax(logits, axis=-1)
            token_logit = mx.take_along_axis(logits, token[..., None], axis=-1)[..., 0]
            token_prob = mx.exp(token_logit - mx.logsumexp(logits, axis=-1))
            return token, token_prob

        if temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        token = mx.random.categorical(logits.astype(mx.float32), axis=-1)
        token_logit = mx.take_along_axis(logits, token[..., None], axis=-1)[..., 0]
        token_prob = mx.exp(token_logit - mx.logsumexp(logits, axis=-1))
        return token, token_prob

    def _project_hidden(self, hidden_states: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            out = _small_row_gemv_weight(
                self.model.embed_tokens.weight,
                hidden_states,
                max_sequence_length=8,
            )
            if out is not None:
                return out
            return self.model.embed_tokens.as_linear(hidden_states)
        out = _small_row_linear(
            self.diffusion_head,
            hidden_states,
            max_sequence_length=8,
        )
        if out is not None:
            return out
        if (
            isinstance(self.diffusion_head, nn.Linear)
            and self.small_sequence_head_chunks > 1
            and 2 <= hidden_states.shape[-2] <= 16
        ):
            weight_chunks = mx.split(
                self.diffusion_head.weight, self.small_sequence_head_chunks, axis=0
            )
            return mx.concatenate(
                [mx.matmul(hidden_states, weight.T) for weight in weight_chunks],
                axis=-1,
            )
        return self.diffusion_head(hidden_states)

    def _greedy_sample_hidden(
        self,
        hidden_states: mx.array,
        return_prob: bool = False,
        chunked_score: bool = False,
    ) -> mx.array | Tuple[mx.array, mx.array]:
        if return_prob and chunked_score:
            scored = self._chunked_greedy_score_hidden(hidden_states, return_prob=True)
            if scored is not None:
                return scored
        logits = self._project_hidden(hidden_states)
        if return_prob:
            return self._sample_with_temperature_topk_topp(logits, temperature=0.0)
        return self._sample_tokens(logits, temperature=0.0)

    def _chunked_greedy_score_hidden(
        self, hidden_states: mx.array, return_prob: bool
    ) -> Optional[mx.array | Tuple[mx.array, mx.array]]:
        if hidden_states.shape[-2] > 32:
            return None
        weight = (
            self.model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.diffusion_head.weight
        )
        if not self.config.tie_word_embeddings and not isinstance(
            self.diffusion_head, nn.Linear
        ):
            return None
        return _chunked_greedy_score_weight(
            weight,
            hidden_states,
            chunks=self.greedy_score_chunks,
            return_prob=return_prob,
        )

    def _sample_from_hidden(
        self,
        hidden_states: mx.array,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_prob: bool = False,
        chunked_score: bool = False,
    ) -> mx.array | Tuple[mx.array, mx.array]:
        if temperature == 0.0:
            return self._greedy_sample_hidden(
                hidden_states,
                return_prob=return_prob,
                chunked_score=chunked_score,
            )

        logits = self._project_hidden(hidden_states)
        if return_prob:
            return self._sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        return self._sample_tokens(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def _sample_tokens(
        self,
        logits: mx.array,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> mx.array:
        if temperature == 0.0:
            return mx.argmax(logits, axis=-1)

        if temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        return mx.random.categorical(logits.astype(mx.float32), axis=-1)

    @staticmethod
    def _trim_cache(cache, max_length: int) -> None:
        for layer_cache in cache:
            excess = max(0, int(layer_cache.offset) - int(max_length))
            if excess:
                layer_cache.trim(excess)

    def load_linear_spec_lora(self, adapter_path: str | Path) -> bool:
        adapter_path = Path(adapter_path)
        adapter_file = adapter_path / "adapter_model.safetensors"
        if not adapter_file.exists():
            return False
        weights = mx.load(str(adapter_file))
        rank = 128
        scale = 4.0

        for layer_idx, layer in enumerate(self.model.layers):
            o_proj = layer.self_attn.o_proj
            if not isinstance(o_proj, DraftLoRALinear):
                o_proj = DraftLoRALinear(o_proj, rank=rank, scale=scale)
                layer.self_attn.o_proj = o_proj

            prefix = "base_model.model.encoder.layers." f"{layer_idx}.self_attn.o_proj"
            key_a = f"{prefix}.lora_A.weight"
            key_b = f"{prefix}.lora_B.weight"
            if key_a not in weights or key_b not in weights:
                return False
            o_proj.lora_a = weights[key_a].T.astype(o_proj.lora_dtype)
            o_proj.lora_b = weights[key_b].T.astype(o_proj.lora_dtype)

        self._linear_spec_lora_loaded = True
        return True

    def set_linear_spec_lora_enabled(self, enabled: bool) -> None:
        for layer in self.model.layers:
            o_proj = layer.self_attn.o_proj
            if isinstance(o_proj, DraftLoRALinear):
                o_proj.enabled = enabled

    def generate(
        self,
        inputs: mx.array,
        temperature: float = 0.0,
        block_length: Optional[int] = None,
        steps: Optional[int] = None,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: Optional[float] = None,
        min_threshold: Optional[float] = None,
        editing_threshold: Optional[float] = None,
        max_post_steps: Optional[int] = None,
        eos_id: Optional[int] = None,
        mask_id: Optional[int] = None,
        num_to_transfer: Optional[int] = None,
        max_transfer_per_step: Optional[int] = None,
        stability_steps: Optional[int] = None,
        visualize: bool = False,
        tokenizer: Optional[Any] = None,
        skip_special_tokens: bool = False,
        stats: Optional[Dict[str, float]] = None,
        linear_speculative: bool = False,
        **kwargs,
    ) -> mx.array:
        generation_mode = kwargs.get("generation_mode")
        if generation_mode in ("linear_speculative", "linear_spec"):
            linear_speculative = True

        if inputs.shape[0] != 1:
            raise ValueError(
                "Nemotron Labs Diffusion generation currently supports batch size 1."
            )

        eos_id = self.config.eos_token_id if eos_id is None else eos_id
        mask_id = self.config.mask_token_id if mask_id is None else mask_id

        def config_default(name, fallback):
            value = getattr(self.config, name, None)
            return fallback if value is None else value

        block_length = int(
            block_length
            if block_length is not None
            else config_default(
                "default_block_length",
                config_default("block_size", 32),
            )
        )
        steps = int(
            steps
            if steps is not None
            else config_default("default_diffusion_steps", 32)
        )
        threshold = (
            threshold
            if threshold is not None
            else config_default("default_diffusion_threshold", 0.95)
        )
        if editing_threshold is None:
            editing_threshold = config_default(
                "default_diffusion_editing_threshold", 0.9
            )
        max_post_steps = int(
            max_post_steps
            if max_post_steps is not None
            else config_default("default_diffusion_max_post_steps", 16)
        )
        num_to_transfer = int(
            num_to_transfer
            if num_to_transfer is not None
            else config_default("default_diffusion_num_to_transfer", 1)
        )
        if max_transfer_per_step is None:
            max_transfer_per_step = config_default(
                "default_diffusion_max_transfer_per_step", None
            )
        stability_steps = int(
            stability_steps
            if stability_steps is not None
            else config_default("default_diffusion_stability_steps", 2)
        )
        if linear_speculative:
            if not self._linear_spec_lora_loaded:
                model_path = getattr(self, "model_path", None)
                if model_path is not None:
                    self.load_linear_spec_lora(Path(model_path) / "linear_spec_lora")
            output, _ = self.linear_spec_generate(
                inputs,
                max_new_tokens=gen_length,
                block_length=block_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                mask_token_id=mask_id,
                eos_token_id=eos_id,
                threshold=0.0,
                stats=stats,
            )
            return output[:, inputs.shape[1] :]

        eos_token_ids = (
            set(eos_id) if isinstance(eos_id, (list, tuple, set)) else {eos_id}
        )
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")
        steps = max(1, int(steps))
        sampler = kwargs.get("sampler")
        ar_weight = kwargs.get("ar_weight", 0.0)
        head_scoring = kwargs.get("head_scoring")
        ar_weight = float(ar_weight)
        if ar_weight < 0.0 or ar_weight > 1.0:
            raise ValueError("ar_weight must be between 0.0 and 1.0.")
        default_sampler_name = getattr(
            self.config, "default_diffusion_sampler", "native"
        )
        sampler_name = (sampler or default_sampler_name).lower()
        sampler_aliases = {
            "default": default_sampler_name.lower(),
            "optimized": "confidence_threshold_bound",
            "threshold_bound": "confidence_threshold_bound",
            "bound": "confidence_threshold_bound",
            "hf": "native",
            "upstream": "native",
            "confidence_threshold": "native",
            "threshold": "native",
            "threshold_ref": "confidence_threshold_ref",
            "ref": "confidence_threshold_ref",
            "cumulative": "cumulative_error",
        }
        sampler_name = sampler_aliases.get(sampler_name, sampler_name)
        valid_samplers = {
            "native",
            "fixed",
            "confidence_threshold_ref",
            "confidence_threshold_bound",
            "cumulative_error",
        }
        if sampler_name not in valid_samplers:
            raise ValueError(
                "Unsupported Nemotron diffusion sampler "
                f"{sampler!r}. Expected one of {sorted(valid_samplers)}."
            )
        sampling_scaling_factor_arg = kwargs.get(
            "sampling_scaling_factor", kwargs.get("factor")
        )
        if sampling_scaling_factor_arg is None:
            sampling_scaling_factor = (
                getattr(self.config, "default_diffusion_sampling_scaling_factor", 2.0)
                if sampler_name == "confidence_threshold_bound"
                else 1.0
            )
        else:
            sampling_scaling_factor = float(sampling_scaling_factor_arg)
        if min_threshold is None and sampler_name == "confidence_threshold_bound":
            min_threshold = getattr(self.config, "default_diffusion_min_threshold", 0.4)
        if min_threshold is not None:
            min_threshold = float(min_threshold)
        transformers_parity_arg = kwargs.get("transformers_parity")
        transformers_parity = (
            sampler_name == "native"
            if transformers_parity_arg is None
            else bool(transformers_parity_arg)
        )
        self._set_transformers_parity_runtime(transformers_parity)
        head_scoring_name = (head_scoring or "full").lower()
        head_scoring_aliases = {
            "default": "full",
            "full_logits": "full",
            "project_full_logits": "full",
            "chunked_masked": "chunked",
        }
        head_scoring_name = head_scoring_aliases.get(
            head_scoring_name, head_scoring_name
        )
        if head_scoring_name not in {"full", "chunked"}:
            raise ValueError(
                "Unsupported Nemotron head_scoring "
                f"{head_scoring!r}. Expected 'full' or 'chunked'."
            )
        use_chunked_scoring = head_scoring_name == "chunked"
        if max_transfer_per_step is not None:
            max_transfer_per_step = min(
                block_length, max(1, int(max_transfer_per_step))
            )
        if stats is not None:
            stats["diffusion_sampler"] = sampler_name
            stats["diffusion_head_scoring"] = (
                "chunked_masked"
                if use_chunked_scoring and self.greedy_score_chunks > 1
                else "project_full_logits"
            )
            stats["diffusion_min_threshold"] = (
                float(min_threshold) if min_threshold is not None else float("nan")
            )
            stats["diffusion_sampling_scaling_factor"] = sampling_scaling_factor
            stats["diffusion_transformers_parity"] = float(transformers_parity)
            for key in (
                "diffusion_blocks",
                "diffusion_denoise_nfe",
                "diffusion_post_block_nfe",
                "diffusion_confidence_steps",
                "diffusion_argmax_only_steps",
                "diffusion_masked_rows_scored",
                "diffusion_accepted_tokens",
                "diffusion_mixed_ar_forwards",
            ):
                stats.setdefault(key, 0.0)

        def add_stat(key: str, value: float = 1.0) -> None:
            if stats is not None:
                stats[key] = stats.get(key, 0.0) + float(value)

        visualizer = DiffusionUnmaskingVisualizer(
            active=visualize and sys.stdout.isatty(),
            mask_id=mask_id,
            eos_token_ids=eos_token_ids,
            tokenizer=tokenizer,
            skip_special_tokens=skip_special_tokens,
        )

        generated_blocks = []
        prompt_tic = time.perf_counter()
        recorded_prompt_time = False
        cache = self.make_cache()
        prefill_hidden = self.model(
            inputs,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        )
        next_token = self._sample_from_hidden(
            prefill_hidden[:, -1:, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        mx.eval(next_token)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(inputs.size)
            recorded_prompt_time = True

        total_generated = 0
        end_length = None
        num_blocks = (gen_length + block_length - 1) // block_length
        for _ in range(num_blocks):
            remaining = gen_length - total_generated
            if remaining <= 0:
                break
            add_stat("diffusion_blocks")
            current_block_length = min(block_length, remaining)
            block_positions = mx.arange(current_block_length)
            block = mx.full((1, current_block_length), mask_id, dtype=inputs.dtype)
            block[:, 0] = next_token[:, 0]
            if visualizer.active:
                preview = (
                    mx.concatenate(generated_blocks + [block], axis=1)
                    if generated_blocks
                    else block
                )
                visualizer.visualize(preview, force=True)

            denoise_steps = max(1, min(steps, current_block_length))
            denoise_range = range(denoise_steps) if current_block_length > 1 else ()
            masked_count = max(0, current_block_length - 1)
            for step_idx in denoise_range:
                if masked_count == 0:
                    break
                mask_index = block == mask_id
                force_completion = step_idx == denoise_steps - 1
                add_stat("diffusion_denoise_nfe")
                add_stat("diffusion_masked_rows_scored", masked_count)
                masked_positions = mx.sort(
                    mx.where(mask_index[0], block_positions, current_block_length)
                )[:masked_count]
                hidden_states = self.model(
                    block,
                    cache=cache,
                    use_cache=False,
                    use_causal_mask=False,
                )
                if ar_weight > 0.0:
                    causal_hidden_states = self.model(
                        block,
                        cache=cache,
                        use_cache=False,
                        use_causal_mask=True,
                    )
                    shifted_causal_hidden_states = mx.concatenate(
                        [hidden_states[:, :1, :], causal_hidden_states[:, :-1, :]],
                        axis=1,
                    )
                    hidden_states = (
                        (1.0 - ar_weight) * hidden_states
                        + ar_weight * shifted_causal_hidden_states
                    ).astype(hidden_states.dtype)
                    add_stat("diffusion_mixed_ar_forwards")
                masked_hidden_states = mx.take(hidden_states, masked_positions, axis=1)
                need_confidence = not force_completion and masked_count > 1
                add_stat(
                    "diffusion_confidence_steps"
                    if need_confidence
                    else "diffusion_argmax_only_steps"
                )
                if need_confidence:
                    sampled_tokens, token_probs = self._sample_from_hidden(
                        masked_hidden_states,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        return_prob=True,
                        chunked_score=use_chunked_scoring,
                    )
                else:
                    sampled_tokens = self._sample_from_hidden(
                        masked_hidden_states,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    token_probs = None
                if stats is not None and not recorded_prompt_time:
                    if token_probs is None:
                        mx.eval(sampled_tokens)
                    else:
                        mx.eval(sampled_tokens, token_probs)
                    stats["prompt_time"] = time.perf_counter() - prompt_tic
                    stats["prompt_tokens"] = float(inputs.size)
                    recorded_prompt_time = True

                position_matches = block_positions[None, :] == masked_positions[:, None]
                sampled_block = mx.sum(
                    mx.where(
                        position_matches,
                        sampled_tokens[0, :, None],
                        mx.zeros(
                            (masked_count, current_block_length), dtype=block.dtype
                        ),
                    ),
                    axis=0,
                    keepdims=True,
                ).astype(block.dtype)
                sampled_block = mx.where(mask_index, sampled_block, block)

                if force_completion or masked_count == 1:
                    transfer_mask = mask_index
                    accepted_count = masked_count
                elif threshold is not None:
                    sorted_indices = mx.argsort(-token_probs, axis=-1)
                    sorted_confidence = mx.take_along_axis(
                        token_probs, sorted_indices, axis=-1
                    )
                    sorted_block_positions = mx.take_along_axis(
                        masked_positions[None, :], sorted_indices, axis=-1
                    )
                    sorted_positions = mx.arange(masked_count)[None, :]
                    transfer_limit = masked_count
                    if sampler_name == "fixed":
                        transfer_limit = min(masked_count, max(1, int(num_to_transfer)))
                    if max_transfer_per_step is not None:
                        transfer_limit = min(transfer_limit, max_transfer_per_step)

                    if sampler_name == "native":
                        keep_sorted = sorted_confidence >= threshold
                    elif sampler_name == "fixed":
                        keep_sorted = sorted_positions < transfer_limit
                        keep_sorted = keep_sorted & (sorted_confidence >= threshold)
                    elif sampler_name == "confidence_threshold_ref":
                        positional_threshold = 1.0 - sampling_scaling_factor / (
                            sorted_positions.astype(sorted_confidence.dtype) + 2.0
                        )
                        positional_threshold = mx.where(
                            sorted_positions == 0,
                            mx.array(
                                mx.finfo(sorted_confidence.dtype).min,
                                dtype=sorted_confidence.dtype,
                            ),
                            positional_threshold,
                        )
                        criteria = (sorted_confidence >= threshold) & (
                            sorted_confidence >= positional_threshold
                        )
                        keep_sorted = mx.cumprod(
                            criteria.astype(mx.int32), axis=1
                        ).astype(mx.bool_)
                        keep_sorted = keep_sorted & (sorted_positions < transfer_limit)
                    elif sampler_name == "cumulative_error":
                        confidence_floor = mx.array(
                            1e-12, dtype=sorted_confidence.dtype
                        )
                        cumulative_log_confidence = mx.cumsum(
                            mx.log(mx.maximum(sorted_confidence, confidence_floor)),
                            axis=1,
                        )
                        keep_sorted = cumulative_log_confidence >= mx.log(
                            mx.array(max(float(threshold), 1e-12))
                        )
                        keep_sorted = keep_sorted & (sorted_positions < transfer_limit)
                    else:
                        positional_threshold = 1.0 - sampling_scaling_factor / (
                            sorted_positions.astype(sorted_confidence.dtype) + 2.0
                        )
                        positional_threshold = mx.where(
                            sorted_positions == 0,
                            mx.array(
                                mx.finfo(sorted_confidence.dtype).min,
                                dtype=sorted_confidence.dtype,
                            ),
                            positional_threshold,
                        )
                        lower_bound = 0.5 if min_threshold is None else min_threshold
                        keep_sorted = (sorted_confidence >= threshold) | (
                            (sorted_confidence >= lower_bound)
                            & (sorted_confidence >= positional_threshold)
                        )
                        if max_transfer_per_step is not None:
                            keep_sorted = keep_sorted & (
                                sorted_positions < transfer_limit
                            )
                    keep_sorted = keep_sorted | (sorted_positions == 0)
                    kept_positions = mx.where(
                        keep_sorted, sorted_block_positions, current_block_length
                    )
                    transfer_mask = (
                        block_positions[None, None, :] == kept_positions[..., None]
                    ).any(axis=1) & mask_index
                    accepted_count = int(transfer_mask.sum().item())
                else:
                    remaining_steps = max(1, denoise_steps - step_idx)
                    transfer_count = max(
                        1, (masked_count + remaining_steps - 1) // remaining_steps
                    )
                    if max_transfer_per_step is not None:
                        transfer_count = min(transfer_count, max_transfer_per_step)
                    _, indices = _topk(token_probs, min(transfer_count, masked_count))
                    transfer_positions = mx.take_along_axis(
                        masked_positions[None, :], indices, axis=-1
                    )
                    transfer_mask = (
                        block_positions[None, None, :] == transfer_positions[..., None]
                    ).any(axis=1) & mask_index
                    accepted_count = min(transfer_count, masked_count)
                block = mx.where(transfer_mask, sampled_block, block)
                add_stat("diffusion_accepted_tokens", accepted_count)
                if visualizer.active and bool(transfer_mask.any().item()):
                    preview = (
                        mx.concatenate(generated_blocks + [block], axis=1)
                        if generated_blocks
                        else block
                    )
                    visualizer.visualize(preview)
                if force_completion:
                    break
                if end_length is not None:
                    break
                masked_count -= accepted_count
                if masked_count == 0:
                    break

            generated_block = block[:, :current_block_length]
            generated_blocks.append(generated_block)
            total_generated += current_block_length
            if eos_early_stop and end_length is None:
                eos_index = _first_token_index(generated_block[0], eos_token_ids)
                if eos_index is not None:
                    end_length = total_generated - current_block_length + eos_index + 1
                    break
            if end_length is not None:
                break
            if total_generated >= gen_length:
                break

            output_hidden = self.model(
                block,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            )
            add_stat("diffusion_post_block_nfe")
            next_token = self._sample_from_hidden(
                output_hidden[:, -1:, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        generated = (
            mx.concatenate(generated_blocks, axis=1)
            if generated_blocks
            else mx.zeros((1, 0), dtype=inputs.dtype)
        )
        end = end_length if end_length is not None else generated.shape[1]
        if stats is not None:
            stats["diffusion_generated_tokens"] = float(end)
            stats["diffusion_total_nfe"] = stats.get(
                "diffusion_denoise_nfe", 0.0
            ) + stats.get("diffusion_post_block_nfe", 0.0)
            denoise_nfe = stats.get("diffusion_denoise_nfe", 0.0)
            if denoise_nfe:
                stats["diffusion_tokens_per_denoise_forward"] = (
                    stats.get("diffusion_accepted_tokens", 0.0) / denoise_nfe
                )
        if visualizer.active:
            generated_ids = generated[0].tolist()
            visualizer.finish()
            if tokenizer is not None:
                final_text = tokenizer.decode(
                    generated_ids[:end], skip_special_tokens=skip_special_tokens
                )
            else:
                final_text = " ".join(str(token_id) for token_id in generated_ids[:end])
            print(final_text, end="", flush=True)
            if stats is not None:
                stats["text_already_printed"] = True
        return generated[:, :end]

    def ar_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        stats: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> tuple[mx.array, int]:
        self._set_transformers_parity_runtime(False)
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )

        prompt_tic = time.perf_counter()
        cache = self.make_cache()
        prefill_hidden = self.model(
            prompt_ids,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        )
        next_token = self._sample_from_hidden(
            prefill_hidden[:, -1:, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        mx.eval(next_token)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(prompt_ids.size)

        generated = []
        nfe = 0
        for _ in range(max_new_tokens):
            nfe += 1
            generated.append(next_token)
            if bool(
                mx.array(
                    [token in eos_token_ids for token in next_token[:, 0].tolist()]
                )
                .all()
                .item()
            ):
                break
            next_hidden = self.model(
                next_token,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            )
            next_token = self._sample_from_hidden(
                next_hidden[:, -1:, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        if not generated:
            return prompt_ids, nfe
        return (
            mx.concatenate([prompt_ids, mx.concatenate(generated, axis=1)], axis=1),
            nfe,
        )

    def linear_spec_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        threshold: float = 0.0,
        stats: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> tuple[mx.array, int]:
        self._set_transformers_parity_runtime(False)
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch size 1.")
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")
        max_draft_window = min(block_length, 32)
        base_draft_window = min(max_draft_window, 8)
        draft_window = base_draft_window

        mask_token_id = (
            self.config.mask_token_id if mask_token_id is None else mask_token_id
        )
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )

        prompt_tic = time.perf_counter()
        cache = self.make_cache()
        prefill_hidden = self.model(
            prompt_ids,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        )
        next_token = self._sample_from_hidden(
            prefill_hidden[:, -1:, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        mx.eval(next_token)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(prompt_ids.size)

        generated = [next_token]
        total_generated = 1
        nfe = 1

        if next_token.item() in eos_token_ids:
            return mx.concatenate([prompt_ids, next_token], axis=1), nfe

        while total_generated < max_new_tokens:
            cache_len = cache[0].offset
            current_block_length = min(draft_window, max_new_tokens - total_generated)
            block = mx.full(
                (1, current_block_length), mask_token_id, dtype=prompt_ids.dtype
            )
            block[:, 0] = next_token[:, 0]

            while bool((block == mask_token_id).any().item()):
                self.set_linear_spec_lora_enabled(True)
                draft_hidden = self.model(
                    block,
                    cache=cache,
                    use_cache=False,
                    use_causal_mask=False,
                )
                nfe += 1
                is_mask = block == mask_token_id
                if threshold > 0:
                    draft_tokens, draft_probs = self._sample_from_hidden(
                        draft_hidden,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        return_prob=True,
                    )
                    neg_large = mx.array(
                        mx.finfo(draft_probs.dtype).min, dtype=draft_probs.dtype
                    )
                    draft_conf = mx.where(is_mask, draft_probs, neg_large)
                    unmask = draft_conf >= threshold
                    if not bool(unmask.any().item()):
                        _, best_idx = _topk(draft_conf, 1)
                        positions = mx.arange(current_block_length)
                        unmask = (positions[None, None, :] == best_idx[..., None]).any(
                            axis=1
                        )
                    block = mx.where(unmask, draft_tokens, block)
                else:
                    draft_tokens = self._sample_from_hidden(
                        draft_hidden,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    block = mx.where(is_mask, draft_tokens, block)
                    break

            self.set_linear_spec_lora_enabled(False)
            verify_hidden = self.model(
                block,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            )
            nfe += 1
            ar_tokens = self._sample_from_hidden(
                verify_hidden,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            ar_token_ids = ar_tokens[0].tolist()
            block_ids = block[0].tolist()
            accepted = 1
            for i in range(current_block_length - 1):
                if ar_token_ids[i] == block_ids[i + 1]:
                    accepted += 1
                else:
                    break
            accepted = min(accepted, max_new_tokens - total_generated)
            accepted_tokens = ar_tokens[:, :accepted]
            generated.append(accepted_tokens)
            total_generated += accepted

            self._trim_cache(cache, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            eos_index = _first_token_index(accepted_tokens[0], eos_token_ids)
            if eos_index is not None:
                generated[-1] = accepted_tokens[:, : eos_index + 1]
                break
            if accepted == current_block_length and draft_window < max_draft_window:
                draft_window = min(max_draft_window, draft_window * 2)
            elif (
                accepted <= max(1, current_block_length // 2)
                and draft_window > base_draft_window
            ):
                draft_window = max(base_draft_window, draft_window // 2)

        return (
            mx.concatenate([prompt_ids, mx.concatenate(generated, axis=1)], axis=1),
            nfe,
        )

    def stream_linear_spec_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        threshold: float = 0.0,
        stats: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self._set_transformers_parity_runtime(False)
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch size 1.")
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")
        max_draft_window = min(block_length, 32)
        base_draft_window = min(max_draft_window, 8)
        draft_window = base_draft_window

        mask_token_id = (
            self.config.mask_token_id if mask_token_id is None else mask_token_id
        )
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )

        prompt_tic = time.perf_counter()
        cache = self.make_cache()
        prefill_hidden = self.model(
            prompt_ids,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        )
        next_token = self._sample_from_hidden(
            prefill_hidden[:, -1:, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        mx.eval(next_token)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(prompt_ids.size)

        yield next_token
        total_generated = 1

        if next_token.item() in eos_token_ids:
            return

        while total_generated < max_new_tokens:
            cache_len = cache[0].offset
            current_block_length = min(draft_window, max_new_tokens - total_generated)
            block = mx.full(
                (1, current_block_length), mask_token_id, dtype=prompt_ids.dtype
            )
            block[:, 0] = next_token[:, 0]

            while bool((block == mask_token_id).any().item()):
                self.set_linear_spec_lora_enabled(True)
                draft_hidden = self.model(
                    block,
                    cache=cache,
                    use_cache=False,
                    use_causal_mask=False,
                )
                is_mask = block == mask_token_id
                if threshold > 0:
                    draft_tokens, draft_probs = self._sample_from_hidden(
                        draft_hidden,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        return_prob=True,
                    )
                    neg_large = mx.array(
                        mx.finfo(draft_probs.dtype).min, dtype=draft_probs.dtype
                    )
                    draft_conf = mx.where(is_mask, draft_probs, neg_large)
                    unmask = draft_conf >= threshold
                    if not bool(unmask.any().item()):
                        _, best_idx = _topk(draft_conf, 1)
                        positions = mx.arange(current_block_length)
                        unmask = (positions[None, None, :] == best_idx[..., None]).any(
                            axis=1
                        )
                    block = mx.where(unmask, draft_tokens, block)
                else:
                    draft_tokens = self._sample_from_hidden(
                        draft_hidden,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    block = mx.where(is_mask, draft_tokens, block)
                    break

            self.set_linear_spec_lora_enabled(False)
            verify_hidden = self.model(
                block,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            )
            ar_tokens = self._sample_from_hidden(
                verify_hidden,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            ar_token_ids = ar_tokens[0].tolist()
            block_ids = block[0].tolist()
            accepted = 1
            for i in range(current_block_length - 1):
                if ar_token_ids[i] == block_ids[i + 1]:
                    accepted += 1
                else:
                    break
            accepted = min(accepted, max_new_tokens - total_generated)
            accepted_tokens = ar_tokens[:, :accepted]

            self._trim_cache(cache, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            eos_index = _first_token_index(accepted_tokens[0], eos_token_ids)
            if eos_index is not None:
                accepted_tokens = accepted_tokens[:, : eos_index + 1]
            mx.eval(accepted_tokens)
            yield accepted_tokens
            total_generated += accepted_tokens.shape[1]
            if eos_index is not None:
                break
            if accepted == current_block_length and draft_window < max_draft_window:
                draft_window = min(max_draft_window, draft_window * 2)
            elif (
                accepted <= max(1, current_block_length // 2)
                and draft_window > base_draft_window
            ):
                draft_window = max(base_draft_window, draft_window // 2)

    def sanitize(self, weights):
        if self.config.tie_word_embeddings:
            weights.pop("diffusion_head.weight", None)

        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
            and not k.endswith(".self_attn.k_scale")
            and not k.endswith(".self_attn.v_scale")
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        return [KVCache() for _ in self.layers]
