from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import mlx.nn
import numpy as np

PrecisionName = Literal["bf16", "1bit", "2bit"]

_SUPPORTED_BITS = (1, 2)
_SUPPORTED_SCALE_DTYPES = {"bfloat16": mx.bfloat16}
_SUPPORTED_GROUP_SIZES = (128,)
_DEFAULT_AXES_DIMS_ROPE = (32, 32, 32, 32)
DEFAULT_QUANT_GROUP_SIZE = 128


@dataclass(frozen=True)
class Flux2KleinBlockSpec:
    dim: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    mlp_ratio: float = 3.0
    layer_norm_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    rope_theta: int = 2000
    axes_dims_rope: tuple[int, ...] = _DEFAULT_AXES_DIMS_ROPE

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.dim * self.mlp_ratio)

    @property
    def qkv_mlp_out_dim(self) -> int:
        return (3 * self.dim) + (2 * self.mlp_hidden_dim)

    @property
    def out_proj_in_dim(self) -> int:
        return self.dim + self.mlp_hidden_dim

    @property
    def modulation_single_out_dim(self) -> int:
        return 3 * self.dim

    @property
    def modulation_double_out_dim(self) -> int:
        return 6 * self.dim

    @classmethod
    def from_transformer_config(cls, config_path: Path) -> "Flux2KleinBlockSpec":
        config = json.loads(config_path.read_text())
        dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
        eps = float(config["eps"])
        return cls(
            dim=dim,
            num_heads=int(config["num_attention_heads"]),
            head_dim=int(config["attention_head_dim"]),
            mlp_ratio=float(config["mlp_ratio"]),
            layer_norm_eps=eps,
            rms_norm_eps=eps,
            rope_theta=int(config.get("rope_theta", 2000)),
            axes_dims_rope=tuple(config.get("axes_dims_rope", _DEFAULT_AXES_DIMS_ROPE)),
        )


@dataclass
class PackedWeight:
    """Pre-quantized weight triple consumed by `_make_linear` as a fast path
    that skips `quantize_affine_nbit`. Shapes match `quantize_affine_nbit`'s
    outputs: packed (rows, cols // (32 // bits)), scales (rows, cols // group_size),
    biases (rows, cols // group_size).
    """

    packed: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int


WeightOrPacked = mx.array | PackedWeight


@dataclass
class SingleBlockWeights:
    modulation: mx.array
    qkv_mlp_proj: WeightOrPacked
    out_proj: WeightOrPacked
    norm_q: mx.array
    norm_k: mx.array


@dataclass
class DoubleBlockWeights:
    modulation_img: mx.array
    modulation_txt: mx.array
    to_q: WeightOrPacked
    to_k: WeightOrPacked
    to_v: WeightOrPacked
    add_q_proj: WeightOrPacked
    add_k_proj: WeightOrPacked
    add_v_proj: WeightOrPacked
    to_out: WeightOrPacked
    to_add_out: WeightOrPacked
    ff_linear_in: WeightOrPacked
    ff_linear_out: WeightOrPacked
    ff_context_linear_in: WeightOrPacked
    ff_context_linear_out: WeightOrPacked
    norm_q: mx.array
    norm_k: mx.array
    norm_added_q: mx.array
    norm_added_k: mx.array


def single_block_weight_keys(block_index: int) -> dict[str, str]:
    prefix = f"single_transformer_blocks.{block_index}.attn"
    return {
        "modulation": "single_stream_modulation.linear.weight",
        "qkv_mlp_proj": f"{prefix}.to_qkv_mlp_proj.weight",
        "out_proj": f"{prefix}.to_out.weight",
        "norm_q": f"{prefix}.norm_q.weight",
        "norm_k": f"{prefix}.norm_k.weight",
    }


def double_block_weight_keys(block_index: int) -> dict[str, str]:
    prefix = f"transformer_blocks.{block_index}"
    attn = f"{prefix}.attn"
    return {
        "modulation_img": "double_stream_modulation_img.linear.weight",
        "modulation_txt": "double_stream_modulation_txt.linear.weight",
        "to_q": f"{attn}.to_q.weight",
        "to_k": f"{attn}.to_k.weight",
        "to_v": f"{attn}.to_v.weight",
        "add_q_proj": f"{attn}.add_q_proj.weight",
        "add_k_proj": f"{attn}.add_k_proj.weight",
        "add_v_proj": f"{attn}.add_v_proj.weight",
        "to_out": f"{attn}.to_out.0.weight",
        "to_add_out": f"{attn}.to_add_out.weight",
        "ff_linear_in": f"{prefix}.ff.linear_in.weight",
        "ff_linear_out": f"{prefix}.ff.linear_out.weight",
        "ff_context_linear_in": f"{prefix}.ff_context.linear_in.weight",
        "ff_context_linear_out": f"{prefix}.ff_context.linear_out.weight",
        "norm_q": f"{attn}.norm_q.weight",
        "norm_k": f"{attn}.norm_k.weight",
        "norm_added_q": f"{attn}.norm_added_q.weight",
        "norm_added_k": f"{attn}.norm_added_k.weight",
    }


def _require_supported_bits(bits: int) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {_SUPPORTED_BITS}, got {bits}")


def _require_divisible(value: int, divisor: int, label: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{label}={value} must be divisible by {divisor}")


def _silu(x: mx.array) -> mx.array:
    return mlx.nn.silu(x)


def _layer_norm_affine(
    x: mx.array, weight: mx.array, bias: mx.array, eps: float
) -> mx.array:
    return mx.fast.layer_norm(x, weight, bias, eps)


def _rms_norm_with_weight(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _scale_dtype_from_name(name: str) -> mx.Dtype:
    if name not in _SUPPORTED_SCALE_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_SCALE_DTYPES))
        raise ValueError(f"Unsupported scale dtype '{name}'. Supported: {supported}")
    return _SUPPORTED_SCALE_DTYPES[name]


def _ensure_batched_hidden_states(x: mx.array) -> tuple[mx.array, bool]:
    if x.ndim == 2:
        return x[None, :, :], True
    if x.ndim != 3:
        raise ValueError(f"Expected hidden states with ndim=2 or 3, got ndim={x.ndim}")
    return x, False


def _restore_hidden_states(x: mx.array, squeezed: bool) -> mx.array:
    return x[0] if squeezed else x


def _apply_rope_bshd(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    out_dtype = x.dtype
    x_f = x.astype(mx.float32)
    cos_b = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    sin_b = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
    x2 = x_f.reshape(*x_f.shape[:-1], -1, 2)
    real = x2[..., 0]
    imag = x2[..., 1]
    out0 = real * cos_b + (-imag) * sin_b
    out1 = imag * cos_b + real * sin_b
    out = mx.stack([out0, out1], axis=-1).reshape(*x_f.shape)
    return out.astype(out_dtype)


_FUSED_SINGLE_NORM_ROPE_SOURCE = """
    uint global_tid = thread_position_in_grid.x;
    uint pos = global_tid / 32;
    uint tid = global_tid % 32;

    const int S_DIM = {S_DIM};
    const int H_DIM = {H_DIM};
    const int D_DIM = {D_DIM};
    const int D_HALF = D_DIM / 2;
    const int FUSED_DIM = {FUSED_DIM};
    const int DIM = {DIM};
    const float eps = {EPS}f;
    const int ELEMS = D_DIM / 32;

    uint b = pos / (S_DIM * H_DIM);
    uint s = (pos / H_DIM) % S_DIM;
    uint h = pos % H_DIM;

    uint fused_base = b * S_DIM * FUSED_DIM + s * FUSED_DIM;
    uint q_col = h * D_DIM;
    uint k_col = DIM + h * D_DIM;
    uint v_col = 2 * DIM + h * D_DIM;

    float qv[{ELEMS}], kv[{ELEMS}], vv[{ELEMS}];
    float sq_q = 0.0f, sq_k = 0.0f;

    for (int i = 0; i < ELEMS; i++) {{
        uint d = tid * ELEMS + i;
        qv[i] = float(fused[fused_base + q_col + d]);
        kv[i] = float(fused[fused_base + k_col + d]);
        vv[i] = float(fused[fused_base + v_col + d]);
        sq_q += qv[i] * qv[i];
        sq_k += kv[i] * kv[i];
    }}

    float total_sq_q = simd_sum(sq_q);
    float total_sq_k = simd_sum(sq_k);
    float inv_rms_q = metal::rsqrt(total_sq_q / float(D_DIM) + eps);
    float inv_rms_k = metal::rsqrt(total_sq_k / float(D_DIM) + eps);

    for (int i = 0; i < ELEMS; i++) {{
        uint d = tid * ELEMS + i;
        qv[i] = qv[i] * inv_rms_q * float(norm_q[d]);
        kv[i] = kv[i] * inv_rms_k * float(norm_k[d]);
    }}

    uint cos_base = s * D_HALF;
    for (int p = 0; p < ELEMS; p += 2) {{
        uint d = tid * ELEMS + p;
        uint d_half = d >> 1;
        float c = float(cos_vals[cos_base + d_half]);
        float sn = float(sin_vals[cos_base + d_half]);
        float q0 = qv[p], q1 = qv[p + 1];
        float k0 = kv[p], k1 = kv[p + 1];
        qv[p]     = q0 * c - q1 * sn;
        qv[p + 1] = q1 * c + q0 * sn;
        kv[p]     = k0 * c - k1 * sn;
        kv[p + 1] = k1 * c + k0 * sn;
    }}

    uint out_base = b * H_DIM * S_DIM * D_DIM + h * S_DIM * D_DIM + s * D_DIM;
    for (int i = 0; i < ELEMS; i++) {{
        uint d = tid * ELEMS + i;
        q_out[out_base + d] = T(qv[i]);
        k_out[out_base + d] = T(kv[i]);
        v_out[out_base + d] = T(vv[i]);
    }}
"""


_FUSED_DOUBLE_NORM_ROPE_SOURCE = """
    uint global_tid = thread_position_in_grid.x;
    uint pos = global_tid / 32;
    uint tid = global_tid % 32;

    const int S_IMG = {S_IMG};
    const int S_TXT = {S_TXT};
    const int S_TOTAL = S_IMG + S_TXT;
    const int H_DIM = {H_DIM};
    const int D_DIM = {D_DIM};
    const int D_HALF = D_DIM / 2;
    const int DIM = {DIM};
    const float eps = {EPS}f;
    const int ELEMS = D_DIM / 32;

    uint b = pos / (S_TOTAL * H_DIM);
    uint s = (pos / H_DIM) % S_TOTAL;
    uint h = pos % H_DIM;

    bool is_txt = (s < S_TXT);
    uint src_s = is_txt ? s : (s - S_TXT);
    uint q_col = h * D_DIM;
    uint k_col = DIM + h * D_DIM;
    uint v_col = 2 * DIM + h * D_DIM;

    float qv[{ELEMS}], kv[{ELEMS}], vv[{ELEMS}];
    float sq_q = 0.0f, sq_k = 0.0f;

    int src_stride = is_txt ? (S_TXT * 3 * DIM) : (S_IMG * 3 * DIM);
    uint base = b * src_stride + src_s * 3 * DIM;

    for (int i = 0; i < ELEMS; i++) {{
        uint d = tid * ELEMS + i;
        if (is_txt) {{
            qv[i] = float(txt_qkv[base + q_col + d]);
            kv[i] = float(txt_qkv[base + k_col + d]);
            vv[i] = float(txt_qkv[base + v_col + d]);
        }} else {{
            qv[i] = float(img_qkv[base + q_col + d]);
            kv[i] = float(img_qkv[base + k_col + d]);
            vv[i] = float(img_qkv[base + v_col + d]);
        }}
        sq_q += qv[i] * qv[i];
        sq_k += kv[i] * kv[i];
    }}

    float total_sq_q = simd_sum(sq_q);
    float total_sq_k = simd_sum(sq_k);
    float inv_rms_q = metal::rsqrt(total_sq_q / float(D_DIM) + eps);
    float inv_rms_k = metal::rsqrt(total_sq_k / float(D_DIM) + eps);

    for (int i = 0; i < ELEMS; i++) {{
        uint d = tid * ELEMS + i;
        if (is_txt) {{
            qv[i] = qv[i] * inv_rms_q * float(norm_added_q[d]);
            kv[i] = kv[i] * inv_rms_k * float(norm_added_k[d]);
        }} else {{
            qv[i] = qv[i] * inv_rms_q * float(norm_q[d]);
            kv[i] = kv[i] * inv_rms_k * float(norm_k[d]);
        }}
    }}

    uint cos_base = s * D_HALF;
    for (int p = 0; p < ELEMS; p += 2) {{
        uint d = tid * ELEMS + p;
        uint d_half = d >> 1;
        float c = float(cos_vals[cos_base + d_half]);
        float sn = float(sin_vals[cos_base + d_half]);
        float q0 = qv[p], q1 = qv[p + 1];
        float k0 = kv[p], k1 = kv[p + 1];
        qv[p]     = q0 * c - q1 * sn;
        qv[p + 1] = q1 * c + q0 * sn;
        kv[p]     = k0 * c - k1 * sn;
        kv[p + 1] = k1 * c + k0 * sn;
    }}

    uint out_base = b * H_DIM * S_TOTAL * D_DIM + h * S_TOTAL * D_DIM + s * D_DIM;
    for (int i = 0; i < ELEMS; i++) {{
        uint d = tid * ELEMS + i;
        q_out[out_base + d] = T(qv[i]);
        k_out[out_base + d] = T(kv[i]);
        v_out[out_base + d] = T(vv[i]);
    }}
"""


_fused_single_kernel_cache: dict[str, Any] = {}
_fused_double_kernel_cache: dict[str, Any] = {}


def _get_fused_double_norm_rope_kernel(
    spec: "Flux2KleinBlockSpec", img_seq: int, txt_seq: int
) -> Any:
    key = f"{img_seq}_{txt_seq}_{spec.dim}_{spec.num_heads}_{spec.head_dim}_{spec.rms_norm_eps}"
    if key not in _fused_double_kernel_cache:
        elems = spec.head_dim // 32
        if elems < 2 or (elems % 2) != 0:
            raise ValueError(
                f"head_dim={spec.head_dim} incompatible with 32-wide SIMD group; "
                "kernel requires head_dim to be a multiple of 64."
            )
        source = _FUSED_DOUBLE_NORM_ROPE_SOURCE.format(
            S_IMG=img_seq,
            S_TXT=txt_seq,
            H_DIM=spec.num_heads,
            D_DIM=spec.head_dim,
            DIM=spec.dim,
            EPS=spec.rms_norm_eps,
            ELEMS=elems,
        )
        safe_key = key.replace(".", "_").replace("-", "n").replace("+", "p")
        _fused_double_kernel_cache[key] = mx.fast.metal_kernel(
            name=f"flux2_fused_double_norm_rope_{safe_key}",
            input_names=[
                "img_qkv",
                "txt_qkv",
                "norm_q",
                "norm_k",
                "norm_added_q",
                "norm_added_k",
                "cos_vals",
                "sin_vals",
            ],
            output_names=["q_out", "k_out", "v_out"],
            source=source,
        )
    return _fused_double_kernel_cache[key]


def _get_fused_single_norm_rope_kernel(
    spec: "Flux2KleinBlockSpec", seq_len: int
) -> Any:
    fused_dim = spec.qkv_mlp_out_dim
    key = f"{seq_len}_{spec.dim}_{spec.num_heads}_{spec.head_dim}_{fused_dim}_{spec.rms_norm_eps}"
    if key not in _fused_single_kernel_cache:
        elems = spec.head_dim // 32
        if elems < 2 or (elems % 2) != 0:
            raise ValueError(
                f"head_dim={spec.head_dim} incompatible with 32-wide SIMD group; "
                "kernel requires head_dim to be a multiple of 64."
            )
        source = _FUSED_SINGLE_NORM_ROPE_SOURCE.format(
            S_DIM=seq_len,
            H_DIM=spec.num_heads,
            D_DIM=spec.head_dim,
            FUSED_DIM=fused_dim,
            DIM=spec.dim,
            EPS=spec.rms_norm_eps,
            ELEMS=elems,
        )
        safe_key = key.replace(".", "_").replace("-", "n").replace("+", "p")
        _fused_single_kernel_cache[key] = mx.fast.metal_kernel(
            name=f"flux2_fused_single_norm_rope_{safe_key}",
            input_names=["fused", "norm_q", "norm_k", "cos_vals", "sin_vals"],
            output_names=["q_out", "k_out", "v_out"],
            source=source,
        )
    return _fused_single_kernel_cache[key]


def _require_supported_group_size(group_size: int) -> None:
    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"group_size must be one of {_SUPPORTED_GROUP_SIZES}, got {group_size}"
        )


def quantize_affine_nbit(
    weight: mx.array,
    *,
    bits: int,
    group_size: int = DEFAULT_QUANT_GROUP_SIZE,
    scale_dtype: str = "bfloat16",
) -> tuple[mx.array, mx.array, mx.array]:
    _require_supported_bits(bits)
    _require_supported_group_size(group_size)
    mx_scale_dtype = _scale_dtype_from_name(scale_dtype)

    weight_np = np.asarray(weight.astype(mx.float32))
    if weight_np.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got ndim={weight_np.ndim}")

    rows, cols = weight_np.shape
    _require_divisible(cols, group_size, "in_features")

    max_q = (1 << bits) - 1
    vals_per_u32 = 32 // bits
    _require_divisible(cols, vals_per_u32, "in_features")

    grouped = weight_np.reshape(rows, cols // group_size, group_size)
    g_min = np.min(grouped, axis=2)
    g_max = np.max(grouped, axis=2)
    scales = np.maximum((g_max - g_min) / max_q, 1e-7)
    biases = g_min

    q = np.round((grouped - biases[:, :, None]) / scales[:, :, None])
    q = np.clip(q, 0, max_q).astype(np.uint32)
    q_flat = q.reshape(rows, -1)
    q_grouped = q_flat.reshape(rows, -1, vals_per_u32)
    shifts = (np.arange(vals_per_u32, dtype=np.uint32) * bits).reshape(
        1, 1, vals_per_u32
    )
    packed = np.sum(np.left_shift(q_grouped, shifts), axis=2, dtype=np.uint32)

    return (
        mx.array(packed),
        mx.array(scales).astype(mx_scale_dtype),
        mx.array(biases).astype(mx_scale_dtype),
    )


@lru_cache(maxsize=None)
def _require_native_quantized_matmul(bits: int, group_size: int) -> None:
    """Verify that the active MLX runtime supports native quantized matmul at
    the requested (bits, group_size). Raises at first call if the kernel is
    not available; result is cached so subsequent calls are free.
    """
    _require_supported_bits(bits)
    _require_supported_group_size(group_size)
    probe_weight = mx.arange(group_size, dtype=mx.float32).reshape(1, group_size) - (
        group_size // 2
    )
    packed_weight, scales, biases = quantize_affine_nbit(
        probe_weight,
        bits=bits,
        group_size=group_size,
    )
    probe_x = mx.ones((1, group_size), dtype=mx.bfloat16)
    probe_out = mx.quantized_matmul(
        probe_x,
        packed_weight,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
    )
    mx.eval(probe_out)


class DenseLinearKernel:
    def __init__(self, weight: mx.array):
        self.weight = weight.astype(mx.bfloat16)

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        if x.ndim == 1:
            x = x[None, :]
        flat_x = x.reshape((-1, x.shape[-1])).astype(mx.bfloat16)
        output = flat_x @ self.weight.transpose()
        return output.reshape((*original_shape[:-1], self.out_features))


@dataclass
class QuantizedLinearKernel:
    packed_weight: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int

    @classmethod
    def from_weight(
        cls,
        weight: mx.array,
        *,
        bits: int,
        group_size: int,
        scale_dtype: str = "bfloat16",
    ) -> "QuantizedLinearKernel":
        _require_native_quantized_matmul(bits, group_size)
        packed_weight, scales, biases = quantize_affine_nbit(
            weight,
            bits=bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
        )
        return cls(
            packed_weight=packed_weight,
            scales=scales,
            biases=biases,
            bits=bits,
            group_size=group_size,
        )

    @property
    def out_features(self) -> int:
        return int(self.scales.shape[0])

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        if x.ndim == 1:
            x = x[None, :]
        flat_x = x.reshape((-1, x.shape[-1])).astype(mx.bfloat16)
        output = mx.quantized_matmul(
            flat_x,
            self.packed_weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
            bits=self.bits,
        )
        return output.reshape((*original_shape[:-1], self.out_features))


LinearKernel = DenseLinearKernel | QuantizedLinearKernel


def _fuse_quantized_linears(
    linears: list[QuantizedLinearKernel],
) -> QuantizedLinearKernel:
    return QuantizedLinearKernel(
        packed_weight=mx.concatenate(
            [linear.packed_weight for linear in linears], axis=0
        ),
        scales=mx.concatenate([linear.scales for linear in linears], axis=0),
        biases=mx.concatenate([linear.biases for linear in linears], axis=0),
        bits=linears[0].bits,
        group_size=linears[0].group_size,
    )


def _fuse_dense_linears(linears: list[DenseLinearKernel]) -> DenseLinearKernel:
    return DenseLinearKernel(
        mx.concatenate([linear.weight for linear in linears], axis=0)
    )


def _fuse_linears(linears: list[LinearKernel]) -> LinearKernel:
    if all(isinstance(linear, QuantizedLinearKernel) for linear in linears):
        return _fuse_quantized_linears(linears)
    if all(isinstance(linear, DenseLinearKernel) for linear in linears):
        return _fuse_dense_linears(linears)
    raise TypeError("Cannot fuse mixed linear kernel types")


def _make_linear(
    weight: WeightOrPacked, precision: PrecisionName, group_size: int
) -> LinearKernel:
    if isinstance(weight, PackedWeight):
        _require_native_quantized_matmul(weight.bits, weight.group_size)
        return QuantizedLinearKernel(
            packed_weight=weight.packed,
            scales=weight.scales,
            biases=weight.biases,
            bits=weight.bits,
            group_size=weight.group_size,
        )
    if precision == "bf16":
        return DenseLinearKernel(weight)
    bits = 1 if precision == "1bit" else 2
    return QuantizedLinearKernel.from_weight(weight, bits=bits, group_size=group_size)


def _swiglu_projected(x: mx.array) -> mx.array:
    gate, value = mx.split(x, 2, axis=-1)
    return _silu(gate) * value


class SingleFlux2Block:
    def __init__(
        self,
        *,
        spec: Flux2KleinBlockSpec,
        weights: SingleBlockWeights,
        precision: PrecisionName,
        group_size: int,
    ):
        self.spec = spec
        # Modulation linear stays Dense regardless of `precision`.
        self.modulation = DenseLinearKernel(weights.modulation)
        self.qkv_mlp_proj = _make_linear(weights.qkv_mlp_proj, precision, group_size)
        self.out_proj = _make_linear(weights.out_proj, precision, group_size)
        self.norm_q = weights.norm_q.astype(mx.bfloat16)
        self.norm_k = weights.norm_k.astype(mx.bfloat16)

    def prepare_modulation(self, temb: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if temb.ndim == 1:
            temb = temb[None, :]
        mod = self.modulation(_silu(temb)).reshape((temb.shape[0], 1, 3, self.spec.dim))
        shift = mod[:, :, 0, :]
        scale = mod[:, :, 1, :]
        gate = mod[:, :, 2, :]
        norm_w = (1.0 + scale).reshape(-1)
        norm_b = shift.reshape(-1)
        return norm_w, norm_b, gate.reshape(-1)

    def forward_from_modulation(
        self,
        hidden_states: mx.array,
        modulation: tuple[mx.array, mx.array, mx.array],
        rotary_cos: mx.array,
        rotary_sin: mx.array,
    ) -> mx.array:
        norm_w, norm_b, gate = modulation
        norm_hidden_states = _layer_norm_affine(
            hidden_states, norm_w, norm_b, self.spec.layer_norm_eps
        )

        fused = self.qkv_mlp_proj(norm_hidden_states)
        mlp_hidden = fused[:, :, 3 * self.spec.dim :]

        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        H = self.spec.num_heads
        D = self.spec.head_dim

        kernel = _get_fused_single_norm_rope_kernel(self.spec, seq_len)
        query, key, value = kernel(
            inputs=[fused, self.norm_q, self.norm_k, rotary_cos, rotary_sin],
            template=[("T", mx.bfloat16)],
            grid=(batch_size * seq_len * H * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (batch_size, H, seq_len, D),
                (batch_size, H, seq_len, D),
                (batch_size, H, seq_len, D),
            ],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
        )

        attn_output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(D),
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            (batch_size, seq_len, self.spec.dim)
        )
        mlp_output = _swiglu_projected(mlp_hidden)
        fused_out = mx.concatenate([attn_output, mlp_output], axis=-1)
        block_output = self.out_proj(fused_out)
        return hidden_states + gate * block_output

    def forward(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        rotary_cos: mx.array,
        rotary_sin: mx.array,
    ) -> mx.array:
        hidden_states, squeezed = _ensure_batched_hidden_states(hidden_states)
        output = self.forward_from_modulation(
            hidden_states,
            self.prepare_modulation(temb),
            rotary_cos,
            rotary_sin,
        )
        return _restore_hidden_states(output, squeezed)


class DoubleFlux2Block:
    def __init__(
        self,
        *,
        spec: Flux2KleinBlockSpec,
        weights: DoubleBlockWeights,
        precision: PrecisionName,
        group_size: int,
    ):
        self.spec = spec
        # Modulation linears stay Dense regardless of `precision`.
        self.modulation_img = DenseLinearKernel(weights.modulation_img)
        self.modulation_txt = DenseLinearKernel(weights.modulation_txt)
        img_qkv_parts = [
            _make_linear(weights.to_q, precision, group_size),
            _make_linear(weights.to_k, precision, group_size),
            _make_linear(weights.to_v, precision, group_size),
        ]
        txt_qkv_parts = [
            _make_linear(weights.add_q_proj, precision, group_size),
            _make_linear(weights.add_k_proj, precision, group_size),
            _make_linear(weights.add_v_proj, precision, group_size),
        ]
        self.img_qkv = _fuse_linears(img_qkv_parts)
        self.txt_qkv = _fuse_linears(txt_qkv_parts)
        self.to_out = _make_linear(weights.to_out, precision, group_size)
        self.to_add_out = _make_linear(weights.to_add_out, precision, group_size)
        self.ff_linear_in = _make_linear(weights.ff_linear_in, precision, group_size)
        self.ff_linear_out = _make_linear(weights.ff_linear_out, precision, group_size)
        self.ff_context_linear_in = _make_linear(
            weights.ff_context_linear_in, precision, group_size
        )
        self.ff_context_linear_out = _make_linear(
            weights.ff_context_linear_out, precision, group_size
        )
        self.norm_q = weights.norm_q.astype(mx.bfloat16)
        self.norm_k = weights.norm_k.astype(mx.bfloat16)
        self.norm_added_q = weights.norm_added_q.astype(mx.bfloat16)
        self.norm_added_k = weights.norm_added_k.astype(mx.bfloat16)

    def prepare_modulation(
        self,
        linear: LinearKernel,
        temb: mx.array,
    ) -> tuple[tuple[mx.array, mx.array, mx.array], ...]:
        if temb.ndim == 1:
            temb = temb[None, :]
        mod = linear(_silu(temb)).reshape((temb.shape[0], 1, 2, 3, self.spec.dim))
        shift_msa = mod[:, :, 0, 0, :]
        scale_msa = mod[:, :, 0, 1, :]
        gate_msa = mod[:, :, 0, 2, :]
        shift_mlp = mod[:, :, 1, 0, :]
        scale_mlp = mod[:, :, 1, 1, :]
        gate_mlp = mod[:, :, 1, 2, :]
        return (
            (
                (1.0 + scale_msa).reshape(-1),
                shift_msa.reshape(-1),
                gate_msa.reshape(-1),
            ),
            (
                (1.0 + scale_mlp).reshape(-1),
                shift_mlp.reshape(-1),
                gate_mlp.reshape(-1),
            ),
        )

    def _feedforward(
        self, x: mx.array, linear_in: LinearKernel, linear_out: LinearKernel
    ) -> mx.array:
        return linear_out(_swiglu_projected(linear_in(x)))

    def forward_from_modulation(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        img_modulation: tuple[tuple[mx.array, mx.array, mx.array], ...],
        txt_modulation: tuple[tuple[mx.array, mx.array, mx.array], ...],
        rotary_cos: mx.array,
        rotary_sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        (w_msa, b_msa, gate_msa), (w_mlp, b_mlp, gate_mlp) = img_modulation
        (c_w_msa, c_b_msa, c_gate_msa), (c_w_mlp, c_b_mlp, c_gate_mlp) = txt_modulation

        norm_hidden_states = _layer_norm_affine(
            hidden_states, w_msa, b_msa, self.spec.layer_norm_eps
        )
        norm_encoder_hidden_states = _layer_norm_affine(
            encoder_hidden_states, c_w_msa, c_b_msa, self.spec.layer_norm_eps
        )

        batch_size = int(hidden_states.shape[0])
        img_seq = int(hidden_states.shape[1])
        txt_seq = int(encoder_hidden_states.shape[1])
        H = self.spec.num_heads
        D = self.spec.head_dim

        img_qkv = self.img_qkv(norm_hidden_states)
        txt_qkv = self.txt_qkv(norm_encoder_hidden_states)

        kernel = _get_fused_double_norm_rope_kernel(self.spec, img_seq, txt_seq)
        full_query, full_key, full_value = kernel(
            inputs=[
                img_qkv,
                txt_qkv,
                self.norm_q,
                self.norm_k,
                self.norm_added_q,
                self.norm_added_k,
                rotary_cos,
                rotary_sin,
            ],
            template=[("T", mx.bfloat16)],
            grid=(batch_size * (img_seq + txt_seq) * H * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (batch_size, H, txt_seq + img_seq, D),
                (batch_size, H, txt_seq + img_seq, D),
                (batch_size, H, txt_seq + img_seq, D),
            ],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
        )

        attn = mx.fast.scaled_dot_product_attention(
            full_query,
            full_key,
            full_value,
            scale=1.0 / math.sqrt(D),
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(
            (batch_size, txt_seq + img_seq, self.spec.dim)
        )
        context_attn_output = self.to_add_out(attn[:, :txt_seq])
        attn_output = self.to_out(attn[:, txt_seq:])

        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = _layer_norm_affine(
            hidden_states, w_mlp, b_mlp, self.spec.layer_norm_eps
        )
        hidden_states = hidden_states + gate_mlp * self._feedforward(
            norm_hidden_states,
            self.ff_linear_in,
            self.ff_linear_out,
        )

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = _layer_norm_affine(
            encoder_hidden_states, c_w_mlp, c_b_mlp, self.spec.layer_norm_eps
        )
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self._feedforward(
            norm_encoder_hidden_states,
            self.ff_context_linear_in,
            self.ff_context_linear_out,
        )

        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        rotary_cos: mx.array,
        rotary_sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        hidden_states, squeezed_hidden = _ensure_batched_hidden_states(hidden_states)
        encoder_hidden_states, squeezed_encoder = _ensure_batched_hidden_states(
            encoder_hidden_states
        )
        if squeezed_hidden != squeezed_encoder:
            raise ValueError(
                "hidden_states and encoder_hidden_states must have matching batch structure"
            )
        enc_out, hid_out = self.forward_from_modulation(
            hidden_states,
            encoder_hidden_states,
            self.prepare_modulation(self.modulation_img, temb),
            self.prepare_modulation(self.modulation_txt, temb),
            rotary_cos,
            rotary_sin,
        )
        return (
            _restore_hidden_states(enc_out, squeezed_hidden),
            _restore_hidden_states(hid_out, squeezed_hidden),
        )
