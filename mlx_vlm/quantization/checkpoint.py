"""Convert source checkpoint quantization formats to MLX weight layouts."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx


def _e4m3_decode_table() -> mx.array:
    """Return a 256-entry ``float32`` LUT mapping every E4M3FN byte to its value.

    OCP ``E4M3FN``: 1 sign / 4 exponent (bias 7) / 3 mantissa bits,
    no infinities, and ``(exp=15, mant=7)`` reserved for NaN (max finite
    magnitude is therefore 448). This matches the byte convention MLX uses for
    ``nvfp4`` scales (verified empirically).
    """
    table = []
    for byte in range(256):
        sign = (byte >> 7) & 1
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0:
            value = (mantissa / 8.0) * 2.0**-6  # subnormal
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table.append(-value if sign else value)
    return mx.array(table, dtype=mx.float32)


_E4M3_DECODE_LUT = _e4m3_decode_table()


def _transform_modelopt_nvfp4_weights(
    weights: Dict[str, mx.array],
    quantization_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, mx.array], Optional[Dict[str, Any]]]:
    """Convert ModelOpt NVFP4 and mixed NVFP4/FP8 checkpoints.

    ModelOpt's mixed export uses ``weight_scale_2`` to identify NVFP4
    linears and a lone ``weight_scale`` for FP8 linears. MLX can load the
    former natively; the latter are decoded to dense weights because its FP8
    mode uses block scales rather than ModelOpt's per-tensor/channel scale.
    """
    if quantization_config is None:
        return weights, None
    quant_method = quantization_config.get("quant_method")
    is_modelopt = quant_method in {"modelopt", "modelopt_mixed"}
    is_fp8_with_nvfp4_experts = (
        quant_method == "fp8" and quantization_config.get("moe_quant_algo") == "NVFP4"
    )
    if not (is_modelopt or is_fp8_with_nvfp4_experts) or quantization_config.get(
        "quant_algo"
    ) not in {
        "NVFP4",
        "NVFP4_AWQ",
        "W4A16_NVFP4",
        "MIXED_PRECISION",
    }:
        return weights, None

    scale_2_suffix = ".weight_scale_2"
    nvfp4_prefixes = {
        key[: -len(scale_2_suffix)] for key in weights if key.endswith(scale_2_suffix)
    }
    scale_suffix = ".weight_scale"
    scaled_prefixes = {
        key[: -len(scale_suffix)] for key in weights if key.endswith(scale_suffix)
    }
    fp8_prefixes = scaled_prefixes - nvfp4_prefixes
    if not nvfp4_prefixes and not fp8_prefixes:
        return weights, None

    nvfp4_consumed = {
        f"{prefix}.{suffix}"
        for prefix in nvfp4_prefixes
        for suffix in ("weight", "weight_scale", "input_scale")
    }
    fp8_consumed = {
        f"{prefix}.{suffix}"
        for prefix in fp8_prefixes
        for suffix in ("weight", "input_scale")
    }
    transformed = {}
    # Materialize FP8 conversions in batches to bound the lazy graph size.
    pending: List[mx.array] = []

    def _flush(force: bool = False) -> None:
        if pending and (force or len(pending) >= 256):
            mx.eval(pending)
            pending.clear()

    for key, value in weights.items():
        # ModelOpt emits per-layer FP8 KV-cache scales when kv_cache_quant_algo
        # is set. MLX quantizes the KV cache at runtime and has no parameter to
        # hold them, so drop them rather than fail the strict load.
        if key.endswith(".k_scale") or key.endswith(".v_scale"):
            continue
        if key.endswith(scale_2_suffix):
            prefix = key[: -len(scale_2_suffix)]
            weight_key = f"{prefix}.weight"
            scale_key = f"{prefix}.weight_scale"
            if weight_key not in weights or scale_key not in weights:
                raise ValueError(f"Missing ModelOpt NVFP4 tensors for {prefix}.")

            weight = weights[weight_key]
            scale = weights[scale_key]
            if (
                weight.dtype != mx.uint8
                or scale.dtype != mx.uint8
                or weight.ndim not in (2, 3)
                or scale.ndim != weight.ndim
                or value.size not in (1, weight.shape[0] if weight.ndim == 3 else 1)
            ):
                raise ValueError(f"Invalid ModelOpt NVFP4 tensors for {prefix}.")
            if (
                weight.shape[:-1] != scale.shape[:-1]
                or weight.shape[-1] != 8 * scale.shape[-1]
            ):
                raise ValueError(f"Invalid ModelOpt NVFP4 scale shape for {prefix}.")

            if quantization_config.get("quant_algo") == "NVFP4_AWQ":
                # Some ModelOpt AWQ exports contain E4M3 NaN block scales
                # (0x7F/0xFF). Match their reference loader's recovery: zero
                # those blocks instead of poisoning every downstream layer.
                nan_scales = (scale & 0x7F) == 0x7F
                nan_count = mx.sum(nan_scales).item()
                if nan_count:
                    logging.warning(
                        "Zeroing %d NaN NVFP4 AWQ block scales for %s",
                        nan_count,
                        prefix,
                    )
                    scale = mx.where(nan_scales, mx.array(0, mx.uint8), scale)

            transformed[weight_key] = weight.view(mx.uint32)
            transformed[f"{prefix}.scales"] = scale
            transformed[key] = value.astype(mx.float32).squeeze()
        elif key.endswith(scale_suffix) and key[: -len(scale_suffix)] in fp8_prefixes:
            prefix = key[: -len(scale_suffix)]
            weight_key = f"{prefix}.weight"
            if weight_key not in weights or weights[weight_key].dtype != mx.uint8:
                raise ValueError(f"Invalid ModelOpt FP8 tensors for {prefix}.")
            if not mx.issubdtype(value.dtype, mx.floating):
                raise ValueError(f"Invalid ModelOpt FP8 scale for {prefix}.")
            transformed[weight_key] = _dequantize_compressed_tensors_fp8_weight(
                weights[weight_key], value
            )
            pending.append(transformed[weight_key])
            _flush()
        elif key in nvfp4_consumed or key in fp8_consumed:
            continue
        else:
            transformed[key] = value

    _flush(force=True)

    quantization = None
    if nvfp4_prefixes:
        quantization = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    return transformed, quantization


def _transform_compressed_tensors_nvfp4_weights(
    weights: Dict[str, mx.array],
    quantization_config: Dict[str, Any],
) -> Dict[str, mx.array]:
    """Preserve packed FP4 weights, FP8 block scales and the global factor.

    Compressed-tensors reconstructs weights as ``fp4 * scale / global_scale``.
    Keep ``weight_global_scale`` in FP32 instead of rounding the effective
    block scales back to FP8.
    """
    packed_suffix = ".weight_packed"

    new_weights = {}
    for key in list(weights.keys()):
        if key.endswith(packed_suffix):
            prefix = key[: -len(packed_suffix)]
            packed = weights[key]
            scale = weights[f"{prefix}.weight_scale"]
            global_scale = weights[f"{prefix}.weight_global_scale"].astype(mx.float32)

            new_weights[f"{prefix}.weight"] = packed.view(mx.uint32)
            new_weights[f"{prefix}.scales"] = scale
            new_weights[f"{prefix}.weight_global_scale"] = global_scale.squeeze()
        elif key.endswith((".weight_scale",) + _COMPRESSED_TENSORS_DROP_SUFFIXES):
            continue
        else:
            new_weights[key] = weights[key]

    return new_weights


def _transform_compressed_tensors_int4_weights(
    weights: Dict[str, mx.array],
    quantization_config: Dict[str, Any],
) -> Tuple[Dict[str, mx.array], Dict[str, Any]]:
    """Remap compressed-tensors INT4 ``pack-quantized`` weights to MLX affine.

    A symmetric int4 ``pack-quantized`` checkpoint stores, per quantized Linear:

    - ``<p>.weight_packed`` ``int32`` ``[out, in // 8]``
      (8x int4 per word, LSB-first)
    - ``<p>.weight_scale``  (``bf16``/``float``)
      ``[out, in // group_size]``
    - ``<p>.weight_shape``  ``int64`` ``[2]`` (unused by MLX)

    MLX affine ``QuantizedLinear`` uses the same int4 packing and dequantizes as
    ``w * scale + bias``. Symmetric int4 stores values in ``[0, 15]``
    representing ``[-8, 7]``, i.e. ``value = packed - 8``, so we set
    ``bias = -8 * scale``. The packed ``int32`` is bit-identical to MLX's
    ``uint32`` layout (reinterpreted via ``view``).
    """
    weights_cfg = (
        quantization_config.get("config_groups", {})
        .get("group_0", {})
        .get("weights", {})
    )
    group_size = weights_cfg.get("group_size", 32)
    bits = weights_cfg.get("num_bits", 4)
    packed_suffix = ".weight_packed"

    new_weights = {}
    for key in list(weights.keys()):
        if key.endswith(packed_suffix):
            prefix = key[: -len(packed_suffix)]
            scale = weights[f"{prefix}.weight_scale"]
            new_weights[f"{prefix}.weight"] = weights[key].view(mx.uint32)
            new_weights[f"{prefix}.scales"] = scale
            new_weights[f"{prefix}.biases"] = -(2 ** (bits - 1)) * scale
        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            # Consumed alongside their ``.weight_packed`` (shape is unused).
            continue
        else:
            new_weights[key] = weights[key]

    return new_weights, {"group_size": group_size, "bits": bits, "mode": "affine"}


# Compressed-tensors auxiliary tensors with no MLX consumer. They carry
# activation / kv-cache / shape metadata; left in place they reach
# ``model.load_weights(strict=True)`` as unexpected keys and abort startup.
_COMPRESSED_TENSORS_DROP_SUFFIXES = (
    ".weight_shape",
    ".weight_global_scale",  # emitted with its packed NVFP4 weight
    ".weight_zero_point",
    ".input_global_scale",
    ".input_scale",
    ".k_scale",
    ".v_scale",
)


def _compressed_tensors_group_weights(
    quantization_config: Dict[str, Any], ct_format: str
) -> Dict[str, Any]:
    """Return the ``weights`` sub-config of the first group with ``ct_format``."""
    for group in (quantization_config.get("config_groups") or {}).values():
        if isinstance(group, dict) and group.get("format") == ct_format:
            return group.get("weights") or {}
    return {}


def _dequantize_compressed_tensors_fp8_weight(
    weight: mx.array,
    scale: mx.array,
    block_structure: Optional[List[int]] = None,
) -> mx.array:
    """Dequantize an fp8 ``float-quantized`` weight to dense.

    ``float-quantized`` stores the weight as ``float8_e4m3fn`` -- which
    ``mx.load`` surfaces as raw ``uint8`` E4M3 bytes -- plus either a
    per-output-channel or blockwise ``weight_scale``. MLX has no matching FP8
    mode, so decode the E4M3 codes and rescale into a dense tensor. The dense
    weight is emitted in ``weight_scale``'s dtype (typically ``bfloat16``).
    """
    if weight.dtype != mx.uint8 or weight.ndim < 2:
        raise ValueError(
            "Compressed-tensors FP8 weights must be E4M3 byte matrices; "
            f"got dtype={weight.dtype}, shape={weight.shape}."
        )

    out_dtype = scale.dtype if scale.dtype != mx.float32 else mx.bfloat16
    decoded = _E4M3_DECODE_LUT[weight.astype(mx.uint32)]  # float32 [out, in]
    scale = scale.astype(mx.float32)
    if block_structure is None:
        if scale.ndim == 1:
            scale = scale[:, None]
        return (decoded * scale).astype(out_dtype)

    if len(block_structure) != 2 or any(size <= 0 for size in block_structure):
        raise ValueError(
            "Compressed-tensors FP8 block_structure must contain two positive "
            f"dimensions; got {block_structure}."
        )
    block_rows, block_cols = block_structure
    *batch_shape, rows, cols = weight.shape
    row_blocks = (rows + block_rows - 1) // block_rows
    col_blocks = (cols + block_cols - 1) // block_cols
    expected_scale_shape = (*batch_shape, row_blocks, col_blocks)
    if scale.shape != expected_scale_shape:
        raise ValueError(
            "Compressed-tensors FP8 scale shape does not match its weight: "
            f"weight={weight.shape}, scales={scale.shape}, "
            f"block_structure={block_structure}, expected={expected_scale_shape}."
        )

    pad_rows = row_blocks * block_rows - rows
    pad_cols = col_blocks * block_cols - cols
    if pad_rows or pad_cols:
        decoded = mx.pad(
            decoded,
            [(0, 0)] * len(batch_shape) + [(0, pad_rows), (0, pad_cols)],
        )
    decoded = decoded.reshape(
        *batch_shape,
        row_blocks,
        block_rows,
        col_blocks,
        block_cols,
    )
    decoded = (decoded * scale[..., :, None, :, None]).reshape(
        *batch_shape,
        rows + pad_rows,
        cols + pad_cols,
    )
    return decoded[..., :rows, :cols].astype(out_dtype)


def _transform_compressed_tensors_mixed_weights(
    weights: Dict[str, mx.array],
    quantization_config: Dict[str, Any],
) -> Tuple[Dict[str, mx.array], Optional[Dict[str, Any]]]:
    """Route compressed-tensors mixed-precision or pure FP8 weights.

    A ``mixed-precision`` export keeps a single top-level ``format`` and puts
    the real formats in per-group ``config_groups``. A pure ``float-quantized``
    export uses the same FP8 tensor layout without any packed weights. Rather
    than match each group's regex ``targets``/``ignore`` against module paths
    (fragile -- the HF names differ from MLX's), route every quantized Linear
    by the tensors it actually carries, which is exactly what the group
    assignment produced:

    - ``.weight_packed`` + ``.weight_global_scale`` -> NVFP4
      (keeps block scales and the global divisor separate)
    - ``.weight_packed`` alone -> INT4 ``pack-quantized`` (folded to ``affine``)
    - ``.weight_scale`` without ``.weight_packed`` -> channel-wise fp8
      ``float-quantized`` (dequantized to a dense weight)

    Layers under ``ignore`` (vision, ``linear_attn``, ``mtp``) carry a plain
    ``.weight`` with no scale and pass through untouched, so the later
    ``nn.quantize`` pass -- which only quantizes modules that have a ``.scales``
    tensor -- leaves them dense.

    Only one MLX-native quantized mode may coexist with the dense/fp8 layers
    (``nn.quantize`` applies a single ``group_size``/``bits``/``mode``). The
    reported models pair NVFP4 with dense fp8, so this holds; a checkpoint that
    mixes e.g. NVFP4 and INT4 raises rather than silently mis-quantizing.
    """
    packed_prefixes = {
        key[: -len(".weight_packed")]
        for key in weights
        if key.endswith(".weight_packed")
    }
    # A ``.weight_scale`` with no packed sibling is a float-quantized fp8 layer.
    fp8_prefixes = {
        key[: -len(".weight_scale")] for key in weights if key.endswith(".weight_scale")
    } - packed_prefixes

    int4_cfg = _compressed_tensors_group_weights(quantization_config, "pack-quantized")
    int4_bits = int4_cfg.get("num_bits", 4)
    int4_group_size = int4_cfg.get("group_size", 32)
    fp8_cfg = _compressed_tensors_group_weights(quantization_config, "float-quantized")
    fp8_block_structure = fp8_cfg.get("block_structure")

    new_weights: Dict[str, mx.array] = {}
    native_quant: Dict[str, Dict[str, Any]] = {}
    pending: List[mx.array] = []

    def flush(force: bool = False) -> None:
        if pending and (force or len(pending) >= 256):
            mx.eval(pending)
            pending.clear()

    for key in list(weights):
        if key not in weights:
            continue
        value = weights[key]
        if key.endswith(".weight_packed"):
            prefix = key[: -len(".weight_packed")]
            scale = weights[f"{prefix}.weight_scale"]
            global_key = f"{prefix}.weight_global_scale"
            if global_key in weights:  # NVFP4
                global_scale = weights[global_key].astype(mx.float32)
                new_weights[f"{prefix}.weight"] = value.view(mx.uint32)
                new_weights[f"{prefix}.scales"] = scale
                new_weights[global_key] = global_scale.squeeze()
                native_quant["nvfp4"] = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
            else:  # INT4 symmetric pack-quantized
                new_weights[f"{prefix}.weight"] = value.view(mx.uint32)
                new_weights[f"{prefix}.scales"] = scale
                new_weights[f"{prefix}.biases"] = -(2 ** (int4_bits - 1)) * scale
                native_quant["affine"] = {
                    "group_size": int4_group_size,
                    "bits": int4_bits,
                    "mode": "affine",
                }
            continue
        if key.endswith(".weight_scale"):
            prefix = key[: -len(".weight_scale")]
            if prefix in fp8_prefixes:
                weight_key = f"{prefix}.weight"
                source_weight = weights.pop(weight_key)
                source_scale = weights.pop(key)
                new_weights[f"{prefix}.weight"] = (
                    _dequantize_compressed_tensors_fp8_weight(
                        source_weight,
                        source_scale,
                        fp8_block_structure,
                    )
                )
                pending.append(new_weights[f"{prefix}.weight"])
                flush()
            # NVFP4 / INT4 scales are consumed with their ``.weight_packed``.
            continue
        if key.endswith(".weight") and key[: -len(".weight")] in fp8_prefixes:
            continue  # emitted (dequantized to dense) at its ``.weight_scale``
        if any(key.endswith(suffix) for suffix in _COMPRESSED_TENSORS_DROP_SUFFIXES):
            continue
        new_weights[key] = value

    flush(force=True)

    if not native_quant:
        return new_weights, None
    if len(native_quant) > 1:
        raise NotImplementedError(
            "mixed-precision compressed-tensors checkpoint mixes multiple "
            f"MLX-native quantized modes {sorted(native_quant)}; only one native "
            "mode alongside dense/fp8 layers is supported."
        )
    return new_weights, next(iter(native_quant.values()))


def _transform_compressed_tensors_weights(
    weights: Dict[str, mx.array],
    quantization_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, mx.array], Optional[Dict[str, Any]]]:
    """Transform compressed-tensors weights before model-specific sanitization.

    This runs before MoE expert stacking so per-expert tensors have already been
    renamed to MLX-native ``.weight`` / ``.scales`` / ``.biases`` keys.
    """
    if quantization_config is None:
        return weights, None
    if quantization_config.get("quant_method") != "compressed-tensors":
        return weights, None

    # Mixed-precision and pure float-quantized exports both use
    # ``.weight``/``.weight_scale`` for FP8 tensors. Route these before the
    # packed-weight guard because a pure FP8 checkpoint has no
    # ``.weight_packed`` tensors at all.
    if quantization_config.get("format") in {
        "mixed-precision",
        "float-quantized",
    }:
        return _transform_compressed_tensors_mixed_weights(weights, quantization_config)

    if not any(key.endswith(".weight_packed") for key in weights):
        return weights, None

    weights_cfg = (
        quantization_config.get("config_groups", {})
        .get("group_0", {})
        .get("weights", {})
    )
    ct_format = quantization_config.get("format") or quantization_config.get(
        "config_groups", {}
    ).get("group_0", {}).get("format")
    quant_type = weights_cfg.get("type")

    if ct_format == "nvfp4-pack-quantized":
        return _transform_compressed_tensors_nvfp4_weights(
            weights, quantization_config
        ), {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    if ct_format == "pack-quantized" and quant_type == "int":
        return _transform_compressed_tensors_int4_weights(weights, quantization_config)

    return weights, None


def transform_quantized_weights(
    weights: Dict[str, mx.array],
    quantization_config: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, mx.array], Optional[Dict[str, Any]]]:
    """Convert source weights before model-specific sanitization."""
    weights, quantization = _transform_modelopt_nvfp4_weights(
        weights, quantization_config
    )
    if quantization is None:
        return _transform_compressed_tensors_weights(weights, quantization_config)
    return weights, quantization


def merge_quantization_configs(
    weights: Dict[str, mx.array],
    quantization: Dict[str, Any],
    expert_quantization: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay NVFP4 expert settings using sanitized parameter paths."""
    merged = dict(quantization)
    for key in weights:
        if key.endswith(".weight_scale_2"):
            merged[key.removesuffix(".weight_scale_2")] = dict(expert_quantization)
    return merged
