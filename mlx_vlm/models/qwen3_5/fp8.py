"""Qwen fine-grained FP8 checkpoint conversion helpers.

Qwen's Transformers FP8 checkpoints store E4M3 weight bytes alongside an
arbitrary BF16 inverse scale for each 128x128 block.  MLX MXFP8 instead uses
an E8M0 scale for every 32-value group and every output row, so the source
weights cannot be reinterpreted directly (unlike DeepSeek V4's UE8M0
checkpoints).  Reconstruct each source block lazily and immediately requantize
it to the native MLX layout.
"""

import mlx.core as mx

FP8_BLOCK_SIZE = 128
MLX_MXFP8_QUANTIZATION = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def make_quantization_config(config: dict) -> dict | None:
    """Return the native MLX config for a supported Qwen FP8 checkpoint."""
    quantization = config.get("quantization_config") or {}
    is_supported = (
        config.get("model_type") in {"qwen3_5", "qwen3_5_moe"}
        and isinstance(quantization, dict)
        and quantization.get("quant_method") == "fp8"
        and quantization.get("fmt", "e4m3") == "e4m3"
        and quantization.get("weight_block_size") == [128, 128]
    )
    return dict(MLX_MXFP8_QUANTIZATION) if is_supported else None


def _dequantize_qwen_fp8_weight(
    weight: mx.array,
    scale_inv: mx.array,
    *,
    block_size: int = FP8_BLOCK_SIZE,
) -> mx.array:
    if weight.dtype != mx.uint8 or weight.ndim < 2:
        raise ValueError(
            "Qwen fine-grained FP8 weights must be E4M3 byte matrices loaded as "
            f"uint8; got dtype={weight.dtype}, shape={weight.shape}."
        )
    if scale_inv.ndim != weight.ndim or not mx.issubdtype(scale_inv.dtype, mx.floating):
        raise ValueError(
            "Qwen fine-grained FP8 scales must be a floating-point block "
            f"grid; got dtype={scale_inv.dtype}, shape={scale_inv.shape}."
        )

    *batch_shape, rows, cols = weight.shape
    expected_scale_shape = (
        *batch_shape,
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if scale_inv.shape != expected_scale_shape:
        raise ValueError(
            "Qwen fine-grained FP8 scale shape does not match its weight: "
            f"weight={weight.shape}, scales={scale_inv.shape}, "
            f"expected={expected_scale_shape}."
        )

    pad_rows = (-rows) % block_size
    pad_cols = (-cols) % block_size
    decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
    if pad_rows or pad_cols:
        decoded = mx.pad(
            decoded,
            [(0, 0)] * len(batch_shape) + [(0, pad_rows), (0, pad_cols)],
        )

    decoded = decoded.reshape(
        *batch_shape,
        (rows + pad_rows) // block_size,
        block_size,
        (cols + pad_cols) // block_size,
        block_size,
    )
    decoded = (decoded * scale_inv[..., :, None, :, None]).reshape(
        *batch_shape, rows + pad_rows, cols + pad_cols
    )
    return decoded[..., :rows, :cols]


def quantize_qwen_fp8_weight(
    weight: mx.array, scale_inv: mx.array
) -> tuple[mx.array, mx.array]:
    """Convert one Qwen block-FP8 tensor to native MLX MXFP8 lazily."""
    if weight.shape[-1] % MLX_MXFP8_QUANTIZATION["group_size"] != 0:
        raise ValueError(
            "Qwen FP8 weight input dimension must be divisible by the MLX "
            f"MXFP8 group size; got shape={weight.shape}."
        )

    restored = _dequantize_qwen_fp8_weight(weight, scale_inv)
    quantized = mx.quantize(restored, **MLX_MXFP8_QUANTIZATION)
    if len(quantized) != 2:
        raise ValueError("MLX MXFP8 quantization unexpectedly produced biases.")
    return quantized


def convert_qwen_fp8_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Replace Qwen FP8 inverse-scale pairs with MLX weight/scales pairs.

    Regular linears use ``.weight`` / ``.weight_scale_inv`` while fused MoE
    expert tensors use bare ``gate_up_proj`` / ``gate_up_proj_scale_inv``
    names. Preserve the bare expert name until the MoE sanitizer splits it.
    """
    scale_keys = [key for key in weights if key.endswith("_scale_inv")]
    if not scale_keys:
        return weights

    converted = dict(weights)
    for scale_key in scale_keys:
        weight_key = scale_key[: -len("_scale_inv")]
        if weight_key not in converted:
            raise ValueError(f"Missing FP8 weight for scale tensor {scale_key!r}.")

        weight = converted.pop(weight_key)
        scale_inv = converted.pop(scale_key)
        packed, scales = quantize_qwen_fp8_weight(weight, scale_inv)
        converted[weight_key] = packed
        if weight_key.endswith(".weight"):
            scales_key = weight_key[: -len(".weight")] + ".scales"
        else:
            scales_key = weight_key + "_scales"
        converted[scales_key] = scales

    return converted
