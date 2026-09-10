"""Fine-grained FP8 checkpoint conversion helpers.

Some Transformers FP8 checkpoints store E4M3 weight bytes alongside an
arbitrary floating-point inverse scale for each 128x128 block. MLX MXFP8 uses
an E8M0 scale for every 32-value group and every output row, so the source
weights cannot be reinterpreted directly. Reconstruct each source block lazily
and immediately requantize it to the native MLX layout.
"""

import mlx.core as mx

FINE_GRAINED_FP8_BLOCK_SIZE = 128
MLX_MXFP8_QUANTIZATION = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def make_quantization_config(config: dict) -> dict | None:
    """Return the native MLX config for a supported block-FP8 checkpoint."""
    quantization = config.get("quantization_config") or {}
    is_supported = (
        isinstance(quantization, dict)
        and quantization.get("quant_method") == "fp8"
        and quantization.get("fmt", "e4m3") == "e4m3"
        and quantization.get("weight_block_size") == [128, 128]
    )
    return dict(MLX_MXFP8_QUANTIZATION) if is_supported else None


def _dequantize_fp8_weight(
    weight: mx.array,
    scale_inv: mx.array,
    *,
    block_size: int = FINE_GRAINED_FP8_BLOCK_SIZE,
) -> mx.array:
    if weight.dtype != mx.uint8 or weight.ndim != 2:
        raise ValueError(
            "Fine-grained FP8 weights must be 2D E4M3 bytes loaded as "
            f"uint8; got dtype={weight.dtype}, shape={weight.shape}."
        )
    if scale_inv.ndim != 2 or not mx.issubdtype(scale_inv.dtype, mx.floating):
        raise ValueError(
            "Fine-grained FP8 scales must be a 2D floating-point block "
            f"grid; got dtype={scale_inv.dtype}, shape={scale_inv.shape}."
        )

    rows, cols = weight.shape
    expected_scale_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if scale_inv.shape != expected_scale_shape:
        raise ValueError(
            "Fine-grained FP8 scale shape does not match its weight: "
            f"weight={weight.shape}, scales={scale_inv.shape}, "
            f"expected={expected_scale_shape}."
        )

    pad_rows = (-rows) % block_size
    pad_cols = (-cols) % block_size
    decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
    if pad_rows or pad_cols:
        decoded = mx.pad(decoded, ((0, pad_rows), (0, pad_cols)))

    decoded = decoded.reshape(
        (rows + pad_rows) // block_size,
        block_size,
        (cols + pad_cols) // block_size,
        block_size,
    )
    decoded = (decoded * scale_inv[:, None, :, None]).reshape(
        rows + pad_rows, cols + pad_cols
    )
    return decoded[:rows, :cols]


def _quantize_fp8_weight(
    weight: mx.array,
    scale_inv: mx.array,
    target_quantization: dict | None = None,
) -> tuple[mx.array, ...]:
    """Convert one block-FP8 tensor to a native MLX quantized layout lazily."""
    target_quantization = target_quantization or MLX_MXFP8_QUANTIZATION
    group_size = target_quantization["group_size"]
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            "FP8 weight input dimension must be divisible by the target "
            f"group size {group_size}; got shape={weight.shape}."
        )

    restored = _dequantize_fp8_weight(weight, scale_inv)
    return mx.quantize(restored, **target_quantization)


def transform_fp8_weights(
    weights: dict[str, mx.array],
    config: dict,
    target_quantization: dict | None = None,
) -> tuple[dict[str, mx.array], dict | None]:
    """Convert a compatible block-FP8 checkpoint to an MLX weight layout.

    The source format is detected from configuration rather than model type.
    Unsupported FP8 layouts pass through unchanged for their own loader or
    sanitizer to handle. By default the target is native MXFP8; callers may
    request another native MLX quantization layout to avoid an intermediate
    requantization round trip.
    """
    source_quantization = make_quantization_config(config)
    if source_quantization is None:
        return weights, None
    quantization = dict(target_quantization or source_quantization)

    scale_keys = [key for key in weights if key.endswith(".weight_scale_inv")]
    if not scale_keys:
        return weights, quantization

    converted = dict(weights)
    for scale_key in scale_keys:
        weight_key = scale_key[: -len("_scale_inv")]
        if weight_key not in converted:
            raise ValueError(f"Missing FP8 weight for scale tensor {scale_key!r}.")

        weight = converted.pop(weight_key)
        scale_inv = converted.pop(scale_key)
        quantized = _quantize_fp8_weight(weight, scale_inv, quantization)
        packed, scales = quantized[:2]
        converted[weight_key] = packed
        prefix = weight_key[: -len(".weight")]
        converted[prefix + ".scales"] = scales
        if len(quantized) == 3:
            converted[prefix + ".biases"] = quantized[2]

    return converted, quantization
