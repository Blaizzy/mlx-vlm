"""Fine-grained FP8 checkpoint conversion for GLM-5-Next."""

from ..qwen3_5.fp8 import (
    MLX_MXFP8_QUANTIZATION,
    convert_qwen_fp8_weights,
    quantize_qwen_fp8_weight,
)


def make_quantization_config(config: dict) -> dict | None:
    quantization = config.get("quantization_config") or {}
    is_supported = (
        config.get("model_type") == "glm5_next"
        and isinstance(quantization, dict)
        and quantization.get("quant_method") == "fp8"
        and quantization.get("fmt", "e4m3") == "e4m3"
        and quantization.get("weight_block_size") == [128, 128]
    )
    return dict(MLX_MXFP8_QUANTIZATION) if is_supported else None


convert_glm5_next_fp8_weights = convert_qwen_fp8_weights
quantize_glm5_next_fp8_weight = quantize_qwen_fp8_weight


__all__ = [
    "convert_glm5_next_fp8_weights",
    "make_quantization_config",
    "quantize_glm5_next_fp8_weight",
]
