from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx import nn

from mlx_vlm.fp8 import _dequantize_fp8_weight, make_quantization_config

from ..flux2.weights import load_vae
from ..llada2_moe.config import ModelConfig as TextConfig
from .conditioning import QueryFormer, TextProjection
from .config import LLaDAImageTransformerConfig, read_config
from .sigvq import SigVQ, sanitize_sigvq_weights
from .text_encoder import LLaDAImageTextEncoder
from .transformer import LLaDAImageTransformer, sanitize_transformer_weights


def load_safetensors(directory: Path) -> dict[str, mx.array]:
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if indexes:
        filenames = sorted(set(read_config(indexes[0])["weight_map"].values()))
        files = [directory / filename for filename in filenames]
    else:
        files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    weights = {}
    for file in files:
        shard = mx.load(str(file))
        if weights.keys() & shard.keys():
            raise ValueError(f"Duplicate weight keys in {file}")
        weights.update(shard)
    return weights


def apply_weights(model: nn.Module, weights: dict[str, mx.array]):
    # Never silently leave missing or mismatched checkpoint tensors initialized.
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def restore_fp8_weights(
    weights: dict[str, mx.array], config: dict
) -> dict[str, mx.array]:
    """Restore the published E4M3 block weights for BF16 MLX computation.

    Denoiser linears use Transformers' 2D block layout. The text encoder stores
    an additional leading expert dimension and names its scales ``*_scale``.
    """
    experts = config.get("llada_fp8_experts", {})
    if experts.get("enabled") and (
        experts.get("format") != "e4m3fn"
        or experts.get("weight_granularity") != "per_expert_2d_block"
        or experts.get("weight_block_size") != [128, 128]
    ):
        raise ValueError("Unsupported LLaDA-Image FP8 expert format")
    block_fp8 = make_quantization_config(config) is not None
    for key in list(weights):
        if key.endswith(".weight_scale_inv"):
            if not block_fp8:
                raise ValueError(
                    "Missing or unsupported FP8 quantization configuration"
                )
            weight_key = key.removesuffix("_scale_inv")
        elif ".mlp.experts." in key and key.endswith("_scale"):
            if not experts.get("enabled"):
                raise ValueError("FP8 expert scales require llada_fp8_experts metadata")
            weight_key = key.removesuffix("_scale")
        else:
            continue
        if weight_key not in weights:
            raise ValueError(f"Missing FP8 weight for {key}")
        value, scale = weights.pop(weight_key), weights.pop(key)
        shape = value.shape
        if value.ndim == 3:
            if value.shape[1] % 128 or scale.shape != (
                shape[0],
                shape[1] // 128,
                (shape[2] + 127) // 128,
            ):
                raise ValueError(f"Invalid FP8 expert block shapes for {weight_key}")
            value = value.reshape(-1, shape[-1])
            scale = scale.reshape(-1, scale.shape[-1])
        restored = (
            _dequantize_fp8_weight(value, scale).reshape(shape).astype(mx.bfloat16)
        )
        # Bound conversion memory instead of retaining all FP32 reconstruction graphs.
        mx.eval(restored)
        weights[weight_key] = restored
    if any(value.dtype == mx.uint8 for value in weights.values()):
        raise ValueError("An FP8 weight is missing its block scales")
    return weights


def sanitize_text_encoder_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    result = {}
    for key, value in weights.items():
        if key == "model.lm_head.weight":
            continue
        key = key.removeprefix("model.language_model.")
        if ".mlp.experts." in key:
            key = key.replace(".mlp.experts.", ".mlp.switch_mlp.") + ".weight"
        result[key] = value
    return result


def load_text_encoder(root: Path) -> LLaDAImageTextEncoder:
    raw_config = read_config(root / "text_encoder" / "config.json")
    config = TextConfig.from_dict(raw_config)
    model = LLaDAImageTextEncoder(config)
    return apply_weights(
        model,
        sanitize_text_encoder_weights(
            restore_fp8_weights(load_safetensors(root / "text_encoder"), raw_config)
        ),
    )


def load_lm_head(root: Path) -> nn.Linear:
    # Reading the index keeps the other text shards lazy; only this matrix is used.
    weight = load_safetensors(root / "text_encoder")["model.lm_head.weight"]
    model = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    return apply_weights(model, {"weight": weight})


def load_sigvq(root: Path, *, include_encoder: bool = True) -> SigVQ:
    model = SigVQ(
        read_config(root / "sigvq" / "config.json"), include_encoder=include_encoder
    )
    return apply_weights(
        model,
        sanitize_sigvq_weights(
            load_safetensors(root / "sigvq"), include_encoder=include_encoder
        ),
    )


def load_queryformer(root: Path) -> QueryFormer:
    model = QueryFormer(read_config(root / "queryformer" / "config.json"))
    return apply_weights(model, load_safetensors(root / "queryformer"))


def load_text_projection(root: Path) -> TextProjection:
    model = TextProjection(read_config(root / "text_projection" / "config.json"))
    return apply_weights(model, load_safetensors(root / "text_projection"))


def load_transformer(root: Path) -> LLaDAImageTransformer:
    raw_config = read_config(root / "transformer" / "config.json")
    config = LLaDAImageTransformerConfig.from_dict(raw_config)
    model = LLaDAImageTransformer(config)
    return apply_weights(
        model,
        sanitize_transformer_weights(
            restore_fp8_weights(load_safetensors(root / "transformer"), raw_config)
        ),
    )


__all__ = [
    "load_lm_head",
    "load_queryformer",
    "load_sigvq",
    "load_text_encoder",
    "load_text_projection",
    "load_transformer",
    "load_vae",
]
