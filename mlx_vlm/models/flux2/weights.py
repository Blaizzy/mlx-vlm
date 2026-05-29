from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_vlm.models.flux2.config import Flux2Variant
from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.qwen.text_encoder import Qwen3TextEncoder
from mlx_vlm.models.flux2.transformer import Flux2Transformer
from mlx_vlm.models.flux2.vae import Flux2VAE

FULL_DECODER_CHANNELS = (128, 256, 512, 512)


def load_text_encoder(
    model_path: str | Path, variant: Flux2Variant
) -> Qwen3TextEncoder:
    raw = _load_safetensors(Path(model_path).expanduser() / "text_encoder")
    weights = {}
    for key, value in raw.items():
        if not key.startswith("model."):
            continue
        stripped = key[len("model.") :]
        if stripped.startswith(("embed_tokens.", "layers.", "norm.")):
            weights[stripped] = value.astype(ModelConfig.precision)
    text_encoder = Qwen3TextEncoder(**variant.text_encoder_overrides)
    text_encoder.update(tree_unflatten(list(weights.items())))
    return text_encoder


def load_transformer(model_path: str | Path, variant: Flux2Variant) -> Flux2Transformer:
    raw = _load_safetensors(Path(model_path).expanduser() / "transformer")
    weights = {}
    for key, value in raw.items():
        mapped = key
        mapped = mapped.replace(
            "time_guidance_embed.timestep_embedder.", "time_guidance_embed."
        )
        mapped = mapped.replace(".to_out.0.", ".to_out.")
        weights[mapped] = value.astype(ModelConfig.precision)
    transformer = Flux2Transformer(**variant.transformer_overrides)
    transformer.update(tree_unflatten(list(weights.items())))
    return transformer


def load_vae(model_path: str | Path, *, include_encoder: bool = False) -> Flux2VAE:
    raw = _load_safetensors(Path(model_path).expanduser() / "vae")
    weights = {}
    for key, value in raw.items():
        if key.endswith(".num_batches_tracked"):
            continue
        if not (
            key.startswith("decoder.")
            or key.startswith("post_quant_conv.")
            or key.startswith("bn.")
            or (include_encoder and key.startswith("encoder."))
            or (include_encoder and key.startswith("quant_conv."))
        ):
            continue
        mapped = key.replace(".to_out.0.", ".to_out.")
        tensor = value.astype(ModelConfig.precision)
        if tensor.ndim == 4:
            tensor = tensor.transpose(0, 2, 3, 1)
        weights[mapped] = tensor
    vae = Flux2VAE(
        decoder_block_out_channels=FULL_DECODER_CHANNELS,
        include_encoder=include_encoder,
        encoder_block_out_channels=FULL_DECODER_CHANNELS,
    )
    vae.update(tree_unflatten(list(weights.items())))
    return vae


def _load_safetensors(directory: Path) -> dict[str, mx.array]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing weight directory: {directory}")
    weights: dict[str, mx.array] = {}
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    for file in files:
        weights.update(mx.load(str(file)))
    return weights
