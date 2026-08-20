from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx

from mlx_vlm.models.qwen3_vl.config import ModelConfig as Qwen3VLConfig
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel

from .text_encoder import MageFlowTextEncoder
from .transformer import MageFlowTransformer
from .vae import MageVAE


def _load_safetensors(directory: Path) -> dict[str, mx.array]:
    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    return weights


def sanitize_transformer_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    sanitized = {}
    for key, value in weights.items():
        key = key.replace(".img_mod.1.", ".img_mod.linear.")
        key = key.replace(".txt_mod.1.", ".txt_mod.linear.")
        key = key.replace(".img_mlp.net.0.proj.", ".img_mlp.linear_in.")
        key = key.replace(".img_mlp.net.2.", ".img_mlp.linear_out.")
        key = key.replace(".txt_mlp.net.0.proj.", ".txt_mlp.linear_in.")
        key = key.replace(".txt_mlp.net.2.", ".txt_mlp.linear_out.")
        key = key.replace(".attn.to_out.0.", ".attn.to_out.")
        sanitized[key] = value
    return sanitized


def load_transformer(model_path: str | Path) -> MageFlowTransformer:
    root = Path(model_path).expanduser()
    config = json.loads((root / "transformer" / "config.json").read_text())
    transformer = MageFlowTransformer(
        in_channels=int(config.get("in_channels", 128)),
        out_channels=int(config.get("out_channels", 128)),
        context_in_dim=int(config.get("context_in_dim", 2560)),
        hidden_size=int(config.get("hidden_size", 3072)),
        num_heads=int(config.get("num_heads", 24)),
        depth=int(config.get("depth", 12)),
        axes_dim=tuple(config.get("axes_dim", (16, 56, 56))),
        theta=float(config.get("theta", 10000)),
    )
    weights = sanitize_transformer_weights(_load_safetensors(root / "transformer"))
    transformer.load_weights(list(weights.items()), strict=True)
    return transformer


def _map_vae_key(key: str) -> str | None:
    if key.startswith("student.dconv_encoder."):
        key = "dconv_encoder." + key.removeprefix("student.dconv_encoder.")
    elif key.startswith("pipeline.y_embedder.encoder."):
        return None
    elif key.startswith("pipeline."):
        key = "decoder_model." + key.removeprefix("pipeline.")
    else:
        return None

    key = key.replace(".adaLN_modulation.1.", ".adaLN_modulation.linear.")
    key = key.replace(".ca.1.", ".ca_conv.")
    key = key.replace(".t_embedder.mlp.0.", ".t_embedder.linear_1.")
    key = key.replace(".t_embedder.mlp.2.", ".t_embedder.linear_2.")
    key = key.replace(".x_embedder.embedder.0.", ".x_embedder.linear.")
    key = re.sub(
        r"(\.dec_net\.res_blocks\.\d+)\.mlp\.0\.",
        r"\1.linear_1.",
        key,
    )
    key = re.sub(
        r"(\.dec_net\.res_blocks\.\d+)\.mlp\.2\.",
        r"\1.linear_2.",
        key,
    )
    return key


def sanitize_vae_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    sanitized = {}
    for raw_key, value in weights.items():
        key = _map_vae_key(raw_key)
        if key is None or raw_key.endswith(".num_batches_tracked"):
            continue
        if value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        sanitized[key] = value
    return sanitized


def load_vae(model_path: str | Path, *, include_encoder: bool = True) -> MageVAE:
    root = Path(model_path).expanduser()
    vae = MageVAE(include_encoder=include_encoder)
    weights = sanitize_vae_weights(_load_safetensors(root / "vae"))
    if not include_encoder:
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("dconv_encoder.")
        }
    vae.load_weights(list(weights.items()), strict=True)
    return vae


def load_text_encoder(
    model_path: str | Path, *, max_length: int = 2048
) -> MageFlowTextEncoder:
    root = Path(model_path).expanduser()
    text_root = root / "text_encoder"
    config = json.loads((text_root / "config.json").read_text())
    model_config = Qwen3VLConfig.from_dict(config)
    model = Qwen3VLModel(model_config)
    weights = _load_safetensors(text_root)
    weights = model.sanitize(weights)
    weights = model.vision_tower.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    return MageFlowTextEncoder(
        model=model,
        model_path=root,
        max_length=max_length,
    )


__all__ = [
    "load_text_encoder",
    "load_transformer",
    "load_vae",
    "sanitize_transformer_weights",
    "sanitize_vae_weights",
]
