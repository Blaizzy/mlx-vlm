"""Load Ming-Image-0.1-Design components from a diffusers checkpoint into MLX."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import MingImageConfig
from .text_encoder import MingImageTextEncoder
from .transformer import MingImageTransformer, sanitize_transformer_weights
from .vae import build_vae, sanitize_vae_weights


def _load_shards(directory: Path) -> dict[str, mx.array]:
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors under {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    return weights


def _apply(model: nn.Module, weights: list[tuple[str, mx.array]]) -> nn.Module:
    model.load_weights(weights, strict=True)
    model.eval()
    return model


def load_transformer(
    model_path: str | Path, config: MingImageConfig | None = None
) -> MingImageTransformer:
    root = Path(model_path).expanduser()
    config = config or MingImageConfig.from_model_path(root)
    model = MingImageTransformer(config.dit)
    weights = sanitize_transformer_weights(_load_shards(root / "transformer"))
    return _apply(model, list(weights.items()))


def load_vae(model_path: str | Path, config: MingImageConfig | None = None):
    root = Path(model_path).expanduser()
    config = config or MingImageConfig.from_model_path(root)
    model = build_vae(config.vae)
    weights = sanitize_vae_weights(_load_shards(root / "vae"))
    return _apply(model, list(weights.items()))


def _stack_experts(weights: dict[str, mx.array], prefix: str, num_experts: int) -> None:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        first = f"{prefix}.experts.0.{proj}.weight"
        if first not in weights:
            continue
        stacked = mx.stack(
            [
                weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                for e in range(num_experts)
            ]
        )
        weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked


def _sanitize_mllm(
    weights: dict[str, mx.array],
    num_layers: int,
    num_experts: int,
    first_k_dense: int,
) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith(("model.lm_head", "linear_proj", "vision.")):
            continue
        if not key.startswith("model.model."):
            continue
        k = "mllm." + key[len("model.model.") :]
        if ".mlp.audio_gate." in k:
            continue
        k = k.replace(".attention.q_norm.", ".attention.query_layernorm.")
        k = k.replace(".attention.k_norm.", ".attention.key_layernorm.")
        k = k.replace(".mlp.gate.weight", ".mlp.gate.gate_proj.weight")
        k = k.replace(".mlp.image_gate.weight", ".mlp.image_gate.gate_proj.weight")
        out[k] = value
    for layer in range(first_k_dense, num_layers):
        _stack_experts(out, f"mllm.layers.{layer}.mlp", num_experts)
    return out


def _sanitize_connector(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith("model.layers."):
            out["connector." + key[len("model.") :]] = value
        elif key == "model.norm.weight":
            out["connector.norm.weight"] = value
    return out


def _sanitize_bridge(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key == "query_tokens_dict.16x16":
            out["query_tokens"] = value
        elif key.startswith(("proj_in.", "proj_out.")):
            out[key] = value
        elif key == "proj_directvlm.0.weight":
            out["directvlm_norm.weight"] = value
        elif key.startswith("proj_directvlm.1."):
            out["directvlm_proj." + key[len("proj_directvlm.1.") :]] = value
    return out


def load_text_encoder(
    model_path: str | Path, config: MingImageConfig | None = None
) -> MingImageTextEncoder:
    root = Path(model_path).expanduser()
    config = config or MingImageConfig.from_model_path(root)
    model = MingImageTextEncoder(config)
    weights: dict[str, mx.array] = {}
    weights.update(
        _sanitize_mllm(
            _load_shards(root / "mllm"),
            config.mllm.num_hidden_layers,
            config.mllm.num_experts,
            config.mllm.first_k_dense_replace,
        )
    )
    weights.update(_sanitize_connector(_load_shards(root / "connector")))
    weights.update(_sanitize_bridge(_load_shards(root / "mlp")))
    return _apply(model, list(weights.items()))


__all__ = ["load_text_encoder", "load_transformer", "load_vae"]
