from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten

from mlx_vlm.models.bonsai.constants import ModelConfig
from mlx_vlm.models.bonsai.klein_fast import (
    Flux2KleinFastTransformer,
    Flux2KleinMegakernelSpec,
    find_packed_artifact_dir,
    load_klein_fast_packed_weights_from_disk,
)
from mlx_vlm.models.bonsai.qwen.text_encoder import Qwen3TextEncoder
from mlx_vlm.models.bonsai.vae import BonsaiVAE

SMALL_DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
SMALL_DECODER_FILE = "diffusion_pytorch_model.safetensors"


def load_text_encoder_4bit(model_path: str | Path) -> Qwen3TextEncoder:
    root = Path(model_path).expanduser() / "text_encoder-mlx-4bit"
    raw = mx.load(str(root / "model.safetensors"))
    stripped = {k[len("model.") :]: v for k, v in raw.items() if k.startswith("model.")}
    nested = tree_unflatten(list(stripped.items()))
    text_encoder = Qwen3TextEncoder(hidden_size=2560, intermediate_size=9728)
    nn.quantize(
        text_encoder,
        class_predicate=lambda _, module: hasattr(module, "to_quantized"),
        bits=4,
        group_size=64,
    )
    text_encoder.update(nested)
    return text_encoder


def load_transformer(
    model_path: str | Path, precision: str
) -> Flux2KleinFastTransformer:
    root = Path(model_path).expanduser()
    spec = Flux2KleinMegakernelSpec()
    packed_dir = find_packed_artifact_dir(root)
    if packed_dir is None:
        raise FileNotFoundError(
            f"Missing transformer-packed-mflux artifact under {root}"
        )
    weights = load_klein_fast_packed_weights_from_disk(
        packed_dir, spec, dtype=mx.bfloat16
    )
    transformer = Flux2KleinFastTransformer(
        weights=weights,
        precision=precision,
        patch_size=1,
        in_channels=spec.in_channels,
        out_channels=spec.in_channels,
        num_layers=spec.num_double_blocks,
        num_single_layers=spec.num_single_blocks,
        attention_head_dim=spec.head_dim,
        num_attention_heads=spec.num_heads,
        joint_attention_dim=spec.context_dim,
        timestep_guidance_channels=256,
        mlp_ratio=spec.mlp_ratio,
        axes_dims_rope=spec.axes_dims_rope,
        rope_theta=spec.rope_theta,
        guidance_embeds=False,
        layer_norm_eps=spec.layer_norm_eps,
        rms_norm_eps=spec.rms_norm_eps,
    )
    raw: dict[str, mx.array] = {}
    for shard in packed_dir.glob("*.safetensors"):
        if not shard.name.startswith("._"):
            raw.update(mx.load(str(shard)))
    transformer.time_guidance_embed.linear_1.weight = raw[
        "time_guidance_embed.timestep_embedder.linear_1.weight"
    ].astype(mx.bfloat16)
    transformer.time_guidance_embed.linear_2.weight = raw[
        "time_guidance_embed.timestep_embedder.linear_2.weight"
    ].astype(mx.bfloat16)
    mx.eval(
        transformer.time_guidance_embed.linear_1.weight,
        transformer.time_guidance_embed.linear_2.weight,
    )
    return transformer


def load_vae() -> BonsaiVAE:
    vae = BonsaiVAE()
    path = hf_hub_download(repo_id=SMALL_DECODER_REPO, filename=SMALL_DECODER_FILE)
    raw = mx.load(path)
    mapped: dict = {}
    for key, value in raw.items():
        if not (
            key.startswith("decoder.")
            or key.startswith("post_quant_conv.")
            or key.startswith("bn.")
        ):
            continue
        if key.endswith(".num_batches_tracked"):
            continue
        key = key.replace(".to_out.0.", ".to_out.")
        tensor = value.astype(ModelConfig.precision)
        if tensor.ndim == 4:
            tensor = tensor.transpose(0, 2, 3, 1)
        _set_nested_value(mapped, key, tensor)
    vae.update(mapped)
    return vae


def _set_nested_value(tree: dict, path: str, value: mx.array) -> None:
    parts = path.split(".")
    current = tree
    i = 0
    while i < len(parts) - 1:
        part = parts[i]
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            current.setdefault(part, [])
            index = int(parts[i + 1])
            while len(current[part]) <= index:
                current[part].append({})
            current = current[part][index]
            i += 2
        else:
            current = current.setdefault(part, {})
            i += 1
    current[parts[-1]] = value
