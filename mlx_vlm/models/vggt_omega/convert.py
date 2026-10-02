"""Convert an official VGGT-Omega ``.pt`` checkpoint to an MLX model directory.

The checkpoint is read without torch. The output holds ``config.json``,
``preprocessor_config.json`` and ``model.safetensors`` in MLX layout
(``bias_mask`` folded into the qkv biases, channel-last convs). With
``--dtype bfloat16`` every tensor is bf16 except the two RoPE ``periods``
buffers, which stay float32 (bf16 would move the rotation frequencies).

Usage:
    python -m mlx_vlm.models.vggt_omega.convert \
        --hf-repo facebook/VGGT-Omega --checkpoint vggt_omega_1b_512.pt \
        --mlx-path VGGT-Omega-1B-512-bf16 --dtype bfloat16
"""

import argparse
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ...utils import MODEL_CONVERSION_DTYPES, save_config, save_weights
from ..sam3d_objects.checkpoint import to_safetensors
from .config import ModelConfig
from .vggt_omega import Model

# Checkpoint buffers that inference does not use.
_UNUSED_KEYS = ("aggregator.patch_embed.mask_token",)
# ``ConvTranspose2d`` kernels; every other 4-D weight is a ``Conv2d``.
_CONV_TRANSPOSE_KEY = re.compile(r"\.resize_layers\.[01]\.weight$")


def _resolution_from_name(name: str) -> Optional[int]:
    """``vggt_omega_1b_512.pt`` -> 512."""
    match = re.search(r"_(\d{3,4})(?=[_.])", Path(name).name)
    return int(match[1]) if match else None


def to_mlx_layout(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Fold each ``qkv.bias_mask`` into its bias, move conv kernels to
    channel-last and drop unused buffers."""
    out = {}
    for k, v in weights.items():
        if k in _UNUSED_KEYS or k.endswith(".bias_mask"):
            continue
        mask = weights.get(k + "_mask")
        if mask is not None:
            v = v * mask.astype(v.dtype)
        if v.ndim == 4 and k.endswith(".weight"):
            if _CONV_TRANSPOSE_KEY.search(k):
                v = v.transpose(1, 2, 3, 0)  # ConvTranspose2d: (in, out, kh, kw)
            else:
                v = v.transpose(0, 2, 3, 1)  # Conv2d: (out, in, kh, kw)
        out[k] = v
    return out


def convert_state_dict(
    weights: Dict[str, mx.array], config: ModelConfig, dtype: str
) -> Model:
    """Official state dict -> model in MLX layout and ``dtype`` (strict load)."""
    model = Model(config)
    cast = getattr(mx, dtype)
    weights = {
        # The RoPE periods stay float32 whatever the output dtype.
        k: v if k.endswith("rope_embed.periods") else v.astype(cast)
        for k, v in to_mlx_layout(weights).items()
    }
    model.load_weights(list(weights.items()), strict=True)
    return model


def save_model(model: Model, path: Path):
    """Write the weights, ``config.json`` and ``preprocessor_config.json``."""
    config = model.config
    save_weights(path, model, donate_weights=True)
    save_config(config.to_dict(), path / "config.json")
    preprocessor = {
        "image_resolution": config.image_resolution,
        "mode": "balanced",
        "patch_size": config.patch_size,
    }
    save_config(preprocessor, path / "preprocessor_config.json")


def convert(
    checkpoint: str,
    mlx_path: str,
    dtype: str = "bfloat16",
    image_resolution: Optional[int] = None,
    hf_repo: Optional[str] = None,
) -> Path:
    if hf_repo:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(hf_repo, checkpoint)
    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out) as tmp:
        source = Path(tmp) / "source.safetensors"
        to_safetensors(checkpoint, source)
        weights = mx.load(str(source))

        def has(prefix):
            return any(k.startswith(prefix) for k in weights)

        config = ModelConfig(
            enable_camera=has("camera_head."),
            enable_depth=has("dense_head."),
            enable_alignment=has("text_alignment_head."),
            image_resolution=image_resolution
            or _resolution_from_name(checkpoint)
            or ModelConfig.image_resolution,
        )
        save_model(convert_state_dict(weights, config, dtype), out)
    print(f"Saved {dtype} weights to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the .pt (or its --hf-repo name)"
    )
    parser.add_argument("--hf-repo", default=None, help="Download --checkpoint here")
    parser.add_argument("--mlx-path", required=True, help="Output directory")
    parser.add_argument("--dtype", default="bfloat16", choices=MODEL_CONVERSION_DTYPES)
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=None,
        help="Processor resolution (default: parsed from the file name, else 512)",
    )
    parser.add_argument("--upload-repo", default=None, help="HF repo id to upload to")
    args = parser.parse_args()
    out = convert(
        args.checkpoint, args.mlx_path, args.dtype, args.image_resolution, args.hf_repo
    )
    if args.upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload_repo, exist_ok=True)
        api.upload_folder(repo_id=args.upload_repo, folder_path=str(out))
        print(f"Uploaded to https://huggingface.co/{args.upload_repo}")


if __name__ == "__main__":
    main()
