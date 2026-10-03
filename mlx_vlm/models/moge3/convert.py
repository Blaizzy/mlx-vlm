"""Convert an official MoGe-3 ``model.pt`` checkpoint to an MLX model repo.

The official checkpoints (e.g. ``Ruicheng/moge-3-vitl``) store a torch pickle
with ``model_config`` + ``model`` state dict. This script writes the standard
MLX layout: ``config.json`` + ``model.safetensors`` (+ optional Hub upload).

Usage:
    python -m mlx_vlm.models.moge3.convert \
        --torch-ckpt /path/to/model.pt \
        --mlx-path /path/to/out \
        --dtype float32 \
        [--upload-repo mlx-community/moge-3-vitl-mlx-fp32]
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from ...utils import MODEL_CONVERSION_DTYPES
from .moge3 import _CONV_TRANSPOSE_KEY


def convert_weights(state_dict):
    """Torch state dict -> MLX weights (channel-last conv layouts)."""
    weights = {}
    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()
        if key in ("encoder.image_mean", "encoder.image_std"):
            arr = arr.reshape(1, 1, 1, 3)
        elif arr.ndim == 4 and _CONV_TRANSPOSE_KEY.search(key):
            arr = arr.transpose(1, 2, 3, 0)  # ConvTranspose2d: (in, out, kh, kw)
        elif arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)  # Conv2d: (out, in, kh, kw)
        weights[key] = mx.array(arr)
    return weights


def convert(
    torch_ckpt: str, mlx_path: str, dtype: str = "float32", upload_repo: str = None
):
    import torch

    checkpoint = torch.load(torch_ckpt, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]

    weights = convert_weights(checkpoint["model"])
    if dtype != "float32":
        cast = getattr(mx, dtype)
        weights = {
            k: (v.astype(cast) if mx.issubdtype(v.dtype, mx.floating) else v)
            for k, v in weights.items()
        }

    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    (out / "config.json").write_text(
        json.dumps({"model_type": "moge3", **model_config}, indent=2)
    )
    (out / "preprocessor_config.json").write_text(
        json.dumps({"resize_to": None}, indent=2)
    )
    print(f"Saved {len(weights)} tensors to {out}")

    if upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(upload_repo, exist_ok=True)
        api.upload_folder(repo_id=upload_repo, folder_path=str(out))
        print(f"Uploaded to https://huggingface.co/{upload_repo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-ckpt", required=True, help="Path to model.pt")
    parser.add_argument("--mlx-path", required=True, help="Output directory")
    parser.add_argument("--dtype", default="float32", choices=MODEL_CONVERSION_DTYPES)
    parser.add_argument("--upload-repo", default=None, help="HF repo id to upload to")
    args = parser.parse_args()
    convert(args.torch_ckpt, args.mlx_path, args.dtype, args.upload_repo)


if __name__ == "__main__":
    main()
