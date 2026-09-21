"""Convert a Qwen-Image-2.1 diffusers checkpoint to MLX (optionally quantized).

Loads each component through the MLX loaders, quantizes the transformer and text
encoder (the VAE is convolutional and stays in its loaded precision), saves the
weights in MLX layout, and copies the processor/scheduler assets so the result
loads with no runtime remap. Mirrors mage_flow/convert.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import save_weights, upload_to_hub

from .download import download_model, validate_model_layout
from .weights import load_text_encoder, load_transformer, load_vae

_CARD = """---
license: other
license_name: qwen-research
base_model: Qwen/Qwen-Image-2.1
library_name: mlx
pipeline_tag: text-to-image
tags:
- mlx
- qwen-image
- text-to-image
---

# Qwen-Image-2.1 (MLX{quant})

MLX conversion of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
a single-stream DiT text-to-image + image-editing model with a Qwen3-VL text
encoder and a 64-channel RGBA VAE. Governed by the Qwen Research License.
"""


def _save_component(out_dir: Path, model, src_dir: Path, quant: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_weights(out_dir, model, donate_weights=True)
    config = json.loads((src_dir / "config.json").read_text())
    # Weights are saved in final MLX layout, so loaders must skip the
    # diffusers->MLX remap for this checkpoint.
    config["mlx_format"] = True
    if quant is not None:
        config["quantization"] = quant
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def convert(
    model: str,
    output: str | Path,
    *,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    upload_repo: str | None = None,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    src = Path(model).expanduser()
    if not src.exists():
        src = download_model(model, token=token, revision=revision)
    src = validate_model_layout(src)
    out = Path(output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    quant = {"group_size": group_size, "bits": bits, "mode": mode} if quantize else None

    transformer = load_transformer(src)
    if quantize:
        transformer, _ = quantize_model(transformer, {}, group_size, bits, mode)
    _save_component(out / "transformer", transformer, src / "transformer", quant)
    del transformer

    text_encoder = load_text_encoder(src)
    if quantize:
        text_encoder, _ = quantize_model(text_encoder, {}, group_size, bits, mode)
    _save_component(out / "text_encoder", text_encoder, src / "text_encoder", quant)
    del text_encoder

    vae = load_vae(src)
    _save_component(out / "vae", vae, src / "vae", None)
    del vae

    for sub in ("processor", "scheduler"):
        if (src / sub).exists():
            shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)
    if (src / "model_index.json").exists():
        shutil.copy2(src / "model_index.json", out / "model_index.json")
    (out / "README.md").write_text(
        _CARD.format(quant=f", {bits}-bit" if quantize else "")
    )

    if upload_repo is not None:
        upload_to_hub(str(out), upload_repo)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen-Image-2.1 to MLX.")
    parser.add_argument("--model", default="qwen-image-2.1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--upload-repo", default=None)
    args = parser.parse_args()
    out = convert(
        args.model,
        args.output,
        quantize=not args.no_quantize,
        bits=args.bits,
        group_size=args.group_size,
        upload_repo=args.upload_repo,
    )
    print(f"Saved MLX checkpoint to {out}")


if __name__ == "__main__":
    main()


__all__ = ["convert"]
