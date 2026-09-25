"""Convert a Ming-Image-0.1-Design checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import save_weights, upload_to_hub

from .weights import load_text_encoder, load_transformer, load_vae

_CARD = """---
license: mit
base_model: inclusionAI/Ming-Image-0.1-Design
library_name: mlx
pipeline_tag: text-to-image
tags:
- mlx
- ming-image
- text-to-image
---

# Ming-Image-0.1-Design (MLX{quant})

MLX conversion of [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design),
a 6B text-to-image model for text-rich design with RGBA / transparent-background
output.
"""


def _skip_dit(path: str, _module) -> bool:
    return "t_embedder" not in path and "cap_embedder" not in path


def _skip_encoder(path: str, _module) -> bool:
    return not (
        path.endswith("gate.gate_proj") or path.endswith("image_gate.gate_proj")
    )


def _write_config(out_dir: Path, base: dict, quant: dict | None) -> None:
    base = dict(base)
    base["mlx_format"] = True
    if quant is not None:
        base["quantization"] = quant
    (out_dir / "config.json").write_text(json.dumps(base, indent=2))


def convert(
    model: str,
    output: str | Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    upload_repo: str | None = None,
) -> Path:
    src = Path(model).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Ming-Image checkpoint not found: {src}")
    out = Path(output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    quant = {"group_size": group_size, "bits": bits, "mode": mode}

    transformer = load_transformer(src)
    transformer, _ = quantize_model(
        transformer, {}, group_size, bits, mode, quant_predicate=_skip_dit
    )
    (out / "transformer").mkdir(parents=True, exist_ok=True)
    save_weights(out / "transformer", transformer, donate_weights=True)
    _write_config(
        out / "transformer",
        json.loads((src / "transformer" / "config.json").read_text()),
        quant,
    )
    del transformer

    encoder = load_text_encoder(src)
    encoder, _ = quantize_model(
        encoder, {}, group_size, bits, mode, quant_predicate=_skip_encoder
    )
    (out / "text_encoder").mkdir(parents=True, exist_ok=True)
    save_weights(out / "text_encoder", encoder, donate_weights=True)
    _write_config(out / "text_encoder", {}, quant)
    del encoder

    vae = load_vae(src)
    (out / "vae").mkdir(parents=True, exist_ok=True)
    save_weights(out / "vae", vae, donate_weights=True)
    _write_config(
        out / "vae", json.loads((src / "vae" / "config.json").read_text()), None
    )
    del vae

    # Config-only copies so from_model_path / layout detection still resolve.
    for sub in ("mllm", "connector", "mlp"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / sub / "config.json", out / sub / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = src / "mllm" / name
        if source.exists():
            shutil.copy2(source, out / "mllm" / name)
    shutil.copytree(src / "scheduler", out / "scheduler", dirs_exist_ok=True)

    (out / "README.md").write_text(_CARD.format(quant=f", {bits}-bit" if bits else ""))
    if upload_repo is not None:
        upload_to_hub(str(out), upload_repo)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ming-Image-0.1-Design to MLX."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--upload-repo", default=None)
    args = parser.parse_args()
    out = convert(
        args.model,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
        upload_repo=args.upload_repo,
    )
    print(f"Saved MLX checkpoint to {out}")


if __name__ == "__main__":
    main()


__all__ = ["convert"]
