"""Save original MOSS-Image components as a standalone MLX checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ...generate.image import IMAGE_DOWNLOAD_PATTERNS
from ...utils import get_model_path, save_weights
from .config import detect_moss_image_layout
from .weights import load_text_encoder, load_transformer, load_vae, read_config


def model_card(source_id=None):
    source_metadata = f"base_model: {source_id}\n" if source_id is not None else ""
    source_description = (
        f"Converted from [{source_id}](https://huggingface.co/{source_id})."
        if source_id is not None
        else "Converted from a local MOSS-Image checkpoint."
    )
    return f"""---
library_name: mlx-vlm
license: apache-2.0
{source_metadata}pipeline_tag: text-to-image
tags:
- mlx
- moss-image
---

# MOSS-Image for MLX

{source_description}
Requires mlx-vlm with MOSS-Image support. Inference runs in MLX without PyTorch
or Diffusers. This checkpoint supports text-to-image generation.

```python
from mlx_vlm.generate import generate_image, load_image_generation_model

model = load_image_generation_model("/path/to/this/checkpoint")
result = generate_image(model, "A red ceramic teapot on a wooden table",
                        width=1024, height=1024, seed=42)
result.save("moss.png")
```

```sh
mlx_vlm.generate --output-modality image --model /path/to/this/checkpoint \\
  --prompt "A red ceramic teapot on a wooden table" \\
  --size 1024x1024 --seed 42 --output moss.png
```

Defaults: 50 steps, guidance 3.0, and block 22 skipped in the negative branch.
Both image dimensions must be positive multiples of 16. The VAE uses fp32.
The text encoder is released before loading the diffusion transformer.
"""


def convert_moss_image(model_path, output_path, *, dtype=None, source_id=None):
    source, destination = Path(model_path).expanduser(), Path(output_path).expanduser()
    if not detect_moss_image_layout(source):
        raise ValueError(f"Not a MOSS-Image checkpoint: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("MOSS-Image conversion output must differ from its source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"MOSS-Image conversion output is not empty: {destination}"
        )
    if dtype not in (None, "bfloat16", "float16", "float32"):
        raise ValueError(f"Unsupported MOSS-Image dtype: {dtype}")
    destination.mkdir(parents=True, exist_ok=True)
    index = read_config(source / "model_index.json")
    if source_id is None:
        source_id = index.get("source_model")
    if source_id is not None:
        index["source_model"] = source_id
    index["mlx_vlm_format"] = "moss_image"
    (destination / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    for name in ("tokenizer", "scheduler"):
        target = destination / name
        target.mkdir()
        for file in sorted((source / name).iterdir()):
            if file.is_file() and file.suffix in {".json", ".jinja", ".txt", ".model"}:
                shutil.copy2(file, target / file.name)
    if (source / "LICENSE").exists():
        shutil.copy2(source / "LICENSE", destination / "LICENSE")
    (destination / "README.md").write_text(model_card(source_id))
    for name, loader in (
        ("text_encoder", load_text_encoder),
        ("transformer", load_transformer),
        ("vae", load_vae),
    ):
        print(f"[INFO] Converting MOSS-Image {name}", flush=True)
        model = loader(source)
        if dtype is not None and name != "vae":
            # Preserve the recurrent decay parameters in fp32, like the
            # original Qwen checkpoint. The VAE always decodes in fp32.
            model.load_weights(
                [
                    (
                        key,
                        (
                            value.astype(getattr(mx, dtype))
                            if mx.issubdtype(value.dtype, mx.floating)
                            and not key.endswith((".A_log", ".dt_bias"))
                            else value
                        ),
                    )
                    for key, value in tree_flatten(model.parameters())
                ]
            )
        config = read_config(source / name / "config.json")
        config["mlx_vlm_format"] = "moss_image"
        if name == "vae":
            config["dtype"] = "float32"
        elif dtype is not None:
            config["dtype"] = dtype
            if name == "text_encoder":
                config["text_config"]["dtype"] = dtype
        if name == "text_encoder":
            config.pop("vision_config", None)
            config["architectures"] = ["MOSSImageTextEncoder"]
        elif name == "vae":
            config["include_encoder"] = False
        component_path = destination / name
        save_weights(component_path, model, donate_weights=True)
        (component_path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        del model
        gc.collect()
        mx.clear_cache()
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Convert MOSS-Image to a standalone MLX checkpoint"
    )
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--source-id",
        help="Original Hugging Face model ID for a local checkpoint's model card.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    args = parser.parse_args()
    source = get_model_path(
        args.hf_path,
        revision=args.revision,
        allow_patterns=[*IMAGE_DOWNLOAD_PATTERNS, "LICENSE"],
    )
    convert_moss_image(
        source,
        args.mlx_path,
        dtype=args.dtype,
        source_id=(
            args.source_id
            or (args.hf_path if not Path(args.hf_path).exists() else None)
        ),
    )


if __name__ == "__main__":
    main()
