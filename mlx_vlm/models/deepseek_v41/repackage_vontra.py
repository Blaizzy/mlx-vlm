"""Repackage Vontra's 2-bit MLX checkpoint into mlx-vlm-native layout.

Vontra's tensors already use upstream key names and MLX-native 2-bit affine
packing, so this is a pure namespace move (``layers.*`` and friends gain the
``language_model.`` prefix via :meth:`Model.sanitize`) plus our config,
tokenizer, and engram map. No precision changes: bit-identical weights.

Usage (runs on the conversion host, streams shard by shard)::

    python -m mlx_vlm.models.deepseek_v41.repackage_vontra \\
        --source /Users/neywa/DeepSeek-V4.1-Flash-MLX-2bit-MTP \\
        --output /Users/neywa/DeepSeek-V4.1-Flash-MLX-2bit \\
        --engram-map /Users/neywa/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit/engram_token_map.json
"""

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from .config import ModelConfig
from .deepseek_v41 import Model


def _read_index(source: Path) -> Dict[str, str]:
    with open(source / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def output_config(
    quantization: Optional[dict] = None,
) -> dict:
    """Our load config: full model config plus the 2-bit affine recipe."""
    config = ModelConfig().to_dict()
    config["model_type"] = "deepseek_v41"
    recipe = {"group_size": 64, "bits": 2, "mode": "affine"}
    if quantization is not None:
        recipe.update(
            {
                k: quantization[k]
                for k in ("group_size", "bits", "mode")
                if k in quantization
            }
        )
    config["quantization"] = dict(recipe)
    config["quantization_config"] = dict(recipe)
    return config


def repackage_vontra(
    source: str,
    output: str,
    *,
    engram_map: Optional[str] = None,
) -> Path:
    """Copy Vontra shards into our layout with our config. Returns output dir."""
    source_path = Path(source)
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"Output is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    weight_map = _read_index(source_path)
    by_file: Dict[str, list] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(filename, []).append(key)

    model = Model(ModelConfig())
    output_map: dict[str, str] = {}
    total_size = 0
    for filename in sorted(by_file):
        tensors = dict(mx.load(str(source_path / filename)))
        renamed = model.sanitize(tensors)
        for key, value in tensors.items():
            if key.startswith("mtp."):
                renamed[key] = value
        if len(renamed) != len(tensors):
            raise ValueError(f"Key collision repackaging {filename}.")
        mx.save_safetensors(
            str(output_path / filename),
            renamed,
            metadata={"format": "mlx"},
        )
        for key, value in renamed.items():
            output_map[key] = filename
            total_size += value.nbytes
        del tensors, renamed
        gc.collect()
        mx.clear_cache()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_map.items())),
    }
    (output_path / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    with open(output_path / "config.json", "w") as f:
        json.dump(output_config(), f, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = source_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)
    if engram_map is not None:
        shutil.copy2(engram_map, output_path / "engram_token_map.json")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repackage Vontra 2-bit weights into mlx-vlm-native layout."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engram-map", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    output = repackage_vontra(**vars(args))
    print(f"Wrote repackaged model to {output}")


if __name__ == "__main__":
    main()
