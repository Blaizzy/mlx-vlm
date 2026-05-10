import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import mlx.core as mx
from safetensors import safe_open

from ....utils import get_model_path
from .qwen3_5_mtp import Qwen3_5MTPDraftModel


def _safetensor_files(model_path: Path) -> list[Path]:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        data = json.load(f)
    return data.get("weight_map", {})


def _iter_mtp_keys(model_path: Path) -> Iterable[tuple[Path, list[str]]]:
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith("mtp."):
                by_file.setdefault(filename, []).append(key)
        if by_file:
            for filename, keys in by_file.items():
                yield model_path / filename, keys
            return

    for file in _safetensor_files(model_path):
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if key.startswith("mtp.")]
        if keys:
            yield file, keys


def _is_mlx_safetensors(file: Path) -> bool:
    with safe_open(file, framework="mlx") as f:
        metadata = f.metadata() or {}
    return metadata.get("format") == "mlx"


def _load_selected_tensors(file: Path, keys: list[str]) -> Dict[str, mx.array]:
    tensors = {}
    with safe_open(file, framework="mlx") as f:
        for key in keys:
            tensors[key] = mx.array(f.get_tensor(key))
    return tensors


def split_qwen3_5_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write Qwen3.5 native MTP tensors into a standalone drafter folder."""
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = source_path / "config.json"
    with open(config_path) as f:
        source_config = json.load(f)

    text_config = dict(source_config.get("text_config") or {})
    if not text_config:
        raise ValueError(f"{source_path} does not contain a text_config.")

    selected = {}
    source_is_mlx = False
    for file, keys in _iter_mtp_keys(source_path):
        source_is_mlx = source_is_mlx or _is_mlx_safetensors(file)
        selected.update(_load_selected_tensors(file, keys))

    if not selected:
        raise ValueError(f"No mtp.* tensors found in {source_path}.")

    if not source_is_mlx:
        selected = Qwen3_5MTPDraftModel.sanitize(None, selected)
    else:
        selected = {
            key[len("mtp.") :] if key.startswith("mtp.") else key: value
            for key, value in selected.items()
        }

    mx.save_safetensors(
        str(output_path / "model.safetensors"),
        selected,
        metadata={"format": "mlx"},
    )

    draft_config = {
        "model_type": "qwen3_5_mtp",
        "text_config": text_config,
        "block_size": int(
            block_size or text_config.get("mtp_num_hidden_layers", 1) + 2
        ),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", True)),
    }
    if any(key.endswith(".scales") for key in selected):
        quantization = source_config.get("mtplx_mtp_quantization")
        if quantization is None:
            quantization = source_config.get("quantization")
        if quantization is not None:
            draft_config["quantization"] = quantization
            draft_config["quantization_config"] = quantization

    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Qwen3.5 native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_qwen3_5_mtp(**vars(args))
    print(f"Wrote Qwen3.5 MTP drafter to {output}")


if __name__ == "__main__":
    main()
