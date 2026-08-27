import argparse
import glob
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import mlx.core as mx
from safetensors import safe_open

from ....models.longcat_flash_sparse.config import ModelConfig as LongcatSparseConfig
from ....utils import get_model_path
from .longcat_flash_sparse_mtp import LongcatFlashSparseMTPDraftModel

_MTP_PREFIXES = ("model.mtp.", "mtp.")


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        return json.load(f).get("weight_map", {})


def _is_mtp_key(key: str) -> bool:
    return any(key.startswith(p) for p in _MTP_PREFIXES)


def _iter_mtp_keys(model_path: Path) -> Iterable[tuple[Path, list[str]]]:
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if _is_mtp_key(key):
                by_file.setdefault(filename, []).append(key)
        for filename, keys in by_file.items():
            yield model_path / filename, keys
        return

    for path in glob.glob(str(model_path / "*.safetensors")):
        file = Path(path)
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if _is_mtp_key(key)]
        if keys:
            yield file, keys


def _load_selected_tensors(file: Path, keys: list[str]) -> Dict[str, mx.array]:
    try:
        with safe_open(file, framework="mlx") as f:
            return {key: mx.array(f.get_tensor(key)) for key in keys}
    except TypeError:
        shard = mx.load(str(file))
        return {key: shard[key] for key in keys}


def split_longcat_flash_sparse_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write LongCat-Flash-Lite-Sparse native MTP tensors into a standalone drafter."""
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)
    text_config = dict(source_config.get("text_config") or source_config)
    text_config.setdefault("model_type", "longcat_flash_sparse")

    selected: Dict[str, mx.array] = {}
    for file, keys in _iter_mtp_keys(source_path):
        selected.update(_load_selected_tensors(file, keys))
    if not selected:
        raise ValueError(f"No mtp.* tensors found in {source_path}.")

    ctx = SimpleNamespace(args=LongcatSparseConfig.from_dict(text_config))
    selected = LongcatFlashSparseMTPDraftModel.sanitize(ctx, selected)

    mx.save_safetensors(
        str(output_path / "model.safetensors"), selected, metadata={"format": "mlx"}
    )

    draft_config = {
        "model_type": "longcat_flash_sparse_mtp",
        "text_config": text_config,
        "block_size": int(block_size or text_config.get("mtp_num_layers", 3) + 1),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
    }
    if any(key.endswith(".scales") for key in selected):
        quantization = source_config.get("quantization")
        if quantization is not None:
            draft_config["quantization"] = quantization
            draft_config["quantization_config"] = quantization
    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json", "tokenization_llama.py"):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split LongCat-Flash-Lite-Sparse MTP tensors into an MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_longcat_flash_sparse_mtp(**vars(args))
    print(f"Wrote LongCat-Flash-Lite-Sparse MTP drafter to {output}")


if __name__ == "__main__":
    main()
