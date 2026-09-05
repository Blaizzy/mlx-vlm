"""Build a proposal-only compact head from a quantized Qwen LM head."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
from safetensors import safe_open

from ....utils import get_model_path
from .compact_head import (
    COMPACT_HEAD_MODEL_TYPE,
    COMPACT_HEAD_SCHEMA_VERSION,
    CompactProposalHead,
    tokenizer_vocab_sha256,
    vocab_ids_sha256,
)

_TENSOR_NAMES = ("weight", "scales", "biases")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint_files(model_path: Path) -> List[Path]:
    return sorted(
        path.resolve()
        for path in model_path.glob("*.safetensors")
        if path.name != "consolidated.safetensors"
    )


def _tensor_file_map(model_path: Path) -> Dict[str, Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("model.safetensors.index.json has no weight_map")
        root = model_path.resolve()
        result = {}
        for key, value in weight_map.items():
            if not isinstance(value, str) or Path(value).is_absolute():
                raise ValueError("checkpoint index contains an invalid shard path")
            path = (root / value).resolve()
            if not path.is_relative_to(root):
                raise ValueError("checkpoint index shard escapes the model directory")
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint shard does not exist: {path}")
            result[key] = path
        return result

    result: Dict[str, Path] = {}
    for path in _checkpoint_files(model_path):
        with safe_open(path, framework="mlx") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"checkpoint tensor appears more than once: {key}")
                result[key] = path
    return result


def _find_affine_lm_head(tensor_files: Dict[str, Path]) -> Dict[str, Tuple[str, Path]]:
    candidates = []
    for key in tensor_files:
        if key != "lm_head.weight" and not key.endswith(".lm_head.weight"):
            continue
        prefix = key[: -len("weight")]
        keys = {name: prefix + name for name in _TENSOR_NAMES}
        if all(name in tensor_files for name in keys.values()):
            candidates.append(keys)
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one affine LM head with weight, scales, and biases"
        )
    keys = candidates[0]
    return {name: (key, tensor_files[key]) for name, key in keys.items()}


def _load_head_tensors(
    head_tensors: Dict[str, Tuple[str, Path]],
) -> Dict[str, mx.array]:
    by_file: Dict[Path, List[Tuple[str, str]]] = {}
    for name, (key, path) in head_tensors.items():
        by_file.setdefault(path, []).append((name, key))

    loaded: Dict[str, mx.array] = {}
    for path, items in by_file.items():
        try:
            with safe_open(path, framework="mlx") as handle:
                loaded.update(
                    {name: mx.array(handle.get_tensor(key)) for name, key in items}
                )
        except (AttributeError, RuntimeError, TypeError):
            shard = mx.load(str(path))
            loaded.update({name: shard[key] for name, key in items})
    return loaded


def _validate_vocab_ids(vocab_ids: Sequence[int], vocab_size: int) -> List[int]:
    values = list(vocab_ids)
    if not values:
        raise ValueError("compact vocabulary must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("compact vocabulary IDs must be integers")
    if len(values) != len(set(values)):
        raise ValueError("compact vocabulary IDs must be unique")
    values.sort()
    if values[0] < 0 or values[-1] >= vocab_size:
        raise ValueError("compact vocabulary ID exceeds target vocabulary")
    return values


def select_vocab_ids(
    *,
    prefix_size: int,
    added_token_ids: Sequence[int],
    vocab_size: int,
    row_multiple: int = 32,
) -> List[int]:
    """Select a low-ID prefix plus added tokens and shape-alignment rows."""
    if prefix_size <= 0 or prefix_size >= vocab_size:
        raise ValueError("vocabulary prefix must be between zero and vocab size")
    if row_multiple <= 0:
        raise ValueError("row multiple must be positive")
    selected = list(range(prefix_size))
    selected.extend(
        sorted({token_id for token_id in added_token_ids if token_id >= prefix_size})
    )
    selected_set = set(selected)
    next_id = prefix_size
    while len(selected) % row_multiple:
        while next_id in selected_set:
            next_id += 1
        if next_id >= vocab_size:
            raise ValueError("cannot align compact rows within target vocabulary")
        selected.append(next_id)
        selected_set.add(next_id)
    return _validate_vocab_ids(selected, vocab_size)


def _resolved_revision(model_path: Path) -> Optional[str]:
    parts = model_path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )


def _resolve_model_path(
    source: str,
    *,
    revision: Optional[str],
    subdir: Optional[str],
    force_download: bool,
) -> Tuple[Path, Path]:
    allow_patterns = None
    if subdir:
        allow_patterns = [
            f"{subdir}/*.json",
            f"{subdir}/*.safetensors",
            f"{subdir}/*.model",
            f"{subdir}/*.tiktoken",
        ]
    source_root = get_model_path(
        source,
        revision=revision,
        force_download=force_download,
        allow_patterns=allow_patterns,
    )
    return source_root, source_root / subdir if subdir else source_root


def build_compact_proposal_head(
    source: str | Path,
    output: str | Path,
    *,
    vocab_ids: Sequence[int],
    revision: Optional[str] = None,
    subdir: Optional[str] = None,
    force_download: bool = False,
    tokenizer=None,
) -> Path:
    """Copy selected packed affine rows into a compact proposal-head sidecar."""
    source_string = str(source)
    source_is_local = Path(source_string).exists()
    source_root, model_path = _resolve_model_path(
        source_string,
        revision=revision,
        subdir=subdir,
        force_download=force_download,
    )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"source model has no config.json: {model_path}")

    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config") or config
    model_type = text_config.get("model_type") or config.get("model_type")
    if not model_type or not str(model_type).startswith(("qwen3_5", "qwen3_next")):
        raise ValueError("compact proposal builder supports Qwen3.5-family models")
    hidden_size = int(text_config["hidden_size"])
    vocab_size = int(text_config["vocab_size"])
    selected = _validate_vocab_ids(vocab_ids, vocab_size)

    tensor_files = _tensor_file_map(model_path)
    head_tensors = _find_affine_lm_head(tensor_files)
    source_tensors = _load_head_tensors(head_tensors)
    weight = source_tensors["weight"]
    scales = source_tensors["scales"]
    biases = source_tensors["biases"]
    if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
        raise ValueError("source affine LM-head tensors must be rank 2")
    if weight.dtype != mx.uint32:
        raise ValueError("source affine LM-head weight must use packed uint32 values")
    if scales.shape != biases.shape or int(scales.shape[0]) != int(weight.shape[0]):
        raise ValueError("source affine LM-head tensor shapes do not match")
    if int(weight.shape[0]) < vocab_size:
        raise ValueError("source LM head has fewer rows than the configured vocabulary")

    packed_bits = int(weight.shape[1]) * 32
    if packed_bits % hidden_size:
        raise ValueError("cannot infer source LM-head bit width")
    bits = packed_bits // hidden_size
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported source LM-head bit width: {bits}")
    if int(scales.shape[1]) <= 0 or hidden_size % int(scales.shape[1]):
        raise ValueError("cannot infer source LM-head group size")
    group_size = hidden_size // int(scales.shape[1])
    quantization = config.get("quantization") or text_config.get("quantization") or {}
    if quantization.get("mode", "affine") != "affine":
        raise ValueError("source LM head must use affine quantization")
    if "bits" in quantization and int(quantization["bits"]) != bits:
        raise ValueError("source LM-head bit width differs from config")
    if "group_size" in quantization and int(quantization["group_size"]) != group_size:
        raise ValueError("source LM-head group size differs from config")

    selected_array = mx.array(selected, dtype=mx.int32)
    compact = {
        "weight": mx.take(weight, selected_array, axis=0),
        "scales": mx.take(scales, selected_array, axis=0),
        "biases": mx.take(biases, selected_array, axis=0),
        "vocab_ids": selected_array,
    }
    mx.eval(*compact.values())

    if tokenizer is None:
        tokenizer = _load_tokenizer(model_path)
    tokenizer_hash = tokenizer_vocab_sha256(tokenizer)
    CompactProposalHead(
        **compact,
        group_size=group_size,
        bits=bits,
        mode="affine",
        full_vocab_size=vocab_size,
        target_model_type=str(model_type),
        target_tokenizer_sha256=tokenizer_hash,
    )

    source_paths = sorted({path for _, path in head_tensors.values()})
    source_kind = "local" if source_is_local else "huggingface"
    sidecar_config = {
        "bits": bits,
        "group_size": group_size,
        "hidden_size": hidden_size,
        "mode": "affine",
        "model_type": COMPACT_HEAD_MODEL_TYPE,
        "purpose": "proposal-only greedy MTP projection; never target verification",
        "rows": len(selected),
        "schema_version": COMPACT_HEAD_SCHEMA_VERSION,
        "selection": {
            "mapping_sha256": vocab_ids_sha256(selected),
            "maximum_token_id": selected[-1],
            "minimum_token_id": selected[0],
            "ordering": "ascending_token_id",
        },
        "source": {
            "config_sha256": _sha256_file(config_path),
            "kind": source_kind,
            "model": None if source_is_local else source_string,
            "requested_revision": revision,
            "resolved_revision": _resolved_revision(source_root),
            "subdir": subdir,
            "tensor_files": [
                {
                    "path": path.relative_to(model_path.resolve()).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in source_paths
            ],
            "tensor_keys": {name: key for name, (key, _) in head_tensors.items()},
        },
        "target": {
            "model_type": str(model_type),
            "tokenizer_vocab_sha256": tokenizer_hash,
            "vocab_size": vocab_size,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        weights_path = temporary / "compact_head.safetensors"
        mx.save_safetensors(
            str(weights_path),
            compact,
            metadata={
                "format": "mlx",
                "purpose": "proposal-only",
            },
        )
        _write_json(temporary / "config.json", sidecar_config)
        _write_json(
            temporary / "manifest.json",
            [
                {
                    "bytes": (temporary / name).stat().st_size,
                    "path": name,
                    "sha256": _sha256_file(temporary / name),
                }
                for name in ("compact_head.safetensors", "config.json")
            ],
        )
        temporary.rename(output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def _read_vocab_ids(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--vocab-ids must contain a JSON array")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--subdir")
    parser.add_argument("--force-download", action="store_true")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--vocab-ids", type=Path)
    selection.add_argument("--vocab-prefix", type=int)
    parser.add_argument("--row-multiple", type=int, default=32)
    parser.add_argument(
        "--include-added-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include tokenizer-added IDs with --vocab-prefix (default: true).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, model_path = _resolve_model_path(
        args.source,
        revision=args.revision,
        subdir=args.subdir,
        force_download=args.force_download,
    )
    tokenizer = _load_tokenizer(model_path)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config") or config
    vocab_size = int(text_config["vocab_size"])
    if args.vocab_ids is not None:
        vocab_ids = _read_vocab_ids(args.vocab_ids)
    else:
        added_ids = (
            list(tokenizer.get_added_vocab().values())
            if args.include_added_tokens
            else []
        )
        vocab_ids = select_vocab_ids(
            prefix_size=args.vocab_prefix,
            added_token_ids=added_ids,
            vocab_size=vocab_size,
            row_multiple=args.row_multiple,
        )
    output = build_compact_proposal_head(
        args.source,
        args.output,
        vocab_ids=vocab_ids,
        revision=args.revision,
        subdir=args.subdir,
        force_download=False,
        tokenizer=tokenizer,
    )
    print(f"Wrote compact Qwen proposal head to {output}")


if __name__ == "__main__":
    main()
