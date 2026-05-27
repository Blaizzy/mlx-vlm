import argparse
import glob
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import mlx.core as mx
from safetensors import safe_open

from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config
from ....utils import _load_safetensors, get_model_path
from .deepseek_v4_mtp import DeepseekV4MTPDraftModel


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


def _load_selected_tensors(file: Path, keys: list[str]) -> Dict[str, mx.array]:
    tensors = {}
    try:
        with safe_open(file, framework="mlx") as f:
            for key in keys:
                tensors[key] = mx.array(f.get_tensor(key))
    except (AttributeError, RuntimeError, TypeError):
        shard = _load_safetensors(str(file))
        tensors = {key: shard[key] for key in keys}
    return tensors


def _text_config(source_config: dict) -> dict:
    return dict(source_config.get("text_config") or source_config)


def _module_from_scales_key(key: str) -> str:
    return key[: -len(".scales")]


def _quantization_from_weights(weights: Dict[str, mx.array]) -> Optional[dict]:
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {"group_size": 64, "bits": 8, "mode": "affine"}

    for key in weights:
        if not key.endswith(".scales"):
            continue
        module = _module_from_scales_key(key)
        if "decoder.ffn.switch_mlp." in module and module.endswith("_proj"):
            quantization[module] = mxfp4
        elif (
            module in ("e_proj", "h_proj")
            or "decoder.ffn.shared_experts." in module
            or "decoder.attn.w" in module
        ):
            quantization[module] = mxfp8

    return quantization if len(quantization) > 3 else None


def split_deepseek_v4_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write DeepSeek-V4 native MTP tensors into a standalone drafter folder."""
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = source_path / "config.json"
    with open(config_path) as f:
        source_config = json.load(f)
    text_config = _text_config(source_config)

    selected = {}
    for file, keys in _iter_mtp_keys(source_path):
        selected.update(_load_selected_tensors(file, keys))

    if not selected:
        raise ValueError(f"No mtp.* tensors found in {source_path}.")

    sanitize_context = SimpleNamespace(args=DeepseekV4Config.from_dict(text_config))
    selected = DeepseekV4MTPDraftModel.sanitize(sanitize_context, selected)

    mx.save_safetensors(
        str(output_path / "model.safetensors"),
        selected,
        metadata={"format": "mlx"},
    )

    depth = int(text_config.get("num_nextn_predict_layers", 1))
    draft_config = {
        "model_type": "deepseek_v4_mtp",
        "text_config": text_config,
        "block_size": int(block_size or depth + 1),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
    }
    quantization = _quantization_from_weights(selected)
    if quantization is not None:
        draft_config["quantization"] = quantization
        draft_config["quantization_config"] = quantization

    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split DeepSeek-V4 native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_deepseek_v4_mtp(**vars(args))
    print(f"Wrote DeepSeek-V4 MTP drafter to {output}")


if __name__ == "__main__":
    main()
