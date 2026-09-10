"""Convert a facebook/sapiens2 HF checkpoint (bare .safetensors) into an mlx-vlm
model directory (config.json + sanitized safetensors).

Usage:
  python -m mlx_vlm.models.sapiens2.convert \
      --hf-repo facebook/sapiens2-seg-0.4b \
      --out ./sapiens2-seg-0.4b-mlx \
      --dtype bfloat16

The HF repos ship as a single safetensors file with no accompanying config —
task + size are inferred from the repo name (or supplied via CLI flags).
"""

import argparse
import dataclasses
import json
import re
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from .config import SIZES, TASKS, ModelConfig
from .sapiens2 import Model


_NAME_PATTERN = re.compile(
    r"sapiens2-(?P<task>pose|seg|normal|pointmap)-(?P<size>[0-9.]+b(?:-4k)?)"
)


def infer_task_size(repo_id: str):
    m = _NAME_PATTERN.search(repo_id)
    if not m:
        raise ValueError(
            f"Could not infer task/size from '{repo_id}'. "
            "Pass --task and --size explicitly."
        )
    return m["task"], m["size"]


def _asdict_maybe_config(obj):
    if dataclasses.is_dataclass(obj):
        return {
            k: _asdict_maybe_config(v)
            for k, v in dataclasses.asdict(obj).items()
        }
    if isinstance(obj, (list, tuple)):
        return [_asdict_maybe_config(x) for x in obj]
    return obj


def build_config_dict(task: str, size: str) -> dict:
    cfg = ModelConfig(task=task, size=size)
    d = _asdict_maybe_config(cfg)
    # Housekeeping: drop framework-compat Nones that confuse JSON readers
    d.pop("text_config", None)
    d.pop("vision_config", None)
    return d


def convert(
    hf_repo: str,
    out_dir: Path,
    dtype: str = "bfloat16",
    task: str = None,
    size: str = None,
):
    if task is None or size is None:
        inferred_task, inferred_size = infer_task_size(hf_repo)
        task = task or inferred_task
        size = size or inferred_size
    assert task in TASKS, f"unknown task {task}"
    assert size in SIZES, f"unknown size {size}"

    print(f"[convert] repo={hf_repo} task={task} size={size} dtype={dtype}")
    local = Path(snapshot_download(hf_repo))
    st_files = list(local.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors in {local}")
    print(f"[convert] source: {st_files[0]}")

    weights = mx.load(str(st_files[0]))
    sanitized = Model.sanitize(weights)

    # Cast to target dtype
    target_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[dtype]
    cast = {k: v.astype(target_dtype) if v.dtype != target_dtype else v
            for k, v in sanitized.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mx.save_safetensors(str(out_dir / "model.safetensors"), cast)
    cfg_dict = build_config_dict(task=task, size=size)
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    # Minimal preprocessor config (1024 x 768, ImageNet stats on [0, 255])
    preproc = {
        "image_size": [1024, 768],
        "image_mean": [123.675, 116.28, 103.53],
        "image_std": [58.395, 57.12, 57.375],
    }
    (out_dir / "preprocessor_config.json").write_text(json.dumps(preproc, indent=2))

    print(f"[convert] wrote {len(cast)} tensors to {out_dir}/model.safetensors")
    print(f"[convert] config: {out_dir}/config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", required=True, help="e.g. facebook/sapiens2-seg-0.4b")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--task", default=None, help="override task inference")
    ap.add_argument("--size", default=None, help="override size inference")
    args = ap.parse_args()
    convert(args.hf_repo, args.out, args.dtype, args.task, args.size)


if __name__ == "__main__":
    main()
