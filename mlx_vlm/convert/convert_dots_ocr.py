from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Iterable

import re

import numpy as np
from safetensors import safe_open

HERE = pathlib.Path(__file__).parent


def iter_safetensors(path: str | os.PathLike) -> Iterable[str]:
    p = pathlib.Path(path)
    if p.is_file() and p.suffix == ".safetensors":
        yield str(p)
    elif p.is_dir():
        for f in sorted(p.glob("*.safetensors")):
            yield str(f)
    else:
        raise FileNotFoundError(f"No .safetensors found at {path}")


def list_vision_keys(st_path: str | os.PathLike) -> list[str]:
    keys = []
    with safe_open(st_path, framework="np") as f:
        for k in f.keys():
            if k.startswith("vision_tower"):
                keys.append(k)
    return keys


def cli_scan(target: str):
    all_keys: set[str] = set()
    files = list(iter_safetensors(target))
    if not files:
        print(f"[scan] no .safetensors under {target}")
        return 1
    for f in files:
        ks = list_vision_keys(f)
        print(f"[scan] {os.path.basename(f)}: {len(ks)} vision keys")
        all_keys.update(ks)
    inv = sorted(all_keys)
    print(json.dumps({"files": files, "unique_vision_keys": len(inv)}, indent=2))
    for k in inv[:10]:
        print("  ", k)
    return 0


# --------- Key mapping ---------


def map_key_hf_to_mlx(hf_key: str) -> str | None:
    if not hf_key.startswith("vision_tower."):
        return None
    k = hf_key.replace("vision_tower.", "vision.")
    k = k.replace("patch_embed.patchifier.", "patch.")
    k = k.replace("post_trunk_norm.", "post.")
    k = re.sub(r"^vision.blocks.(\d+).", r"vision.blocks.\1.", k)

    allowed_suffixes = (
        "patch.proj.weight",
        "patch.norm.weight",
        "post.weight",
        "attn.qkv.weight",
        "attn.proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
        "mlp.fc3.weight",
        "norm1.weight",
        "norm2.weight",
        "merger.ln.weight",
        "merger.mlp.0.weight",
        "merger.mlp.2.weight",
        "merger.mlp.0.bias",
        "merger.mlp.2.bias",
    )
    if any(k.endswith(s) for s in allowed_suffixes):
        return k
    return None


def convert_dir_or_file_to_npz(target: str, out_npz: str) -> dict:
    arrays: dict[str, np.ndarray] = {}
    num_skipped = 0
    files = list(iter_safetensors(target))
    for st in files:
        with safe_open(st, framework="np") as handle:
            for hf_key in handle.keys():
                if not hf_key.startswith("vision_tower"):
                    continue
                mlx_key = map_key_hf_to_mlx(hf_key)
                if mlx_key is None:
                    num_skipped += 1
                    continue
                arrays[mlx_key] = np.array(handle.get_tensor(hf_key))
    if not arrays:
        raise RuntimeError(f"No mapped vision weights found under {target}")
    out_dir = os.path.dirname(out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(out_npz, **arrays)
    return {"written": out_npz, "tensors": len(arrays), "skipped": num_skipped}


def cli_convert(target: str, out_npz: str):
    info = convert_dir_or_file_to_npz(target, out_npz)
    print(
        f"[convert] wrote {info['tensors']} tensors to {info['written']}"
        f" (skipped {info['skipped']})"
    )
    return 0


def preview_npz(npz_path: str, limit: int = 12):
    import numpy as np

    z = np.load(npz_path)
    print(f"[preview] {npz_path}: {len(z.files)} tensors")
    for name in sorted(z.files)[:limit]:
        arr = z[name]
        print(f"  {name:48s} {arr.shape} {arr.dtype}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "scan":
        sys.exit(cli_scan(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "to-npz":
        sys.exit(cli_convert(sys.argv[2], sys.argv[3]))
    if len(sys.argv) >= 3 and sys.argv[1] == "preview":
        sys.exit(preview_npz(sys.argv[2]))
    print("Usage:")
    print("  python -m mlx_vlm.convert.convert_dots_ocr scan /path/to/model_or_dir")
    print(
        "  python -m mlx_vlm.convert.convert_dots_ocr to-npz /path/to/model_or_dir "
        "weights/dots_ocr_vision.npz"
    )
    print(
        "  python -m mlx_vlm.convert.convert_dots_ocr preview "
        "weights/dots_ocr_vision.npz"
    )
    sys.exit(1)
