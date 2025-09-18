from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Iterable

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


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "scan":
        sys.exit(cli_scan(sys.argv[2]))
    print("Usage:")
    print("  python -m mlx_vlm.convert.convert_dots_ocr scan /path/to/model_or_dir")
    sys.exit(1)
