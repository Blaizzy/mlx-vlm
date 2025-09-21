from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Iterable

import re

import numpy as np
from safetensors import deserialize, safe_open

HERE = pathlib.Path(__file__).parent

_DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": "BF16",
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
}


def iter_safetensors(path: str | os.PathLike) -> Iterable[str]:
    """Recursively yield all safetensors files under path."""

    p = pathlib.Path(path)
    if p.is_file() and p.suffix == ".safetensors":
        yield str(p)
        return
    if p.is_dir():
        for f in sorted(p.rglob("*.safetensors")):
            yield str(f)
        return
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


def _bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    raw_uint32 = raw.astype(np.uint32) << 16
    return raw_uint32.view(np.float32)


def _tensor_to_numpy(data: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    np_dtype = _DTYPE_MAP.get(dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported dtype {dtype}")
    if np_dtype == "BF16":
        raw = np.frombuffer(data, dtype=np.uint16)
        arr = _bf16_to_float32(raw)
    else:
        arr = np.frombuffer(data, dtype=np_dtype)
    arr = arr.reshape(shape)
    if arr.dtype in (np.float16, np.float64) or np_dtype == "BF16":
        arr = arr.astype(np.float32, copy=False)
    return arr.copy()


def convert_dir_or_file_to_npz(target: str, out_npz: str) -> dict:
    arrays: dict[str, np.ndarray] = {}
    num_skipped = 0
    files = list(iter_safetensors(target))
    for st in files:
        with open(st, "rb") as handle:
            content = handle.read()
        for hf_key, tensor in deserialize(content):
            if not hf_key.startswith("vision_tower"):
                continue
            mlx_key = map_key_hf_to_mlx(hf_key)
            if mlx_key is None:
                num_skipped += 1
                continue
            arr = _tensor_to_numpy(tensor["data"], tensor["dtype"], tensor["shape"])
            arrays[mlx_key] = arr
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


def scan_text_prefixes(target: str, limit: int = 40):
    import collections

    counter = collections.Counter()
    files = list(iter_safetensors(target))
    if not files:
        print(f"[text-scan] no .safetensors under {target}")
        return []
    for st_path in files:
        with safe_open(st_path, framework="np") as handle:
            for key in handle.keys():
                if key.startswith("vision_tower"):
                    continue
                top = key.split(".", 1)[0]
                counter[top] += 1
    tops = counter.most_common()
    print("[text-scan] top prefixes:", tops[:limit])
    return tops


def discover_projector(target: str):
    """Find potential projector weights (vision -> text adapters)."""

    candidates: list[tuple[str, tuple[int, ...]]] = []
    files = list(iter_safetensors(target))
    if not files:
        print(f"[projector] no .safetensors under {target}")
        return []

    for st_path in files:
        with safe_open(st_path, framework="np") as handle:
            for key in handle.keys():
                if "vision_tower" in key:
                    continue
                if any(p in key for p in ("proj", "projector", "vision_proj", "visual_proj")):
                    try:
                        tensor = handle.get_tensor(key)
                        candidates.append((key, tuple(tensor.shape)))
                    except Exception:
                        continue

    candidates = sorted(set(candidates), key=lambda x: x[0])
    print("[projector] candidates:", candidates[:20])
    return candidates


def iter_tensor_entries(target: str):
    for st_path in iter_safetensors(target):
        with open(st_path, "rb") as handle:
            data = handle.read()
        for key, tensor in deserialize(data):
            yield st_path, key, tensor


def sample_model_keys(target: str, limit: int = 100) -> list[str]:
    keys: list[str] = []
    for _, key, _ in iter_tensor_entries(target):
        if key.startswith("model."):
            keys.append(key)
            if len(keys) >= limit:
                break
    return keys


def infer_hidden_size_and_projector(target: str):
    hidden_size = None
    style = "unknown"
    projectors: list[tuple[str, tuple[int, ...]]] = []

    for _, key, tensor in iter_tensor_entries(target):
        if key.endswith("model.embed_tokens.weight"):
            hidden_size = int(tensor["shape"][1])
            break

    keys = sample_model_keys(target, limit=80)
    if any(".self_attn.q_proj.weight" in k for k in keys):
        style = "llama"
    if any(".attention.wq.weight" in k for k in keys):
        style = "qwen"
    if any(".attention.query_key_value.weight" in k for k in keys):
        style = "phi"

    vision_dim = 1536
    exclude = (".attention.", ".attn.", ".mlp.", ".norm", "embed_tokens", "lm_head")
    hints = (
        "project",
        "vision",
        "visual",
        "connector",
        "mm_",
        "multi_modal",
        "image",
    )

    for _, key, tensor in iter_tensor_entries(target):
        if not key.startswith("model."):
            continue
        if any(token in key for token in exclude):
            continue
        if hidden_size is None:
            continue
        shape = tuple(int(x) for x in tensor["shape"])
        if len(shape) != 2:
            continue
        shape_matches = shape in ((hidden_size, vision_dim), (vision_dim, hidden_size))
        name_matches = any(h in key.lower() for h in hints)
        if shape_matches and name_matches:
            projectors.append((key, shape))

    identity_ok = hidden_size == vision_dim and not projectors
    info = {
        "hidden_size": hidden_size,
        "style": style,
        "projectors": projectors,
        "identity_ok": identity_ok,
    }
    print(info)
    return info


def convert_projector_to_npz(target: str, out_npz: str):
    import numpy as np

    info = infer_hidden_size_and_projector(target)
    hidden_size = info["hidden_size"]
    projectors = info["projectors"]
    identity_ok = info["identity_ok"]

    if projectors:
        key, _ = projectors[0]
        array = None
        for _, key_entry, tensor in iter_tensor_entries(target):
            if key_entry == key:
                array = _tensor_to_numpy(tensor["data"], tensor["dtype"], tensor["shape"])
                break
        if array is None:
            raise RuntimeError(f"Projector tensor {key} not found at load time")
        if array.shape[0] == 1536:
            array = array.T
        np.savez(out_npz, **{"projector.proj.weight": array})
        print(f"[projector] wrote {out_npz} key={key} -> shape={array.shape}")
        return {
            "npz": out_npz,
            "key": key,
            "shape": tuple(array.shape),
            "hidden_size": hidden_size,
        }

    if identity_ok:
        array = np.eye(1536, dtype=np.float32)
        np.savez(out_npz, **{"projector.proj.weight": array})
        print("[projector] wrote", out_npz, "identity 1536x1536 (fallback)")
        return {
            "npz": out_npz,
            "key": "identity",
            "shape": (1536, 1536),
            "hidden_size": hidden_size,
        }

    raise RuntimeError("No projector found and hidden size != 1536; cannot fallback to identity")


def write_text_summary(
    target: str,
    out_json: str = "mlx_vlm/convert/reports/dots_text_summary.json",
):
    import json
    import os

    prefix_counts = scan_text_prefixes(target, limit=200)
    projector_candidates = discover_projector(target)

    payload = {
        "prefix_counts": [
            {"prefix": prefix, "count": count} for prefix, count in prefix_counts
        ],
        "projector_candidates": [
            {"key": key, "shape": list(shape)} for key, shape in projector_candidates
        ],
    }

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_json, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("[summary] wrote", out_json)
    return out_json


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "scan":
        sys.exit(cli_scan(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "to-npz":
        sys.exit(cli_convert(sys.argv[2], sys.argv[3]))
    if len(sys.argv) >= 3 and sys.argv[1] == "preview":
        sys.exit(preview_npz(sys.argv[2]))
    if len(sys.argv) >= 3 and sys.argv[1] == "scan-text":
        scan_text_prefixes(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "discover-proj":
        discover_projector(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "summarize-text":
        write_text_summary(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) >= 4 and sys.argv[1] == "convert-proj":
        convert_projector_to_npz(sys.argv[2], sys.argv[3])
        sys.exit(0)
    print("Usage:")
    print("  python -m mlx_vlm.convert.convert_dots_ocr scan /path/to/model_or_dir")
    print("  python -m mlx_vlm.convert.convert_dots_ocr scan-text /path/to/model_or_dir")
    print("  python -m mlx_vlm.convert.convert_dots_ocr discover-proj /path/to/model_or_dir")
    print(
        "  python -m mlx_vlm.convert.convert_dots_ocr summarize-text /path/to/model_or_dir"
    )
    print(
        "  python -m mlx_vlm.convert.convert_dots_ocr to-npz /path/to/model_or_dir "
        "weights/dots_ocr_vision.npz"
    )
    print(
        "  python -m mlx_vlm.convert.convert_dots_ocr preview "
        "weights/dots_ocr_vision.npz"
    )
    sys.exit(1)
