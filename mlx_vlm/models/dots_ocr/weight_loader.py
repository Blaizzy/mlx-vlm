from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import mlx.core as mx
import numpy as np


_MERGER_ALIAS = {"0": "mlp0", "2": "mlp2"}


def _resolve_path(root: Any, dotted: str) -> Tuple[Any, str]:
    """Traverse attributes/indexes and return (parent_obj, final_attr)."""
    parts = dotted.split(".")
    cur = root
    idx = 0
    last = len(parts) - 1

    while idx < last:
        part = parts[idx]
        nxt = parts[idx + 1] if idx + 1 <= last else None

        if part == "mlp" and nxt in _MERGER_ALIAS:
            parts[idx] = _MERGER_ALIAS[nxt]
            parts.pop(idx + 1)
            last -= 1
            continue

        if re.fullmatch(r"\d+", part):
            target = int(part)
            cur = cur[target]
        else:
            cur = getattr(cur, part)
        idx += 1

    final = parts[-1]
    if final == "mlp":
        raise AttributeError("Final path cannot be 'mlp'")
    if final in _MERGER_ALIAS:
        final = _MERGER_ALIAS[final]
    return cur, final


def _assign_array(param_obj: Any, name: str, np_arr: np.ndarray) -> None:
    """Assign numpy array into MLX module attribute as mx.array."""
    target = getattr(param_obj, name, None)
    target_shape = tuple(target.shape) if target is not None else None

    if target_shape and np_arr.shape != target_shape:
        if np_arr.ndim == 4 and len(target_shape) == 4:
            # Torch Conv2d weights come as [out, in, kh, kw]; MLX expects [out, kh, kw, in]
            if (
                np_arr.shape[0] == target_shape[0]
                and np_arr.shape[1] == target_shape[3]
                and np_arr.shape[2] == target_shape[1]
                and np_arr.shape[3] == target_shape[2]
            ):
                np_arr = np_arr.transpose(0, 2, 3, 1)
            else:
                raise ValueError(
                    f"Unsupported conv weight layout: {np_arr.shape} -> {target_shape}"
                )
        elif np_arr.ndim == 2 and len(target_shape) == 2:
            # Handle potential transpose mismatch for Linear layers
            if (
                np_arr.shape[0] == target_shape[1]
                and np_arr.shape[1] == target_shape[0]
            ):
                np_arr = np_arr.T
            else:
                raise ValueError(
                    f"Unsupported linear weight layout: {np_arr.shape} -> {target_shape}"
                )
        else:
            raise ValueError(
                "Shape mismatch when assigning weight: "
                f"expected {target_shape}, got {np_arr.shape}"
            )

    dtype = target.dtype if target is not None else None
    setattr(param_obj, name, mx.array(np_arr, dtype=dtype))


def load_npz_into_vision(vision_module: Any, npz_path: str) -> Dict[str, int]:
    """Load weights from NPZ into DotsVisionTransformer_MLX instance."""
    loaded = 0
    skipped = 0
    missing = 0

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

    with np.load(npz_path) as z:
        for key in z.files:
            if not key.startswith("vision."):
                skipped += 1
                continue
            if not key.endswith(allowed_suffixes):
                skipped += 1
                continue

            path = key[len("vision.") :]
            try:
                parent, attr = _resolve_path(vision_module, path)
            except (AttributeError, IndexError, KeyError, TypeError):
                missing += 1
                continue

            try:
                _assign_array(parent, attr, z[key])
            except ValueError:
                missing += 1
                continue
            loaded += 1

    return {"loaded": loaded, "skipped": skipped, "missing": missing}
