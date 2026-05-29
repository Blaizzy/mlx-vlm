"""Compare our MLX LocateAnything port against the HF reference dump.

Loads ~/la_parity_ref.npz (produced by la_parity_ref.py on the GPU box),
feeds the *identical* pixel_values + grid to our MLX vision tower + connector
in fp32, and reports numerical parity (max/mean abs diff, cosine similarity).
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from mlx_vlm.utils import load_model

REF = os.path.expanduser("~/la_parity_ref.npz")
META = os.path.expanduser("~/la_parity_ref.json")
MODEL_PATH = os.path.expanduser("~/models/LocateAnything-3B")


def stats(name, a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(
        f"  {name:16s} n={a.shape[0]:>8d}  max|d|={diff.max():.4e}  "
        f"mean|d|={diff.mean():.4e}  cos={cos:.6f}  "
        f"ref[range]=[{b.min():.3f},{b.max():.3f}]"
    )
    return cos


def main():
    ref = np.load(REF)
    meta = json.load(open(META)) if os.path.exists(META) else {}
    gh = ref["image_grid_hws"]
    h, w = int(gh[0][0]), int(gh[0][1])
    print(f"ref grid={h}x{w}  pixel_values={ref['pixel_values'].shape}")
    print(f"ref decoded: {meta.get('decoded', '<none>')}")

    model = load_model(Path(MODEL_PATH))
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

    pv = mx.array(ref["pixel_values"].astype(np.float32))
    grid = mx.array(gh.astype(np.int64))

    vt_list = model.vision_tower(
        pv.transpose(0, 2, 3, 1), grid_thw=grid, grid_shapes=[(h, w)]
    )
    merged_dim = vt_list[0].shape[-1] * vt_list[0].shape[-2]
    vit_mlx = mx.concatenate(vt_list, axis=0).reshape(-1, merged_dim)
    mlp_mlx = model.multi_modal_projector(vt_list)

    print("\n=== Numerical parity (MLX fp32 vs HF fp32, identical inputs) ===")
    cos_v = stats("vision_model", np.array(vit_mlx), ref["vit_embeds"])
    cos_m = stats("mlp1(connector)", np.array(mlp_mlx), ref["mlp_out"])

    print("\n=== Verdict ===")
    ok = cos_v > 0.999 and cos_m > 0.999
    print(
        f"  vision cos={cos_v:.6f}  connector cos={cos_m:.6f}  -> "
        f"{'PARITY OK' if ok else 'MISMATCH - investigate'}"
    )


if __name__ == "__main__":
    main()
