#!/usr/bin/env python3
"""
dots_ocr_mask_visualizer.py

Visualize cu_seqlens and the resulting block-diagonal attention mask
for batches of images with variable (H, W) patch grids.

Usage:
  python dots_ocr_mask_visualizer.py --hw 64x48 32x32 80x20 --out mask.png

This will create:
  - A printed cu_seqlens array (e.g., [0, 3072, 4096, 5696] for HW=[64*48, 32*32, 80*20])
  - A saved image showing the block-diagonal attention mask
    (white = allowed attention, black = masked).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_hw_list(hw_list):
    shapes = []
    for s in hw_list:
        if 'x' not in s.lower():
            raise ValueError(f"Invalid --hw entry '{s}'. Use 'HxW' like 64x48.")
        h_str, w_str = s.lower().split('x')
        H = int(h_str.strip())
        W = int(w_str.strip())
        shapes.append((H, W))
    return shapes

def build_cu_seqlens(hw_shapes):
    cu = [0]
    total = 0
    for H, W in hw_shapes:
        total += H * W
        cu.append(total)
    return np.array(cu, dtype=np.int32)

def block_diag_mask_from_cu(cu):
    total = int(cu[-1])
    mask = np.zeros((total, total), dtype=bool)
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i+1])
        mask[start:end, start:end] = True
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", nargs="+", required=True, help="List of HxW patch grids, e.g. 64x48 32x32")
    ap.add_argument("--out", type=str, default="mask.png", help="Output filename for the mask image")
    args = ap.parse_args()

    shapes = parse_hw_list(args.hw)
    cu = build_cu_seqlens(shapes)
    print("H, W per image:", shapes)
    print("HW per image:  ", [H*W for H, W in shapes])
    print("cu_seqlens:     ", cu.tolist())

    mask = block_diag_mask_from_cu(cu)  # True = allowed
    # Visualize: True(allowed)=1 (white), False(masked)=0 (black)
    img = mask.astype(np.float32)

    plt.figure(figsize=(6,6))
    plt.imshow(img, interpolation="nearest")
    plt.title("Block-diagonal attention mask")
    plt.xlabel("Key sequence index")
    plt.ylabel("Query sequence index")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved mask visualization to: {args.out}")

if __name__ == "__main__":
    main()
