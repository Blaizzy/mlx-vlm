#!/usr/bin/env python3
"""Plot NIAH results as heatmaps."""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_single_needle(results: list[dict], model: str, kv_bits, ax=None):
    single = [r for r in results if r["dataset"] == "single_needle"]
    if not single:
        return

    contexts = sorted(
        set(r["context_length"] for r in single), key=lambda x: int(x.replace("k", ""))
    )
    depths = sorted(set(r["depth"] for r in single))

    # Build accuracy grid: rows=depths (top=1.0, bottom=0.0), cols=contexts
    grid = np.full((len(depths), len(contexts)), np.nan)
    for r in single:
        ci = contexts.index(r["context_length"])
        di = depths.index(r["depth"])
        grid[di, ci] = 100.0 if r["correct"] else 0.0

    # Flip so depth 1.0 is at top
    grid = grid[::-1]
    depths_display = depths[::-1]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(8, len(contexts) * 1.2), max(5, len(depths) * 0.9))
        )
    else:
        fig = ax.figure

    cmap = plt.cm.RdYlGn
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Annotate each cell
    for i in range(len(depths_display)):
        for j in range(len(contexts)):
            val = grid[i, j]
            if not np.isnan(val):
                color = "white" if val < 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts, fontsize=11)
    ax.set_yticks(range(len(depths_display)))
    ax.set_yticklabels([f"{d:.0%}" for d in depths_display], fontsize=11)
    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Needle Depth", fontsize=12)

    kv_label = f" | KV {kv_bits}-bit" if kv_bits else ""
    ax.set_title(f"NIAH Single Needle — {model}{kv_label}", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=11)

    return fig


def plot_multi_needle(results: list[dict], model: str, kv_bits, ax=None):
    multi = [r for r in results if r["dataset"] == "multi_needle"]
    if not multi:
        return

    contexts = sorted(
        set(r["context_length"] for r in multi), key=lambda x: int(x.replace("k", ""))
    )

    # Build grid: rows=needle_id, cols=context
    all_needle_ids = []
    for r in sorted(multi, key=lambda x: int(x["context_length"].replace("k", ""))):
        for nd in r["needle_details"]:
            if nd["id"] not in all_needle_ids:
                all_needle_ids.append(nd["id"])

    grid = np.full((len(all_needle_ids), len(contexts)), np.nan)
    for r in multi:
        ci = contexts.index(r["context_length"])
        for nd in r["needle_details"]:
            if nd["id"] in all_needle_ids:
                ni = all_needle_ids.index(nd["id"])
                grid[ni, ci] = 100.0 if nd["found"] else 0.0

    # Labels for needles (short)
    needle_labels = [nid.replace("needle_", "Needle ") for nid in all_needle_ids]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(8, len(contexts) * 1.2), max(4, len(all_needle_ids) * 0.8))
        )
    else:
        fig = ax.figure

    cmap = plt.cm.RdYlGn
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(len(needle_labels)):
        for j in range(len(contexts)):
            val = grid[i, j]
            if not np.isnan(val):
                color = "white" if val < 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="gray")

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts, fontsize=11)
    ax.set_yticks(range(len(needle_labels)))
    ax.set_yticklabels(needle_labels, fontsize=11)
    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Needle", fontsize=12)

    kv_label = f" | KV {kv_bits}-bit" if kv_bits else ""
    ax.set_title(f"NIAH Multi Needle — {model}{kv_label}", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=11)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot NIAH heatmaps")
    parser.add_argument("results_json", help="Path to NIAH results JSON")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (default: show interactively)",
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    model = data.get("model", "Unknown")
    kv_bits = data.get("kv_bits")
    results = data["results"]

    has_single = any(r["dataset"] == "single_needle" for r in results)
    has_multi = any(r["dataset"] == "multi_needle" for r in results)

    if has_single and has_multi:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_single_needle(results, model, kv_bits, ax=ax1)
        plot_multi_needle(results, model, kv_bits, ax=ax2)
        fig.tight_layout(pad=2.0)
    elif has_single:
        fig = plot_single_needle(results, model, kv_bits)
    elif has_multi:
        fig = plot_multi_needle(results, model, kv_bits)
    else:
        print("No results to plot.")
        sys.exit(1)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
