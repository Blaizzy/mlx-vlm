# Grounded Reasoning with MLX-VLM

<div align="center">

![MLX-VLM](https://img.shields.io/badge/MLX--VLM-Grounded%20Reasoning-blue)
![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)
![Apple Silicon](https://img.shields.io/badge/optimized-Apple%20Silicon-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

Grounded visual reasoning on Apple Silicon by combining **Falcon Perception** (segmentation) with a local **VLM orchestrator** (Gemma4 or any mlx-vlm model).

<p align="center">
  <i>Pixel-precise answers to visual questions — entirely on your Mac.</i>
</p>

## Overview

Instead of asking a VLM to guess spatial relationships from pixels, this system grounds every conclusion to segmentation masks:

```
User query + Image
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  Orchestrator VLM  (Gemma4 / any VLM)  │
 │  Reasons, plans tool calls, synthesises │
 └──────────────┬───────────────────────────┘
                │  tool calls
      ┌─────────┴──────────────────┐
      ▼                            ▼
 ground_expression          compute_relations
 Falcon Perception          numpy IoU, centroid,
 (segmentation)             size ratio, distance
```

The VLM receives both a **Set-of-Marks image** (coloured numbered masks) and **structured numerical metadata** (area, centroid, bbox) — enabling precise spatial reasoning without guessing.

## Models

| Model | Memory | Recommended for |
|-------|--------|-----------------|
| `tiiuae/Falcon-Perception` | ~2 GB | Always — perception backbone |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | ~14 GB | M3 Pro / M3 Max (best quality) |
| `google/gemma-4-e4b-it` | ~16 GB | M3 Pro / M3 Max |
| `google/gemma-4-e2b-it` | ~5 GB | M3 base (8 GB) |

## Getting Started

### Prerequisites

- macOS on Apple Silicon (M series)
- Python 3.10+

### Installation

```bash
pip install mlx-vlm
```

### Install dependencies

```bash
cd grounded_reasoning
pip install -r requirements.txt
```

## Usage

### Notebook (recommended)

Open `demo.ipynb` for a step-by-step walkthrough with examples and visualisations.

### CLI

```bash
python agent.py \
    --image path/to/image.jpg \
    --query "Which duck is flying the highest?" \
    --vlm-model mlx-community/gemma-4-26b-a4b-it-4bit
```

### Python API

```python
from mlx_vlm import load
from agent import LocalVLMClient, run_agent, run_baseline

# Load models (once, reuse across queries)
fp_model, fp_processor = load("tiiuae/Falcon-Perception")
vlm_client = LocalVLMClient("mlx-community/gemma-4-26b-a4b-it-4bit")

# Run grounded reasoning
result = run_agent(image, "Which duck is flying highest?",
                   fp_model, fp_processor, vlm_client)

print(result.answer)
print(f"Grounded on masks: {result.supporting_mask_ids}")
result.final_image.show()

# Compare against plain VLM baseline
baseline = run_baseline(image, "Which duck is flying highest?", vlm_client)
print(baseline)
```

## How It Works

1. **VLM plans** — the orchestrator reads the image and query, then calls `ground_expression` with a short noun phrase.
2. **Falcon Perception segments** — returns binary masks + spatial metadata (centroid, area, bbox) for all matching instances.
3. **VLM reasons** — sees the Set-of-Marks overlay and the JSON metadata; can call `compute_relations` for exact pairwise stats, or `get_crop` to zoom in.
4. **VLM answers** — calls `answer` with its conclusion and the supporting mask IDs, which are rendered onto the final image.

## Available Tools

| Tool | Purpose | Returns |
|------|---------|---------|
| `ground_expression(expression)` | Segment all instances matching a noun phrase | SoM image + JSON metadata |
| `get_crop(mask_id)` | Zoom into a small or overlapping mask | Cropped image |
| `compute_relations(mask_ids)` | Pairwise spatial stats between masks | JSON (IoU, position, size) |
| `answer(response, supporting_mask_ids)` | Return final answer and terminate | — |

## File Structure

```
grounded_reasoning/
├── agent.py          # Agent loop, LocalVLMClient, run_agent, run_baseline
├── fp_tools.py       # Falcon Perception interface and metadata utilities
├── viz.py            # Set-of-Marks rendering and crop helpers
├── requirements.txt
├── README.md
└── demo.ipynb        # Interactive walkthrough
```

## Privacy & Performance

- **Fully local** — no API keys, no data leaves your Mac.
- **Unified memory** — both models share Apple Silicon's memory pool.
- **Metal cache cleared** after each VLM call to keep memory usage stable.

## Contributing

Community contributions are welcome! See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

<p align="center">
  <i>Built on <a href="https://github.com/Blaizzy/mlx-vlm">mlx-vlm</a> and <a href="https://github.com/tiiuae/Falcon-Perception">Falcon Perception</a></i>
</p>
