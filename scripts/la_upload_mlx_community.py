"""Clean, document, and upload LocateAnything-3B MLX variants to mlx-community.

For each variant: strip the PyTorch remote-code .py files + training artifacts,
drop config auto_map, copy the NVIDIA LICENSE, write an attribution model card,
do a quick load sanity check (mlx-vlm), then upload to the HF repo.

Usage: uv run python scripts/la_upload_mlx_community.py [--dry-run]
"""

import json
import os
import shutil
import sys
from pathlib import Path

SRC = Path(os.path.expanduser("~/models/LocateAnything-3B"))
ORG = "mlx-community"
DRY = "--dry-run" in sys.argv

VARIANTS = [
    {
        "repo": "LocateAnything-3B-bf16",
        "dir": "~/models/LocateAnything-3B-mlx-bf16",
        "quant": "MLX format, bfloat16 (unquantized)",
        "note": "Numerically matches the original bf16 checkpoint; identical grounding output.",
    },
    {
        "repo": "LocateAnything-3B-8bit",
        "dir": "~/models/LocateAnything-3B-mlx-8bit",
        "quant": "MLX 8-bit (~9.4 bits/weight)",
        "note": "Grounding output is byte-identical to the bf16 model in our tests.",
    },
    {
        "repo": "LocateAnything-3B-4bit",
        "dir": "~/models/LocateAnything-3B-mlx-4bit-mixed",
        "quant": "MLX mixed 4/8-bit (`mixed_4_8`, ~6.7 bits/weight)",
        "note": (
            "Box coordinates stay accurate (within ~1-2 quant levels of bf16); "
            "semantic labels may generalize (e.g. `object` instead of `remote`). "
            "Pure 4-bit was not released because quantizing the tied "
            "`embed_tokens`/`lm_head` destroys coordinate-token precision."
        ),
    },
]

REMOTE_CODE = [
    "configuration_locateanything.py",
    "configuration_qwen2.py",
    "modeling_locateanything.py",
    "modeling_qwen2.py",
    "modeling_vit.py",
    "processing_locateanything.py",
    "image_processing_locateanything.py",
    "generate_utils.py",
    "mask_magi_utils.py",
    "mask_sdpa_utils.py",
]
TRAINING_ARTIFACTS = ["training_args.bin", "trainer_state.json", "all_results.json"]


def card(repo: str, quant: str, note: str) -> str:
    return f"""---
license: other
license_name: nvidia-license
license_link: https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE
language:
- en
base_model:
- nvidia/LocateAnything-3B
pipeline_tag: image-text-to-text
library_name: mlx-vlm
tags:
- mlx
- vision
- object-detection
- grounding
- locateanything
- nvidia
- eagle
---

# {ORG}/{repo}

{quant} conversion of [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B),
a vision-language model for fast, high-quality visual grounding (object detection,
referring-expression grounding, pointing, GUI/text localization). Converted with
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) for Apple Silicon.

{note}

## Requirements

> **Note:** LocateAnything support in `mlx-vlm` currently lives in a pull request
> and is **not yet in a released `mlx-vlm`**. Until it merges, install from the
> branch that adds the `locateanything` model:
>
> ```bash
> pip install "git+https://github.com/beshkenadze/mlx-vlm@feat/locateanything-3b"
> ```

## Usage

```bash
python -m mlx_vlm.generate --model {ORG}/{repo} \\
  --image http://images.cocodataset.org/val2017/000000039769.jpg \\
  --prompt "Detect all objects in the image." --max-tokens 128 --temperature 0.0
```

Output is structured coordinate tokens, e.g.
`<ref>remote</ref><box><64><152><273><244></box>` with coordinates quantized to
`<0>`..`<1000>` (normalized). Decoding modes: autoregressive (`slow`, default) and
**Parallel Box Decoding** (`fast`/`hybrid`, ~2x faster) via `generation_mode`.

## Attribution & license

- Derived from **nvidia/LocateAnything-3B** — released under the
  [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE):
  **non-commercial, research/academic use only** (commercial use not permitted
  except by NVIDIA). Redistribution must retain this license and attribution.
- Vision encoder: **MoonViT-SO-400M** (MIT). Language model: **Qwen2.5-3B-Instruct**
  (Qwen Research License). Part of the [Eagle VLM](https://github.com/NVlabs/EAGLE) family.

The `LICENSE` file from the source model is included in this repo.
"""


def prep(d: Path, v: dict) -> bool:
    removed = []
    for name in REMOTE_CODE + TRAINING_ARTIFACTS:
        p = d / name
        if p.exists():
            p.unlink()
            removed.append(name)
    # Drop auto_map (this is an MLX model, not a transformers remote-code model)
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("auto_map", None)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    # LICENSE + card
    if (SRC / "LICENSE").exists():
        shutil.copy(SRC / "LICENSE", d / "LICENSE")
    (d / "README.md").write_text(card(v["repo"], v["quant"], v["note"]))
    print(
        f"  cleaned ({len(removed)} files removed), wrote README + LICENSE, dropped auto_map"
    )
    return True


def load_check(d: Path) -> bool:
    try:
        from mlx_vlm.utils import load

        model, processor = load(str(d), trust_remote_code=True)
        ok = getattr(model.config, "model_type", None) == "locateanything"
        print(
            f"  load check: model_type={model.config.model_type} processor={type(processor).__name__} -> {'OK' if ok else 'FAIL'}"
        )
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  load check FAILED: {type(e).__name__}: {e}")
        return False


def upload(d: Path, repo: str):
    from huggingface_hub import HfApi

    api = HfApi()
    full = f"{ORG}/{repo}"
    api.create_repo(full, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(d), repo_id=full, repo_type="model")
    print(f"  UPLOADED -> https://huggingface.co/{full}")


def main():
    for v in VARIANTS:
        d = Path(os.path.expanduser(v["dir"]))
        print(f"\n### {v['repo']}  ({d})")
        if not d.exists():
            print("  SKIP: dir missing")
            continue
        prep(d, v)
        if not load_check(d):
            print("  SKIP upload (load check failed)")
            continue
        if DRY:
            print("  [dry-run] would upload")
        else:
            upload(d, v["repo"])
    print("\nDONE")


if __name__ == "__main__":
    main()
