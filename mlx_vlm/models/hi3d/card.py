"""Write the Hub model card beside a converted bundle."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

HUB_METADATA = """---
library_name: mlx
pipeline_tag: image-to-3d
license: other
license_name: hi3d-open-license-1.0
license_link: LICENSE
base_model:
  - hi3d-official/Hi3Dv1.0
tags:
  - mlx
  - mlx-vlm
  - apple-silicon
  - image-to-3d
  - mesh
  - sparc3d
---

"""

SOURCE_URL = "https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/hi3d/"

INSTALL = """
> **Install:** until the `hi3d` model is in an mlx-vlm release, install it from
> source: `pip install git+{source}`.

"""

CONTENTS = """
## Bundle contents

| Prefix | Source | Tensors | Dtypes |
| --- | --- | ---: | --- |
{rows}

`conversion.json` lists every source file with its SHA-256 and each transform.
The Hi3D tensors are unchanged in value and dtype.
"""

PERFORMANCE = """
## Local performance

Hardware: {hardware}. Input: {input}, {resolution} output, {steps} Heun steps
per pass, seed {seed}.

| Stage | Seconds |
| --- | ---: |
{rows}

Peak memory: {peak:.1f} GB. Output: {vertices:,} vertices, {faces:,} faces.
"""


def write_card(destination, source="https://github.com/Blaizzy/mlx-vlm.git"):
    """Build the card from the module README, conversion report and validation."""
    root = Path(destination)
    readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
    readme = re.sub(
        r"(?<=\]\()(?!#|https?://)([^)\s]+)(?=\))",
        lambda match: urljoin(SOURCE_URL, match[0]),
        readme,
    )
    title, _, body = readme.partition("\n")
    card = HUB_METADATA + title + "\n" + INSTALL.format(source=source) + body.lstrip()
    report = json.loads((root / "conversion.json").read_text())
    sources = {
        "transformer_512": "hi3d-official/Hi3Dv1.0",
        "transformer_1024": "hi3d-official/Hi3Dv1.0",
        "vae": "hi3d-official/Hi3Dv1.0",
        "dino": "facebook/dinov2-large",
        "clip": "openai/clip-vit-large-patch14",
    }
    rows = []
    for part, dtypes in report["dtypes"].items():
        prefix = f"conditioner.{part}" if part in ("dino", "clip") else part
        count = sum(dtypes.values())
        kinds = ", ".join(f"{k} ({v})" for k, v in sorted(dtypes.items()))
        rows.append(f"| `{prefix}` | {sources.get(part, part)} | {count} | {kinds} |")
    card += CONTENTS.format(rows="\n".join(rows))
    validation = root / "validation.json"
    if validation.exists():
        bench = json.loads(validation.read_text()).get("benchmark")
        if bench:
            stages = "\n".join(
                f"| {k} | {v:.1f} |" for k, v in bench["timings"].items()
            )
            card += PERFORMANCE.format(rows=stages, **bench)
    (root / "README.md").write_text(card, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--source", default="https://github.com/Blaizzy/mlx-vlm.git")
    args = parser.parse_args()
    write_card(args.destination, args.source)


if __name__ == "__main__":
    main()
