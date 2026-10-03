"""Write the Hub model card beside converted weights; the runtime stays in mlx-vlm."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

HUB_METADATA = """---
library_name: mlx
pipeline_tag: image-to-3d
license: other
license_name: sam-license
license_link: LICENSE
base_model:
  - facebook/sam-3d-objects
  - Ruicheng/moge-3-vitl
tags:
  - mlx
  - apple-silicon
  - bfloat16
  - image-to-3d
  - gaussian-splatting
  - inference
---

"""

SOURCE_URL = (
    "https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/sam3d_objects/"
)

PERFORMANCE = """
## Local performance

Hardware: {hardware}. Input: {height}×{width} RGB plus object mask{source};
automatic MoGe-3 depth, {ss} structure steps and {slat} latent steps, both
Gaussian heads and mesh. One warmup was excluded; the same seed produced
identical latent outputs.

| Metric | Result |
| --- | ---: |
| Median latency ({runs} runs) | {median:.2f} s |
| Maximum peak memory | {peak:.2f} GB |

Two additional async requests matched the synchronous latent output while
the event-loop heartbeat continued. These measurements describe one input.
"""


def write_card(destination):
    """Build a Hub card from the module README and optional bundle validation."""
    root = Path(destination)
    readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
    # Source-relative inline links must point back to mlx-vlm on the Hub.
    readme = re.sub(
        r"(?<=\]\()(?!#)([^)\s]+)(?=\))",
        lambda match: urljoin(SOURCE_URL, match[0]),
        readme,
    )
    card = HUB_METADATA + readme
    validation = root / "validation.json"
    if validation.exists():
        report = json.loads(validation.read_text())
        if "benchmark" in report:
            b = report["benchmark"]
            card += PERFORMANCE.format(
                hardware=b["hardware"],
                source=f" ({b['input']})" if b.get("input") else "",
                height=b["input_size"][0],
                width=b["input_size"][1],
                ss=b["steps"][0],
                slat=b["steps"][1],
                runs=len(b["samples"]),
                median=b["median_seconds"],
                peak=b["peak_memory_gb"],
            )
    (root / "README.md").write_text(card, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    write_card(parser.parse_args().destination)


if __name__ == "__main__":
    main()
