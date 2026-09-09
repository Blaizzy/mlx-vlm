"""Detect document regions and save reading-ordered JSON plus an overlay.

Run from the repository root with ``python -m examples.pp_doclayout_v3``.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mlx_vlm.utils import get_model_path, load_model


def annotate(image, records):
    """Draw normalized [y0, x0, y1, x1] boxes on a copy of the input."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(12, image.width // 75))
    colors = ("#005bbb", "#a5246e", "#087e55", "#a35a00")
    for record in records:
        y0, x0, y1, x1 = record["bbox"]
        box = (
            x0 * image.width / 1000,
            y0 * image.height / 1000,
            x1 * image.width / 1000,
            y1 * image.height / 1000,
        )
        order = record["reading_order"]
        color = colors[(order - 1) % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        label = f"{order}. {record['label']} ({record['score']:.3f})"
        _, _, right, bottom = draw.textbbox((0, 0), label, font=font)
        position = (
            max(0, min(box[0], image.width - right - 4)),
            max(0, min(box[1] - bottom - 4, image.height - bottom - 4)),
        )
        bounds = draw.textbbox(position, label, font=font)
        draw.rectangle(
            (bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2),
            fill="white",
        )
        draw.text(position, label, font=font, fill=color)
    return overlay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HashNuke/pp-doclayout-v3-mlx")
    parser.add_argument("--revision", help="Optional Hugging Face model revision.")
    parser.add_argument("--image", type=Path, default=Path("examples/images/paper.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("layout-output"))
    parser.add_argument("--output-json", type=Path, help="Path to the detection JSON.")
    parser.add_argument(
        "--output-image", type=Path, help="Path to the annotated image."
    )
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    json_path = args.output_json or args.output_dir / f"{args.image.stem}_layout.json"
    image_path = args.output_image or args.output_dir / f"{args.image.stem}_layout.png"
    if any(
        left.resolve() == right.resolve()
        or (left.exists() and right.exists() and left.samefile(right))
        for left, right in combinations((args.image, json_path, image_path), 2)
    ):
        parser.error("Input image and output paths must be different.")

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    model = load_model(get_model_path(args.model, revision=args.revision))
    records = sorted(
        model.detect(image, conf=args.conf), key=lambda item: item["reading_order"]
    )
    result = {
        "model": args.model,
        "revision": args.revision,
        "image": args.image.as_posix(),
        "width": image.width,
        "height": image.height,
        "confidence_threshold": args.conf,
        "image_size": 1024,
        "regions": records,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Also catch differently cased output aliases that did not exist initially.
    if image_path.exists() and image_path.samefile(json_path):
        parser.error("JSON and image output paths must be different.")
    annotate(image, records).save(image_path)
    print(f"Detected {len(records)} regions; saved {json_path} and {image_path}")


if __name__ == "__main__":
    main()
