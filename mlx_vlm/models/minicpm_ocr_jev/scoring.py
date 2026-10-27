"""The supplied Jev value-head input contract: original, OCR text, rendering."""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
from PIL import Image

VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
USER_TAIL = (
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。"
)


def build_prompt(ocr_text):
    return (
        f"<|im_start|>user\n{VISION_PLACEHOLDER}OCR文本：\n{ocr_text}\n"
        f"{VISION_PLACEHOLDER}{USER_TAIL}<|im_end|>\n"
    )


def load_image(image):
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    image = image.convert("RGB")
    width, height = image.size
    if max(width, height) / min(width, height) <= 199:
        return image
    size = (max(width, math.ceil(height / 199)), max(height, math.ceil(width / 199)))
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - width) // 2, (size[1] - height) // 2))
    return canvas


def score(model, processor, orig_image, render_image, ocr_text):
    """Return P(consistent), preserving the reference prompt and image order."""
    from ...utils import prepare_inputs

    inputs = prepare_inputs(
        processor,
        images=[load_image(orig_image), load_image(render_image)],
        prompts=build_prompt(ocr_text),
        add_special_tokens=False,
    )
    logits = model(**inputs).logits.astype(mx.float32)
    probs = mx.softmax(logits, axis=-1)
    mx.eval(logits, probs)
    return {
        "prob_1": probs[0, 1].item(),
        "prob_0": probs[0, 0].item(),
        "logit_0": logits[0, 0].item(),
        "logit_1": logits[0, 1].item(),
        "n_tokens": inputs["input_ids"].shape[1],
    }


def main():
    from ...utils import load

    parser = argparse.ArgumentParser(
        description="Score OCR consistency with MiniCPM-OCR Jev."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--orig-image", required=True)
    parser.add_argument("--render-image", required=True)
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--ocr-text")
    text.add_argument("--ocr-text-file")
    args = parser.parse_args()
    ocr_text = (
        args.ocr_text
        if args.ocr_text is not None
        else Path(args.ocr_text_file).read_text()
    )
    model, processor = load(args.model)
    print(
        json.dumps(
            score(model, processor, args.orig_image, args.render_image, ocr_text),
            indent=2,
        )
    )
