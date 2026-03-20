#!/usr/bin/env python3
"""Test batch_generate across models that previously failed.

Run all:   python -m mlx_vlm.tests.test_batch_failures
Run one:   python -m mlx_vlm.tests.test_batch_failures gemma3n

Each model is tested with 2 prompts + 2 images of different shapes.
"""

import gc
import sys
import traceback

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.generate import batch_generate

IMAGES = [
    "examples/images/cats.jpg",
    "examples/images/desktop_setup.png",
]
PROMPTS = ["What do you see?", "Describe this image."]

MODEL_DIR = "/Volumes/Extreme SSD/Models/mlx-vlm"

# Map model_type -> local weight path (relative to MODEL_DIR)
MODELS = {
    "gemma3n": "gemma-3n-E2B-it-4bit",
    "hunyuan_vl": "HunyuanOCR",
    "jina_vlm": "jinaai--jina-vlm-mlx",
    "florence2": "Florence-2-base-ft-6bit",
    "llama4": "Llama-4-Scout-17B-16E-Instruct-4bit",
    "deepseek_vl_v2": "deepseek-vl2-4bit",
}


def test_model(model_type: str, model_path: str) -> bool:
    """Test batch_generate for a single model. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type} ({model_path})")
    print(f"{'='*60}")

    try:
        model, processor = load(model_path)
        print(f"  Loaded model OK")

        response = batch_generate(
            model,
            processor,
            images=IMAGES,
            prompts=PROMPTS,
            max_tokens=20,
            verbose=True,
        )

        for i, text in enumerate(response.texts):
            print(f"  [{i}] {text[:100]}...")

        print(f"  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    finally:
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    # Allow running a single model: python -m mlx_vlm.tests.test_batch_failures gemma3n
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = list(MODELS.keys())

    results = {}
    for name in targets:
        if name not in MODELS:
            print(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
            continue
        path = f"{MODEL_DIR}/{MODELS[name]}"
        results[name] = test_model(name, path)

    print(f"\n{'='*60}")
    print("Summary:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
