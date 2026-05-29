"""AR-oracle verification for LocateAnything Parallel Box Decoding.

Loads LocateAnything-3B, prepares the COCO cats image + "Detect all objects"
prompt once, then runs PBD in slow / fast / hybrid modes and prints the decoded
box tokens + tokens/sec for each. By design hybrid must match slow exactly; fast
should match within +/-1 quant level.
"""

import os
import time
from pathlib import Path

import mlx.core as mx

from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load, prepare_inputs

MODEL_PATH = os.path.expanduser("~/models/LocateAnything-3B")
IMAGE = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = "Detect all objects in the image."


def main():
    model, processor = load(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer

    formatted = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    inputs = prepare_inputs(
        processor,
        images=IMAGE,
        prompts=formatted,
        image_token_index=getattr(model.config, "image_token_index", None),
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    grid_kwargs = {
        k: inputs[k] for k in ("image_grid_hws", "_grid_shapes") if k in inputs
    }
    if "image_token_id" in inputs:
        grid_kwargs["image_token_id"] = inputs["image_token_id"]

    print(f"prompt tokens={input_ids.shape[1]}")

    results = {}
    for mode in ("slow", "fast", "hybrid"):
        mx.metal.clear_cache() if hasattr(mx, "metal") else None
        tic = time.perf_counter()
        gen = model.pbd_generate(
            input_ids,
            pixel_values=pixel_values,
            generation_mode=mode,
            max_tokens=256,
            **grid_kwargs,
        )
        dt = time.perf_counter() - tic
        text = tokenizer.decode(gen, skip_special_tokens=False)
        tps = len(gen) / dt if dt > 0 else 0.0
        results[mode] = (text, len(gen), tps)
        print(f"\n=== {mode} ===")
        print(f"tokens={len(gen)} time={dt:.2f}s tps={tps:.2f}")
        print(text)

    print("\n=== VERDICT ===")
    slow_txt = results["slow"][0]
    print(f"hybrid == slow : {results['hybrid'][0] == slow_txt}")
    print(f"fast   == slow : {results['fast'][0] == slow_txt}")


if __name__ == "__main__":
    main()
