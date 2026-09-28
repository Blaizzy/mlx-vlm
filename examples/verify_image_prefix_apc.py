"""Real-model Qwen3.5/Qwen3.8 image-prefix regression and timing probe.

Run on an idle Apple Silicon GPU:
  python examples/verify_image_prefix_apc.py --model mlx-community/Qwen3.8-27B-4bit
  python examples/verify_image_prefix_apc.py --model mlx-community/Qwen3.8-Flash-Next-4bit

Quantized MoE models such as Qwen3.8-Flash-Next change their first-token
distribution with the prefill chunking alone, so the cached result is compared
with cold runs at several prefill step sizes and must match one of them.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mlx_vlm import load, stream_generate
from mlx_vlm.apc import APCManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    args = parser.parse_args()
    model, processor = load(args.model)
    manager = APCManager(overrides={"memory_max_gb": 4})
    with tempfile.TemporaryDirectory(prefix="image-prefix-apc-") as directory:
        root = Path(directory)
        for color in ("red", "green", "blue"):
            Image.new("RGB", (128, 128), color).save(root / f"{color}.png")

        encoded = []
        original = type(model).get_input_embeddings

        def embeddings(self, input_ids, pixel_values=None, **kwargs):
            encoded.append(0 if pixel_values is None else pixel_values.shape[0])
            return original(self, input_ids, pixel_values, **kwargs)

        type(model).get_input_embeddings = embeddings

        def run(label, messages, images, use_cache=True, prefill_step_size=2048):
            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded.clear()
            start = time.monotonic()
            first = None
            first_probs = None
            text = []
            for result in stream_generate(
                model,
                processor,
                prompt,
                image=[str(root / f"{name}.png") for name in images] or None,
                max_tokens=20,
                temperature=0,
                prefill_step_size=prefill_step_size,
                apc_manager=manager if use_cache else None,
                apc_tenant="image-prefix-test",
                apc_image_prefix=True,
            ):
                if first is None:
                    first = time.monotonic() - start
                    first_probs = result.logprobs.astype(mx.float32)
                    first_probs = first_probs - mx.logsumexp(first_probs)
                    mx.eval(first_probs)
                text.append(result.text)
            row = {
                "case": label,
                "ttft_s": first,
                "text": "".join(text),
                "prompt_tokens": result.prompt_tokens,
                "cached_tokens": result.cached_tokens,
                "encoded_rows": list(encoded),
                "resident_bytes": manager.resident_bytes(),
            }
            print(json.dumps(row), flush=True)
            return row, first_probs

        def user(images, text):
            return {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(root / f"{name}.png")}
                    for name in images
                ]
                + [{"type": "text", "text": text}],
            }

        system = {
            "role": "system",
            "content": "Reference material: "
            + "The archive contains routine background facts. " * 400,
        }
        question = "List the colors of all images in conversation order, using only color words."
        base = [
            system,
            user(["red"], "What color is this image? Reply with one color word."),
        ]
        text_base = [system, user([], "Say OK.")]
        cases = [
            (
                "text_to_image",
                text_base,
                [],
                [
                    *text_base,
                    {"role": "assistant", "content": "OK."},
                    user(["green"], question),
                ],
                ["green"],
                ["green"],
            ),
            (
                "image_to_image",
                base,
                ["red"],
                [
                    *base,
                    {"role": "assistant", "content": "Red."},
                    user(["green"], question),
                ],
                ["red", "green"],
                ["red", "green"],
            ),
            (
                "image_to_two_images",
                base,
                ["red"],
                [
                    *base,
                    {"role": "assistant", "content": "Red."},
                    user(["green", "blue"], question),
                ],
                ["red", "green", "blue"],
                ["red", "green", "blue"],
            ),
        ]
        try:
            run("kernel_warmup", [user([], "Say OK.")], [])
            for name, past, past_images, extended, extended_images, expected in cases:
                manager.clear()
                run(name + "_past", past, past_images)
                warm, log_warm = run(name + "_warm", extended, extended_images)
                assert warm["cached_tokens"] > 0
                assert warm["encoded_rows"] == [
                    256 * (len(extended_images) - len(past_images))
                ]
                kls = {}
                for step in (2048, 512, 256):
                    cold, log_cold = run(
                        f"{name}_cold_step{step}",
                        extended,
                        extended_images,
                        False,
                        step,
                    )
                    for row in (warm, cold):
                        words = row["text"].lower()
                        positions = [words.index(color) for color in expected]
                        assert positions == sorted(positions), row
                    kls[step] = mx.sum(mx.exp(log_cold) * (log_cold - log_warm)).item()
                print(
                    json.dumps({"case": name, "first_token_kl_by_cold_step": kls}),
                    flush=True,
                )
                assert min(kls.values()) < 0.05

            # Same path, changed old content: no image state from that path may
            # survive. A checkpoint before the first image can still be used.
            manager.clear()
            run("before_change", base, ["red"])
            Image.new("RGB", (128, 128), "blue").save(root / "red.png")
            changed, _ = run("changed_old_image", cases[1][3], ["red", "green"])
            assert changed["encoded_rows"] == [512]
            # The stale red image must not survive. Whether the answer also
            # lists green varies with prefill chunking on quantized MoE models.
            assert "blue" in changed["text"].lower()
            assert "red" not in changed["text"].lower()
            assert changed["cached_tokens"] <= 2800
            Image.new("RGB", (128, 128), "red").save(root / "red.png")

            manager.clear()
            pair = [system, user(["red", "green"], question)]
            run("before_reorder", pair, ["red", "green"])
            swapped, _ = run(
                "reordered_history",
                [system, user(["green", "red"], question)],
                ["green", "red"],
            )
            assert swapped["encoded_rows"] == [512]
            assert swapped["text"].lower().index("green") < swapped[
                "text"
            ].lower().index("red")
            assert swapped["cached_tokens"] <= 2800
        finally:
            type(model).get_input_embeddings = original
            manager.clear()
            manager.close()


if __name__ == "__main__":
    main()
