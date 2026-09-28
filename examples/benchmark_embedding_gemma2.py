"""Benchmark a local EmbeddingGemma 2 checkpoint, including quantized conversions.

Uses the EAP extras cat image, speech recording, and video. Timings are warm,
synchronized MLX forward passes; preprocessing and model loading are excluded.
Run checkpoints sequentially to avoid GPU contention. JSON includes input sizes,
individual samples, throughput, weight bytes, and peak MLX active memory.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from transformers import AutoProcessor

from mlx_vlm.embedding_loader import load_embedding_model


def to_mlx(batch):
    return {key: mx.array(value) for key, value in batch.items()}


def prepare_cases(processor, assets):
    cases = {}
    passage = (
        "Mars is the fourth planet from the Sun. Its reddish appearance comes "
        "from iron minerals in its soil. Robotic missions have explored its "
        "surface, atmosphere, ancient river valleys, and polar ice caps. "
    )
    for batch_size, length in ((1, 128), (32, 128), (1, 512), (32, 512)):
        cases[f"text_{batch_size}x{length}"] = to_mlx(
            processor.tokenizer(
                ["title: none | text: " + passage * 32] * batch_size,
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="np",
            )
        )
    contents = {
        "image": [{"type": "image", "url": str(assets / "cat.jpg")}],
        "audio": [{"type": "audio", "url": str(assets / "speech.wav")}],
        "video": [{"type": "video", "url": str(assets / "sample_video.mp4")}],
    }
    for name, content in contents.items():
        cases[name] = to_mlx(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                return_dict=True,
                return_tensors="np",
            )
        )
    contents["text"] = [{"type": "text", "text": "A photo of a cat"}]
    cases["mixed_4"] = to_mlx(
        processor.apply_chat_template(
            [[{"role": "user", "content": content}] for content in contents.values()],
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )
    )
    mx.eval(cases)
    return cases


def measure(model, inputs, warmups, repeats):
    def forward():
        result = model(**inputs)
        mx.eval(result.last_hidden_state, result.text_embeds)
        return result.text_embeds

    for _ in range(warmups):
        forward()
    mx.synchronize()
    mx.reset_peak_memory()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = forward()
        samples.append(time.perf_counter() - start)
        assert np.isfinite(np.array(result)).all()
        del result
    median = statistics.median(samples)
    batch, padded_length = inputs["input_ids"].shape
    return {
        "batch_size": batch,
        "padded_tokens": padded_length,
        "valid_tokens": int(inputs["attention_mask"].sum().item()),
        "frames": (
            inputs["num_frames_per_video"].tolist()
            if "num_frames_per_video" in inputs
            else []
        ),
        "median_ms": median * 1000,
        "p90_ms": float(np.percentile(samples, 90)) * 1000,
        "samples_ms": [sample * 1000 for sample in samples],
        "embeddings_per_second": batch / median,
        "peak_active_bytes": mx.get_peak_memory(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, required=True, help="EAP extras assets directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("warmups and repeats must be positive")
    processor = AutoProcessor.from_pretrained(args.model_path)
    start = time.perf_counter()
    cases = prepare_cases(processor, args.assets)
    preparation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = load_embedding_model(args.model_path)
    load_seconds = time.perf_counter() - start
    parameters = tree_flatten(model.parameters())
    config = json.loads((args.model_path / "config.json").read_text())
    report = {
        "device": mx.device_info(),
        "mlx_version": mx.__version__,
        "tf32": os.environ.get("MLX_ENABLE_TF32", "default"),
        "model_path": str(args.model_path),
        "quantization": config.get("quantization"),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "weight_bytes": sum(value.nbytes for _, value in parameters),
        "text_weight_bytes": sum(
            value.nbytes
            for key, value in parameters
            if key.startswith("language_model.")
        ),
        "checkpoint_weight_bytes": sum(
            path.stat().st_size for path in args.model_path.glob("*.safetensors")
        ),
        "preparation_seconds": preparation_seconds,
        "load_seconds": load_seconds,
        "cases": {},
    }
    gc.collect()
    mx.clear_cache()
    for name, inputs in cases.items():
        result = measure(model, inputs, args.warmups, args.repeats)
        report["cases"][name] = result
        print(name, json.dumps(result), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    # The documentation's retrieval smoke test, outside the timed region.
    retrieval = to_mlx(
        processor.tokenizer(
            [
                "task: search result | query: Which planet is known as the Red Planet?",
                "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
                "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            ],
            padding=True,
            return_tensors="np",
        )
    )
    embeddings = model(**retrieval).text_embeds
    venus, mars = (embeddings[:1] @ embeddings[1:].T).tolist()[0]
    assert mars > venus, "The documented Mars passage must outrank Venus"
    report["retrieval_smoke"] = {"venus_similarity": venus, "mars_similarity": mars}
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
