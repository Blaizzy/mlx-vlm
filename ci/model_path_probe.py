from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ci.change_rules import load_yaml_mapping


def checkpoint(job: Mapping[str, Any]) -> tuple[str, str]:
    value = job.get("hf_checkpoint")
    if not isinstance(value, Mapping):
        raise ValueError("job has no hf_checkpoint")
    repo = value.get("repo")
    revision = value.get("revision")
    if not isinstance(repo, str) or not repo:
        raise ValueError("hf_checkpoint repo is required")
    if not isinstance(revision, str) or not revision:
        raise ValueError("hf_checkpoint revision is required")
    return repo, revision


def cached_checkpoint(repo: str, revision: str) -> Path | None:
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME")
        if home:
            hub = str(Path(home) / "hub")
    if not hub:
        return None
    slug = "models--" + repo.replace("/", "--")
    snapshot = Path(hub) / slug / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def prepare_processor(processor: Any, config: Mapping[str, Any]) -> None:
    vision = config.get("vision_config", {})
    if not isinstance(vision, Mapping):
        vision = {}
    values = {
        "patch_size": vision.get("patch_size"),
        "vision_feature_select_strategy": config.get("vision_feature_select_strategy"),
    }
    for name, value in values.items():
        if hasattr(processor, name) and getattr(processor, name) is None and value:
            setattr(processor, name, value)
    if (
        hasattr(processor, "num_additional_image_tokens")
        and processor.num_additional_image_tokens == 0
        and vision.get("model_type") == "clip_vision_model"
    ):
        processor.num_additional_image_tokens = 1


def formatted_prompt(processor: Any, config: Mapping[str, Any], prompt: str) -> str:
    from mlx_vlm.prompt_utils import apply_chat_template

    try:
        return apply_chat_template(processor, config, prompt, num_images=1)
    except TypeError as error:
        if "concatenate str" not in str(error):
            raise
    messages = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
        return_messages=True,
    )
    image_token = str(getattr(processor, "image_token", "<image>"))
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, Mapping):
                    parts.append(str(item))
                elif item.get("type") in {"image", "image_url", "input_image"}:
                    parts.append(image_token)
                elif item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text") or item.get("content") or ""))
            content = "\n".join(part for part in parts if part)
        normalized.append({"role": message.get("role", "user"), "content": content})
    return processor.tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
    )


def summarize(results: Sequence[Any], started: float, ttft_ms: float) -> dict[str, Any]:
    if not results:
        raise RuntimeError("generation produced no results")
    last = results[-1]
    text = "".join(str(getattr(item, "text", "")) for item in results).strip()
    token_ids = [
        int(item.token) for item in results if getattr(item, "token", None) is not None
    ]
    identity = {
        "finish_reason": last.finish_reason,
        "generation_tokens": int(last.generation_tokens),
        "text": text,
        "token_ids": token_ids,
    }
    return {
        "generated_text": text,
        "output_hash": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
        "prompt_tokens": int(last.prompt_tokens),
        "generation_tokens": int(last.generation_tokens),
        "prefill_tps": round(float(last.prompt_tps), 3),
        "decode_tps": round(float(last.generation_tps), 3),
        "ttft_ms": round(ttft_ms, 3),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "peak_memory_gib": round(float(last.peak_memory), 4),
        "finish_reason": last.finish_reason,
    }


def generate(model, processor, formatted: str, image: Path, max_tokens: int):
    from mlx_vlm import stream_generate

    started = time.perf_counter()
    first_token_at = None
    results = []
    for result in stream_generate(
        model,
        processor,
        formatted,
        image=str(image),
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        results.append(result)
    return summarize(
        results,
        started,
        ((first_token_at or time.perf_counter()) - started) * 1000,
    )


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one measured run is required")
    hashes = {str(run["output_hash"]) for run in runs}
    if len(hashes) != 1:
        raise RuntimeError("deterministic runs produced different output hashes")
    result = dict(runs[-1])
    for name in (
        "prompt_tokens",
        "generation_tokens",
        "prefill_tps",
        "decode_tps",
        "ttft_ms",
        "embedding_tps",
        "embedding_latency_ms",
        "wall_ms",
        "peak_memory_gib",
    ):
        if all(name in run for run in runs):
            result[name] = round(statistics.median(float(run[name]) for run in runs), 4)
    result["runs"] = [dict(run) for run in runs]
    return result


def embedding_scenario(
    job: Mapping[str, Any], scenarios: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    configured = job.get("scenarios")
    if not isinstance(configured, list):
        raise ValueError("ModelPath work has no scenarios")
    if "embedding" not in configured:
        return None
    scenario = scenarios.get("embedding")
    if not isinstance(scenario, Mapping):
        raise ValueError("embedding scenario is not configured")
    value = scenario.get("input")
    if not isinstance(value, Mapping):
        raise ValueError("embedding scenario has no input")
    texts = value.get("texts")
    if (
        not isinstance(texts, list)
        or len(texts) != 3
        or any(not isinstance(text, str) or not text for text in texts)
    ):
        raise ValueError("embedding scenario requires three texts")
    return scenario


def embed(model: Any, processor: Any, texts: Sequence[str]) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

    tokenizer = getattr(processor, "tokenizer", processor)
    max_length = min(int(getattr(tokenizer, "model_max_length", 512) or 512), 8192)
    mx.reset_peak_memory()
    started = time.perf_counter()
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    output = model(
        mx.array(encoded["input_ids"]),
        attention_mask=mx.array(encoded["attention_mask"]),
    )
    embeddings = getattr(output, "text_embeds", None)
    if embeddings is None:
        raise ValueError("embedding model produced no text embeddings")
    mx.eval(embeddings)
    elapsed = time.perf_counter() - started
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise ValueError("embedding model produced an invalid output shape")
    norms = np.linalg.norm(array, axis=-1)
    denominator = np.maximum(norms[:, None] * norms[None, :], 1e-12)
    similarities = array @ array.T / denominator
    finite = bool(np.isfinite(array).all())
    normalized = bool(np.allclose(norms, np.ones_like(norms), atol=1e-3))
    related_above_unrelated = bool(similarities[0, 1] > similarities[0, 2])
    prompt_tokens = int(np.asarray(encoded["attention_mask"]).sum())
    wall_ms = elapsed * 1000
    return {
        "output_hash": hashlib.sha256(array.tobytes()).hexdigest()[:16],
        "output_shape": list(array.shape),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": 0,
        "embedding_tps": round(prompt_tokens / elapsed, 3) if elapsed else 0.0,
        "embedding_latency_ms": round(wall_ms, 3),
        "wall_ms": round(wall_ms, 3),
        "peak_memory_gib": round(float(mx.get_peak_memory()) / 2**30, 4),
        "finish_reason": "embedding",
        "correctness_checks": {
            "finite_embeddings": finite,
            "normalized_embeddings": normalized,
            "related_pair_similarity_above_unrelated": related_above_unrelated,
        },
        "similarities": {
            "related": round(float(similarities[0, 1]), 6),
            "unrelated": round(float(similarities[0, 2]), 6),
        },
    }


def embedding_run(
    job: Mapping[str, Any],
    scenario: Mapping[str, Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    from mlx_vlm.embedding_loader import load_embedding_model
    from mlx_vlm.models.pooling import read_pooling_config
    from mlx_vlm.utils import get_model_path, load_processor

    repo, revision = checkpoint(job)
    local_checkpoint = os.environ.get("CI_CHECKPOINT_PATH")
    cached = cached_checkpoint(repo, revision)
    source = Path(local_checkpoint) if local_checkpoint else cached
    if source is None:
        source = get_model_path(
            repo,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
    model = load_embedding_model(source)
    model.pooling_config = read_pooling_config(source)
    processor = load_processor(
        source,
        add_detokenizer=False,
        trust_remote_code=False,
    )
    texts = scenario["input"]["texts"]
    for _ in range(warmup):
        embed(model, processor, texts)
    findings = aggregate([embed(model, processor, texts) for _ in range(iterations)])
    checks = findings.get("correctness_checks", {})
    if not isinstance(checks, Mapping) or not all(checks.values()):
        raise RuntimeError("embedding correctness contract failed")
    findings.update(
        {
            "model": repo,
            "revision": revision,
            "scenario": "embedding",
        }
    )
    return findings


def run(
    job: Mapping[str, Any],
    scenarios: Mapping[str, Any],
    image: Path,
    prompt: str,
    max_tokens: int,
    warmup: int = 1,
    iterations: int = 3,
) -> dict[str, Any]:
    scenario = embedding_scenario(job, scenarios)
    if scenario is not None:
        return embedding_run(job, scenario, warmup, iterations)

    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    repo, revision = checkpoint(job)
    local_checkpoint = os.environ.get("CI_CHECKPOINT_PATH")
    cached = cached_checkpoint(repo, revision)
    source = Path(local_checkpoint) if local_checkpoint else cached or repo
    source_revision = revision if isinstance(source, str) else None
    model, processor = load(
        source,
        revision=source_revision,
        trust_remote_code=False,
        processor_config={"trust_remote_code": False},
    )
    config = load_config(
        source,
        revision=source_revision,
        trust_remote_code=False,
    )
    prepare_processor(processor, config)
    formatted = formatted_prompt(processor, config, prompt)

    for _ in range(warmup):
        generate(model, processor, formatted, image, max_tokens)
    findings = aggregate(
        [
            generate(model, processor, formatted, image, max_tokens)
            for _ in range(iterations)
        ]
    )
    findings.update(
        {
            "model": repo,
            "revision": revision,
            "prompt": prompt,
            "image": image.name,
        }
    )
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Describe the animal in this image in one sentence."
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    findings = run(
        json.loads(args.job.read_text()),
        load_yaml_mapping(args.scenarios).get("scenarios", {}),
        args.image,
        args.prompt,
        args.max_tokens,
        args.warmup,
        args.iterations,
    )
    output = args.output or Path(os.environ.get("CI_JOB_FINDINGS", "findings.json"))
    output.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    print(json.dumps(findings, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
