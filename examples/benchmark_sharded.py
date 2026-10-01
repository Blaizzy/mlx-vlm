"""Fixed-length text benchmark for a tensor-sharded MLX-VLM model.

Launch with mlx.launch, using the same hostfile as sharded_generate.py.
Reports aggregate batch throughput and each rank's peak MLX allocation.
Synthetic prompts use independent caches; EOS is ignored for fixed-length timing.

Example (matching model and checkout paths on both hosts)::

    mlx.launch --hostfile hosts.json --backend jaccl \
        --env MLX_METAL_FAST_SYNCH=1 -- \
        python examples/benchmark_sharded.py --backend jaccl \
        --model /path/to/DeepSeek-V4.1-Flash-4bit --contexts 8192 \
        --batch-sizes 4 --max-tokens 1024 --output-dir /tmp/deepseek-benchmark
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import socket
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.generate import BatchGenerator
from mlx_vlm.generate.ar import _split_prompt_kwargs_per_row
from mlx_vlm.utils import sharded_load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["ring", "jaccl"], required=True)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[1024, 8192, 32768, 65536, 131072, 250000],
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--quality-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    group = mx.distributed.init(strict=True, backend=args.backend)
    rank = group.rank()
    if group.size() != 2:
        raise ValueError("This benchmark requires two ranks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"rank-{rank}.jsonl"

    def emit(stage, **data):
        record = dict(
            stage=stage,
            rank=rank,
            host=socket.gethostname(),
            backend=args.backend,
            **data,
        )
        line = json.dumps(record, sort_keys=True)
        with result_path.open("a") as out:
            out.write(line + "\n")
        print(line, flush=True)
        return record

    def barrier():
        mx.eval(mx.distributed.all_sum(mx.array(1), group=group, stream=mx.cpu))

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(4 * 1024**3)
    mx.random.seed(0)
    model, processor = sharded_load(args.model, tensor_group=group)
    lm = model.language_model
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    text_config = getattr(model.config, "text_config", model.config)
    limit = text_config.max_position_embeddings
    if max(args.contexts) + args.max_tokens > limit:
        raise ValueError(f"Requested prompt plus output exceeds {limit} positions")
    tokenizer.stopping_criteria.eos_token_ids = []
    emit(
        "loaded",
        world_size=group.size(),
        mlx=mx.__version__,
        mlx_vlm=importlib.metadata.version("mlx-vlm"),
        fast_synch=os.environ.get("MLX_METAL_FAST_SYNCH"),
        generation_stream="default_gpu",
        device=mx.device_info(),
        model=args.model,
        weight_gb=sum(p.nbytes for _, p in tree_flatten(lm.parameters())) / 1e9,
        contexts=args.contexts,
        batch_sizes=args.batch_sizes,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        kv_cache="unquantized",
        sampling="greedy",
        ignore_eos=True,
    )

    if args.quality_smoke:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one short sentence.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        cache = lm.make_cache()
        inputs = mx.array([token_ids])
        generated = []
        for _ in range(32):
            logits = lm(inputs, cache=cache).logits[:, -1]
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
            generated.append(token.item())
            inputs = token[:, None]
        digest = hashlib.sha256(json.dumps(generated).encode()).digest()
        hashes = mx.distributed.all_gather(
            mx.array(list(digest), dtype=mx.uint8), group=group, stream=mx.cpu
        ).tolist()
        assert hashes[:32] == hashes[32:]
        emit(
            "quality_smoke",
            text=tokenizer.decode(generated),
            tokens=generated,
            ranks_agree=True,
        )
        del cache, inputs, logits, token
        gc.collect()
        mx.clear_cache()

    # Vary prose order by row, keeping deterministic token IDs on both hosts.
    sentences = [
        "The research team measured network latency and memory bandwidth during inference.",
        "Each computer stores part of the model and exchanges partial results with its peer.",
        "A long document contains observations, experiments, methods, and conclusions.",
        "The garden has apple trees, blue flowers, and a quiet path beside the river.",
        "Engineers compare throughput at different batch sizes and sequence lengths.",
        "The library catalog records the author, publication date, and subject of every book.",
        "Attention connects the current question with information from earlier passages.",
        "A careful report distinguishes measured results from predictions and assumptions.",
    ]
    corpora = []
    for row in range(max(args.batch_sizes)):
        rng = random.Random(2000 + row)
        corpus = tokenizer.encode(
            " ".join(rng.choice(sentences) for _ in range(1024)),
            add_special_tokens=False,
        )
        corpora.append(corpus)
    suffix = tokenizer.encode(
        "\nExplain the technical ideas above in detail.\n", add_special_tokens=False
    )

    def prompts(context, batch):
        output = []
        for row in range(batch):
            prefix = tokenizer.encode(
                f"Document {row + 1}:\n", add_special_tokens=False
            )
            n = context - len(prefix) - len(suffix)
            corpus = corpora[row]
            output.append(
                prefix + (corpus * ((n + len(corpus) - 1) // len(corpus)))[:n] + suffix
            )
        assert all(len(p) == context for p in output)
        return output

    def run_case(context, batch, tokens, warmup=False):
        gc.collect()
        mx.clear_cache()
        lm._rope_deltas = None
        lm._position_ids = None
        inputs = prompts(context, batch)
        gen = BatchGenerator(
            lm,
            processor,
            max_tokens=tokens,
            prefill_batch_size=batch,
            completion_batch_size=batch,
            prefill_step_size=args.prefill_step_size,
            compute_logprobs=False,
            # Keep embedding preparation and generation on the same GPU stream.
            stream=mx.default_stream(mx.gpu),
        )
        features = model.get_input_embeddings(mx.array(inputs), None)
        uids = gen.insert(
            inputs,
            max_tokens=tokens,
            prompt_kwargs=_split_prompt_kwargs_per_row(
                {k: v for k, v in features.to_dict().items() if v is not None}, batch
            ),
        )
        output_tokens = {uid: [] for uid in uids}
        mx.synchronize(gen.stream)
        barrier()
        mx.reset_peak_memory()
        start = time.perf_counter()
        prompt_end = None
        last_progress = start
        while gen.has_work:
            prompt_responses, responses = gen.next()
            if prompt_responses:
                mx.synchronize(gen.stream)
                prompt_end = time.perf_counter()
                if not warmup:
                    emit(
                        "prefill_complete",
                        context=context,
                        batch=batch,
                        seconds=prompt_end - start,
                        prompt_tps=context * batch / (prompt_end - start),
                        peak_memory_gb=mx.get_peak_memory() / 1e9,
                    )
            for response in responses:
                output_tokens[response.uid].append(response.token)
            now = time.perf_counter()
            if rank == 0 and not warmup and now - last_progress >= 30:
                pending = gen._prompt_batch
                emit(
                    "progress",
                    context=context,
                    batch=batch,
                    elapsed_s=now - start,
                    prefill_columns=getattr(
                        pending,
                        "_processed_prompt_columns",
                        context if prompt_end else 0,
                    ),
                    generated_tokens=sum(map(len, output_tokens.values())),
                    peak_memory_gb=mx.get_peak_memory() / 1e9,
                )
                last_progress = now
        mx.synchronize(gen.stream)
        end = time.perf_counter()
        peak = mx.get_peak_memory() / 1e9
        gen.close()
        counts = [len(output_tokens[uid]) for uid in uids]
        assert counts == [tokens] * batch, counts
        assert prompt_end is not None
        digest = hashlib.sha256(
            json.dumps(output_tokens, sort_keys=True).encode()
        ).digest()
        hashes = mx.distributed.all_gather(
            mx.array(list(digest), dtype=mx.uint8), group=group, stream=mx.cpu
        ).tolist()
        assert hashes[:32] == hashes[32:], "Output tokens differ across ranks"
        record = dict(
            context=context,
            batch=batch,
            prompt_tokens=context * batch,
            generation_tokens=tokens * batch,
            output_tokens_per_request=tokens,
            prompt_seconds=prompt_end - start,
            generation_seconds=end - prompt_end,
            prompt_tps=context * batch / (prompt_end - start),
            generation_tps=tokens * batch / (end - prompt_end),
            generation_tps_per_request=tokens / (end - prompt_end),
            elapsed_seconds=end - start,
            peak_memory_gb=peak,
            token_hash=digest.hex(),
            ranks_agree=True,
        )
        del gen
        gc.collect()
        mx.clear_cache()
        if not warmup:
            emit("completed", **record)
        return record

    completed = set()
    for line in result_path.read_text().splitlines():
        record = json.loads(line)
        if (
            record["stage"] == "completed"
            and record["output_tokens_per_request"] == args.max_tokens
        ):
            completed.add((record["context"], record["batch"]))
    for batch in args.batch_sizes:
        emit("warmup", batch=batch)
        run_case(128, batch, 8, warmup=True)
    for context in args.contexts:
        for batch in args.batch_sizes:
            skip = mx.distributed.all_sum(
                mx.array(int((context, batch) in completed)), group=group, stream=mx.cpu
            ).item()
            if skip == group.size():
                continue
            emit("start", context=context, batch=batch)
            run_case(context, batch, args.max_tokens)
    barrier()
    emit("suite_complete")


if __name__ == "__main__":
    main()
