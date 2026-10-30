"""Run one Mistral Large 4 request with experts sharded across MLX devices.

Launch with mlx.launch; every rank must have its assigned checkpoint at --model.
See mlx_vlm/models/mistral_large4/README.md for staging and hostfile instructions.
"""

import argparse
import json
import socket
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import wired_limit
from PIL import Image
from transformers import AutoProcessor

from mlx_vlm.utils import load_model


def generate(model, processor, group, args):
    if args.image:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning_effort,
        )
        inputs = processor(
            text=text, images=[Image.open(args.image)], return_tensors="np"
        )
        ids = mx.array(inputs["input_ids"])
        pixels = inputs["pixel_values"]
        pixels = (
            [mx.array(x) for x in pixels]
            if isinstance(pixels, list)
            else mx.array(pixels)
        )
        embeddings = model.get_input_embeddings(
            ids, pixels, image_sizes=mx.array(inputs["image_sizes"])
        ).inputs_embeds
        mx.eval(embeddings)
    else:
        ids = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning_effort,
        )
        ids, embeddings = mx.array([ids]), None

    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
    cache = model.language_model.make_cache()
    start = time.perf_counter()
    logits = model.language_model(ids, inputs_embeds=embeddings, cache=cache).logits[
        :, -1
    ]
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    prefill_seconds = time.perf_counter() - start
    tokens = []
    start = time.perf_counter()
    for i in range(args.max_tokens):
        value = int(token.item())
        tokens.append(value)
        if value == processor.tokenizer.eos_token_id or i + 1 == args.max_tokens:
            break
        logits = model.language_model(token[:, None], cache=cache).logits[:, -1]
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
    decode_seconds = time.perf_counter() - start

    # Check lengths before gathering variable-sized output buffers.
    lengths = mx.distributed.all_gather(mx.array([len(tokens)], mx.int32), group=group)
    if not bool(mx.all(lengths == len(tokens))):
        raise RuntimeError("Distributed ranks generated different output lengths")
    gathered = mx.distributed.all_gather(mx.array(tokens, mx.int32), group=group)
    agree = bool(mx.all(gathered.reshape(group.size(), -1) == mx.array(tokens)[None]))
    if not agree:
        raise RuntimeError("Distributed ranks generated different token sequences")
    return dict(
        host=socket.gethostname(),
        rank=group.rank(),
        world_size=group.size(),
        mlx=mx.__version__,
        backend=args.backend,
        prompt=args.prompt,
        image=str(args.image) if args.image else None,
        prompt_tokens=ids.size,
        generated_tokens=len(tokens),
        token_ids=tokens,
        text=processor.tokenizer.decode(tokens),
        prefill_seconds=prefill_seconds,
        prefill_tps=ids.size / prefill_seconds,
        decode_seconds=decode_seconds,
        decode_tps=max(len(tokens) - 1, 0) / decode_seconds,
        peak_gb=mx.get_peak_memory() / 1e9,
        active_gb=mx.get_active_memory() / 1e9,
        ranks_agree=agree,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", default="jaccl")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low"
    )
    parser.add_argument(
        "--output", type=Path, help="Directory for per-rank result JSON files"
    )
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    group = mx.distributed.init(backend=args.backend, strict=True)
    mx.set_cache_limit(512 * 1024**2)
    model = load_model(args.model, lazy=True)
    model.shard(group)
    mx.eval(model.parameters())
    processor = AutoProcessor.from_pretrained(args.model)
    # Match the normal MLX generation path and the MiniMax distributed runner.
    # Without a wired working set, large models incur substantial residency costs.
    with wired_limit(model):
        result = generate(model, processor, group, args)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{group.rank()}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
