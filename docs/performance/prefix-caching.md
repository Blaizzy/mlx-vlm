# Automatic Prefix Caching (APC)

Automatic Prefix Caching reuses block-level K/V cache state across requests that share the same prefix. It is useful for repeated long documents, long chat histories, or retrieval contexts where each request appends a short new suffix.

APC has two tiers:

- **Warm memory**: keeps reusable `APCBlock` tensors in process memory. This is the fastest path, but it keeps both the reusable block pool and the runtime `KVCache`.
- **Warm disk**: persists cached prefixes as safetensors shards so they survive process restarts. Warm-disk reads build the layer-major prompt cache directly without promoting restored blocks into the `APCBlock` pool; writes can still populate both memory and disk tiers.

## Python Script

Use `APCManager` directly when calling `stream_generate`:

```python
from pathlib import Path

from mlx_vlm import load, stream_generate
from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.prompt_utils import apply_chat_template

model_id = "Qwen/Qwen3-VL-4B-Instruct"
model, processor = load(model_id)

disk = DiskBlockStore(
    Path("~/.cache/mlx-vlm/caching").expanduser(),
    namespace=model_id,
    max_bytes=3 * (1 << 30),  # 3 GB disk cap; use None for uncapped
)
apc = APCManager(num_blocks=4096, block_size=16, disk=disk)

document = Path("long_document.txt").read_text()

try:
# First request computes the full prefix and stores reusable K/V blocks.
    prompt1 = apply_chat_template(
        processor,
        model.config,
        prompt=f"{document}\n\nSummarize the key decisions.",
        num_images=0,
    )
    for _ in stream_generate(
        model, processor, prompt1, max_tokens=128, temperature=0.0, apc_manager=apc
    ):
        pass

# Second request shares the same document prefix and only prefills the suffix.
    prompt2 = apply_chat_template(
        processor,
        model.config,
        prompt=f"{document}\n\nList the open engineering risks.",
        num_images=0,
    )
    for chunk in stream_generate(
        model, processor, prompt2, max_tokens=128, temperature=0.0, apc_manager=apc
    ):
        print(chunk.text, end="", flush=True)

    print(apc.stats_snapshot())
finally:
    apc.close()
```

To compare cold, warm-memory, warm-disk, and disk-eviction behavior with a
model, use the same direct API path:

```python
import os
import tempfile
import time
from pathlib import Path

from mlx_vlm import load, stream_generate
from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.prompt_utils import apply_chat_template

model_id = "Qwen/Qwen3-VL-4B-Instruct"
contexts = [8000, 20000, 50000, 100000]
disk_cap_gb = 0  # 0 means uncapped
shard_max_blocks = 256
context_sweep_max_tokens = 1  # one token is enough to measure prefill reuse

test_prompt_tokens = 8000
fill_prompts = 80
eviction_disk_cap_gb = 3.0

os.environ["APC_DISK_SHARD_MAX_BLOCKS"] = str(shard_max_blocks)

model, processor = load(model_id)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor


def disk_cap_bytes(gb: float):
    return None if gb <= 0 else int(gb * (1 << 30))


def make_context(target_tokens: int, seed: int = 0) -> str:
    line = (
        f"Document {seed}: APC benchmark content with deterministic facts, "
        "dates, identifiers, and repeated technical notes.\n"
    )
    line_tokens = max(1, len(tokenizer.encode(line, add_special_tokens=False)))
    text = line * max(1, target_tokens // line_tokens)
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += line
    return text


def make_prompt(context: str, question: str) -> str:
    return apply_chat_template(
        processor,
        model.config,
        prompt=f"{context}\n\n{question}",
        num_images=0,
    )


def run_once(apc: APCManager, context: str, question: str, max_tokens: int = 32):
    prompt = make_prompt(context, question)
    apc.reset_stats()

    last = None
    output = []
    start = time.perf_counter()
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        apc_manager=apc,
    ):
        output.append(chunk.text)
        last = chunk

    if last is None:
        raise RuntimeError("generation returned no chunks")

    return {
        "wall_s": time.perf_counter() - start,
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": last.prompt_tps,
        "generation_tps": last.generation_tps,
        "apc": apc.stats_snapshot(),
        "text": "".join(output).strip(),
    }


def print_result(label: str, result: dict) -> None:
    stats = result["apc"]
    print(
        f"{label:<12} "
        f"prompt_tokens={result['prompt_tokens']:>7} "
        f"prompt_tps={result['prompt_tps']:>8.1f} "
        f"gen_tps={result['generation_tps']:>7.1f} "
        f"matched={stats.get('matched_tokens', 0):>7} "
        f"disk_hits={stats.get('disk_hits', 0):>5} "
        f"disk_evictions={stats.get('disk_evictions', 0):>5}"
    )


def open_apc(cache_root: Path, namespace: str, disk_gb: float) -> APCManager:
    disk = DiskBlockStore(
        cache_root,
        namespace=namespace,
        max_bytes=disk_cap_bytes(disk_gb),
    )
    return APCManager(num_blocks=4096, block_size=16, disk=disk)


def run_context_sweep() -> None:
    print("cold / warm-memory / warm-disk")
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp)
        for target_tokens in contexts:
            context = make_context(target_tokens)
            namespace = f"{model_id}-context-{target_tokens}"
            apc = open_apc(cache_root, namespace, disk_cap_gb)
            try:
                print(f"\ncontext ~= {target_tokens} text tokens")
                print_result(
                    "cold",
                    run_once(
                        apc,
                        context,
                        "Summarize the key decisions.",
                        max_tokens=context_sweep_max_tokens,
                    ),
                )
                print_result(
                    "warm-memory",
                    run_once(
                        apc,
                        context,
                        "List the open engineering risks.",
                        max_tokens=context_sweep_max_tokens,
                    ),
                )
            finally:
# Closing waits for queued disk writes before reopening the disk tier.
                apc.close()

            apc = open_apc(cache_root, namespace, disk_cap_gb)
            try:
                print_result(
                    "warm-disk",
                    run_once(
                        apc,
                        context,
                        "Extract the implementation timeline.",
                        max_tokens=context_sweep_max_tokens,
                    ),
                )
            finally:
                apc.close()


def run_disk_eviction_workload() -> None:
    print("\ndisk eviction workload")
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp)
        namespace = f"{model_id}-eviction"
        test_context = make_context(test_prompt_tokens, seed=0)

        apc = open_apc(cache_root, namespace, eviction_disk_cap_gb)
        try:
            print_result(
                "seed",
                run_once(apc, test_context, "Summarize the retained test prefix."),
            )
        finally:
            apc.close()

        apc = open_apc(cache_root, namespace, eviction_disk_cap_gb)
        try:
            for i in range(fill_prompts):
                fill_context = make_context(test_prompt_tokens, seed=i + 1)
                run_once(
                    apc,
                    fill_context,
                    f"Summarize filler document {i + 1}.",
                    max_tokens=1,
                )
                if (i + 1) % 10 == 0:
                    stats = apc.stats_snapshot()
                    print(
                        f"filled={i + 1:>3} "
                        f"disk_gb={stats.get('disk_bytes', 0) / (1 << 30):.2f} "
                        f"disk_evictions={stats.get('disk_evictions', 0)}"
                    )
        finally:
            apc.close()

        apc = open_apc(cache_root, namespace, eviction_disk_cap_gb)
        try:
            print_result(
                "post-fill",
                run_once(
                    apc,
                    test_context,
                    "Check whether the retained test prefix still restores.",
                ),
            )
        finally:
            apc.close()


run_context_sweep()
run_disk_eviction_workload()
```

## Server

Enable in-memory APC for the server with environment variables:

```sh
APC_ENABLED=1 \
APC_NUM_BLOCKS=4096 \
mlx_vlm.server --model Qwen/Qwen3-VL-4B-Instruct --port 8080
```

APC works with KV-cache quantization (`--kv-bits`):

```sh
APC_ENABLED=1 \
APC_NUM_BLOCKS=4096 \
mlx_vlm.server --model Qwen/Qwen3-VL-4B-Instruct --kv-bits 8 --port 8080
```

Enable the persistent disk tier:

```sh
APC_ENABLED=1 \
APC_NUM_BLOCKS=4096 \
APC_DISK_PATH=~/.cache/mlx-vlm/caching \
APC_DISK_MAX_GB=3 \
APC_DISK_SHARD_MAX_BLOCKS=256 \
mlx_vlm.server --model Qwen/Qwen3-VL-4B-Instruct --port 8080
```

Repeated requests with the same long prefix will hit APC automatically:

```sh
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-APC-Tenant: demo" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [{
      "role": "user",
      "content": "Paste a long shared document here.\n\nNow answer question A."
    }],
    "max_tokens": 128
  }'
```

Use the same `X-APC-Tenant` value for requests that may share cached prefixes. Use different tenant values to isolate cache entries between users or workspaces.

Inspect and reset APC state:

```sh
curl http://localhost:8080/v1/cache/stats
curl -X POST http://localhost:8080/v1/cache/reset
```

Common APC environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APC_ENABLED` | `0` | Set to `1` to enable APC |
| `APC_NUM_BLOCKS` | `2048` | Number of in-memory APC blocks |
| `APC_BLOCK_SIZE` | `16` | Tokens per APC block |
| `APC_DISK_PATH` | unset | Directory for persistent disk shards |
| `APC_DISK_MAX_GB` | `0` | Disk cap in GB; `0` means uncapped |
| `APC_DISK_SHARD_MAX_BLOCKS` | `256` | Max blocks per disk segment shard |
| `APC_MAX_POOL_TENSORS` | `450000` | Stops adding memory blocks before the Metal resource limit; disk writes continue |
| `APC_LAYER_MAJOR_MEMORY_MIN_TOKENS` | `50000` | Store long warm-memory prefixes as compact layer-major snapshots instead of per-block tensors |
| `APC_HASH` | `fast` | Set to `sha256` for a stable cryptographic hash |
| `APC_TRACE` | unset | Set to `1` for greppable store/reject/self-check log lines |

APC is disabled automatically for models that use a custom cache layout. APC works with `--kv-bits` (including TurboQuant): the live KV cache stays quantized; the reusable APC pool stores dequantized float K/V, so pool size does not shrink with quant.
When APC is enabled on the server, a non-fatal layout self-check runs at model load.
