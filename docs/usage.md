# Usage

## Command Line Interface (CLI)

Generate output from a model:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

## Chat UI with Gradio

Launch the chat interface:

```bash
python -m mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

## Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## MoE Offloading

Run a mixture-of-experts checkpoint that is larger than available RAM by paging routed experts from disk. Repack the checkpoint into an offloaded store once, then load it as usual — `load()` detects the offloaded layout automatically:

```bash
python -m mlx_vlm moe_offload --build /path/to/checkpoint --out /path/to/offloaded
python -m mlx_vlm.generate --model /path/to/offloaded --prompt "Explain how photosynthesis works." --max-tokens 100
```

Serving works the same way. Bound the resident expert set with `--expert-cache-gb`, and inspect cache hits/misses/evictions at `/v1/moe-offload/stats`:

```bash
python -m mlx_vlm.server --model /path/to/offloaded --expert-cache-gb 8
```

## Speculative Decoding

The rebuilt path supports GLM-5.3-Flash with its native MTP head.

```bash
python -m mlx_vlm.split_mtp --model zai-org/GLM-5.3-Flash \
    --output GLM-5.3-Flash-MTP-FP8 --q-mode mxfp8

mlx_vlm.generate --model zai-org/GLM-5.3-Flash \
    --draft-model GLM-5.3-Flash-MTP-FP8 --draft-kind mtp \
    --draft-block-size 2 --temperature 0 \
    --prompt "Explain why the sky is blue."

mlx_vlm.server --model zai-org/GLM-5.3-Flash \
    --draft-model GLM-5.3-Flash-MTP-FP8 --draft-kind mtp
```

The verification block size includes one target token. Start with `2` (one
draft), then measure larger blocks on your workload. The former DFlash,
EAGLE, and other MTP implementations have been removed.
See [speculative decoding](speculative-decoding.md) for the cache contract,
sampling behavior, current limits, and FP8 parity benchmark.

## Server (FastAPI)

```bash
python -m mlx_vlm.server
```

See `README.md` for a complete `curl` example.

### Live settings (`/v1/settings`)

Read and change a curated set of server settings at runtime, without a
restart. `GET` lists the settings the server accepts; `PATCH` changes them.

```bash
# list the available settings and their current values
curl http://127.0.0.1:8080/v1/settings

# merge: only the settings you list are changed
curl -X PATCH http://127.0.0.1:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"kv_quant_scheme": "turboquant"}'

# replace: reset everything to its boot-time default, then apply these
curl -X PATCH http://127.0.0.1:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"op": "replace", "values": {"apc_enabled": true}}'
```

Changes take effect on the next request. Most settings reload the affected
model first — KV, APC, and speculative-decoding settings reload text models,
`vision_cache_size` reloads image models — while `max_kv_size` and
`token_queue_timeout` apply to new requests without a reload.

The response reports which settings were applied and which were rejected;
unknown names and invalid values are rejected and never applied.

## Distributed Inference

mlx-vlm supports distributed inference across multiple computers. It works by sharding the language model (not the vision tower), because the LLM is much larger and vision embeddings only need to be computed once.

The parallel implementation is compatible with mlx-lm sharding primitives.

The following command shows how you can run Kimi K2.6, a 1T parameter model, on several computers. For a smaller option, you can try `[mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16](https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16)`.

```bash
mlx.launch \
    --hostfile ring-thunderbolt.json \
    --backend jaccl \
    --hostfile /path/to/hosts.json \
    --env MLX_METAL_FAST_SYNCH=1 \
    -- \
    mlx-vlm/examples/sharded_generate.py \
    --model moonshotai/Kimi-K2.6 \
    --prompt "Describe this image" \
    --image mx-vlm/examples/images/scene_1.jpg
```

We recommend you use the JACCL protocol over Thunderbolt. For more information, please refer to [the MLX distributed communication guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).
