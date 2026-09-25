# mlx-vlm

mlx-vlm is a fast and simple local inference engine that supports vision, text, audio, video, omni, embedding, and re-ranking models through a single CLI, an OpenAI/Anthropic-compatible server, and a Python API built ontop of [mlx](https://github.com/ml-explore/mlx).



## Documentation

- **[Getting Started](getting-started.md)** — install mlx-vlm and run your first model from the CLI or Python.
- **[Models](models.md)** — the catalog of supported model families, by modality.
- **[CLI Reference](cli/index.md)** — every command and flag: `generate`, `server`, `convert`, `chat`, `lora`, and specialized tools.
- **[Server](server.md)** — the OpenAI-compatible FastAPI server.
- **[Fine-tuning](fine-tuning.md)** — LoRA and QLoRA training.
- **Performance** — [Speculative Decoding](performance/speculative-decoding.md), [Prefix Caching](performance/prefix-caching.md), [Quantization](performance/quantization.md), and [KV Cache Quantization](performance/kv-cache-quantization.md).

## Installation

```sh
pip install -U mlx-vlm
```

See [Getting Started](getting-started.md) for optional extras and the full quickstart.
