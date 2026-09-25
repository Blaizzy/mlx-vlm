# MLX-VLM

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on Apple silicon using [MLX](https://github.com/ml-explore/mlx). It supports vision, text, audio, video, omni, embedding, and re-ranking models through a single CLI, an OpenAI-compatible server, and a Python API.

## Documentation

- **[Getting Started](getting-started.md)** — install MLX-VLM and run your first model from the CLI or Python.
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
