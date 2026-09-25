# CLI Reference

mlx-vlm ships several command-line tools. This reference is written to be a practical, agent-facing map of the CLI: what each command does, its full flag set (grouped), and copy-paste examples.

## Invocation

Each command is available as an installed console script and through the `mlx_vlm` module dispatcher:

```sh
mlx_vlm.generate --help              # console script (installed by pip)
python -m mlx_vlm generate --help    # module dispatcher
```

The dispatcher also accepts `generate_image`, `generate_video`, and `generate_audio` as shortcuts for `generate --output-modality <kind>`.

!!! note
    The older dotted form `python -m mlx_vlm.generate` is **deprecated** — use one of the forms above. The one exception is fine-tuning, which is only available as `python -m mlx_vlm.lora` (it has no console script and is not part of the dispatcher).

Every command supports `--help`, the **authoritative, version-current** list of flags; the tables in these pages mirror it as of the current release.

## Which command?

| Goal | Command |
|------|---------|
| Run inference on text / image / audio / video | [`generate`](generate.md) |
| Generate images or other media | [`generate`](generate.md#image-media-generation) |
| Serve an OpenAI-compatible API | [`server`](server.md) |
| Convert & quantize a Hugging Face model to MLX | [`convert`](convert.md) |
| Interactive chat in the terminal | [`chat`](chat.md) |
| Chat in the browser (Gradio) | [Gradio chat UI](#gradio-chat-ui) |
| Fine-tune with LoRA / QLoRA / full | [`lora`](lora.md) |
| Repack a MoE checkpoint for expert offloading | [MoE offloading](#moe-offloading) |
| Evaluations & model-specific tools | [Specialized CLIs](specialized.md) |

## Conventions

- **Model argument** — a Hugging Face repo id (e.g. `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`) or a local path. Used by `--model` (`generate`/`server`/`chat`) and `--hf-path` (`convert`).
- **Media inputs** — `--image` and `--audio` accept multiple space-separated values, and each may be a URL or a local file path.
- **Greedy decoding** — pass `--temperature 0`.
- **Remote code** — some models require `--trust-remote-code`.
- **Extras** — quote the package when installing optional groups, e.g. `pip install -U 'mlx-vlm[ui]'`. See [Getting Started](../getting-started.md#optional-extras).

## Gradio chat UI

`mlx_vlm.chat_ui` launches a browser chat interface. It requires the optional `ui` extra (`pip install -U 'mlx-vlm[ui]'`).

```sh
mlx_vlm.chat_ui --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | Hugging Face repo id or local path to load |

## MoE offloading

`mlx_vlm.moe_offload` repacks a Mixture-of-Experts checkpoint so experts can be streamed from disk, reducing resident memory.

```sh
mlx_vlm.moe_offload --build <mlx_model_path> --out <offload_path>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--build` | — | Path to the MLX MoE checkpoint to repack |
| `--out` | — | Output path for the offload-ready checkpoint |
| `--resident-shard-gb` | — | Target size (GB) of the resident (always-in-memory) shard |

For evaluation harnesses and per-model generate/convert scripts, see [Specialized CLIs](specialized.md).
