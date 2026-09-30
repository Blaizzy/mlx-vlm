# CLI Reference

MLX-VLM provides several command line entry points:

Recommended invocation style:

```bash
python -m mlx_vlm <subcommand> ...
```

For example:

```bash
python -m mlx_vlm convert --help
python -m mlx_vlm generate --help
```

- `mlx_vlm.decide` – predict typed decisions from text and named questions.
- `mlx_vlm.convert` – convert Hugging Face models to MLX format.
- `mlx_vlm.generate` – run inference on images, audio, or video.
- `mlx_vlm.chat_ui` – start an interactive Gradio UI.
- `mlx_vlm.server` – run the FastAPI server.
- `mlx_vlm.moe_offload` – repack a MoE checkpoint for expert offloading.

Each command accepts `--help` for full usage information.

Decision CLI and HTTP examples, including `--decision-model` preloading, are in
the [decision models guide](https://github.com/Blaizzy/mlx-vlm#decision-models).
