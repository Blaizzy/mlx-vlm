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

- `mlx_vlm.convert` – convert Hugging Face models to MLX format.
- `mlx_vlm.generate` – run inference on images.
- `mlx_vlm.video_generate` – generate from a video file.
- `mlx_vlm.smolvlm_video_generate` – lightweight video generation.
- `mlx_vlm.chat_ui` – start an interactive Gradio UI.
- `mlx_vlm.server` – run the FastAPI server.

Each command accepts `--help` for full usage information.
