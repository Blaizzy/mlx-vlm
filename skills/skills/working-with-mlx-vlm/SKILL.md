---
name: working-with-mlx-vlm
description: Use this skill when the user wants to run MLX-VLM inference, use mlx_vlm.generate, start mlx_vlm.server, call the OpenAI-compatible API, use the chat UI, process image/audio/video inputs, choose a model, or debug basic local execution on Apple silicon.
---

# Working With MLX-VLM

Use MLX-VLM for local multimodal inference on Apple silicon. Start by identifying the user's desired surface: CLI generation, Python API, local server, chat UI, or app/client integration.

**Related skills**: `Skill("mlx-vlm-skills:model-integration")` for adding or debugging model families; `Skill("mlx-vlm-skills:serving-and-performance")` for throughput, memory, batching, caching, quantization, or speculative decoding.

## Orientation

- Check the repo first: `README.md`, `docs/usage.md`, `docs/cli_reference.md`, and model-specific READMEs in `mlx_vlm/models/<family>/README.md`.
- Prefer `python -m mlx_vlm <subcommand>` in examples unless the installed console script is clearly available.
- Do not download or run a large model unless the user asked for it. Prefer an existing local path, Hugging Face cache entry, or a small smoke-test model.
- Match the model to the modality. Image, audio, video, OCR, detection, segmentation, and image-generation models often have model-specific prompt or processor requirements.

## Choose The Surface

Use CLI generation for quick checks:

```bash
python -m mlx_vlm generate \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --prompt "Describe this image" \
  --image /path/to/image.jpg \
  --max-tokens 128
```

Use Python when the user needs programmatic control:

```python
from mlx_vlm import load
from mlx_vlm.generate import generate

model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
result = generate(
    model,
    processor,
    "Describe this image",
    image="/path/to/image.jpg",
    max_tokens=128,
)
print(result.text)
```

Use the server for client/app integration:

```bash
python -m mlx_vlm server --model mlx-community/Qwen2-VL-2B-Instruct-4bit --port 8080
curl http://localhost:8080/v1/models
```

Use `python -m mlx_vlm chat_ui` when the user wants an interactive Gradio workflow.

## Server/API Notes

- Main endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/health`, `/metrics`, and `/unload`.
- The server loads one model at a time and can preload via `--model`.
- Use `--trust-remote-code` or `MLX_TRUST_REMOTE_CODE=true` only when the model requires it and the user accepts that trust boundary.
- For OpenAI-compatible clients, test `/v1/models` before sending generation requests.

## Validation

- For CLI/Python changes, run the smallest smoke test that exercises the requested modality.
- For server changes, start the server, check `/health` or `/v1/models`, then send one minimal request.
- If generation quality matters, use deterministic settings where possible: low temperature, bounded `--max-tokens`, and a fixed prompt/input.
- If a failure mentions unsupported `model_type`, missing weights, processor mismatch, or key-shape mismatch, switch to `Skill("mlx-vlm-skills:model-integration")`.
