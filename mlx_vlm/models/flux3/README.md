# FLUX 3 Image Klein

Generate images and edit photos on your Mac with Black Forest Labs' [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein). Describe what you want in a short prompt, or supply up to ten reference images with an editing instruction. Prompt expansion is enabled automatically for both tasks.

## Models

| Model | Alias | Default steps | Guidance |
|---|---|---:|---:|
| `black-forest-labs/flux-3-image-klein` | `flux3`, `flux3-klein` | 16 | 1 |
| `black-forest-labs/flux-3-image-klein-base` | `flux3-klein-base` | 50 | 3.5 |

Use `flux3-klein` for generation and editing with the default 16 steps. Keep guidance at `1` for this model. The base model defaults to 50 steps and guidance `3.5`.

## Setup

```sh
pip install -U mlx-vlm
```

Accept the model's access conditions on Hugging Face and authenticate your account before the first download. The image model and its [prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) download automatically on first use and require approximately 39 GB of disk space together. You can also use local model directories.

Use the standard BF16 checkpoints. The CUDA FP8/NVFP4 variants are unsupported.

## Command line

Generate an image:

```sh
python -m mlx_vlm generate_image \
  --model black-forest-labs/flux-3-image-klein \
  --prompt "A studio photograph of a red ceramic teapot" \
  --size 512x512 --seed 42 --output outputs/flux3.png
```

Edit it with a short instruction:

```sh
python -m mlx_vlm generate_image \
  --task edit --model flux3-klein \
  --image outputs/flux3.png \
  --prompt "Make the teapot cobalt blue" \
  --seed 42 --output outputs/flux3-edit.png
```

Repeat `--image` to combine multiple references. Refer to them as "the first image", "the second image", and so on in your prompt.

Requested image dimensions must be multiples of 16, from 64 to 4096 pixels per side. If you omit `--size` for an edit, the first reference determines the output size. Large reference images are resized automatically while preserving their aspect ratios.

You can also use `python -m mlx_vlm generate --output-modality image` with the same options.

## Python

```python
from mlx_vlm.generate.image import generate_image, load_image_model

model = load_image_model("flux3-klein")
result = generate_image(
    model,
    "A studio photograph of a red ceramic teapot",
    width=1024,
    height=1024,
    seed=42,
    output_path="outputs/flux3.png",
)
print(result.metadata["revised_prompt"])

editor = load_image_model("flux3-klein", task="edit")
result = generate_image(
    editor,
    "Make the teapot cobalt blue",
    task="edit",
    image_paths=("outputs/flux3.png",),
    seed=42,
    output_path="outputs/flux3-edit.png",
)
```

Use `result.image` to get a PIL image or `result.array` for the pixel array. Add `num_images=2` to a generation call to create a batch; `result.images` contains the individual PIL images.

## Prompt options

For editing, describe the change you want. The prompt upsampler looks at the reference images and fills in the scene description automatically. The expanded caption is available in `result.metadata["revised_prompt"]` and the API's `revised_prompt` field.

If you already have a detailed scene caption, disable expansion:

| Interface | Option |
|---|---|
| Python | `generate_image(model, caption, upsample_prompt=False)` |
| CLI | `--gen-kwargs '{"upsample_prompt": false}'` |
| API | `"upsample_prompt": false` |

To use a local upsampler or a compatible MLX conversion, set `prompt_expansion_model` when loading or generating in Python, use `--prompt-expansion-model /path/to/pup` in the CLI, or pass the `prompt_expansion_model` API field.

To place a subject in a particular part of the image, pass `boxes` in Python or an API request. Coordinates are `[top, left, bottom, right]`, normalized from 0 to 1:

```python
result = generate_image(
    model,
    "A red teapot on the left side of a wooden table",
    boxes=[{
        "id": "teapot",
        "desc": "A red ceramic teapot",
        "bbox": [0.2, 0.05, 0.8, 0.45],
    }],
    output_path="outputs/flux3-layout.png",
)
```

## API

Start the server with the image model:

```sh
python -m mlx_vlm server --image-model flux3-klein
```

Generate an image:

```sh
curl http://localhost:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "flux3-klein",
    "prompt": "A studio photograph of a red ceramic teapot",
    "size": "512x512",
    "seed": 42
  }'
```

For editing, use `/v1/images/edits` and include `"image": "/path/to/reference.png"` or a list of reference paths. Responses include a base64 PNG and the expanded prompt. To save files on the server instead, set `"response_format": "path"` and optionally provide `output_path` or `output_dir`.

## License

Based on [Black Forest Labs' reference code](https://github.com/black-forest-labs/flux-3-image-klein), copyright 2026 Black Forest Labs, Inc., under the [Apache License 2.0](https://github.com/black-forest-labs/flux-3-image-klein/blob/main/LICENSE). The image model weights use the FLUX Kommunity License; Qwen3-VL and the prompt upsampler are Apache-2.0.
