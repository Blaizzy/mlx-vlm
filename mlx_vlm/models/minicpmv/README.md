# MiniCPM-V 4.6

Last updated: 2026-04-24 16:07 (UTC+8)  
Maintained by: Codex

`mlx_vlm.models.minicpmv` provides an MLX backend adaptation for **MiniCPM-V 4.6**.
The language backbone reuses `Qwen3.5`, while the vision stack uses
`SigLIP2 + MiniCPM NaViT-style bucket positional embeddings`.

The MiniCPM-V 4.6 vision path implemented here includes the two-stage visual
compression used by the original model:

- `vit_merger`: inserted after ViT layer `6`, reducing the token count by `4x`
- `merger`: applied after the full ViT output, reducing the token count by another `4x`
- with the default `downsample_mode="16x"`, a single `448x448` image produces `64` visual tokens

## Status

The current adaptation covers:

- conversion from a local Hugging Face checkpoint into MLX format
- image inference through the `mlx-vlm` CLI
- the `slice_mode=false` single-image path
- the default `slice_mode=true` slicing path
- alignment between `<image_id>...</image_id>`, `<slice>...</slice>`, and `image_bound`

The implementation has been sanity-checked locally on:

- single-image OCR prompts
- dense infographic-style images
- multi-panel poster or advertisement layouts

## Model Directories

The local directory names used during development are:

- original Hugging Face checkpoint: `minicpm-v-4_6-0420-rlaif-thinking`
- current validated MLX checkpoint: `minicpm-v-4_6-mlx`

For ongoing development, it is recommended to keep using
`minicpm-v-4_6-mlx` as the canonical MLX export directory.

If you prefer a different output directory, simply replace the `--model`
or `--mlx-path` argument in the commands below.

## Conversion

Use the current CLI entrypoint form to avoid the deprecated
`python -m mlx_vlm.convert ...` invocation.

If your environment is already activated:

```bash
python -m mlx_vlm convert \
  --hf-path minicpm-v-4_6-0420-rlaif-thinking \
  --mlx-path minicpm-v-4_6-mlx
```

If you prefer to run without activating the environment first:

```bash
conda run -n mlx-vlm python -m mlx_vlm convert \
  --hf-path minicpm-v-4_6-0420-rlaif-thinking \
  --mlx-path minicpm-v-4_6-mlx
```

After conversion, the output directory should contain at least:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `preprocessor_config.json`

## CLI Inference

### Quick OCR Sanity Check Without Slicing

For logos, simple OCR tasks, or single-subject images, start with
`slice_mode=false`:

```bash
python -m mlx_vlm generate \
  --model minicpm-v-4_6-mlx \
  --image your-image.jpg \
  --prompt "What text appears in this image?" \
  --max-tokens 80 \
  --temperature 0 \
  --processor-kwargs '{"slice_mode": false}'
```

### Recommended Default for Dense Images

For long images, infographics, collages, posters, and screenshots, the default
slicing path usually performs better:

```bash
python -m mlx_vlm generate \
  --model minicpm-v-4_6-mlx \
  --image your-dense-image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 180 \
  --temperature 0
```

## Python Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "minicpm-v-4_6-mlx"
model, processor = load(model_path)

images = ["your-image.jpg"]
prompt = "What text appears in this image?"

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(images),
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=images,
    max_tokens=80,
    temperature=0.0,
    processor_kwargs={"slice_mode": False},
)

print(result.text)
```

## Usage Notes

### When to Use `slice_mode=false`

This mode is a good fit for:

- logos
- single-subject product images
- simple OCR checks
- low-resolution images with straightforward layout

It keeps prompts shorter, runs faster, and is useful for quick regression checks.

### When to Use the Default Slicing Path

The default slicing behavior is usually better for:

- long images
- information-dense infographics
- collage-style advertisements
- screenshots
- posters and document understanding

## Notes

- This README only documents the supported production path and does not describe
  temporary debugging switches used during development.
- `batch_3d_resampler` is not used as an active inference path in this
  MiniCPM-V 4.6 adaptation.
- This document focuses on image inference. Video support is out of scope here.

## Suggested Verification Commands

After making changes, a minimal regression pass should include at least:

```bash
python -m mlx_vlm generate \
  --model minicpm-v-4_6-mlx \
  --image your-image.jpg \
  --prompt "What text appears in this image?" \
  --max-tokens 80 \
  --temperature 0 \
  --processor-kwargs '{"slice_mode": false}'
```

```bash
python -m py_compile \
  mlx_vlm/models/minicpmv/minicpmv.py \
  mlx_vlm/models/minicpmv/processing_minicpmv.py \
  mlx_vlm/models/minicpmv/vision.py \
  mlx_vlm/models/minicpmv/config.py
```
