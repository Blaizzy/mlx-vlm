# MiniCPM-V 4.6

`mlx_vlm.models.minicpmv4_6` provides an MLX backend adaptation for **MiniCPM-V 4.6**.
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
- video inference through sampled frame processing
- alignment between `<image_id>...</image_id>`, `<slice>...</slice>`, and `image_bound`

The implementation has been sanity-checked locally on:

- single-image OCR prompts
- dense infographic-style images
- multi-panel poster or advertisement layouts
- short sampled video clips

## Model Checkpoints

The currently validated checkpoints are:

- original Hugging Face checkpoint: `openbmb/MiniCPM-V-4.6`
- canonical MLX checkpoint: `mlx-community/MiniCPM-V-4.6-bf16`
- quantized MLX checkpoints:
  - `mlx-community/MiniCPM-V-4.6-4bit`
  - `mlx-community/MiniCPM-V-4.6-5bit`
  - `mlx-community/MiniCPM-V-4.6-8bit`
  - `mlx-community/MiniCPM-V-4.6-nvfp4`
  - `mlx-community/MiniCPM-V-4.6-mxfp4`
  - `mlx-community/MiniCPM-V-4.6-mxfp8`

For ongoing development, it is recommended to keep using
`mlx-community/MiniCPM-V-4.6-bf16` as the canonical validated MLX checkpoint.

If you prefer a different output directory, simply replace the `--model`
or `--mlx-path` argument in the commands below.

## Conversion

Use the current CLI entrypoint form to avoid the deprecated
`python -m mlx_vlm.convert ...` invocation.

If your environment is already activated:

```bash
python -m mlx_vlm convert \
  --hf-path openbmb/MiniCPM-V-4.6 \
  --mlx-path MiniCPM-V-4.6-bf16 \
  --dtype bfloat16
```

If you prefer to run without activating the environment first:

```bash
conda run -n mlx-vlm python -m mlx_vlm convert \
  --hf-path openbmb/MiniCPM-V-4.6 \
  --mlx-path MiniCPM-V-4.6-bf16 \
  --dtype bfloat16
```

After conversion, the output directory should contain at least:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `preprocessor_config.json`

## CLI Inference

### Video Sanity Check

For video inputs, keep the first validation pass small by sampling a few frames
and disabling slicing:

```bash
python -m mlx_vlm generate \
  --model mlx-community/MiniCPM-V-4.6-bf16 \
  --video your-video.mp4 \
  --fps 0.25 \
  --prompt "Describe this video briefly." \
  --max-tokens 120 \
  --temperature 0 \
  --processor-kwargs '{"slice_mode": false, "max_num_frames": 4}'
```

### Quick OCR Sanity Check Without Slicing

For logos, simple OCR tasks, or single-subject images, start with
`slice_mode=false`:

```bash
python -m mlx_vlm generate \
  --model mlx-community/MiniCPM-V-4.6-bf16 \
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
  --model mlx-community/MiniCPM-V-4.6-bf16 \
  --image your-dense-image.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 180 \
  --temperature 0
```

## Python Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/MiniCPM-V-4.6-bf16"
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
- Video inputs are sampled into frame images and then passed through the same
  MiniCPM-V visual placeholder alignment used by image inference.

## Suggested Verification Commands

After making changes, a minimal regression pass should include at least:

```bash
python -m mlx_vlm generate \
  --model mlx-community/MiniCPM-V-4.6-bf16 \
  --image your-image.jpg \
  --prompt "What text appears in this image?" \
  --max-tokens 80 \
  --temperature 0 \
  --processor-kwargs '{"slice_mode": false}'
```

```bash
python -m py_compile \
  mlx_vlm/models/minicpmv4_6/minicpmv4_6.py \
  mlx_vlm/models/minicpmv4_6/processing_minicpmv4_6.py \
  mlx_vlm/models/minicpmv4_6/vision.py \
  mlx_vlm/models/minicpmv4_6/config.py
```
