# MiniCPM-OCR 1.0

MiniCPM-OCR 1.0 is a 1.3B document parsing model with a SigLIP vision encoder
and a Qwen3.5 language backbone. This port loads the supplied Transformers
checkpoint directly and supports MLX conversion, generation, image slicing,
`4x`/`16x` visual compression, and adaptive image/tile upsize.

## Convert and run

```bash
mlx_vlm.convert --hf-path /path/to/MiniCPM-OCR-1.0 \
  --mlx-path /path/to/MiniCPM-OCR-1.0-bf16 --dtype bfloat16

mlx_vlm.generate --model /path/to/MiniCPM-OCR-1.0-bf16 \
  --image page.png --prompt "Convert this page to markdown." \
  --temperature 0 --max-tokens 4096
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("/path/to/MiniCPM-OCR-1.0-bf16")
prompt = apply_chat_template(
    processor, model.config, "Convert this page to markdown.", num_images=1,
)
result = generate(model, processor, prompt, image=["page.png"], max_tokens=4096,
                  temperature=0.0)
print(result.text)
```

The checkpoint defaults to `downsample_mode="4x"`. Passing `16x` reduces visual
sequence length. The processor returns the selected mode with its inputs so
placeholder counts and the vision encoder agree. The checkpoint's chat template
is preserved, including its non-thinking prefix.

Adaptive upsize follows the supplied reference processor. Enable it with:

```bash
mlx_vlm.generate --model /path/to/MiniCPM-OCR-1.0-bf16 \
  --image page.png --prompt "Convert this page to markdown." \
  --processor-kwargs '{"adaptive_small_image_upsize": true, "adaptive_tile_upsize": true, "disable_small_image_upsize": true}'
```

Small dense text retains the full resolution budget, tiny crops use a smaller
budget, and other crops can retain their native size. This uses OpenCV, which is
already a project dependency. Each slice gets its own placeholder count. These
settings survive processor save/reload.

Language RMSNorm scales remain FP32 so adding the checkpoint's unit offset does
not round away BF16 weight values; activations retain their input dtype. The
window merger uses tanh GELU, while the final projection uses exact GELU.

For OCR-output assessment, see [MiniCPM-OCR Jev](../minicpm_ocr_jev/README.md).
