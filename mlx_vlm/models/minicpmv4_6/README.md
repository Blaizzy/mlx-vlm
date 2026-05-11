# MiniCPM-V 4.6

MiniCPM-V 4.6 is an edge-oriented vision-language model for efficient image,
multi-image, and video understanding. It combines a SigLIP2-400M vision encoder
with a Qwen3.5-0.8B language model and supports mixed `4x`/`16x` visual token
compression so applications can trade finer detail for faster inference.

## Model

| Model | Repository |
|---|---|
| bf16 | `mlx-community/MiniCPM-V-4.6-bf16` |
| 4-bit | `mlx-community/MiniCPM-V-4.6-4bit` |
| 5-bit | `mlx-community/MiniCPM-V-4.6-5bit` |
| 8-bit | `mlx-community/MiniCPM-V-4.6-8bit` |
| NVFP4 | `mlx-community/MiniCPM-V-4.6-nvfp4` |
| MXFP4 | `mlx-community/MiniCPM-V-4.6-mxfp4` |
| MXFP8 | `mlx-community/MiniCPM-V-4.6-mxfp8` |

## Details

| | |
|---|---|
| **Architecture** | SigLIP2-400M vision encoder + Qwen3.5-0.8B LLM |
| **Visual Compression** | Mixed `4x`/`16x` downsampling, with `16x` as the default efficient mode |
| **Modalities** | Text, image, multi-image, video |
| **Tasks** | Image description, visual question answering, OCR, document understanding, visual reasoning, video understanding |
| **License** | Apache-2.0 |
| **Official Card** | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) |

## CLI Usage

Image inference:

```bash
python -m mlx_vlm.generate \
    --model mlx-community/MiniCPM-V-4.6-bf16 \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-tokens 200 \
    --temperature 0.0
```

Video inference:

```bash
python -m mlx_vlm.generate \
    --model mlx-community/MiniCPM-V-4.6-bf16 \
    --video path/to/video.mp4 \
    --fps 1 \
    --prompt "Describe what happens in this video." \
    --max-tokens 300 \
    --temperature 0.0 \
    --processor-kwargs '{"max_num_frames": 32, "stack_frames": 1, "max_slice_nums": 1, "use_image_id": false}'
```

Use `downsample_mode="4x"` when a task needs finer visual detail:

```bash
python -m mlx_vlm.generate \
    --model mlx-community/MiniCPM-V-4.6-bf16 \
    --image path/to/image.jpg \
    --prompt "Read all visible text." \
    --max-tokens 300 \
    --temperature 0.0 \
    --processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 36}'
```

## Python Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "mlx-community/MiniCPM-V-4.6-bf16"
model, processor = load(model_path)

images = ["path/to/image.jpg"]
prompt = "Describe this image"

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(images),
)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=images,
    max_tokens=200,
    temperature=0.0,
)
print(result.text)
```

## Architecture

- **Vision**: SigLIP2-400M image encoder with MiniCPM-V patch packing and a
  window-attention merger inserted in the vision stack.
- **Language**: Qwen3.5-0.8B language model backbone for text generation and
  multimodal reasoning.
- **Compression**: The default `16x` visual downsampling path is optimized for
  throughput and lower token cost. The `4x` mode keeps more visual tokens for
  detail-heavy OCR, charts, documents, and small objects.
- **Efficiency**: The official model card reports more than 50% lower visual
  encoding FLOPs from the LLaVA-UHD v4 technique, plus roughly 1.5x token
  throughput compared with Qwen3.5-0.8B.
- **Deployment**: The upstream model is designed for broad edge deployment,
  including iOS, Android, and HarmonyOS.

## Notes

- The official card highlights strong single-image, multi-image, and video
  understanding while keeping the model small enough for mobile-oriented use.
- For evaluation-style prompts, keep thinking disabled when the chat template
  exposes that option. The official Transformers examples use
  `enable_thinking=False` for this mode.
- `downsample_mode` controls placeholder counts during preprocessing and the
  vision encoder path during generation. Keep it consistent across processing
  and generation when using low-level APIs.
- For video, the official card recommends `max_slice_nums=1` and
  `use_image_id=False`; the MLX video path follows the same practical defaults
  in the example above.
- The model weights and upstream code are released under Apache-2.0 by OpenBMB.
