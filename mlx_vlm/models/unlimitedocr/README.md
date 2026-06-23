# Unlimited-OCR

Unlimited-OCR is an OCR and document-understanding model from [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR). It extends the DeepSeek-OCR family with a DeepSeek-V2 language backbone, SAM ViT-B global/local visual features, and a CLIP-L style vision tower.

## Model Architecture

```
                    ┌─────────────────────────────────────────────────────────────────────┐
                    │                         Dynamic Resolution                           │
                    ├─────────────────────────────────────────────────────────────────────┤
Local Patches       │  640×640 → SAM (10×10×1024) → CLIP-L (101×1024) → Proj (100×1280)  │
(0-32 patches)      │                                                                      │
                    ├─────────────────────────────────────────────────────────────────────┤
Global View         │ 1024×1024 → SAM (16×16×1024) → CLIP-L (257×1024) → Proj (256×1280) │
(1 image)           │                                                                      │
                    └─────────────────────────────────────────────────────────────────────┘
                                                        ↓
                              [local_patches, global_view, view_separator]
                                                        ↓
                                            Language Model (DeepSeek-V2)
```

The upstream model has two common image modes:

| Mode | Settings | Use case |
|------|----------|----------|
| `gundam` | `base_size=1024`, `image_size=640`, `cropping=True` | Default single-image OCR with dynamic crops |
| `base` | `image_size=1024`, `cropping=False` | Global-view-only OCR; used by upstream for multi-page/PDF workflows |

## Prompt Formats

When using `mlx_vlm.generate`, pass the task text only. The CLI applies the image-token prompt template automatically, so do **not** manually prefix CLI prompts with `<image>`.

### Document to Markdown
Convert a document image to structured markdown format:
```
<|grounding|>Convert the document to markdown.
```

### General OCR
Extract all text from an image with structured layout/grounding:
```
<|grounding|>OCR this image.
```

### Free OCR
Extract text without explicitly requesting structured layout:
```
Free OCR.
```

### Parse Figures
Extract and describe figures/charts in documents:
```
Parse the figure.
```

### Text Localization (Grounding)
Locate specific text in the image and get bounding box coordinates:
```
Locate <|ref|>your text here<|/ref|> in the image.
```

Output commonly uses `<|det|>label [x1, y1, x2, y2]<|/det|>` with coordinates normalized to the 0-1000 range.

## CLI Examples

### Free OCR (default `gundam` mode)
```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "Free OCR." \
    --max-tokens 1000
```

### Document to Markdown
```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image paper_page.png \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max-tokens 2000
```

### Global-view-only `base` mode
```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "Free OCR." \
    --max-tokens 1000 \
    --processor-kwargs '{"cropping": false, "image_size": 1024}'
```

### Multi-page OCR from rendered PDF pages
Render PDF pages to images first, then pass the page images in order. The
Unlimited-OCR prompt template intentionally inserts a single `<image>` token for
all pages, matching upstream `infer_multi` behavior.

```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image page-001.png page-002.png page-003.png \
    --prompt "Multi page parsing." \
    --max-tokens 4000 \
    --processor-kwargs '{"cropping": false, "image_size": 1024}'
```

### Text Localization
```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image table.jpeg \
    --prompt "Locate <|ref|>Total assets<|/ref|> in the image." \
    --max-tokens 100
```

## Python Script Examples

### Basic OCR
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

prompt = "Free OCR."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    temperature=0.0,
)
print(result.text)
```

### Document to Markdown
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

prompt = "<|grounding|>Convert the document to markdown."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model=model,
    processor=processor,
    image="paper_page.png",
    prompt=formatted_prompt,
    max_tokens=2000,
    temperature=0.0,
)
print(result.text)
```

### Global-view-only `base` mode
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

prompt = "Free OCR."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    temperature=0.0,
    cropping=False,
    image_size=1024,
)
print(result.text)
```

### Batch Processing
```python
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

image_dir = Path("documents/")
for image_path in image_dir.glob("*.png"):
    result = generate(
        model=model,
        processor=processor,
        image=str(image_path),
        prompt=formatted_prompt,
        max_tokens=1000,
        temperature=0.0,
    )
    print(f"\n--- {image_path.name} ---")
    print(result.text)
```

## Dynamic Resolution

Unlimited-OCR uses dynamic resolution in its `gundam` mode:

**Default configuration:**
- **Global view**: 1×1024×1024 → 16×16 visual grid with row newlines + view separator
- **Local patches**: 0-32 patches at 640×640 → 10×10 visual grid per patch with row newlines
- **Small images**: images with width and height `<= 640` use only the global view

**How it works:**
1. The image aspect ratio is used to select a local patch grid, up to 32 patches
2. A padded 1024×1024 global view captures page-level context
3. Local 640×640 crops capture fine text details when the image is large
4. Features are packed as `[local_patches, global_view, view_separator]` and inserted at `<image>` token positions

**Token calculation:**
- Global view at 1024: `(16 image tokens + 1 newline) × 16 rows + 1 view separator = 273 tokens`
- Each 640 local patch grid contributes 100 image tokens plus row newlines after tiling
- With local crops: total image tokens grow with the selected patch grid, up to 32 crops

### Controlling Dynamic Resolution

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

prompt = "Free OCR."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# Default upstream-style gundam mode
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    cropping=True,
    image_size=640,
    base_size=1024,
)

# Faster global-view-only base mode
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    cropping=False,
    image_size=1024,
)
```

### CLI Examples for Dynamic Resolution

```bash
# Default: gundam mode with dynamic crops
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "Free OCR." \
    --max-tokens 1000

# Base mode: global view only
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "Free OCR." \
    --max-tokens 1000 \
    --processor-kwargs '{"cropping": false, "image_size": 1024}'
```

## Special Tokens

| Token | Description |
|-------|-------------|
| `<image>` | Image placeholder inserted by the prompt template |
| `<\|grounding\|>` | Enable grounding/structured output mode |
| `<\|ref\|>...<\|/ref\|>` | Mark text to locate in the image |
| `<\|det\|>...<\|/det\|>` | Bounding box output format |
| `<\|User\|>` | User turn marker |
| `<\|Assistant\|>` | Assistant turn marker |

## Tips

1. Use `temperature=0.0` for deterministic OCR output.
2. Increase `max_tokens` for full pages or dense documents.
3. Use `cropping=False, image_size=1024` for faster global-view-only checks.
4. Use default `cropping=True, image_size=640` for large pages where small text detail matters.
5. Do not manually include `<image>` in CLI prompts; `mlx_vlm` inserts it for this model family.

## Limitations

- The current `mlx_vlm` image loader expects image files. For PDFs, render pages to images first before passing them to the CLI.
- Very dense pages may need high `max_tokens` to avoid truncating the output.
- Dynamic cropping can be slower and uses more memory than global-view-only mode.
- Localization coordinates are normalized to 0-1000 and should be scaled to the source image size.

## Implementation Notes

Unlimited-OCR uses Hugging Face `model_type: unlimited-ocr`. The MLX module name is `unlimitedocr`, and `mlx_vlm` remaps the Hugging Face model type automatically.

The processor loads the tokenizer and `processor_config.json` directly to avoid requiring upstream PyTorch remote-code dependencies during MLX inference.

The language model uses Unlimited-OCR's ring sliding-window attention cache for
generation. Prefill tokens are retained, and generated tokens are kept in a
fixed-size decode ring, which bounds decode-time KV-cache growth for long OCR
outputs.
