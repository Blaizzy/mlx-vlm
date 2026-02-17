# DeepSeek-OCR

DeepSeek-OCR is a powerful OCR model based on the SAM + Qwen2 encoder architecture, optimized for document understanding, text extraction, and visual grounding tasks.

## Model Architecture

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                      Dynamic Resolution                          │
                    ├─────────────────────────────────────────────────────────────────┤
Local Patches       │  768×768 → SAM (12×12×896) → Qwen2 (144×896) → Proj (144×1024) │
(1-6 patches)       │                                                                  │
                    ├─────────────────────────────────────────────────────────────────┤
Global View         │ 1024×1024 → SAM (16×16×896) → Qwen2 (256×896) → Proj (256×1024)│
(1 image)           │                                                                  │
                    └─────────────────────────────────────────────────────────────────┘
                                                    ↓
                              [local_patches, global_view, view_separator]
                                                    ↓
                                          Language Model (DeepSeek-V2)
```

## Prompt Formats

### Document to Markdown
Convert a document image to structured markdown format:
```
<image>
<|grounding|>Convert the document to markdown.
```

### General OCR
Extract all text from an image:
```
<image>
<|grounding|>OCR this image.
```

### Free OCR (without layout)
Extract text without preserving layout structure:
```
<image>
Free OCR.
```

### Parse Figures
Extract and describe figures/charts in documents:
```
<image>
Parse the figure.
```

### Image Description
Get a detailed description of the image:
```
<image>
Describe this image in detail.
```

### Text Localization (Grounding)
Locate specific text in the image and get bounding box coordinates:
```
<image>
Locate <|ref|>your text here<|/ref|> in the image.
```

Output format: `<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`

Coordinates are normalized to 0-1000 range.

## CLI Examples

### Document to Markdown
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max-tokens 2000
```

### General OCR
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image receipt.jpg \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000
```

### Free OCR
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image text_image.png \
    --prompt "Free OCR." \
    --max-tokens 500
```

### Text Localization
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image table.jpeg \
    --prompt "Locate <|ref|>Total assets<|/ref|> in the image." \
    --max-tokens 100
```

## Python Script Examples

### Basic OCR
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Load model
model, processor = load("mlx-community/DeepSeek-OCR-bf16")

# OCR prompt
prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# Generate
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

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

prompt = "<|grounding|>Convert the document to markdown."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model=model,
    processor=processor,
    image="paper.pdf",  # Also works with PDF pages
    prompt=formatted_prompt,
    max_tokens=2000,
    temperature=0.0,
)
print(result.text)
```

### Text Localization
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

# Locate specific text
text_to_find = "Total liabilities"
prompt = f"Locate <|ref|>{text_to_find}<|/ref|> in the image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model=model,
    processor=processor,
    image="financial_table.png",
    prompt=formatted_prompt,
    max_tokens=100,
    temperature=0.0,
)

# Parse bounding box from output
# Output format: <|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
import re
match = re.search(r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]', result.text)
if match:
    x1, y1, x2, y2 = map(int, match.groups())
    # Coordinates are normalized to 0-1000
    print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
```

### Batch Processing
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from pathlib import Path

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# Process multiple images
image_dir = Path("documents/")
for image_path in image_dir.glob("*.png"):
    result = generate(
        model=model,
        processor=processor,
        image=str(image_path),
        prompt=formatted_prompt,
        max_tokens=1000,
    )
    print(f"\n--- {image_path.name} ---")
    print(result.text)
```

## Dynamic Resolution

DeepSeek-OCR uses dynamic resolution to handle images of varying sizes efficiently:

**Default Configuration:**
- **Global view**: 1×1024×1024 → 256 visual tokens
- **Local patches**: (1-6)×768×768 → (1-6)×144 visual tokens each
- **View separator**: 1 token

**How it works:**
1. The image is analyzed for aspect ratio to determine the optimal patch grid
2. Local patches (768×768) capture fine details based on the grid layout
3. A global view (1024×1024) captures the overall context
4. Features are concatenated: [local_patches, global_view, view_separator]

**Token calculation:**
- Each local patch: 144 tokens (from 12×12 SAM features via query_768)
- Global view: 256 tokens (from 16×16 SAM features via query_1024)
- View separator: 1 token
- **Total: (num_patches × 144) + 256 + 1 visual tokens**

**Example patch counts by aspect ratio:**
| Image Size | Aspect | Grid | Patches | Total Tokens |
|------------|--------|------|---------|--------------|
| 800×600 | 4:3 | 3×2 | 6 | 1121 |
| 600×800 | 3:4 | 2×3 | 6 | 1121 |
| 1200×400 | 3:1 | 3×1 | 3 | 689 |
| 400×1200 | 1:3 | 1×3 | 3 | 689 |
| 1000×1000 | 1:1 | 1×1 | 1 | 401 |

### Controlling Dynamic Resolution

You can control the number of patches via the `cropping`, `min_patches`, and `max_patches` parameters:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# Default: dynamic resolution with 1-6 patches
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    # cropping=True, min_patches=1, max_patches=6 (defaults)
)

# Global view only (faster, 257 tokens)
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    cropping=False,
)

# Limit patches for balance of speed vs detail
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    cropping=True,
    min_patches=1,
    max_patches=3,
)
```

**Token counts by configuration:**

| Configuration | Patches | Tokens |
|---------------|---------|--------|
| `cropping=False` | 0 | 257 |
| `max_patches=1` | 1 | 401 |
| `max_patches=3` | 1-3 | 401-689 |
| `max_patches=6` (default) | 1-6 | 401-1121 |

### CLI Examples for Dynamic Resolution

```bash
# Default: dynamic resolution with 1-6 patches
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000

# Global view only (faster, 257 tokens)
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000 \
    --processor-kwargs '{"cropping": false}'

# Limit to 3 patches max
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000 \
    --processor-kwargs '{"cropping": true, "max_patches": 3}'
```

## Special Tokens

| Token | Description |
|-------|-------------|
| `<image>` | Image placeholder in the prompt |
| `<\|grounding\|>` | Enable grounding/structured output mode |
| `<\|ref\|>...<\|/ref\|>` | Mark text to locate in the image |
| `<\|det\|>...<\|/det\|>` | Bounding box output format |
| `<\|User\|>` | User turn marker |
| `<\|Assistant\|>` | Assistant turn marker |

## Tips

1. **For best OCR results**, use `<|grounding|>` prefix to enable structured output mode
2. **For tables**, the model outputs HTML table format automatically
3. **Temperature 0.0** is recommended for OCR tasks to get deterministic results
4. **Increase max_tokens** for documents with lots of text (2000+ for full pages)
5. **Localization coordinates** are normalized to 0-1000 range; scale to your image dimensions

## Limitations

- Localization (`<|ref|>...<|/ref|>`) may return full-image coordinates for some queries
- Best suited for document-style images (forms, tables, receipts, papers)
- Free-form questions may produce less reliable outputs compared to structured prompts
- All queries should have period for better performance
