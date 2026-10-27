# Unlimited-OCR

Unlimited-OCR is Baidu's OCR/document-parsing model for **one-shot long-horizon parsing**.

This README only documents the prompt formats and settings that are either shown by upstream or verified in the MLX implementation.

## Unlimited-OCR-2

[`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2) loads through the same module. It keeps the vision stack and adds a trained decoder attention path (`sparse_attention_config.training_stage == "dash"`):

- **page2d RoPE**: every position gets a (token-in-page, page) coordinate. Each prompt image is preceded by a `<PAGE>` marker; the page index restarts for the answer, and a generated `<PAGE>` advances it.
- **DASH block-sparse attention**: keys are grouped in 32-token blocks. Each query attends a 128-token local window plus the 12 remote blocks picked by a small per-layer router (`self_attn.dash_router.*`: block summaries via a learned `head_cls` query, top-16 scores, 1.5-entmax prior as a log-bias). Prefill queries are restricted to the image blocks of their own page.
- Outputs use `<PAGE>`, `<BLOCK>label[x0, y0, x1, y1]<FORK>text`, and `cross_text` / `append_text` labels for paragraphs that continue across a page break.

The processor detects v2 from `config.json` and rewrites the prompt into the trained `<PAGE><image>...` layout, so the prompt text is the same as for v1:

```bash
# One page: gundam mode (1024 global view + up to 16 local crops).
mlx_vlm.generate --model baidu/Unlimited-OCR-2 --image page.png \
    --prompt "Multi page parsing." --max-tokens 8192

# Several pages in one request: one 1024 base view (273 tokens) per page.
mlx_vlm.generate --model baidu/Unlimited-OCR-2 --image page_0001.png page_0002.png \
    --prompt "Multi page parsing." --max-tokens 16384
```

Like upstream `infer`, the crop mode is automatic unless `cropping` is passed: gundam for one page, base for several. Upstream's SGLang example instead sends whole documents with `crop_mode=True` (gundam tiles per page) and a 10-page window; pass `--processor-kwargs '{"cropping": true}'` for that layout.

MLX implementation notes (`dash.py`):

- Decode gathers only the selected blocks and the local window (`12 × 32 + 128 = 512` keys) instead of masking the whole cache, so the per-token attention cost does not grow with the document.
- Block summaries are computed once, when a block completes. Page/RoPE/candidate state for decode stays on device (no host sync per token).
- The decode-step router and selection are `mx.compile`d on a block capacity that changes every 64 blocks.
- `dash_router` weights stay dense when quantizing (`quant_predicate`); the warmup-stage `indexer` weights are dropped. Batch size 1 only.

Generation guards (`guards.py`). The native stack never decodes unguarded, so MLX applies the same rules to Unlimited-OCR-2 requests by default (generated ids only, state kept on device, ~3% decode cost):

- **page-count guard** (`infer_multi` default): EOS is banned until every prompt `<PAGE>` has been answered, EOS is forced once there are more answered pages than prompt pages, and a page longer than `max_page_tokens` (20000) is cut by forcing the next `<PAGE>` (EOS after the last).
- **no-repeat n-gram** (the evaluation setup of `infer.py`): 35-grams over a 1024-token window for several pages, 128 for one, so a loop is broken after a few repeats instead of running to the token budget.

Without them, one wrong token can send greedy decoding into a loop ("The document is a key to the development of the document." repeated) that never reaches EOS and loses every later page. Tune or disable them with generation kwargs:

```bash
mlx_vlm.generate --model baidu/Unlimited-OCR-2 --image p1.png p2.png \
    --prompt "Multi page parsing." --gen-kwargs '{"no_repeat_ngram_size": 0, "page_count_guard": false}'
```

Options: `page_count_guard`, `max_page_tokens`, `no_repeat_ngram_size` (0 disables), `ngram_window`, `ngram_whitelist_token_ids`. On healthy output the guards never fire and the text is unchanged.

## Model Architecture

```
                        input document image(s)
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ Dynamic image preprocessing │
                    ├─────────────────────────────┤
                    │ gundam: 1024 global view    │
                    │         + 640 local crops   │
                    │ base:   1024 global view    │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ SAM ViT-B + CLIP-L features │
                    │ concatenated → 2048-dim     │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ Linear projector → 1280 dim │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │ DeepSeek-V2-style decoder   │
                    │ with R-SWA KV-cache         │
                    └─────────────────────────────┘
```

The released checkpoint configuration uses:

- a `deeplip_b_l` vision stack with `sam_vit_b` and `clip-l-14-224` components;
- a projector from 2048 visual-feature dimensions to the 1280-dimensional language hidden size;
- a 12-layer DeepSeek-V2-style MoE language decoder with `max_position_embeddings=32768`;
- `sliding_window_size=128` for Unlimited-OCR's Reference Sliding Window Attention (R-SWA).

The MLX implementation keeps the prompt/prefill KV cache and uses a small ring buffer for generated-token KV entries, matching the upstream R-SWA decode behavior.

## Prompt Formats

When using `mlx_vlm.generate` or `apply_chat_template`, pass the task text only. MLX inserts the `<image>` token for you.

### Single-image document parsing

Upstream Transformers example:

```
<image>document parsing.
```

MLX prompt text:

```
document parsing.
```

### Multi-page / PDF parsing

Upstream `infer_multi` example:

```
<image>Multi page parsing.
```

MLX prompt text:

```
Multi page parsing.
```

For multi-page inputs, the MLX prompt template intentionally inserts **one** literal `<image>` token for all pages, matching upstream `infer_multi` behavior.

### Output markers

Document-parsing outputs commonly contain layout tags such as:

```
<|det|>title [357, 135, 642, 155]<|/det|>Unlimited OCR Works
<PAGE>
```

The box coordinates are normalized to the page coordinate system used by the model, typically the 0-1000 range.

You can still try other DeepSeek-OCR-style prompts by keeping the same image and
generation settings, changing only `--prompt`, and comparing the outputs against
the upstream-documented `document parsing.` baseline:

```bash
for prompt in \
    "document parsing." \
    "Free OCR." \
    "<|grounding|>OCR this image." \
    "<|grounding|>Convert the document to markdown." \
    "Parse the figure."; do
    mlx_vlm.generate \
        --model baidu/Unlimited-OCR \
        --image document.png \
        --prompt "$prompt" \
        --max-tokens 4096
done

mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "Locate <|ref|>Unlimited OCR Works<|/ref|> in the image." \
    --max-tokens 512
```

These prompts are experimental for Unlimited-OCR; use the output comparison to
decide whether a prompt changes behavior usefully for your document.

## CLI Examples

### Single image (`gundam` mode, default)

`gundam` is the upstream single-image configuration: `base_size=1024`, `image_size=640`, `crop_mode=True`. In MLX this is the default (`cropping=True`, `image_size=640`, `base_size=1024`).

```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "document parsing." \
    --max-tokens 32768
```

### Single image in global-view-only `base` mode

`base` mode disables local crops and uses a 1024×1024 global view.

```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image document.png \
    --prompt "document parsing." \
    --max-tokens 32768 \
    --processor-kwargs '{"cropping": false, "image_size": 1024}'
```

### Multi-page OCR from rendered PDF pages

The documented Unlimited-OCR PDF workflow renders PDF pages to images first, then passes those images in page order. Upstream uses the `base` configuration for multi-page/PDF workflows.

```bash
mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image page_0001.png page_0002.png page_0003.png \
    --prompt "Multi page parsing." \
    --max-tokens 32768 \
    --processor-kwargs '{"cropping": false, "image_size": 1024}'
```

One way to render a PDF to ordered page images is:

```bash
mkdir -p pages
uv run --with pymupdf python - <<'PY'
from pathlib import Path

import fitz

pdf_path = Path("document.pdf")
out_dir = Path("pages")
out_dir.mkdir(exist_ok=True)

# Upstream examples use 300 DPI. Lower this if you need a quicker smoke test.
dpi = 300
matrix = fitz.Matrix(dpi / 72, dpi / 72)

with fitz.open(pdf_path) as doc:
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=matrix, alpha=False).save(out_dir / f"page_{i + 1:04d}.png")
PY

mlx_vlm.generate \
    --model baidu/Unlimited-OCR \
    --image pages/page_*.png \
    --prompt "Multi page parsing." \
    --max-tokens 32768 \
    --processor-kwargs '{"cropping": false, "image_size": 1024}'
```

## Python Script Examples

### Single-image document parsing

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

prompt = apply_chat_template(
    processor,
    model.config,
    "document parsing.",
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=prompt,
    max_tokens=32768,
    temperature=0.0,
)
print(result.text)
```

### Multi-page / rendered-PDF parsing

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("baidu/Unlimited-OCR")

page_images = ["page_0001.png", "page_0002.png", "page_0003.png"]
prompt = apply_chat_template(
    processor,
    model.config,
    "Multi page parsing.",
    num_images=len(page_images),
)

# Matches upstream infer_multi: one literal <image> token for all pages.
assert prompt.count("<image>") == 1
assert prompt.strip() == "<image>Multi page parsing."

result = generate(
    model=model,
    processor=processor,
    image=page_images,
    prompt=prompt,
    max_tokens=32768,
    temperature=0.0,
    cropping=False,
    image_size=1024,
)
print(result.text)
```

## Dynamic Resolution

Unlimited-OCR has two upstream image modes:

| Mode | Upstream settings | MLX kwargs | Upstream use |
|------|-------------------|------------|--------------|
| `gundam` | `base_size=1024`, `image_size=640`, `crop_mode=True` | `base_size=1024`, `image_size=640`, `cropping=True` | Single-image parsing |
| `base` | `base_size=1024`, `image_size=1024`, `crop_mode=False` | `base_size=1024`, `image_size=1024`, `cropping=False` | Multi-page/PDF parsing |

### Token layout

- In `base` mode, each page is padded to 1024×1024 and represented as a 16×16 visual grid, with one newline embedding per row and one view-separator embedding: `(16 + 1) × 16 + 1 = 273` image-token positions per page.
- In `gundam` mode, MLX always adds the 1024×1024 global view. For images larger than 640×640, it also creates dynamic 640×640 local crops using the same crop-grid search as DeepSeek-OCR, capped at 32 crops.
- For small images where both width and height are `<= 640`, `gundam` mode uses only the global view.

### Controlling the mode

```python
# Default upstream single-image mode.
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=prompt,
    max_tokens=32768,
    cropping=True,
    image_size=640,
    base_size=1024,
)

# Upstream multi-page/PDF mode.
result = generate(
    model=model,
    processor=processor,
    image=page_images,
    prompt=prompt,
    max_tokens=32768,
    cropping=False,
    image_size=1024,
    base_size=1024,
)
```

## Generation Defaults

The upstream examples use deterministic sampling. Baidu's reference code also ships a `SlidingWindowNoRepeatNgramProcessor`, but treats it as opt-in: `infer` / `infer_multi` default `no_repeat_ngram_size=0`, while example calls may pass size `35` with a window of `128` for single images or `1024` for multi-page/PDF inputs.

MLX-VLM follows the same opt-in behavior. To keep this model consistent with the rest of the repository, it does not implement or attach an Unlimited-OCR-specific logits processor automatically; if you want this repetition guard, pass your own callable through the existing `logits_processors` argument.

| Setting | Upstream single image | Upstream multi-page/PDF | MLX behavior |
|---------|-----------------------|--------------------------|--------------|
| Prompt | `<image>document parsing.` | `<image>Multi page parsing.` | Template inserts `<image>` automatically |
| Image mode | `gundam` | `base` | Same via `processor_kwargs` / `generate` kwargs |
| Optional n-gram size | `35` in examples | `35` in examples | Caller-supplied via `logits_processors` |
| Optional n-gram window | `128` in examples | `1024` in examples | Caller-supplied via `logits_processors` |
| Output budget | `max_length=32768` | `max_length=32768` | Use `max_tokens` in MLX |

### Optional sliding-window no-repeat n-gram processor

For long OCR runs, especially multi-page PDFs, Baidu's examples may opt in to a sliding-window no-repeat n-gram logits processor. MLX-VLM does not enable this automatically. If your application uses that guard, pass it explicitly through `logits_processors`:

```python
# Define or import a callable logits processor yourself; MLX-VLM does not
# provide an Unlimited-OCR-specific one. A local port of upstream
# SlidingWindowNoRepeatNgramProcessor can use ngram_size=35 and window=1024
# for multi-page/PDF parsing, or window=128 for single-image parsing.
no_repeat_ngram_processor = ...

result = generate(
    model=model,
    processor=processor,
    image=page_images,
    prompt=prompt,
    max_tokens=32768,
    temperature=0.0,
    cropping=False,
    image_size=1024,
    logits_processors=[no_repeat_ngram_processor],
)
```

## Special Tokens and Markers

| Token / marker | Description |
|----------------|-------------|
| `<image>` | Image placeholder inserted by the prompt template |
| `<PAGE>` | Page separator commonly emitted in multi-page parsing output |
| `<\|det\|>...<\|/det\|>` | Layout detection span with a label and normalized box coordinates |
| `<｜begin▁of▁sentence｜>` / `<｜end▁of▁sentence｜>` | BOS / EOS tokens from the released tokenizer |
| `<｜▁pad▁｜>` | Pad token from the released tokenizer |

## Tips

1. Do not manually add `<image>` to `mlx_vlm.generate` prompts; use `document parsing.` or `Multi page parsing.`.
2. Use `cropping=False, image_size=1024` for multi-page or PDF workflows.
3. Render PDFs to images before calling MLX-VLM, and keep filenames zero-padded so shell glob order matches page order.
4. Use a high `max_tokens` value for dense documents or long PDFs; lower it for quick smoke tests.
5. Use `temperature=0.0` for deterministic OCR output.
6. For 4-bit language-model quantization, keep the vision encoder in `float32`; lowering vision precision can hurt OCR quality or make repetition loops much more likely, while a 4-bit LLM can still work with FP32 vision features.

## Limitations

- The documented MLX workflow expects image files; render PDFs to page images first.
- Upstream documents `base` mode for multi-page/PDF parsing. Dynamic-crop `gundam` mode is intended for single-image parsing.
- Very dense or long documents can take substantial time and memory even with R-SWA because the full visual/prompt prefill is retained.
- This README intentionally avoids documenting generic DeepSeek-OCR prompt variants that are not shown in the Baidu Unlimited-OCR repo.

## Implementation Notes

- The upstream repository includes `SlidingWindowNoRepeatNgramProcessor`, but this implementation keeps repetition guards caller-supplied through `logits_processors` rather than adding model-specific sampling code to MLX-VLM.
- Keep the vision stack in FP32 when using quantized language weights; the no-repeat processor can stop repeated decoding earlier, but it does not fix degraded visual features.
