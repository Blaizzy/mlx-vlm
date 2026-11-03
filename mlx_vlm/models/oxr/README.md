# OXR

[SoMark OXR-1.0](https://github.com/SoMarkAI/OXR) (Optical Everything Recognition) is a 1.3B
document-parsing model (SigLIP2-NaViT visual tokenizer + Qwen3-0.6B) that ranks first on
olmOCR-Bench, OmniDocBench and DocParsingBench. It recognizes text (incl. handwriting),
LaTeX formulas, HTML tables, code blocks, chemical equations/structures and stamps.

OXR is designed to run behind a layout model: the official pipeline crops each layout block and
asks OXR to recognize it with a task prompt, then asks it for the reading order. Single-image
"whole page" recognition also works.

## Model

- **Architecture**: Ovis2.5-style — SigLIP2-NaViT ViT (native resolution, 2D RoPE) → 2×2 merge →
  softmax over a 65,532-entry visual vocabulary → soft visual-token embeddings → Qwen3-0.6B.
- **Image sizing**: `smart_resize` to multiples of 32 within `[448², 960²]` pixels (the official
  vLLM defaults); override with `min_pixels` / `max_pixels`.
- **Weights**: the original checkpoint loads directly (`SoMarkAI/OXR-1.0`); `sanitize()` maps it.

## Prompts

| Task | Prompt |
|---|---|
| Text / handwriting | `Recognize the content of the image.` |
| Formula (LaTeX) | `Recognize the formula of the image into KaTex format.` |
| Table (HTML) | `Recognize the table of the image into html format.` |
| Code | `Recognize the code of the image into markdown code format.` |
| Chemical structure (SMILES) | `Recognize the chemical structure of the image into SMILES.` |
| Whole page | `Recognize the whole page content of the image.` |

## CLI Examples

```bash
uv run mlx_vlm.generate \
  --model SoMarkAI/OXR-1.0 \
  --image formula.png \
  --prompt "Recognize the formula of the image into KaTex format." \
  --max-tokens 2048 --temperature 0.0
```

```bash
uv run mlx_vlm.generate \
  --model SoMarkAI/OXR-1.0 \
  --image table.png \
  --prompt "Recognize the table of the image into html format." \
  --max-tokens 8192 --temperature 0.0
```

## Python Example

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("SoMarkAI/OXR-1.0")
prompt = apply_chat_template(
    processor, model.config, "Recognize the content of the image.", num_images=1
)
result = generate(
    model, processor, prompt, image=["paragraph.png"], max_tokens=4096, temperature=0.0
)
print(result.text)
```

## Notes

- Greedy decoding (`temperature=0.0`) matches the official pipeline.
- Validated against the reference PyTorch implementation: greedy outputs are token-identical on
  text, formula, code, whole-page and a 3,603-token HTML table.
- Quantization keeps the visual tokenizer (`vision_tower`, `vision_head`, `vision_embed`) in full
  precision.
- License: [OXR Open Source License](https://github.com/SoMarkAI/OXR/blob/main/LICENSE)
  (Apache-2.0 with additional terms).
