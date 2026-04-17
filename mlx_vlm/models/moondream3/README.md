# Moondream3

Moondream3 is a Mixture-of-Experts (MoE) vision-language model with 9.27B total parameters and ~2B active per token. It combines a SigLIP-based vision encoder with an MoE text decoder for efficient multimodal understanding.

## Model

| | |
|---|---|
| **Model ID** | `moondream/moondream3-preview` |
| **Architecture** | SigLIP ViT (vision) + MoE Transformer Decoder (language) |
| **Total Parameters** | ~9.27B (2B active per token) |
| **Vision Encoder** | 27 layers, 1152 dim, 16 heads, patch size 14, crop size 378 |
| **Language Model** | 24 layers, 2048 dim, 32 heads (layers 0-3 dense, 4-23 MoE with 64 experts, top-8) |
| **Tasks** | Visual question answering, image description, visual reasoning |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model moondream/moondream3-preview \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-tokens 200
```

With custom parameters:

```bash
python -m mlx_vlm.generate \
    --model moondream/moondream3-preview \
    --image path/to/image.jpg \
    --prompt "How many objects are in this image?" \
    --max-tokens 100 \
    --temp 0.0
```

## Python Usage

```python
from mlx_vlm import load, generate

model, processor = load("moondream/moondream3-preview")

output = generate(
    model,
    processor,
    "Describe this image",
    ["path/to/image.jpg"],
    max_tokens=200,
)
print(output)
```

## Architecture

- **Vision**: SigLIP-based ViT encoder with multi-crop support (up to 12 crops). Images are patchified (14x14 patches on 378x378 crops), producing 729 tokens per crop. Global and local crop features are reconstructed and projected via a 2-layer MLP to 2048-dim.
- **Language**: MoE Transformer Decoder with parallel residual connections (`x = x + attn(ln(x)) + mlp(ln(x))`). Layers 0-3 use dense MLPs; layers 4-23 use Mixture-of-Experts with 64 experts and top-8 routing with GeGLU activation.
- **Tau Scaling**: Learned position- and data-dependent temperature scaling on Q and V in attention, unique to Moondream3.
- **RoPE**: Rotary position embeddings applied to the first 32 of 64 head dimensions, with base theta=1.5M.
- **Prefix Attention**: Bidirectional attention for the first 730 tokens (1 BOS + 729 vision tokens) during prefill.

## Notes

- Uses a custom tokenizer from `moondream/starmie-v1` (SuperBPE).
- The model may output a thinking token (`<|md_reserved_4|>`) before the answer.
- Multi-crop processing reconstructs local features with overlap margin trimming and adaptive average pooling.
- Peak memory usage is approximately 24 GB for the full bf16 model.
