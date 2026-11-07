# Granite Vision (4.0, 4.1)

IBM's Granite Vision 4.x models with DeepStack multi-layer feature injection and WindowQFormer projectors. Both combine a SigLIP vision encoder with a Granite language model: Granite 4.0 Vision uses GraniteMoeHybrid, and Granite Vision 4.1 uses a dense Granite language model. The same loader handles both.

## Model

| | |
|---|---|
| **Model IDs** | `ibm-granite/granite-4.0-3b-vision`, `ibm-granite/granite-vision-4.1-4b` |
| **Architecture** | Granite LM (GraniteMoeHybrid in 4.0, dense Granite in 4.1) + SigLIP (vision) + WindowQFormer projectors + DeepStack |
| **Parameters** | ~3B (4.0), ~4B (4.1) |
| **Vision Encoder** | SigLIP (27 layers, 1152 hidden, 384px, patch 16) |
| **Projectors** | 4 DeepStack + 4 Spatial WindowQFormerDownsampler modules |
| **Tasks** | Document understanding, VQA, image description, table/chart analysis |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model ibm-granite/granite-4.0-3b-vision \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

## Python Usage

```python
from mlx_vlm import load, generate

model, processor = load("ibm-granite/granite-4.0-3b-vision")

image = "path/to/image.jpg"
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image."}
    ]}
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

output = generate(model, processor, image=image, prompt=prompt, max_tokens=512)
print(output)
```

## Architecture

- **Vision**: SigLIP encoder (27 layers, patch size 16)
- **Language**: GraniteMoeHybrid with MUP multipliers and fused gate+up MLP (4.0); dense Granite (4.1)
- **DeepStack**: Vision features from layers [-19, -13, -7, -1] injected at LLM layers [9, 6, 3, 0]
- **Spatial Sampling**: 4 offset groups (TL/TR/BL/BR) from last vision layer → LLM layers [12, 15, 18, 21]
- **WindowQFormer**: Windowed cross-attention with area interpolation downsampling (4/8 rate)
- **LoRA**: Adapter weights auto-merged during loading (r=256, alpha=256)
