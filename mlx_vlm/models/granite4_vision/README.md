# Granite 4.0 Vision

IBM's Granite 4.0 Vision model with DeepStack multi-layer feature injection and WindowQFormer projectors. Combines a SigLIP vision encoder with a GraniteMoeHybrid language model.

## Model

| | |
|---|---|
| **Model ID** | `ibm-granite/granite-4.0-3b-vision` |
| **Architecture** | GraniteMoeHybrid LM + SigLIP (vision) + WindowQFormer projectors + DeepStack |
| **Parameters** | ~3B |
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
- **Language**: GraniteMoeHybrid with MUP multipliers and fused gate+up MLP
- **DeepStack**: Vision features from layers [-19, -13, -7, -1] injected at LLM layers [9, 6, 3, 0]
- **Spatial Sampling**: 4 offset groups (TL/TR/BL/BR) from last vision layer → LLM layers [12, 15, 18, 21]
- **WindowQFormer**: Windowed cross-attention with area interpolation downsampling (4/8 rate)
- **LoRA**: Adapter weights auto-merged during loading (r=256, alpha=256)
