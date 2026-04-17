# Granite Vision 3.2

IBM's Granite Vision 3.2 model for document understanding and visual question answering. Combines a SigLIP vision encoder with a Granite language model using the LLaVA-NeXT AnyRes architecture.

## Model

| | |
|---|---|
| **Model ID** | `mlx-community/granite-vision-3.2-2b-bf16` |
| **Architecture** | Granite LM + SigLIP (vision) + MLP projector (LLaVA-NeXT) |
| **Parameters** | ~3B |
| **Vision Encoder** | SigLIP (27 layers, 1152 hidden, 384px, patch 14) |
| **Vision Features** | Multi-layer concat from layers [-24, -20, -12, -1] |
| **Tasks** | Document understanding, VQA, image description, table/chart analysis |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model mlx-community/granite-vision-3.2-2b-bf16 \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

## Python Usage

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/granite-vision-3.2-2b-bf16")

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

- **Vision**: SigLIP encoder (27 transformer layers, no CLS token)
- **Language**: Granite 3.1-2B with MUP multipliers (embedding, attention, residual, logits scaling)
- **Projector**: Multi-layer feature concatenation (4 layers → 4608-dim) → 2-layer MLP → 2048-dim
- **AnyRes**: Variable resolution via image grid pinpoints (27 configurations)
