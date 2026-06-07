# PLaMo 2.1-8B-VL

PLaMo 2.1-8B-VL is a Japanese-English vision-language model from Preferred
Networks, Inc. It combines a PLaMo 2 language backbone with a SigLIP2 vision
encoder and a simple MLP image adapter for visual question answering, visual
grounding, and natural-image understanding.

## Model

| | |
|---|---|
| **Model ID** | `pfnet/plamo-2.1-8b-vl` |
| **Architecture** | PLaMo 2 language model + SigLIP2 vision encoder + MLP image adapter |
| **Parameters** | 8B |
| **Languages** | Japanese, English |
| **Modalities** | Text, single image |
| **Tasks** | Visual question answering, visual grounding, image description |
| **License** | PLaMo community license |
| **Official Card** | [pfnet/plamo-2.1-8b-vl](https://huggingface.co/pfnet/plamo-2.1-8b-vl) |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model pfnet/plamo-2.1-8b-vl \
    --image path/to/image.jpg \
    --prompt "Describe this image." \
    --max-tokens 200 \
    --temperature 0.0
```

For visual question answering:

```bash
python -m mlx_vlm.generate \
    --model pfnet/plamo-2.1-8b-vl \
    --image path/to/image.jpg \
    --prompt "What object is the person holding?" \
    --max-tokens 200 \
    --temperature 0.0
```

## Python Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "pfnet/plamo-2.1-8b-vl"
model, processor = load(model_path)

image = "path/to/image.jpg"
prompt = "Describe this image."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=1,
)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=image,
    max_tokens=200,
    temperature=0.0,
)
print(result.text)
```

## Architecture

- **Vision**: SigLIP2-style image encoder using dynamic image tiling. Images are
  split into tiles before encoding so non-square inputs can retain more detail.
- **Projector**: MLP image adapter that normalizes vision features, applies GELU,
  and maps them into the PLaMo 2 hidden size.
- **Language**: PLaMo 2 language model backbone using the MLX LM PLaMo decoder.
- **Processor**: Model-specific processor with the PLaMo tokenizer, SigLIP image
  preprocessing, and the PLaMo2VL prompt format.

## Notes

- The upstream model card describes the model as optimized for VQA and visual
  grounding on natural images.
- The official model is intended for single-image inputs. Batch size 1 is
  currently supported in this implementation.
- OCR, document understanding, charts, tables, and mathematical expressions are
  not primary target tasks for the upstream model.
- The model is released under the PLaMo community license. Check the official
  model card and license terms before downloading or using the weights.
