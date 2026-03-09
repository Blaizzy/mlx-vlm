# Gemma 4 (gemma-4-e2b-it / gemma-4-e4b-it)

Gemma 4 is a multimodal model from Google that supports text, image, and thinking (chain-of-thought reasoning).

Capabilities:
- **Text generation** — single and multi-turn conversations
- **Thinking mode** — chain-of-thought reasoning with `--enable-thinking`
- **Image understanding** — single and multi-image analysis
- **Function calling** — tool use with text, images, and thinking

## Model

| Model | Parameters | HF ID |
|-------|-----------|-------|
| Gemma 4 E2B Instruct | E2B | `google/gemma-4-e2b-it` |
| Gemma 4 E5B Instruct | E5B | `google/gemma-4-e5b-it` |
| Gemma 4 E2B Base | E2B | `google/gemma-4-e2b` |
| Gemma 4 E5B Base | E5B | `google/gemma-4-e5b` |

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Text generation

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e2b-it \
  --prompt "Write a poem about the ocean." \
  --max-tokens 500 \
  --temperature 0
```

### Image understanding

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e2b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500 \
  --temperature 0
```

### Thinking mode (chain-of-thought)

Enable thinking to get step-by-step reasoning before the final answer:

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e2b-it \
  --prompt "I want to do a car wash that is 50 meters away, should I walk or drive?" \
  --enable-thinking \
  --max-tokens 2000 \
  --temperature 0
```

### Image + thinking

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e2b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --enable-thinking \
  --max-tokens 2000 \
  --temperature 0
```

### Thinking with budget

Limit the number of thinking tokens before forcing the model to answer:

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e2b-it \
  --prompt "Explain quantum entanglement." \
  --enable-thinking \
  --thinking-budget 512 \
  --max-tokens 2000 \
  --temperature 0
```

## Python

### Basic text generation

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e2b-it")

prompt = apply_chat_template(
    processor, model.config, "Write a poem about the ocean."
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=500,
    temperature=0.0,
)
print(result)
```

### Image understanding

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e2b-it")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "Describe this image.",
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=image,
    max_tokens=500,
    temperature=0.0,
)
print(result)
```

### Thinking mode

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e2b-it")

prompt = apply_chat_template(
    processor, model.config,
    "Explain quantum entanglement in simple terms.",
    chat_template_kwargs={"enable_thinking": True},
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=2000,
    temperature=0.0,
)
print(result)
```

### Image + thinking

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e2b-it")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config,
    "What city is shown in this image? Explain your reasoning.",
    num_images=1,
    chat_template_kwargs={"enable_thinking": True},
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=image,
    max_tokens=2000,
    temperature=0.0,
)
print(result)
```

## Notes

- Thinking mode uses `<|channel>...<channel|>` delimiters to separate reasoning from the final answer.
- The E2B model is lighter and works on machines with 16GB+ memory; E4B requires more.
- Audio support is not yet implemented in this MLX port.
