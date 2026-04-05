# Usage

## Command Line Interface (CLI)

Generate output from a model:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

## Chat UI with Gradio

Launch the chat interface:

```bash
python -m mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

## Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## Server (FastAPI)

```bash
python -m mlx_vlm.server
```

See `README.md` for a complete `curl` example.

