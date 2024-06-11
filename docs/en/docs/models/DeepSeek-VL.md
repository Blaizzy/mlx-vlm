DeepSeek-VL possesses general multimodal understanding capabilities, capable of processing logical diagrams, web pages, formula recognition, scientific literature, natural images, and embodied intelligence in complex scenarios.

- Paper: https://arxiv.org/abs/2403.05525
- License: MIT

**Example Usage:**
```python
import mlx.core as mx
from mlx_vlm import load, generate
from .prompt_utils import get_message_json

model_path = "mlx-community/deepseek-vl-7b-chat-4bit"
model, processor = load(model_path)

prompt = processor.apply_chat_template(
    [get_message_json(model.config.model_type, "What are these?")],
    tokenize=False,
    add_generation_prompt=True,
)

output = generate(
    model,
    processor,
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    prompt,
    verbose=True
)
```