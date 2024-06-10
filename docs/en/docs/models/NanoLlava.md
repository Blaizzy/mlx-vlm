nanoLLaVA is a "small but mighty" 1B vision-language model designed to run efficiently on edge devices.

**Example Usage:**
```python
import mlx.core as mx
from mlx_vlm import load, generate
from .prompt_utils import get_message_json

model_path = "mlx-community/nanoLLaVA"
model, processor = load(model_path)

prompt = processor.apply_chat_template(
    [get_message_json(model.config.model_type, "What are these?")],
    tokenize=False,
    add_generation_prompt=True,
)

output = generate(model, processor, "http://images.cocodataset.org/val2017/000000039769.jpg", prompt,  verbose=True)
```
