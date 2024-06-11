LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture.

- Paper: https://llava-vl.github.io/

**Example Usage:**
```python
import mlx.core as mx
from mlx_vlm import load, generate
from .prompt_utils import get_message_json

model_path = "mlx-community/llava-1.5-7b-4bit"
model, processor = load(model_path)

prompt = processor.tokenizer.apply_chat_template(
    [get_message_json(model.config.model_type, "What are these?")],
    tokenize=False,
    add_generation_prompt=True,
)

output = generate(
    model,
    processor,
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    prompt,
    verbose=False
)
```