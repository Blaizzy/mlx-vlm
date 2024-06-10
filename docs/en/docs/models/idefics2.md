### Idefics 2

Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple images, or simply behave as a pure language model without visual inputs. It improves upon Idefics1, significantly enhancing capabilities around OCR, document understanding and visual reasoning.

- Paper: https://huggingface.co/papers/2405.02246
- License: Apache 2.0

**Example Usage:**

```python
import mlx.core as mx
from mlx_vlm import load, generate
from .prompt_utils import get_message_json

model_path = "mlx-community/idefics2-8B-4bit"
model, processor = load(model_path)

prompt = processor.tokenizer.apply_chat_template(
    [get_message_json(model.config.model_type, "What are these?")],
    tokenize=False,
    add_generation_prompt=True,
)

output = generate(model, processor, "http://images.cocodataset.org/val2017/000000039769.jpg", prompt,  verbose=True)
```