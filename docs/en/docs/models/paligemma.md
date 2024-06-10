PaliGemma is a lightweight open vision-language model (VLM) inspired by PaLI-3, and based on open components like the SigLIP vision model and the Gemma language model. It can perform deeper analysis of images and provide useful insights, such as captioning for images and short videos, object detection, and reading text embedded within images.

**Example Usage:**
```python
import mlx.core as mx
from mlx_vlm import load, generate
from .prompt_utils import get_message_json

model_path = "mlx-community/paligemma-3b-mix-448-8bit"
model, processor = load(model_path)

prompt = "What are these?"

output = generate(model, processor, "http://images.cocodataset.org/val2017/000000039769.jpg", prompt, verbose=True)
```