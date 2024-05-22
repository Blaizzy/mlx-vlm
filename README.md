# MLX-VLM

MLX-VLM a package for running Vision LLMs on your Mac using MLX.


## Get started

The easiest way to get started is to install the `mlx-vlm` package:

**With `pip`**:

```sh
pip install mlx-vlm
```

## Inference

**CLI**
```sh
python -m mlx_vlm.generate --model qnguyen3/nanoLLaVA --max-tokens 100 --temp 0.0
```

**Chat UI with Gradio**
```sh
python -m mlx_vlm.chat_ui --model qnguyen3/nanoLLaVA
```

**Script**
```python
import mlx.core as mx
from mlx_vlm import load, generate

model_path = "mlx-community/llava-1.5-7b-4bit"
model, processor = load(model_path)

prompt = processor.tokenizer.apply_chat_template(
    [{"role": "user", "content": f"<image>\nWhat are these?"}],
    tokenize=False,
    add_generation_prompt=True,
)

output = generate(model, processor, "http://images.cocodataset.org/val2017/000000039769.jpg", prompt, verbose=False)
```
