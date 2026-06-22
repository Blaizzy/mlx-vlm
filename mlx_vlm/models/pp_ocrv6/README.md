# PP-OCRv6

PP-OCRv6 models are standalone OCR detection and recognition models from PaddlePaddle. They are loaded directly through `load`; they do not use the chat-generation API.

## Supported Models

- `PaddlePaddle/PP-OCRv6_tiny_det_safetensors`
- `PaddlePaddle/PP-OCRv6_small_det_safetensors`
- `PaddlePaddle/PP-OCRv6_medium_det_safetensors`
- `PaddlePaddle/PP-OCRv6_tiny_rec_safetensors`
- `PaddlePaddle/PP-OCRv6_small_rec_safetensors`
- `PaddlePaddle/PP-OCRv6_medium_rec_safetensors`

## Text Recognition

```python
import mlx.core as mx
from PIL import Image

from mlx_vlm import load

model, processor = load("PaddlePaddle/PP-OCRv6_small_rec_safetensors")
image = Image.open("word.png")

inputs = processor(image)
outputs = model(**inputs)
mx.eval(outputs.logits)

print(processor.post_process_text_recognition(outputs)[0])
```

## Text Detection

```python
import mlx.core as mx
from PIL import Image

from mlx_vlm import load

model, processor = load("PaddlePaddle/PP-OCRv6_small_det_safetensors")
image = Image.open("document.png")

inputs = processor(image)
outputs = model(**inputs)
mx.eval(outputs.logits)

result = processor.post_process_object_detection(
    outputs,
    target_sizes=inputs["target_sizes"],
)[0]
print(result["boxes"])
```
