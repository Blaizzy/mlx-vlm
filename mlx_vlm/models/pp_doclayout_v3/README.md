# PP-DocLayout V3 (MLX)

MLX support for document-layout detection with bounding boxes, class labels,
and reading order. The model does not transcribe text.

- **Original model:** [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors).
- **MLX port:** [HashNuke/pp-doclayout-v3-mlx](https://huggingface.co/HashNuke/pp-doclayout-v3-mlx)

## Detect page regions

Replace `page.png` with your input image:

```python
from mlx_vlm.utils import get_model_path, load_model

model = load_model(get_model_path("HashNuke/pp-doclayout-v3-mlx"))
regions = sorted(
    model.detect("page.png"),
    key=lambda region: region["reading_order"],
)
print(regions)
```

Prints reading-ordered detections. Each region contains `label`, `score`,
`reading_order` (one-based), and `bbox`
(`[y0, x0, y1, x1]`, normalized to 0–1000).
