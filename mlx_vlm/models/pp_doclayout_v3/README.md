# PP-DocLayout V3 (MLX)

MLX support for document-layout detection with bounding boxes, class labels,
and reading order. The model does not transcribe text.

- **Original model:** [PaddlePaddle/PP-DocLayoutV3_safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors).
- **MLX port:** [HashNuke/pp-doclayout-v3-mlx](https://huggingface.co/HashNuke/pp-doclayout-v3-mlx)

## Detect page regions

Run the [example script](../../../examples/pp_doclayout_v3.py) from the repository root:

```sh
python -m examples.pp_doclayout_v3 \
  --image path/to/page.png \
  --output-json output/layout.json \
  --output-image output/layout.png
```

Saves reading-ordered detections as JSON and an annotated image. Each JSON
region contains `label`, `score`, `reading_order` (one-based), and `bbox`
(`[y0, x0, y1, x1]`, normalized to 0–1000).
