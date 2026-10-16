# Muse Spark Open

Meta's Muse Spark Open: a mixture-of-experts language model (816B parameters,
42B active per token) with an image encoder. This port supports text and image
inputs. The audio encoder is not supported.

The vision encoder is the one of Muse Glimmer, except that Muse Spark reads its
position table with the patch grid transposed. For a square grid the two are
the same.

## Checkpoints

- BF16 weights in the Transformers layout, for example
  [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B).
  Convert or quantize them with `mlx_vlm.convert`. As in mlx-lm, quantization
  applies only to the routed experts, except in the first and the last layer,
  and the router and the residual gates stay in FP32.
- Checkpoints converted with `mlx_lm.convert`, for example
  [mlx-community/Muse-Spark-Open-816B-A42B-4bit](https://huggingface.co/mlx-community/Muse-Spark-Open-816B-A42B-4bit).
  Sanitize splits their fused attention projections and converts the stored
  norm gains and residual-gate weights back.

The NVFP4 checkpoint is not supported.

## Usage

```bash
python -m mlx_vlm.generate \
  --model mlx-community/Muse-Spark-Open-816B-A42B-4bit \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 512
```

The model writes a reasoning channel (`to=self`) before the answer
(`to=user`); the processor's `clean_output` keeps only the answer.

## Tool calling

The chat template renders tools in Muse Spark's own format. This port does not
add a tool-call parser for it.
