# mlx_vlm.convert — convert & quantize

Convert a Hugging Face model to MLX format, optionally quantizing (or dequantizing) it in the same pass. The converted model can be saved locally and, with `--upload-repo`, pushed to the Hugging Face Hub.

## Synopsis

```
mlx_vlm.convert [OPTIONS]
```

The console script `mlx_vlm.convert` (or `python -m mlx_vlm convert`) also works. Note: the source prints a deprecation notice for `python -m mlx_vlm.convert` and recommends `mlx_vlm.convert` or `python -m mlx_vlm convert` instead.

## Examples

Plain convert to MLX format:

```
mlx_vlm.convert --hf-path Qwen/Qwen2-VL-2B-Instruct --mlx-path ./qwen2-vl-2b-mlx
```

Convert with 4-bit quantization:

```
mlx_vlm.convert --hf-path Qwen/Qwen2-VL-2B-Instruct --mlx-path ./qwen2-vl-2b-4bit -q --q-bits 4 --q-group-size 64
```

Dequantize an already-quantized model:

```
mlx_vlm.convert --hf-path ./qwen2-vl-2b-4bit --mlx-path ./qwen2-vl-2b-fp16 -d
```

Convert and upload the result to the Hub:

```
mlx_vlm.convert --hf-path Qwen/Qwen2-VL-2B-Instruct --mlx-path ./qwen2-vl-2b-mlx --upload-repo your-username/Qwen2-VL-2B-Instruct-mlx
```

## Options

### Source

| Flag | Default | Description |
| --- | --- | --- |
| `--hf-path`, `--model` | (none) | Path to the model: a local path or a Hugging Face Hub model identifier. |
| `--revision` | `None` | Hugging Face revision (branch), when converting a model from the Hub. |
| `--mlx-path` | `mlx_model` | Path to save the MLX model. |
| `--trust-remote-code` | `False` | Trust remote code. |

### Quantization

| Flag | Default | Description |
| --- | --- | --- |
| `-q`, `--quantize` | `False` | Generate a quantized model. |
| `--q-group-size` | `None` | Group size for quantization. |
| `--q-bits` | `None` | Bits per weight for quantization. |
| `--q-mode` | `affine` | The quantization mode (`affine`, `mxfp4`, `nvfp4`, `mxfp8`). |
| `--quant-method` | `rtn` | Weight quantization method (`rtn`, `awq`). |
| `--quant-predicate` | `None` | Mixed-bit quantization recipe (`mixed_2_6`, `mixed_3_4`, `mixed_3_5`, `mixed_3_6`, `mixed_3_8`, `mixed_4_6`, `mixed_4_8`). |
| `--calibration` | `text` | AWQ calibration inputs: `text` (default) or `multimodal` (image/audio+text). |
| `--calibration-data` | `None` | Optional directory of real images/audio for `--calibration multimodal`. |
| `--dtype` | `None` | Type to save the parameters (`float16`, `bfloat16`, `float32`); defaults to config.json's `torch_dtype` or the current model weights dtype. |

### MTP

| Flag | Default | Description |
| --- | --- | --- |
| `--mtp` | `False` | Also extract the model's native MTP tensors into a standalone drafter. |
| `--mtp-output` | `None` | Output path for the MTP drafter (default: `<mlx-path>-mtp`). |

### Other

| Flag | Default | Description |
| --- | --- | --- |
| `-d`, `--dequantize` | `False` | Dequantize a quantized model. |
| `--upload-repo` | `None` | The Hugging Face repo to upload the model to. |

## See also

- [Quantization](../performance/quantization.md)
