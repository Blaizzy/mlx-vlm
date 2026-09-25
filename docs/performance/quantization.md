# Quantization

Weight and activation quantization for inference. To quantize the KV cache during generation instead, see [KV cache quantization](kv-cache-quantization.md).

## 1-bit Affine Inference

MLX-VLM can load existing affine 1-bit MLX checkpoints without a custom MLX
build. When a checkpoint declares `"bits": 1`, compatible `Linear` and
`Embedding` layers are replaced automatically with an inference-only module
that JIT-compiles its Metal kernel from Python.

The checkpoint must use MLX's packed `uint32` weight layout, include `scales`
and `biases`, and declare a group size of `32`, `64`, or `128`:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 1,
    "mode": "affine"
  }
}
```

Load and generate normally; no extra inference flag is needed:

```python
from mlx_vlm import generate, load

model, processor = load("path/to/1bit-model")
result = generate(model, processor, "Describe this image", image=["image.jpg"])
```

This path is for inference from an already quantized checkpoint. Converting a
floating-point model to 1-bit still requires a quantizer that can produce the
packed weights and affine parameters.


## Activation Quantization (CUDA)

When running on NVIDIA GPUs with MLX CUDA, models quantized with `mxfp8` or `nvfp4` modes require activation quantization to work properly. This converts `QuantizedLinear` layers to `QQLinear` layers which quantize both weights and activations.

### Command Line

Use the `-qa` or `--quantize-activations` flag:

```sh
mlx_vlm.generate --model /path/to/mxfp8-model --prompt "Describe this image" --image /path/to/image.jpg -qa
```

### Python API

Pass `quantize_activations=True` to the `load` function:

```python
from mlx_vlm import load, generate

# Load with activation quantization enabled
model, processor = load(
    "path/to/mxfp8-quantized-model",
    quantize_activations=True
)

# Generate as usual
output = generate(model, processor, "Describe this image", image=["image.jpg"])
```

### Supported Quantization Modes

- `mxfp8` - 8-bit MX floating point
- `nvfp4` - 4-bit NVIDIA floating point

> **Note**: This feature is required for mxfp/nvfp quantized models on CUDA. On Apple Silicon (Metal), these models work without the flag.

