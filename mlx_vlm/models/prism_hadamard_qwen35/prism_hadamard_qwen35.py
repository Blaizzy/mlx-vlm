import math

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from ..qwen3_5 import Model as Qwen3_5Model
from .config import ModelConfig


def hadamard_transform(x, block, signs, *, inverse=False):
    """Apply the pack's signed, normalized blockwise Walsh-Hadamard rotation."""
    shape, dtype = x.shape, x.dtype
    # The pack was calibrated with float32 rotations, including for FP16 lookup.
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class HadamardQuantizedLinear(nn.Module):
    """Standard MLX affine weights with a rotation of the input activations."""

    def __init__(self, input_dims, output_dims, block):
        super().__init__()
        if input_dims % 128:
            raise ValueError("Packed input width must be divisible by 128")
        if block not in (0, 512, 1024, 2048, 4096) or (block and input_dims % block):
            raise ValueError("Invalid Hadamard block size for packed input width")
        self.block = block
        self.bits = 2
        self.group_size = 128
        self.mode = "affine"
        # Allocate packed placeholders directly; never quantize random dense weights.
        self.weight = mx.zeros((output_dims, input_dims // 16), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 128), dtype=mx.float16)
        self.biases = mx.zeros_like(self.scales)
        if block:
            self.signs = mx.ones((input_dims,), dtype=mx.float32)
        self.freeze()

    def __call__(self, x):
        if self.block:
            x = hadamard_transform(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


class HadamardQuantizedEmbedding(HadamardQuantizedLinear):
    """Dequantize selected rows and rotate them back into the embedding basis."""

    def __call__(self, indices):
        shape = indices.shape
        indices = indices.reshape(-1)
        out = (
            mx.dequantize(
                self.weight[indices],
                self.scales[indices],
                self.biases[indices],
                group_size=self.group_size,
                bits=self.bits,
            )
            .reshape(*shape, -1)
            .astype(mx.float16)
        )
        if self.block:
            out = hadamard_transform(out, self.block, self.signs, inverse=True)
        return out

    def as_linear(self, x):
        return super().__call__(x)


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        modules = dict(self.language_model.named_modules())
        replacements = []
        seen = set()
        for record in config.modules:
            path = record["path"]
            if path in seen:
                raise ValueError(f"Duplicate packed module: {path}")
            seen.add(path)
            original = modules.get(path)
            if original is None:
                raise ValueError(f"Unknown packed module: {path}")
            if not isinstance(original, (nn.Linear, nn.Embedding)):
                raise TypeError(f"Unsupported packed module: {path}")
            if record["embedding"] != isinstance(original, nn.Embedding):
                raise ValueError(f"Packed module kind mismatch: {path}")
            if record["dtype"] != "float16":
                raise ValueError(
                    f"Unsupported packed activation dtype: {record['dtype']}"
                )
            if "bias" in original:
                raise ValueError(
                    f"Packed linear with output bias is unsupported: {path}"
                )
            cls = (
                HadamardQuantizedEmbedding
                if record["embedding"]
                else HadamardQuantizedLinear
            )
            rows, width = original.weight.shape
            replacements.append((path, cls(width, rows, record["block"])))
        self.language_model.update_modules(tree_unflatten(replacements))

    def sanitize(self, weights):
        # Schema 2 already uses MLX layouts and offset RMSNorm weights. In
        # particular, no GGUF head permutation or norm offset is needed here.
        valid_signs = []
        for record in self.config.modules:
            if not record["block"]:
                continue
            key = f"language_model.{record['path']}.signs"
            if key not in weights:
                raise ValueError(f"Missing Hadamard sign vector: {key}")
            signs = weights[key]
            valid_signs.append(mx.all((signs == 1) | (signs == -1)))
        if valid_signs and not mx.all(mx.stack(valid_signs)).item():
            raise ValueError("Hadamard signs must contain only -1 and +1")
        return weights
