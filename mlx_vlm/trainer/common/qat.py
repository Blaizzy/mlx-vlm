"""Weight-only affine quantization-aware training for Linear layers."""

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten


class QATLinear(nn.Linear):
    """Keep float weights while simulating packed affine weights in forward passes."""

    def __call__(self, x):
        # Stop the quantizer's gradient; optimize the floating-point master weight.
        weight = mx.stop_gradient(self.weight)
        packed, scales, biases = mx.quantize(
            weight, group_size=self.group_size, bits=self.bits
        )
        quantized = mx.dequantize(
            packed, scales, biases, group_size=self.group_size, bits=self.bits
        )
        weight = self.weight + mx.stop_gradient(quantized - self.weight)
        output = x @ weight.T
        return output + self.bias if "bias" in self else output


def apply_qat(model, *, bits=4, group_size=64):
    """Replace eligible Linear layers locally, preserving checkpoint weight names."""
    replacements = []
    for name, layer in model.named_modules():
        if type(layer) is not nn.Linear or layer.weight.shape[-1] % group_size:
            continue
        output_dims, input_dims = layer.weight.shape
        replacement = QATLinear(input_dims, output_dims, bias="bias" in layer)
        replacement.weight = layer.weight
        if "bias" in layer:
            replacement.bias = layer.bias
        replacement.bits = bits
        replacement.group_size = group_size
        replacements.append((name, replacement))
    if not replacements:
        raise ValueError("No Linear layers support the requested QAT group size.")
    model.update_modules(tree_unflatten(replacements))
    return model


def export_qat_model(model, processor, output_path, *, model_path):
    """Compatibility wrapper; export_model automatically detects QAT layers."""
    from mlx_vlm.export import export_model

    if not any(isinstance(layer, QATLinear) for _, layer in model.named_modules()):
        raise ValueError("Export requires a model prepared with qat=True.")
    return export_model(model, processor, output_path, model_path=model_path)
