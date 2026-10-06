"""NVFP4 linear layers with a separate per-tensor weight scale."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from ..models.switch_layers import QuantizedSwitchLinear


class ScaledQuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Module, global_scale: mx.array):
        super().__init__()
        self.weight = linear.weight
        self.scales = linear.scales
        self.biases = linear.get("biases")
        self.group_size = linear.group_size
        self.bits = linear.bits
        self.mode = linear.mode
        self.weight_scale_2 = global_scale
        self._quantize_activations = isinstance(linear, nn.QQLinear)
        if "bias" in linear:
            self.bias = linear.bias
        self.freeze()

    def __call__(self, x):
        dtype = x.dtype
        # Apply the global scale before rounding the result to the input dtype.
        x = x.astype(mx.float32)
        if self._quantize_activations:
            output = mx.qqmm(
                x,
                self.weight,
                scales=self.scales,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            output = mx.quantized_matmul(
                x,
                self.weight,
                scales=self.scales,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                transpose=True,
            )
        output = (output.astype(mx.float32) * self.weight_scale_2).astype(dtype)
        if "bias" in self:
            output = output + self.bias
        return output


class ScaledQuantizedSwitchLinear(ScaledQuantizedLinear):
    @property
    def input_dims(self):
        return self.scales.shape[-1] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[-2]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        output = mx.gather_qmm(
            x.astype(mx.float32),
            self.weight,
            self.scales,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        scale = self.weight_scale_2[indices][..., None, None]
        output = (output * scale).astype(x.dtype)
        if "bias" in self:
            output = output + self.bias[indices][..., None, :]
        return output


def replace_scaled_quantized_linears(model: nn.Module, weights: dict) -> None:
    def replace(path, module):
        scale = weights.get(f"{path}.weight_scale_2")
        if scale is None:
            return module
        if (
            not isinstance(
                module, (nn.QuantizedLinear, nn.QQLinear, QuantizedSwitchLinear)
            )
            or module.mode != "nvfp4"
        ):
            raise ValueError(f"NVFP4 global scale requires a quantized linear: {path}")
        if isinstance(module, QuantizedSwitchLinear):
            return ScaledQuantizedSwitchLinear(module, scale)
        return ScaledQuantizedLinear(module, scale)

    model.update_modules(
        tree_map_with_path(replace, model.leaf_modules(), is_leaf=nn.Module.is_module)
    )
