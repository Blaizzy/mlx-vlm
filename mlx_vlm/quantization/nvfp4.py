"""NVFP4 linears with global weight scales and optional AWQ input scales."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from ..models.switch_layers import QuantizedSwitchLinear


class ScaledQuantizedLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Module,
        global_scale: mx.array,
        scale_name: str,
        pre_quant_scale: mx.array = None,
    ):
        super().__init__()
        self.weight = linear.weight
        self.scales = linear.scales
        self.biases = linear.get("biases")
        self.group_size = linear.group_size
        self.bits = linear.bits
        self.mode = linear.mode
        self[scale_name] = global_scale
        self._scale_name = scale_name
        self._quantize_activations = isinstance(linear, nn.QQLinear)
        if pre_quant_scale is not None:
            self.pre_quant_scale = pre_quant_scale
        if "bias" in linear:
            self.bias = linear.bias
        self.freeze()

    @property
    def global_scale(self):
        return self[self._scale_name]

    def apply_scale(self, output, scale):
        if self._scale_name == "weight_global_scale":
            return output / scale
        return output * scale

    def __call__(self, x):
        dtype = x.dtype
        # Apply the global scale before rounding the result to the input dtype.
        x = x.astype(mx.float32)
        # ModelOpt AWQ stores input-channel smoothing separately from the
        # packed weights. It must precede both the matmul and optional QQ.
        if "pre_quant_scale" in self:
            x = x * self.pre_quant_scale.astype(mx.float32)
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
        output = self.apply_scale(output.astype(mx.float32), self.global_scale).astype(
            dtype
        )
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
        dtype = x.dtype
        x = x.astype(mx.float32)
        if "pre_quant_scale" in self:
            scale = self.pre_quant_scale.astype(mx.float32)
            if scale.ndim == 2:
                scale = scale[indices][..., None, :]
            x = x * scale
        output = mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        scale = self.global_scale
        if scale.size != 1:
            scale = scale.reshape(self.num_experts, -1)[indices][..., None, :]
        output = self.apply_scale(output, scale).astype(dtype)
        if "bias" in self:
            output = output + self.bias[indices][..., None, :]
        return output


def replace_scaled_quantized_linears(model: nn.Module, weights: dict) -> None:
    def replace(path, module):
        scale_names = [
            name
            for name in ("weight_scale_2", "weight_global_scale")
            if f"{path}.{name}" in weights
        ]
        if not scale_names:
            return module
        if len(scale_names) != 1:
            raise ValueError(f"Conflicting NVFP4 global scales: {path}")
        scale_name = scale_names[0]
        scale = weights[f"{path}.{scale_name}"]
        if (
            not isinstance(
                module, (nn.QuantizedLinear, nn.QQLinear, QuantizedSwitchLinear)
            )
            or module.mode != "nvfp4"
        ):
            raise ValueError(f"NVFP4 global scale requires a quantized linear: {path}")
        pre_quant_scale = weights.get(f"{path}.pre_quant_scale")
        if pre_quant_scale is not None:
            input_dims = module.scales.shape[-1] * module.group_size
            shapes = {(input_dims,), (*module.weight.shape[:-2], input_dims)}
            if pre_quant_scale.shape not in shapes or not mx.issubdtype(
                pre_quant_scale.dtype, mx.floating
            ):
                raise ValueError(f"Invalid NVFP4 pre-quant scale for {path}")
        if isinstance(module, QuantizedSwitchLinear):
            return ScaledQuantizedSwitchLinear(
                module, scale, scale_name, pre_quant_scale
            )
        return ScaledQuantizedLinear(module, scale, scale_name, pre_quant_scale)

    model.update_modules(
        tree_map_with_path(replace, model.leaf_modules(), is_leaf=nn.Module.is_module)
    )
