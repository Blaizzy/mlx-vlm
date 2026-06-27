import mlx.core as mx
import mlx.nn as nn


class ModelOptMXFP8Linear(nn.Linear):
    """Static ModelOpt FP8 linear that quantizes to MLX-native MXFP8."""

    def to_quantized(
        self,
        group_size: int = None,
        bits: int = None,
        mode: str = "affine",
        quantize_input: bool = False,
    ):
        del group_size, bits, mode
        return super().to_quantized(
            group_size=32,
            bits=8,
            mode="mxfp8",
            quantize_input=quantize_input,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        output_dims, input_dims = linear.weight.shape
        layer = cls(input_dims, output_dims, bias="bias" in linear)
        layer.weight = linear.weight
        if "bias" in linear:
            layer.bias = linear.bias
        return layer


class ModelOptNVFP4SwitchLinear(nn.Module):
    """Routed expert linear for ModelOpt NVFP4 W4A16 weights."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        group_size: int = 16,
        bits: int = 4,
    ):
        super().__init__()
        if input_dims % group_size != 0:
            raise ValueError("ModelOpt NVFP4 input dimensions must be divisible by 16.")
        self.group_size = group_size
        self.bits = bits
        self.mode = "nvfp4"
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims * bits // 32), dtype=mx.uint32
        )
        self.scales = mx.zeros(
            (num_experts, output_dims, input_dims // group_size), dtype=mx.uint8
        )
        self.global_scale = mx.ones((num_experts,), dtype=mx.float32)
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        out = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        scale = self["global_scale"][indices].astype(out.dtype)
        while scale.ndim < out.ndim:
            scale = mx.expand_dims(scale, -1)
        return out * scale

    @classmethod
    def from_switch_linear(cls, linear):
        return cls(linear.input_dims, linear.output_dims, linear.num_experts)


def _uses_modelopt_nvfp4(quantization_config):
    if not isinstance(quantization_config, dict):
        return False
    if quantization_config.get("mode") == "nvfp4" and quantization_config.get(
        "group_size"
    ) == 16:
        return True
    if quantization_config.get("quant_method") != "modelopt":
        return False

    for group in (quantization_config.get("config_groups") or {}).values():
        weights = group.get("weights") or {}
        if (
            weights.get("type") == "float"
            and weights.get("num_bits") == 4
            and weights.get("group_size") == 16
        ):
            return True
    return False


def _replace_with_mxfp8_linear(module, name):
    linear = getattr(module, name, None)
    if isinstance(linear, nn.Linear) and not isinstance(linear, ModelOptMXFP8Linear):
        setattr(module, name, ModelOptMXFP8Linear.from_linear(linear))


def install_modelopt_mxfp8_linears(root, quantization_config):
    if not _uses_modelopt_nvfp4(quantization_config):
        return

    layers = getattr(root.language_model.backbone, "layers", [])
    for layer in layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        if getattr(layer, "block_type", None) == "M":
            _replace_with_mxfp8_linear(mixer, "in_proj")
            _replace_with_mxfp8_linear(mixer, "out_proj")
        elif getattr(layer, "block_type", None) == "E":
            shared_experts = getattr(mixer, "shared_experts", None)
            if shared_experts is not None:
                _replace_with_mxfp8_linear(shared_experts, "up_proj")
                _replace_with_mxfp8_linear(shared_experts, "down_proj")


def install_modelopt_nvfp4_switch_linears(root, quantization_config):
    if not _uses_modelopt_nvfp4(quantization_config):
        return

    layers = getattr(root.language_model.backbone, "layers", [])
    for layer in layers:
        if getattr(layer, "block_type", None) != "E":
            continue
        switch_mlp = getattr(layer.mixer, "switch_mlp", None)
        if switch_mlp is None:
            continue
        for name in ("fc1", "fc2"):
            module = getattr(switch_mlp, name, None)
            if module is None or isinstance(module, ModelOptNVFP4SwitchLinear):
                continue
            setattr(
                switch_mlp,
                name,
                ModelOptNVFP4SwitchLinear.from_switch_linear(module),
            )
