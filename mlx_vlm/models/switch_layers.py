import math
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

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
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class OffloadedSwitchGLU(nn.Module):
    """Drop-in for ``SwitchGLU``/``InklingSwitchGLU``: computes only the
    router-selected experts, paged from an on-disk :class:`ExpertStore`
    (see ``mlx_vlm.moe_offload``) instead of holding all experts resident.
    Matches ``SwitchGLU.__call__(x, indices) -> [..., K, D]`` exactly (no
    weighting/sum -- the caller already does that).

    ``gate_scale``/``out_scale`` are Inkling's resident NVFP4 correction
    vectors; ``gate_bias``/``up_bias``/``down_bias`` are ``SwitchLinear``'s
    optional per-expert additive bias (e.g. gpt-oss). ``activation`` is the
    original ``SwitchGLU``'s activation object, called exactly as
    ``activation(x_up, x_gate)`` rather than reimplemented inline -- an inline
    ``silu(gate) * x`` measurably diverged from a real checkpoint's compiled
    activation on real data. ``gate_quant``/``up_quant``/``down_quant`` are
    each a ``(group_size, bits, mode)`` triple, resolved per projection since
    a layer can have different bits per projection (mixed-precision converts).
    """

    def __init__(
        self,
        store: Any,
        layer_id: int,
        gate_quant: Tuple[int, int, str],
        up_quant: Tuple[int, int, str],
        down_quant: Tuple[int, int, str],
        activation: Any = None,
        gate_scale: Optional[mx.array] = None,
        out_scale: Optional[mx.array] = None,
        gate_bias: Optional[mx.array] = None,
        up_bias: Optional[mx.array] = None,
        down_bias: Optional[mx.array] = None,
    ):
        super().__init__()
        self.store, self.layer_id = store, layer_id
        self.gate_quant, self.up_quant, self.down_quant = (
            gate_quant,
            up_quant,
            down_quant,
        )
        self.activation = activation
        self.gate_scale, self.out_scale = gate_scale, out_scale
        self.gate_bias, self.up_bias, self.down_bias = gate_bias, up_bias, down_bias

    def _proj(self, xr, w, scales, biases, quant):
        """One expert's projection. ``scales is None`` means this expert's
        weight was never quantized (a plain bf16/float32 checkpoint) --
        matches ``SwitchLinear.__call__``'s plain matmul in that case."""
        if scales is None:
            return xr @ w.T
        group_size, bits, mode = quant
        return mx.quantized_matmul(
            xr,
            w,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    def __call__(self, x, indices) -> mx.array:
        lead, D = x.shape[:-1], x.shape[-1]
        K = indices.shape[-1]
        xf = x.reshape(-1, D)
        idx = np.asarray(indices).reshape(-1, K)
        N = xf.shape[0]
        out = mx.zeros((N, K, D), dtype=x.dtype)
        for j in np.unique(idx):
            j = int(j)
            tok, slot = np.where(idx == j)
            xr = xf[mx.array(tok)]
            (gw, gsc, gb), (uw, usc, ub), (dw, dsc, db) = self.store.get(
                self.layer_id, j
            )
            x_gate = self._proj(xr, gw, gsc, gb, self.gate_quant)
            if self.gate_bias is not None:
                x_gate = x_gate + self.gate_bias[j].astype(x_gate.dtype)
            if self.gate_scale is not None:
                # one scalar per iteration (every row shares expert j), not a gather.
                x_gate = x_gate * self.gate_scale[j].astype(x_gate.dtype)
            x_up = self._proj(xr, uw, usc, ub, self.up_quant)
            if self.up_bias is not None:
                x_up = x_up + self.up_bias[j].astype(x_up.dtype)
            h = (
                self.activation(x_up, x_gate)
                if self.activation is not None
                else (nn.silu(x_gate) * x_up)
            )
            d = self._proj(h, dw, dsc, db, self.down_quant)
            if self.down_bias is not None:
                d = d + self.down_bias[j].astype(d.dtype)
            if self.out_scale is not None:
                d = d * self.out_scale[j].astype(d.dtype)
            out = out.at[mx.array(tok), mx.array(slot)].add(d)
        # Eval per layer, or a long prefill pins every expert across every
        # layer in one graph until the final eval -- OOMs on a large MoE.
        mx.eval(out)
        return out.reshape(*lead, K, D)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
