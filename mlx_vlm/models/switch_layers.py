import math
from functools import partial
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .activations import swiglu
from .linear import DECODE_BLOCK_SIZE

GLOBAL_SCALE_SUFFIXES = ("weight_scale_2", "weight_global_scale")
EXPERT_WEIGHT_SUFFIXES = (
    "weight",
    "scales",
    "biases",
    *GLOBAL_SCALE_SUFFIXES,
)


def expert_scale_rows(scale, shape):
    """Broadcast a tensor or per-expert factor over its output rows."""
    if scale.size == 1:
        scale = scale.reshape(())
    else:
        scale = scale.reshape(*shape[:-1], -1)
    return mx.broadcast_to(scale, shape)


def expand_expert_scales(weights, prefixes):
    """Keep independent global factors when concatenating output rows."""
    for prefix in prefixes:
        for suffix in GLOBAL_SCALE_SUFFIXES:
            key = f"{prefix}.{suffix}"
            if key in weights:
                weights[key] = expert_scale_rows(
                    weights[key], weights[f"{prefix}.weight"].shape[:-1]
                )


def stack_expert_projection(weights, source, projection, target, num_experts):
    for suffix in (*EXPERT_WEIGHT_SUFFIXES, "bias"):
        if f"{source}.0.{projection}.{suffix}" in weights:
            weights[f"{target}.{suffix}"] = mx.stack(
                [
                    weights.pop(f"{source}.{i}.{projection}.{suffix}")
                    for i in range(num_experts)
                ]
            )


def split_expert_projection(weights, source, targets):
    """Split fused output rows, including each row's quantization parameters."""
    weight_key = f"{source}.weight"
    if source in weights:
        weights[weight_key] = weights.pop(source)
    if weight_key not in weights:
        return
    if weights[weight_key].ndim != 3:
        raise ValueError(
            f"{source} has shape {weights[weight_key].shape}; expected "
            "[num_experts, output_rows, input_columns]"
        )
    shape = weights[weight_key].shape[:-1]
    for suffix in (*EXPERT_WEIGHT_SUFFIXES, "bias"):
        key = f"{source}.{suffix}"
        if key not in weights:
            continue
        value = weights.pop(key)
        if suffix in GLOBAL_SCALE_SUFFIXES:
            value = expert_scale_rows(value, shape)
        axis = -1 if suffix in (*GLOBAL_SCALE_SUFFIXES, "bias") else -2
        parts = mx.split(value, len(targets), axis=axis)
        for target, part in zip(targets, parts):
            weights[f"{target}.{suffix}"] = mx.contiguous(part)


def move_expert_projection(weights, source, target):
    for suffix in (*EXPERT_WEIGHT_SUFFIXES, "bias"):
        key = (
            source if suffix == "weight" and source in weights else f"{source}.{suffix}"
        )
        if key in weights:
            weights[f"{target}.{suffix}"] = weights.pop(key)


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


@partial(mx.compile, shapeless=True)
def _weighted_sum(routed, weights):
    return (
        (routed.astype(weights.dtype) * weights[..., None])
        .sum(axis=-2)
        .astype(routed.dtype)
    )


class SwitchGLU(nn.Module):
    """Selected experts with optional weighting in an explicit accumulation dtype."""

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

    def __call__(
        self,
        x,
        indices,
        weights=None,
        shared=None,
        residual=None,
        *,
        accumulation_dtype=None,
    ) -> mx.array:
        if (
            not self.training
            and accumulation_dtype is not None
            and accumulation_dtype == mx.float32
            and weights is not None
        ):
            from .fast_ops import exact_affine_moe_down_fp32, exact_affine_swiglu

            activated = exact_affine_swiglu(self, x, indices)
            if activated is not None:
                output = exact_affine_moe_down_fp32(
                    self.down_proj, activated, indices, weights, shared
                )
                if output is not None:
                    return self._combine(output, None, None, residual)
        if not self.training and x.ndim == 3 and 1 < x.shape[1] <= DECODE_BLOCK_SIZE:
            from .fast_ops import exact_affine_moe_down, exact_affine_switch_gate_up
            from .linear import tokenwise
            from .quantized_verifier import (
                exact_quantized_moe_hc_expand,
                exact_quantized_selected_linear,
                exact_quantized_switch_linear,
            )

            routed = None
            # The route count per position determines decode's sorting policy.
            if x.shape[0] * indices.shape[-1] < 64:
                gate_up = exact_affine_switch_gate_up(self, x, indices)
                if gate_up is None:
                    up = exact_quantized_switch_linear(self.up_proj, x, indices)
                    gate = exact_quantized_switch_linear(self.gate_proj, x, indices)
                else:
                    up, gate = gate_up
                if up is not None and gate is not None:
                    activated = self.activation(up, gate)
                    if residual is not None and accumulation_dtype is None:
                        output = exact_quantized_moe_hc_expand(
                            self.down_proj,
                            activated,
                            indices,
                            weights,
                            shared,
                            *residual,
                        )
                        if output is not None:
                            return output
                    if (
                        weights is not None
                        and shared is not None
                        and accumulation_dtype is None
                    ):
                        output = exact_affine_moe_down(
                            self.down_proj, activated, indices, weights, shared
                        )
                        if output is not None:
                            return self._combine(output, None, None, residual)
                    routed = exact_quantized_selected_linear(
                        self.down_proj, activated, indices
                    )
            if routed is None:
                routed = tokenwise(self, x, indices)
            return self._combine(
                routed, weights, shared, residual, accumulation_dtype=accumulation_dtype
            )

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

        return self._combine(
            x.squeeze(-2),
            weights,
            shared,
            residual,
            accumulation_dtype=accumulation_dtype,
        )

    @staticmethod
    def _combine(routed, weights, shared, residual=None, *, accumulation_dtype=None):
        if weights is not None:
            if accumulation_dtype is None:
                routed = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
            else:
                routed = _weighted_sum(routed, weights.astype(accumulation_dtype))
        output = routed if shared is None else routed + shared
        if residual is not None:
            from .deepseek_v4.hyper_connection import hc_expand

            output = hc_expand(output, *residual)
        return output


class MoE(nn.Module):
    """Weighted routed experts plus shared experts, with optional residual fusion."""

    def __call__(self, x, *args, residual=None, **kwargs):
        group = getattr(self, "sharding_group", None)
        if group is not None:
            from mlx.nn.layers.distributed import sum_gradients

            x = sum_gradients(group)(x)
        indices, weights = self.gate(x, *args, **kwargs)
        shared = self.shared_experts(x)
        fused_residual = residual if group is None else None
        if isinstance(self.switch_mlp, SwitchGLU):
            output = self.switch_mlp(
                x, indices, weights, shared, residual=fused_residual
            )
        else:
            output = SwitchGLU._combine(
                self.switch_mlp(x, indices), weights, shared, fused_residual
            )
        if group is not None:
            output = mx.distributed.all_sum(output, group=group)
            output = SwitchGLU._combine(output, None, None, residual)
        return output


class OffloadedSwitchGLU(nn.Module):
    """Drop-in for ``SwitchGLU``/``InklingSwitchGLU``: computes only the
    router-selected experts, paged from an on-disk :class:`ExpertStore`
    (see ``mlx_vlm.moe_offload``) instead of holding all experts resident.
    Matches ``SwitchGLU.__call__(x, indices) -> [..., K, D]`` exactly (no
    weighting/sum -- the caller already does that).

    ``gate_scale``/``out_scale`` are Inkling's resident NVFP4 correction
    vectors; ``gate_bias``/``up_bias``/``down_bias`` are ``SwitchLinear``'s
    optional per-expert additive bias (e.g. gpt-oss). ``activation`` is the
    replaced layer's activation object, called exactly as
    ``activation(x_up, x_gate)`` rather than reimplemented inline -- an inline
    ``silu(gate) * x`` measurably diverged from a real checkpoint's compiled
    activation on real data, and silently so for a layer whose gating is not
    SiLU-based at all, so it is required rather than defaulted.
    ``gate_quant``/``up_quant``/``down_quant`` are each a
    ``(group_size, bits, mode)`` triple, resolved per projection since a layer
    can have different bits per projection (mixed-precision converts).
    """

    def __init__(
        self,
        store: Any,
        layer_id: int,
        gate_quant: Tuple[int, int, str],
        up_quant: Tuple[int, int, str],
        down_quant: Tuple[int, int, str],
        activation: Any,
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

    def _apply_selected(self, x, experts, local_indices, selected):
        """Batch a bounded set of experts into three gathered projections."""
        experts = [int(j) for j in experts]
        routed_experts = mx.array(experts)[local_indices]

        def project(value, projection, quant):
            name = ("gate_proj", "up_proj", "down_proj")[projection]
            tensors = [selected[j][projection] for j in experts]
            weight = mx.stack([t[0] for t in tensors])
            scales = [self.store.global_scale(self.layer_id, j, name) for j in experts]
            has_global_scale = any(s is not None for s, _ in scales)
            if tensors[0][1] is None:
                dtype = mx.result_type(value.dtype, weight.dtype)
                # CPU gather_mm only supports FP32, unlike ordinary matmul.
                compute_dtype = mx.float32 if mx.default_device() == mx.cpu else dtype
                output = mx.gather_mm(
                    value.astype(compute_dtype),
                    weight.astype(compute_dtype).swapaxes(-1, -2),
                    rhs_indices=local_indices,
                ).astype(dtype)
            else:
                group_size, bits, mode = quant
                fused_scale = (
                    mode == "nvfp4"
                    and has_global_scale
                    and mx.default_device() == mx.gpu
                    and all(s is None or s.size == 1 for s, _ in scales)
                )
                # MLX expects tensor absmax: 448 * 6 times the checkpoint's
                # multiplicative factor. Fuse it before rounding to BF16.
                global_scale = (
                    mx.stack(
                        [
                            (
                                mx.array(2688.0)
                                if s is None
                                else (2688.0 / s if divide else 2688.0 * s).reshape(())
                            )
                            for s, divide in scales
                        ]
                    )
                    if fused_scale
                    else None
                )
                output = mx.gather_qmm(
                    (
                        value.astype(mx.float32)
                        if has_global_scale and not fused_scale
                        else value
                    ),
                    weight,
                    mx.stack([t[1] for t in tensors]),
                    (
                        mx.stack([t[2] for t in tensors])
                        if tensors[0][2] is not None
                        else None
                    ),
                    rhs_indices=local_indices,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    global_scale=global_scale,
                )
                if has_global_scale and not fused_scale:
                    # Global factors can be scalars or one value per output row.
                    factors = mx.stack(
                        [
                            mx.broadcast_to(
                                s if s is not None else mx.array(1.0),
                                (weight.shape[-2],),
                            )
                            for s, _ in scales
                        ]
                    )[local_indices][:, None, :]
                    divide = mx.array([d for _, d in scales])[local_indices][
                        :, None, None
                    ]
                    output = mx.where(divide, output / factors, output * factors)
                output = output.astype(value.dtype)
            bias = (self.gate_bias, self.up_bias, self.down_bias)[projection]
            if bias is not None:
                output += bias[routed_experts][:, None, :].astype(output.dtype)
            return output

        gate = project(x, 0, self.gate_quant)
        if self.gate_scale is not None:
            gate *= self.gate_scale[routed_experts][:, None, None].astype(gate.dtype)
        up = project(x, 1, self.up_quant)
        output = project(self.activation(up, gate), 2, self.down_quant)
        if self.out_scale is not None:
            output *= self.out_scale[routed_experts][:, None, None].astype(output.dtype)
        return output.squeeze(-2).astype(x.dtype)

    def __call__(self, x, indices) -> mx.array:
        lead, D = x.shape[:-1], x.shape[-1]
        K = indices.shape[-1]
        xf = x.reshape(-1, D)
        idx = np.asarray(indices).reshape(-1, K)
        N = xf.shape[0]
        out = mx.zeros((N, K, D), dtype=x.dtype)
        uniq = np.unique(idx)
        # A large prefill chunk routes over most of num_experts regardless of
        # top-k -- per-expert caching buys nothing there (everything gets
        # touched anyway) and its LRU/eviction bookkeeping just thrashes.
        # Bulk-load the whole layer once instead; decode's later selective
        # get() calls are unaffected (see ExpertStore.get_all).
        selected = (
            self.store.get_all(self.layer_id, uniq)
            if len(uniq) * 2 > self.store.num_experts
            else self.store.get_many(self.layer_id, uniq)
        )
        if N == 1:
            local = mx.array(np.searchsorted(uniq, idx[0]), dtype=mx.uint32)
            return self._apply_selected(xf, uniq, local, selected).reshape(*lead, K, D)
        # Prefill touches many experts. Bound temporary stacked weights to eight
        # experts rather than materializing an entire layer's batched weights.
        for start in range(0, len(uniq), 8):
            group = uniq[start : start + 8]
            tok, slot = np.where(np.isin(idx, group))
            local = mx.array(np.searchsorted(group, idx[tok, slot]), dtype=mx.uint32)
            xr = xf[mx.array(tok)][:, None, :]
            d = self._apply_selected(xr, group, local, selected)
            out = out.at[mx.array(tok), mx.array(slot)].add(d)
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
