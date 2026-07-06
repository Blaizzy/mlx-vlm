import math
from functools import lru_cache
from typing import List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn

_HAS_METAL = mx.metal.is_available()
_HALF_SPLIT = "half_split"
_EVEN_ODD = "even_odd"
_HALF_COS = "half"
_FULL_COS = "full"


class SuScaledRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: Union[List[float], float] = 1.0,
        long_factor: Union[List[float], float] = 1.0,
        short_mscale: float = None,
        long_mscale: float = None,
    ):
        """
        Su Scaled Rotary Embedding layer.

        Args:
            dims (int): The feature dimensions to be rotated.
            base (int, optional): Base for the exponential scaling.
            max_position_embeddings (int, optional): The maximum sequence
              length that this model was trained with. This is used to determine
              the size of the original RoPE embeddings when using long scaling.
              Default: ``131072``.
            original_max_position_embeddings (int, optional): The maximum
              sequence length that this model was trained with. This is used to
              determine the size of the original RoPE embeddings when using long
              scaling. Default: ``4096``.
            short_factor (float or list[float], optional): List of scaling
              factors for sequences of length lesser than
              ``original_max_position_embeddings``. Default: ``1.0``.
            long_factor (float or list[float], optional): List of scaling
              factors for sequences of length greater than
              ``original_max_position_embeddings``.  Default: ``1.0``.
            short_mscale (float, optional): Scale the input prior to embedding.
            long_mscale (float, optional): Scale the input prior to embedding.
        """
        super().__init__()
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dims

        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs

        def default_scale(factor):
            return math.sqrt(
                1 + math.log(factor) / math.log(original_max_position_embeddings)
            )

        factor = max_position_embeddings / original_max_position_embeddings
        self._scale = long_mscale or (1.0 if factor <= 1.0 else default_scale(factor))

    def __call__(self, x, offset: Union[int, mx.array] = 0):
        x = x[...]
        x[..., : self.dim] = self._scale * x[..., : self.dim]
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Llama3RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def extra_repr(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims,
        traditional=False,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001

            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (
                max_val - min_val
            )
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x, offset=0):
        if self.mscale != 1.0:
            x = x[...]
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class ProportionalRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        rotated_dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        factor: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        if rotated_dims > dims:
            raise ValueError("rotated_dims should be smaller than dims")

        exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
        self._freqs = mx.concatenate(
            [
                factor * (base**exponents),
                mx.full(((dims - rotated_dims) // 2,), mx.inf),
            ]
        )

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims,
    base,
    traditional,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
):
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    else:
        rope_type = "default"

    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    if rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )

    if rope_type in ("yarn", "deepseek_yarn", "telechat3-yarn"):
        scaling_factor = scaling_config["factor"]
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            scaling_factor=scaling_factor,
            base=base,
            **rope_kwargs,
        )

    if rope_type == "longrope":
        return SuScaledRoPE(
            dims=dims,
            base=base,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=scaling_config[
                "original_max_position_embeddings"
            ],
            short_factor=scaling_config["short_factor"],
            long_factor=scaling_config["long_factor"],
        )

    if rope_type == "proportional":
        return ProportionalRoPE(
            dims=dims,
            rotated_dims=int(dims * scaling_config.get("partial_rotary_factor", 1.0)),
            traditional=traditional,
            base=base,
            factor=scaling_config.get("factor", 1.0),
        )

    if rope_type == "mrope":
        mrope_section = scaling_config.get("mrope_section", [])
        assert (
            len(mrope_section) == 3
        ), f"MRoPE currently only supports 3 sections, got {len(mrope_section)}."
        return nn.RoPE(dims, traditional=traditional, base=base)

    raise ValueError(f"Unsupported RoPE type {rope_type}")


def _cumulative_splits(lengths: Sequence[int]):
    return mx.cumsum(mx.array(lengths, dtype=mx.int32))[:-1]


def _interleaved_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    for dim, offset in enumerate((1, 2), start=1):
        for idx in range(offset, min(mrope_section[dim] * 3, freq_dim), 3):
            selector[idx] = dim
    return mx.array(selector, dtype=mx.int32)


def _chunked_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    offset = mrope_section[0]
    for dim, length in enumerate(mrope_section[1:], start=1):
        for idx in range(offset, min(offset + length, freq_dim)):
            selector[idx] = dim
        offset += length
    return mx.array(selector, dtype=mx.int32)


@mx.compile
def _selected_mrope_freqs(position_ids, inv_freq, position_selector):
    positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
    return positions.astype(mx.float32) * inv_freq


def mrope_position_selector(style: str, mrope_section: Sequence[int], freq_dim: int):
    if style == "interleaved":
        return _interleaved_position_selector(mrope_section, freq_dim)
    return _chunked_position_selector(mrope_section, freq_dim)


def _selects_frequency_by_position(style: str):
    return style in {"chunked", "interleaved", "split_select"}


def _is_sectioned_style(style: str):
    return style in {"sectioned_half_split", "sectioned_even_odd"}


def _has_mrope_apply_selector(style: str):
    return _selects_frequency_by_position(style) or _is_sectioned_style(style)


def _uses_even_odd_pairing(style: str):
    return style in {"sectioned_even_odd", "split_select", "ernie_3d"}


def _needs_even_odd_layout(style: str):
    return style in {"sectioned_even_odd", "split_select"}


def _pairing_for_style(style: str):
    if _uses_even_odd_pairing(style):
        return _EVEN_ODD
    return _HALF_SPLIT


@lru_cache(maxsize=None)
def _mrope_apply_kernel(
    rotary_dim: int,
    position_ndim: int,
    pairing: str,
):
    if not _HAS_METAL:
        return None

    if position_ndim == 2:
        position_expr = "position_ids[b * q_len + t]"
        selector_source = ""
    else:
        position_expr = "position_ids[(axis * q_bsz + b) * q_len + t]"
        selector_source = "int axis = int(position_selector[freq_idx]);"

    if pairing == _EVEN_ODD:
        pair_source = f"""
        int freq_idx = slot;
        int d = freq_idx * 2;
        int pair_d = d + 1;
        {selector_source}
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        """
    else:
        pair_source = f"""
        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
        {selector_source}
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        """

    source = f"""
        uint elem = thread_position_in_grid.x;

        const int half_dim = {rotary_dim // 2};
        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int slots = half_dim + q_dim - {rotary_dim};
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {{
            return;
        }}

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {{
            int pass_d = {rotary_dim} + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }}

        {pair_source}

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
    """

    return mx.fast.metal_kernel(
        name=f"mrope_apply_{pairing}_{rotary_dim}_{position_ndim}d",
        input_names=["x", "position_ids", "inv_freq", "position_selector"],
        output_names=["x_out"],
        source=source,
    )


def _mrope_apply_cos_sin(x, position_ids, inv_freq, position_selector, pairing):
    """Pure-MLX cos/sin matching the fused MRoPE kernel's angle computation.

    Mirrors ``_mrope_apply_kernel``: ``angle = pos * inv_freq[freq_idx]`` with
    ``pos`` selected per-axis (via ``position_selector``) for 3D position ids.
    Returns cos/sin already laid out for the pairing so a plain rotate matches
    the kernel's element-wise pair rotation.
    """
    half_dim = inv_freq.shape[0]
    if position_ids.ndim == 2:
        # (b, t, half) angles; the same scalar position feeds every freq.
        angle = position_ids.astype(mx.float32)[..., None] * inv_freq
    else:
        # (axis, b, t) -> select the axis feeding each freq, giving (b, t, half).
        positions = mx.take(position_ids, position_selector, axis=0)
        angle = positions.transpose(1, 2, 0).astype(mx.float32) * inv_freq
    cos = mx.cos(angle)[:, None, :, :]
    sin = mx.sin(angle)[:, None, :, :]
    if pairing == _EVEN_ODD:
        cos = mx.repeat(cos, repeats=2, axis=-1)
        sin = mx.repeat(sin, repeats=2, axis=-1)
    else:
        cos = mx.concatenate([cos, cos], axis=-1)
        sin = mx.concatenate([sin, sin], axis=-1)
    return cos, sin, half_dim


def _mrope_apply(q, k, position_ids, inv_freq, position_selector, pairing):
    """Differentiable pure-MLX equivalent of the fused MRoPE kernel apply."""
    cos, sin, _ = _mrope_apply_cos_sin(
        q, position_ids, inv_freq, position_selector, pairing
    )
    rotate_fn = rotate_half_even_odd if pairing == _EVEN_ODD else rotate_half
    return _apply_rotary_embedding(
        q, k, cos, sin, rotate_fn, cast_output=True, compute_dtype=mx.float32
    )


def _fast_mrope_apply(
    kernel,
    q,
    k,
    position_ids,
    inv_freq,
    position_selector,
    pairing=_HALF_SPLIT,
):
    def apply_one(x):
        half_dim = inv_freq.shape[0]
        slots = half_dim + x.shape[-1] - half_dim * 2
        work_size = x.shape[0] * x.shape[1] * x.shape[2] * slots
        (out,) = kernel(
            inputs=[x, position_ids, inv_freq, position_selector],
            template=[("T", x.dtype)],
            grid=(work_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )
        return out

    # Wrap the kernel forward in a custom_function so value_and_grad (training)
    # can differentiate through it: a raw CustomKernel has no VJP, so route the
    # gradient through the pure-MLX equivalent. position_ids/inv_freq/
    # position_selector are position constants (zero cotangent).
    @mx.custom_function
    def apply(q, k, position_ids, inv_freq, position_selector):
        return apply_one(q), apply_one(k)

    @apply.vjp
    def apply_vjp(primals, cotangents, _output):
        q, k, position_ids, inv_freq, position_selector = primals
        _, (dq, dk) = mx.vjp(
            lambda q, k: list(
                _mrope_apply(q, k, position_ids, inv_freq, position_selector, pairing)
            ),
            [q, k],
            list(cotangents),
        )
        return (
            dq,
            dk,
            mx.zeros_like(position_ids),
            mx.zeros_like(inv_freq),
            mx.zeros_like(position_selector),
        )

    return apply(q, k, position_ids, inv_freq, position_selector)


@lru_cache(maxsize=None)
def _rotary_apply_kernel(
    rotary_dim: int,
    pairing: str,
    cos_ndim: int,
    sectioned: bool,
    cos_layout: str,
):
    if not _HAS_METAL:
        return None

    if pairing == _EVEN_ODD:
        pair_source = """
        int freq_idx = slot;
        int d = freq_idx * 2;
        int pair_d = d + 1;
        """
    else:
        pair_source = """
        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
        """

    if pairing == _HALF_SPLIT:
        cos_freq_idx = "freq_idx"
        pair_cos_freq_idx = "pair_d"
    elif cos_layout == _FULL_COS:
        cos_freq_idx = "d"
        pair_cos_freq_idx = "pair_d"
    else:
        cos_freq_idx = "freq_idx"
        pair_cos_freq_idx = "freq_idx"

    if sectioned:
        cos_expr = (
            f"cos[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + " f"{cos_freq_idx}]"
        )
        sin_expr = (
            f"sin[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + " f"{cos_freq_idx}]"
        )
        pair_cos_expr = (
            f"cos[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        pair_sin_expr = (
            f"sin[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        selector_source = "int axis = int(position_selector[freq_idx]);"
        input_names = ["x", "cos", "sin", "position_selector"]
    elif cos_ndim == 4:
        cos_expr = f"cos[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        sin_expr = f"sin[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        pair_cos_expr = (
            f"cos[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        pair_sin_expr = (
            f"sin[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        selector_source = ""
        input_names = ["x", "cos", "sin"]
    else:
        cos_expr = f"cos[(b * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        sin_expr = f"sin[(b * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        pair_cos_expr = f"cos[(b * q_len + t) * {rotary_dim} + {pair_cos_freq_idx}]"
        pair_sin_expr = f"sin[(b * q_len + t) * {rotary_dim} + {pair_cos_freq_idx}]"
        selector_source = ""
        input_names = ["x", "cos", "sin"]

    source = f"""
        uint elem = thread_position_in_grid.x;

        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int half_dim = {rotary_dim // 2};
        const int slots = half_dim + q_dim - {rotary_dim};
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {{
            return;
        }}

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {{
            int pass_d = {rotary_dim} + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }}

        {pair_source}
        {selector_source}
        float c = static_cast<float>({cos_expr});
        float s = static_cast<float>({sin_expr});
        float cp = static_cast<float>({pair_cos_expr});
        float sp = static_cast<float>({pair_sin_expr});

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * cp + xv * sp);
    """

    section_suffix = "sectioned" if sectioned else "preselected"
    return mx.fast.metal_kernel(
        name=(
            f"rotary_apply_{pairing}_{cos_layout}_{section_suffix}_"
            f"{rotary_dim}_{cos_ndim}d"
        ),
        input_names=input_names,
        output_names=["x_out"],
        source=source,
    )


def _precomputed_rotary(
    q,
    k,
    cos,
    sin,
    *,
    pairing,
    cos_layout,
    sectioned,
    mrope_section,
):
    """Differentiable pure-MLX equivalent of ``_fast_rotary_apply``.

    Reproduces the fallback layout applied to cos/sin (half->full expansion for
    even/odd, sectioning for sectioned styles) before the element-wise rotation
    so the gradient matches the kernel forward.
    """
    if sectioned:
        cos, sin = _section_cos_sin(cos, sin, mrope_section)
    else:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    if pairing == _EVEN_ODD:
        if cos_layout == _HALF_COS:
            cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
            sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)
        rotate_fn = rotate_half_even_odd
    else:
        rotate_fn = rotate_half

    return _apply_rotary_embedding(
        q, k, cos, sin, rotate_fn, cast_output=True, compute_dtype=mx.float32
    )


def _fast_rotary_apply(
    kernel,
    q,
    k,
    cos,
    sin,
    position_selector=None,
    *,
    pairing=_HALF_SPLIT,
    cos_layout=_HALF_COS,
    sectioned=False,
    mrope_section=None,
):
    def apply_one(x):
        rotary_dim = cos.shape[-1]
        slots = rotary_dim // 2 + x.shape[-1] - rotary_dim
        work_size = x.shape[0] * x.shape[1] * x.shape[2] * slots
        inputs = [x, cos, sin]
        if position_selector is not None:
            inputs.append(position_selector)
        (out,) = kernel(
            inputs=inputs,
            template=[("T", x.dtype)],
            grid=(work_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )
        return out

    @mx.custom_function
    def apply(q, k, cos, sin):
        return apply_one(q), apply_one(k)

    @apply.vjp
    def apply_vjp(primals, cotangents, _output):
        q, k, cos, sin = primals
        _, (dq, dk) = mx.vjp(
            lambda q, k: list(
                _precomputed_rotary(
                    q,
                    k,
                    cos,
                    sin,
                    pairing=pairing,
                    cos_layout=cos_layout,
                    sectioned=sectioned,
                    mrope_section=mrope_section,
                )
            ),
            [q, k],
            list(cotangents),
        )
        return (dq, dk, mx.zeros_like(cos), mx.zeros_like(sin))

    return apply(q, k, cos, sin)


@lru_cache(maxsize=None)
def _compiled_rotary_apply(
    rotary_dim: int,
    pairing: str,
    cos_ndim: int,
    sectioned: bool,
    cos_layout: str,
    mrope_section: tuple = (),
):
    kernel = _rotary_apply_kernel(
        rotary_dim,
        pairing,
        cos_ndim,
        sectioned,
        cos_layout,
    )
    if kernel is None:
        return None

    section = list(mrope_section) if mrope_section else None

    @mx.compile
    def apply(q, k, cos, sin, position_selector):
        return _fast_rotary_apply(
            kernel,
            q,
            k,
            cos,
            sin,
            position_selector,
            pairing=pairing,
            cos_layout=cos_layout,
            sectioned=sectioned,
            mrope_section=section,
        )

    return apply


@lru_cache(maxsize=None)
def _compiled_mrope_apply(rotary_dim: int, position_ndim: int, pairing: str):
    kernel = _mrope_apply_kernel(rotary_dim, position_ndim, pairing)
    if kernel is None:
        return None

    @mx.compile
    def apply(q, k, position_ids, inv_freq, position_selector):
        return _fast_mrope_apply(
            kernel,
            q,
            k,
            position_ids,
            inv_freq,
            position_selector,
            pairing,
        )

    return apply


def get_mrope_section(
    *,
    rope_scaling: Optional[dict] = None,
    rope_parameters: Optional[dict] = None,
    default: Sequence[int] = (24, 20, 20),
):
    rope_scaling = rope_scaling or {}
    rope_parameters = rope_parameters or {}
    return list(
        rope_parameters.get("mrope_section")
        or rope_scaling.get("mrope_section")
        or default
    )


def compute_inv_freq(dim: int, base: float):
    return 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))


@mx.compile
def _apply_selected_mrope_frequency_layout(freqs, position_selector):
    indices = mx.broadcast_to(
        position_selector[None, None, None, :],
        (1, freqs.shape[1], freqs.shape[2], freqs.shape[3]),
    )
    return mx.take_along_axis(freqs, indices, axis=0)[0]


def mrope_section_selectors(
    mrope_section: Sequence[int],
    position_axes: Sequence[int],
    *,
    interleave_sections: Sequence[int] = (),
):
    mrope_section = list(mrope_section)
    position_axes = list(position_axes)
    interleave_sections = list(interleave_sections)
    if len(position_axes) != len(mrope_section):
        raise ValueError("position_axes must match mrope_section length")

    offsets = [0]
    for length in mrope_section[:-1]:
        offsets.append(offsets[-1] + length)

    position_selector = []
    frequency_selector = []

    interleaved = set(interleave_sections)
    if interleave_sections:
        interleave_length = mrope_section[interleave_sections[0]]
        if any(mrope_section[idx] != interleave_length for idx in interleave_sections):
            raise ValueError("interleaved MRoPE sections must have equal length")
        for local_idx in range(interleave_length):
            for section_idx in interleave_sections:
                position_selector.append(position_axes[section_idx])
                frequency_selector.append(offsets[section_idx] + local_idx)

    for section_idx, length in enumerate(mrope_section):
        if section_idx in interleaved:
            continue
        offset = offsets[section_idx]
        for local_idx in range(length):
            position_selector.append(position_axes[section_idx])
            frequency_selector.append(offset + local_idx)

    return (
        mx.array(position_selector, dtype=mx.int32),
        mx.array(frequency_selector, dtype=mx.int32),
    )


@mx.compile
def compute_selected_mrope_cos_sin(
    position_ids,
    inv_freq,
    position_selector,
    frequency_selector,
):
    positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
    freqs = positions.astype(mx.float32) * mx.take(inv_freq, frequency_selector)
    emb = mx.repeat(freqs, repeats=2, axis=-1)
    return mx.cos(emb), mx.sin(emb)


def apply_mrope_frequency_layout(
    freqs,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
):
    mrope_section = list(mrope_section)

    if _selects_frequency_by_position(style):
        position_selector = mrope_position_selector(
            style,
            mrope_section,
            freqs.shape[-1],
        )
        return _apply_selected_mrope_frequency_layout(freqs, position_selector)
    return freqs


def compute_mrope_frequencies(
    position_ids,
    inv_freq,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
    position_selector=None,
):
    if position_ids.ndim == 2:
        # Text-only positions use the same scalar position for every MRoPE axis,
        # so chunked/interleaved layout selection collapses to the same angles.
        return position_ids.astype(mx.float32)[..., None] * inv_freq

    # Fast path
    if _selects_frequency_by_position(style):
        if position_selector is None:
            position_selector = mrope_position_selector(
                style,
                mrope_section,
                inv_freq.shape[0],
            )
        return _selected_mrope_freqs(position_ids, inv_freq, position_selector)

    # Slow path
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    return apply_mrope_frequency_layout(freqs, mrope_section, style=style)


class MRoPERotaryEmbedding(nn.Module):
    """Shared language-side rotary embedding for MRoPE models.

    ``style`` selects whether frequency layout is applied in the embedding
    itself (Qwen/GLM-OCR style) or deferred to Q/K application
    (PaddleOCR/GLM4V-style sectioned RoPE).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        *,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        mrope_section: Optional[Sequence[int]] = None,
        attention_scaling: float = 1.0,
        cast_output: bool = True,
        style: str = "interleaved",
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.style = style
        self._inv_freq = compute_inv_freq(dim, base)
        self.attention_scaling = attention_scaling
        self.cast_output = cast_output
        self._mrope_section = list(
            mrope_section
            if mrope_section is not None
            else get_mrope_section(
                rope_scaling=rope_scaling,
                rope_parameters=rope_parameters,
            )
        )
        if _has_mrope_apply_selector(style):
            self._position_selector = mrope_position_selector(
                style,
                self.mrope_section,
                self.inv_freq.shape[0],
            )
        else:
            self._position_selector = None
        self.pairing = _pairing_for_style(style)
        self.fused_apply = self.position_selector is not None and _HAS_METAL
        self._compiled_apply = {} if self.fused_apply else None
        self.eval_cached_arrays()

    @property
    def mrope_section(self):
        return self._mrope_section

    @property
    def inv_freq(self):
        return self._inv_freq

    @property
    def position_selector(self):
        return self._position_selector

    def eager_eval_arrays(self):
        if self._position_selector is None:
            return [self._inv_freq]
        return [self._inv_freq, self._position_selector]

    def eval_cached_arrays(self):
        mx.eval(*self.eager_eval_arrays())

    def __call__(self, x, position_ids):
        freqs = compute_mrope_frequencies(
            position_ids,
            self.inv_freq,
            self.mrope_section,
            style=self.style,
            position_selector=self.position_selector,
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        if self.cast_output:
            return cos.astype(x.dtype), sin.astype(x.dtype)
        return cos, sin

    def apply_rotary(
        self,
        q,
        k,
        position_ids,
        *,
        unsqueeze_dim: int = 1,
        cast_output: bool = True,
    ):
        if (
            self.fused_apply
            and unsqueeze_dim == 1
            and position_ids.ndim in (2, 3)
            and q.ndim == 4
            and k.ndim == 4
        ):
            compiled_apply = self._compiled_apply.get(position_ids.ndim)
            if compiled_apply is None:
                compiled_apply = _compiled_mrope_apply(
                    self.dim, position_ids.ndim, self.pairing
                )
                if compiled_apply is not None:
                    self._compiled_apply[position_ids.ndim] = compiled_apply

            if compiled_apply is not None:
                return compiled_apply(
                    q,
                    k,
                    position_ids,
                    self.inv_freq,
                    self.position_selector,
                )

        cos, sin = self(k, position_ids)
        return apply_multimodal_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            mrope_section=self.mrope_section,
            unsqueeze_dim=unsqueeze_dim,
            style=self.style,
            cast_output=cast_output,
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def rotate_half_even_odd(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.flatten(mx.stack([-x2, x1], axis=-1), start_axis=-2, end_axis=-1)


def _apply_rotary_embedding(
    q,
    k,
    cos,
    sin,
    rotate_fn,
    *,
    cast_output: bool = True,
    compute_dtype=None,
):
    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    if compute_dtype is not None:
        q_rot = q_rot.astype(compute_dtype)
        k_rot = k_rot.astype(compute_dtype)

    q_embed = (q_rot * cos) + (rotate_fn(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_fn(k_rot) * sin)

    if cast_output:
        q_embed = q_embed.astype(q.dtype)
        k_embed = k_embed.astype(k.dtype)

    if q_pass.shape[-1] == 0 and k_pass.shape[-1] == 0:
        return q_embed, k_embed

    return (
        mx.concatenate([q_embed, q_pass], axis=-1),
        mx.concatenate([k_embed, k_pass], axis=-1),
    )


@mx.compile
def _apply_interleaved_rotary_pos_emb_axis1(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = q_embed.astype(q.dtype)
    k_embed = k_embed.astype(k.dtype)

    return (
        mx.concatenate([q_embed, q_pass], axis=-1),
        mx.concatenate([k_embed, k_pass], axis=-1),
    )


def _section_frequency_layout(values, mrope_section):
    if len(mrope_section) != 3:
        raise ValueError("sectioned MRoPE expects exactly 3 sections")
    split_indices = _cumulative_splits(list(mrope_section) * 2)
    return mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(values, split_indices, axis=-1))],
        axis=-1,
    )[:, None, :, :]


def _section_cos_sin(cos, sin, mrope_section):
    cos = _section_frequency_layout(cos, mrope_section)
    sin = _section_frequency_layout(sin, mrope_section)
    return cos, sin


def _maybe_fast_precomputed_rotary(
    q,
    k,
    cos,
    sin,
    *,
    pairing: str,
    cos_layout: str = _HALF_COS,
    mrope_section: Optional[Sequence[int]] = None,
    unsqueeze_dim: int = 1,
    cast_output: bool = True,
):
    if (
        not _HAS_METAL
        or unsqueeze_dim != 1
        or not cast_output
        or q.ndim != 4
        or k.ndim != 4
        or cos.ndim not in (3, 4)
        or sin.ndim != cos.ndim
    ):
        return None

    rotary_dim = cos.shape[-1]
    if rotary_dim > q.shape[-1] or rotary_dim > k.shape[-1]:
        return None

    sectioned = mrope_section is not None
    position_selector = None
    if sectioned:
        if cos.ndim != 4:
            return None
        position_selector = mrope_position_selector(
            "chunked",
            mrope_section,
            rotary_dim // 2,
        )

    compiled_apply = _compiled_rotary_apply(
        rotary_dim,
        pairing,
        cos.ndim,
        sectioned,
        cos_layout,
        tuple(mrope_section) if sectioned else (),
    )
    if compiled_apply is None:
        return None

    return compiled_apply(q, k, cos, sin, position_selector)


def apply_rotary_pos_emb_even_odd(q, k, cos, sin, *, cos_layout: str = _HALF_COS):
    """Apply even/odd RoPE from precomputed cos/sin with a Metal fast path."""
    fast = _maybe_fast_precomputed_rotary(
        q,
        k,
        cos,
        sin,
        pairing=_EVEN_ODD,
        cos_layout=cos_layout,
    )
    if fast is not None:
        return fast

    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    if cos_layout == _HALF_COS:
        cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
        sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)

    return _apply_rotary_embedding(
        q,
        k,
        cos,
        sin,
        rotate_half_even_odd,
        compute_dtype=mx.float32,
    )


def apply_multimodal_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    *,
    mrope_section: Optional[Sequence[int]] = None,
    unsqueeze_dim: int = 1,
    style: str = "interleaved",
    cast_output: bool = True,
):
    if style == "interleaved" and unsqueeze_dim == 1 and cast_output:
        return _apply_interleaved_rotary_pos_emb_axis1(q, k, cos, sin)

    if _is_sectioned_style(style):
        if mrope_section is None:
            raise ValueError("mrope_section is required for sectioned MRoPE")
        fast = _maybe_fast_precomputed_rotary(
            q,
            k,
            cos,
            sin,
            pairing=_pairing_for_style(style),
            mrope_section=mrope_section,
            unsqueeze_dim=unsqueeze_dim,
            cast_output=cast_output,
        )
        if fast is not None:
            return fast
        cos, sin = _section_cos_sin(cos, sin, mrope_section)
    else:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    if _needs_even_odd_layout(style):
        cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
        sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)
        rotate_fn = rotate_half_even_odd
    else:
        rotate_fn = rotate_half

    return _apply_rotary_embedding(
        q,
        k,
        cos,
        sin,
        rotate_fn,
        cast_output=cast_output,
    )
