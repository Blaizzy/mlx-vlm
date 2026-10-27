"""Activation fake-quantization — the inference-time half of V4.1's QAT.

The reference *simulates* its training-time quantization on several activations:
they are rounded through FP8 or FP4 and written back before use. Skipping them
changes the numbers the model was trained to see. V4.1 applies:

* **native FP8/FP4 linear inputs** -- FP8 e4m3, blocks of 32, ue8m0 scales;
  retained when the source weights are converted to MLX weight-only formats;

* **window KV** -- FP8 e4m3, blocks of 32, ue8m0 (power-of-two) scales, over the
  *whole* head vector, rope tail included;
* **compressed KV latents** -- FP4 e2m1, blocks of **16**, **e4m3** scales;
* **indexer queries and keys** -- FP4 e2m1, blocks of 32, ue8m0 scales.

Scale semantics (from the reference ``kernel.py``):

* ue8m0: ``2**ceil(log2(amax * (1/max)))`` with amax clamped to a format-specific
  floor first (1e-4 for FP8, 6*2^-126 for FP4). The ceil is computed here by IEEE
  bit manipulation, exactly as ``fast_round_scale`` does, so boundary cases match.
* e4m3-scale: ``amax`` clamped to ``6*2^-9``, then ``amax/6`` rounded to the
  nearest e4m3 value.

FP4 rounding is round-to-nearest-even on the e2m1 grid
{0, .5, 1, 1.5, 2, 3, 4, 6}: ties at .25/1.25/2.5/5 round down, ties at
.75/1.75/3.5 round up.

Formulas follow the layout proven by the PipeNetwork port (Apache-2.0).
"""

import mlx.core as mx

FP8_MAX = 448.0
FP4_MAX = 6.0

DISABLE = False
FP8_MAX_INV = mx.array(1.0 / 448.0, dtype=mx.float32)
FP4_MAX_INV = mx.array(1.0 / 6.0, dtype=mx.float32)

_E2M1_LUT = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)


def _make_fp8_roundtrip_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="deepseek_v41_fp8_roundtrip",
        input_names=["x"],
        output_names=["out"],
        source="""
            uint i = thread_position_in_grid.x;
            float value = float(x[i]);
            // One SIMD group owns one 32-element quantization block.
            float amax = metal::max(metal::simd_max(metal::abs(value)), 1e-4f);
            uint bits = as_type<uint>(amax * (1.0f / 448.0f));
            uint exponent = (bits >> 23) + ((bits & 0x7fffff) != 0);
            float scale = as_type<float>(exponent << 23);
            float v = metal::clamp(value / scale, -448.0f, 448.0f);
            // E4M3 spacing is 2**(exponent-3), floored at 2**-9 for
            // subnormals. rint preserves round-to-nearest-even ties.
            uint vbits = as_type<uint>(metal::abs(v));
            uint qexp = metal::max(int(vbits >> 23) - 3, 118);
            float step = as_type<float>(qexp << 23);
            out[i] = T((metal::rint(v / step) * step) * scale);
        """,
        ensure_row_contiguous=True,
    )


_fp8_roundtrip_kernel = _make_fp8_roundtrip_kernel()


def _log2_ceil_bits(x: mx.array) -> mx.array:
    """ceil(log2(x)) for positive normal fp32 x, via IEEE 754 bits."""
    bits = x.astype(mx.float32).view(mx.uint32)
    exp = ((bits >> 23) & 0xFF).astype(mx.int32) - 127
    man = (bits & 0x7FFFFF).astype(mx.int32)
    return exp + (man != 0).astype(mx.int32)


def _pow2(e: mx.array) -> mx.array:
    """2**e for integer e, via IEEE 754 bits."""
    return ((e + 127) << 23).astype(mx.uint32).view(mx.float32)


@mx.compile
def _ue8m0_scale(amax: mx.array, max_inv: mx.array) -> mx.array:
    """2**ceil(log2(amax * (1/max)))."""
    return _pow2(_log2_ceil_bits(amax * max_inv))


@mx.compile
def _e2m1_round(v: mx.array) -> mx.array:
    """Round |v| <= 6 to the nearest e2m1 value, ties to even, keeping the sign."""
    mag = mx.abs(v)
    idx = mx.zeros(mag.shape, dtype=mx.int32)
    for t in [0.25, 1.25, 2.5, 5.0]:
        idx = idx + (mag > t).astype(mx.int32)
    for t in [0.75, 1.75, 3.5]:
        idx = idx + (mag >= t).astype(mx.int32)
    return mx.sign(v) * _E2M1_LUT[idx]


def _blockify(x: mx.array, block: int):
    n = x.shape[-1]
    if n % block:
        raise ValueError(f"last dim {n} not divisible by block {block}")
    return x.reshape(*x.shape[:-1], n // block, block)


def fake_quant_fp8_ue8m0(x: mx.array, block: int = 32) -> mx.array:
    """FP8 round-trip with pow2 scales."""
    if DISABLE:
        return x
    if (
        _fp8_roundtrip_kernel is not None
        and mx.default_device() == mx.gpu
        and block == 32
        and x.shape[-1] % block == 0
        and x.size
    ):
        return _fp8_roundtrip_kernel(
            inputs=[x],
            template=[("T", x.dtype)],
            grid=(x.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
    dtype = x.dtype
    shape = x.shape
    xb = _blockify(x.astype(mx.float32), block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 1e-4)
    s = _ue8m0_scale(amax, FP8_MAX_INV)
    q = mx.from_fp8(mx.to_fp8(mx.clip(xb / s, -FP8_MAX, FP8_MAX)), mx.float32) * s
    return q.reshape(shape).astype(dtype)


def fake_quant_fp4_ue8m0(x: mx.array, block: int = 32) -> mx.array:
    """FP4 round-trip with ue8m0 scales: the indexer q/k path."""
    if DISABLE:
        return x
    dtype = x.dtype
    shape = x.shape
    xb = _blockify(x.astype(mx.float32), block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 6.0 * 2.0**-126)
    s = _ue8m0_scale(amax, FP4_MAX_INV)
    q = _e2m1_round(mx.clip(xb / s, -FP4_MAX, FP4_MAX)) * s
    return q.reshape(shape).astype(dtype)


def fake_quant_fp4_e4m3(x: mx.array, block: int = 16) -> mx.array:
    """FP4 round-trip with e4m3 scales: the compressed-KV path."""
    if DISABLE:
        return x
    dtype = x.dtype
    shape = x.shape
    xb = _blockify(x.astype(mx.float32), block)
    amax = mx.maximum(mx.max(mx.abs(xb), axis=-1, keepdims=True), 6.0 * 2.0**-9)
    s = mx.from_fp8(mx.to_fp8(amax / FP4_MAX), mx.float32)
    q = _e2m1_round(mx.clip(xb / s, -FP4_MAX, FP4_MAX)) * s
    return q.reshape(shape).astype(dtype)
