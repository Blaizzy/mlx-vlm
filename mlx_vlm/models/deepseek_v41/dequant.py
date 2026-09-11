"""Dequantize the natively-quantized V4.1 release for conversion.

Release formats (``quantization_config``: fp8, weight_block_size [32,32],
scale_fmt ue8m0, expert_dtype fp4; every scale tensor is named ``<w>.scale``):

* **FP8 matmul weights** -- e4m3 ``[out, in]`` with one **ue8m0** scale per
  ``32x32`` block: ``scale[ceil(out/32), ceil(in/32)]``.
* **Routed experts** -- FP4 e2m1 packed two per byte ``[out, in//2]`` (low nibble
  = even input index) with one ue8m0 scale per 32 values along in:
  ``scale[out, in//32]``.
* **Engram tables** -- FP8 e4m3 ``[rows, 256]`` with one ue8m0 scale per 32
  values along the row: ``scale[rows, 8]``.

ue8m0 is exponent-only: byte ``b`` means ``2**(b-127)``. Every scale is an exact
power of two, so applying it is lossless in fp32.

Decoding follows the layout proven by the PipeNetwork port (Apache-2.0).
"""

import mlx.core as mx

FP8_BLOCK = 32
FP4_BLOCK = 32

_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def e8m0_to_float(scale_u8: mx.array) -> mx.array:
    """Decode exponent-only scales: byte b -> 2**(b-127)."""
    e = scale_u8.astype(mx.int32)
    return mx.power(mx.array(2.0, mx.float32), (e - 127).astype(mx.float32))


def _expand_blocks(scale: mx.array, rows: int, cols: int, br: int, bc: int) -> mx.array:
    """Broadcast a per-block scale grid to full [rows, cols], cropping ceil overhang."""
    s = mx.repeat(mx.repeat(scale, br, axis=0), bc, axis=1)
    return s[:rows, :cols]


def dequant_fp8(weight_u8: mx.array, scale_u8: mx.array, dtype=mx.bfloat16) -> mx.array:
    """FP8 e4m3 [out, in] + ue8m0 [ceil(out/32), ceil(in/32)] -> dtype."""
    w = mx.from_fp8(weight_u8, mx.float32)
    rows, cols = w.shape
    s = _expand_blocks(e8m0_to_float(scale_u8), rows, cols, FP8_BLOCK, FP8_BLOCK)
    return (w * s).astype(dtype)


def unpack_fp4(packed: mx.array) -> mx.array:
    """[out, in//2] of two e2m1 nibbles -> float32 [out, in]. Low nibble = even index."""
    b = packed.view(mx.uint8).astype(mx.int32)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    lut = mx.array(_E2M1, mx.float32)
    vals = mx.stack([lut[lo], lut[hi]], axis=-1)
    return vals.reshape(vals.shape[0], -1)


def dequant_fp4(packed: mx.array, scale_u8: mx.array, dtype=mx.bfloat16) -> mx.array:
    """Packed FP4 e2m1 [out, in//2] + ue8m0 [out, in//32] -> dtype."""
    w = unpack_fp4(packed)
    rows, cols = w.shape
    s = _expand_blocks(e8m0_to_float(scale_u8), rows, cols, 1, FP4_BLOCK)
    return (w * s).astype(dtype)


def dequant_fp8_rows(
    weight_u8: mx.array, scale_u8: mx.array, block: int = 32
) -> mx.array:
    """Engram-table layout: e4m3 [..., d] + ue8m0 [..., d//block] -> float32."""
    w = mx.from_fp8(weight_u8, mx.float32)
    s = e8m0_to_float(scale_u8)
    wv = w.reshape(*w.shape[:-1], w.shape[-1] // block, block)
    return (wv * s[..., None]).reshape(w.shape)


def is_fp4_expert(name: str) -> bool:
    """Routed-expert matmuls are the FP4 ones; shared experts are FP8."""
    return ".experts." in name and ".shared_experts." not in name
