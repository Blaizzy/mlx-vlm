"""Fixed-point YUV to RGB conversion matching the FFmpeg 8.1 ARM scaler.

Integer channel rounding follows libswscale's RGB24 NEON and table paths.
This is a numerical compatibility kernel, not a general HDR color transform.
"""

from functools import lru_cache

import mlx.core as mx

# Standard fixed-point YCbCr matrix coefficients, scale 65536, limited range.
COEFFICIENTS = {
    1: (117489, 138438, 13975, 34925),
    2: (104597, 132201, 25675, 53279),
    4: (104448, 132798, 24759, 53109),
    5: (104597, 132201, 25675, 53279),
    6: (104597, 132201, 25675, 53279),
    7: (117579, 136230, 16907, 35559),
    9: (110013, 140363, 12277, 42626),
    10: (110013, 140363, 12277, 42626),
}


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="embedding_gemma2_yuv_to_rgb",
        input_names=["Y", "U", "V", "coeff"],
        output_names=["rgb"],
        source=r"""
            uint x = thread_position_in_grid.x;
            uint y = thread_position_in_grid.y;
            if (x >= W || y >= H) return;
            uint cw = (W + (1 << SX) - 1) >> SX;
            uint ci = (y >> SY) * cw + (x >> SX);
            int yy = int(Y[y * W + x]);
            int u = int(U[ci]);
            int v = int(V[ci]);
            if (DEPTH > 8) {
                const int shift = DEPTH - 8;
                yy = (yy + (1 << (shift - 1))) >> shift;
                if (SY) {
                    // Half-pixel-centered vertical upsampling of 4:2:0 chroma.
                    // Four taps use the reference scaler's B=0, C=0.6 cubic.
                    int base = int(y / 2) - ((y & 1) ? 0 : 1);
                    int ch = (H + 1) >> 1;
                    int accum_u = 0, accum_v = 0;
                    const int even_weights[4] = {-115, 986, 3571, -346};
                    const int odd_weights[4] = {-346, 3571, 986, -115};
                    for (int tap = 0; tap < 4; ++tap) {
                        int row = clamp(base - 1 + tap, 0, ch - 1);
                        int weight = (y & 1) ? odd_weights[tap] : even_weights[tap];
                        int index = row * cw + (x >> SX);
                        accum_u += int(U[index]) * weight;
                        accum_v += int(V[index]) * weight;
                    }
                    u = (accum_u + (1 << (11 + shift))) >> (12 + shift);
                    v = (accum_v + (1 << (11 + shift))) >> (12 + shift);
                } else {
                    u = (u + (1 << (shift - 1))) >> shift;
                    v = (v + (1 << (shift - 1))) >> shift;
                }
                u = clamp(u,0,255);
                v = clamp(v,0,255);
            }
            int r, g, b;
            if (NEON) {
                // Each channel product is truncated separately before the final
                // rounded divide by two, matching the reference's integer path.
                int luma = ((yy * 8 - coeff[5]) * coeff[0]) >> 15;
                u = (u - 128) * 8;
                v = (v - 128) * 8;
                r = (luma + ((v * coeff[1]) >> 15) + 1) >> 1;
                g = (luma + ((u * coeff[2]) >> 15) + ((v * coeff[3]) >> 15) + 1) >> 1;
                b = (luma + ((u * coeff[4]) >> 15) + 1) >> 1;
            } else {
                int ur = ((v * coeff[1]) >> 16) - (coeff[1] >> 9);
                int ug = ((u * coeff[2]) >> 16) - (coeff[2] >> 9);
                int vg = ((v * coeff[3]) >> 16) - (coeff[3] >> 9);
                int ub = ((u * coeff[4]) >> 16) - (coeff[4] >> 9);
                r = (coeff[6] + (yy + ur) * coeff[0] + 32768) >> 16;
                g = (coeff[6] + (yy + ug + vg) * coeff[0] + 32768) >> 16;
                b = (coeff[6] + (yy + ub) * coeff[0] + 32768) >> 16;
            }
            rgb[(y * W + x) * 3 + 0] = uchar(clamp(r, 0, 255));
            rgb[(y * W + x) * 3 + 1] = uchar(clamp(g, 0, 255));
            rgb[(y * W + x) * 3 + 2] = uchar(clamp(b, 0, 255));
        """,
    )


def yuv_to_rgb(planes, meta):
    h, w = planes[0].shape
    crv, cbu, cgu, cgv = COEFFICIENTS.get(meta["colorspace"], COEFFICIENTS[2])
    full = meta["format"].startswith("yuvj")
    cy = 65536 if full else 65536 * 255 // 219
    signed = [crv, -cgu, -cgv, cbu]

    def truncdiv(a, b):
        return (abs(a) // b) * (1 if a >= 0 else -1)

    if full:
        signed = [truncdiv(c * 224, 255) for c in signed]
    neon = meta["depth"] == 8 and w % 16 == 0 and h % 2 == 0
    if neon:
        coeff = [
            (cy * 8192 + 32768) >> 16,
            *[(c * 8192 + 32768) >> 16 for c in signed],
            0 if full else 128,
            0,
        ]
    else:
        increments = [truncdiv(c * 65536 + 32768, cy) for c in signed]
        base = (384 if full else 326) * cy - 384 * 65536 - (0 if full else 16 * 65536)
        coeff = [cy, *increments, 0, base]
    return _kernel()(
        inputs=[*[mx.array(x) for x in planes], mx.array(coeff, dtype=mx.int32)],
        template=[
            ("W", w),
            ("H", h),
            ("SX", meta["log2_chroma_w"]),
            ("SY", meta["log2_chroma_h"]),
            ("DEPTH", meta["depth"]),
            ("NEON", neon),
        ],
        grid=(w, h, 1),
        threadgroup=(min(w, 32), min(h, 8), 1),
        output_shapes=[(h, w, 3)],
        output_dtypes=[mx.uint8],
    )[0]
