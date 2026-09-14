"""Geometry helpers for MoGe-3 (ports of moge.utils.geometry_torch / utils3d)."""

from typing import Optional, Tuple

import mlx.core as mx


def _view_plane_axes(
    width: int, height: int, aspect_ratio: float = None
) -> Tuple[mx.array, mx.array]:
    """Per-axis view-plane coordinates ``u`` (W,) and ``v`` (H,).

    Pixel centers span ``[-w/diag, w/diag]`` x ``[-h/diag, h/diag]``.
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = mx.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    v = mx.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height, height
    )
    return u.astype(mx.float32), v.astype(mx.float32)


def normalized_view_plane_uv(
    width: int, height: int, aspect_ratio: float = None
) -> mx.array:
    """UV (H, W, 2) with left-top (-w/diag, -h/diag) and right-bottom (w/diag, h/diag)."""
    u, v = _view_plane_axes(width, height, aspect_ratio)
    u, v = mx.meshgrid(u, v, indexing="xy")
    return mx.stack([u, v], axis=-1)


def _lm_evaluate(shift, uv, xy, z, fixed):
    """Cost, gradient, Gauss-Newton Hessian and focal at ``shift`` (all (B,))."""
    inv_depth = 1.0 / (z + shift)
    p = xy * inv_depth[..., None]  # xy / (z + s)
    q = -p * inv_depth[..., None]  # d p / d s
    if fixed is None:
        a = (p * uv).sum(axis=(-2, -1), keepdims=True)
        b = mx.maximum((p * p).sum(axis=(-2, -1), keepdims=True), 1e-12)
        f = a / b
        da = (q * uv).sum(axis=(-2, -1), keepdims=True)
        db = 2.0 * (p * q).sum(axis=(-2, -1), keepdims=True)
        df = (da * b - a * db) / (b * b)
    else:
        f = fixed[..., None]
        df = mx.zeros_like(f)
    r = f * p - uv
    j = df * p + f * q
    cost = (r * r).sum(axis=(-2, -1))
    g = (j * r).sum(axis=(-2, -1))
    h = (j * j).sum(axis=(-2, -1))
    return cost, g, h, f.reshape(-1)


@mx.compile
def _lm_step(state, uv, xy, z, fixed, ftol):
    """One damped Gauss-Newton trial per image; acceptance via ``mx.where``."""
    shift, lam, done, cost, g, h, f = state
    step = -g / mx.maximum(h * (1.0 + lam), 1e-30)
    trial = shift + step[:, None]
    cost_t, g_t, h_t, f_t = _lm_evaluate(trial, uv, xy, z, fixed)
    accept = (cost_t < cost) & ~done
    converged = accept & ((cost - cost_t) <= ftol * cost_t)
    tiny = mx.abs(step) < 1e-8 * mx.maximum(mx.abs(shift[:, 0]), 1.0)
    shift = mx.where(accept[:, None], trial, shift)
    cost, g, h, f = (
        mx.where(accept, new, old)
        for new, old in ((cost_t, cost), (g_t, g), (h_t, h), (f_t, f))
    )
    lam = mx.where(done, lam, mx.where(accept, lam / 3.0, lam * 5.0))
    return shift, lam, done | converged | tiny, cost, g, h, f


def _solve_focal_shift(
    uv: mx.array,
    xyz: mx.array,
    focal: Optional[mx.array] = None,
    ftol: float = 1e-3,
    max_iter: int = 30,
) -> Tuple[mx.array, mx.array]:
    xy, z = xyz[..., :2], xyz[..., 2]
    fixed = None if focal is None else focal.reshape(-1, 1)
    ftol = mx.array(ftol, dtype=mx.float32)

    batch = xyz.shape[0]
    shift = mx.zeros((batch, 1), dtype=mx.float32)
    lam = mx.full((batch,), 1e-3, dtype=mx.float32)
    done = mx.zeros((batch,), dtype=mx.bool_)
    state = (shift, lam, done, *_lm_evaluate(shift, uv, xy, z, fixed))
    for _ in range(max_iter):
        state = _lm_step(state, uv, xy, z, fixed, ftol)
    shift, f = state[0], state[-1]
    return shift.reshape(-1), f


def _nearest_indices(in_size: int, out_size: int) -> mx.array:
    """Torch ``F.interpolate(mode='nearest')`` source indices."""
    return mx.floor(mx.arange(out_size) * (in_size / out_size)).astype(mx.int32)


def recover_focal_shift(
    points: mx.array,
    mask: Optional[mx.array] = None,
    focal: Optional[mx.array] = None,
    downsample_size: Tuple[int, int] = (64, 64),
):
    """Recover focal (rel. to half diagonal) and z-shift from a point map.

    - ``points``: (..., H, W, 3); ``mask``: (..., H, W) bool; ``focal``: (...).
    - Returns ``(focal, shift)``, each (...) float32.
    """
    shape = points.shape
    height, width = shape[-3], shape[-2]
    batch_shape = shape[:-3]

    # The solver only needs a 64x64 grid, so build the UV map at that size
    # directly instead of subsampling a full-resolution grid.
    ii = _nearest_indices(height, downsample_size[0])
    jj = _nearest_indices(width, downsample_size[1])
    points = points.reshape(-1, height, width, 3)[:, ii][:, :, jj]
    batch = points.shape[0]
    n = downsample_size[0] * downsample_size[1]
    xyz = points.reshape(batch, n, 3).astype(mx.float32)
    u, v = _view_plane_axes(width, height)
    u, v = mx.meshgrid(u[jj], v[ii], indexing="xy")
    uv = mx.broadcast_to(mx.stack([u, v], axis=-1).reshape(1, n, 2), (batch, n, 2))

    if mask is None:
        valid_count = mx.full((batch,), n)
    else:
        valid = mask.reshape(-1, height, width)[:, ii][:, :, jj].reshape(batch, n, 1)
        valid_count = valid.sum(axis=(-2, -1))
        # Zero invalid samples (and put them at unit depth) so they contribute
        # nothing to the residual and never divide by zero.
        uv = mx.where(valid, uv, 0.0)
        xyz = mx.where(valid, xyz, mx.array([0.0, 0.0, 1.0]))

    focal_flat = None if focal is None else focal.astype(mx.float32).reshape(-1)
    shift, optim_focal = _solve_focal_shift(uv, xyz, focal_flat)

    degenerate = valid_count < 2
    shift = mx.where(degenerate, 0.0, shift).reshape(batch_shape)
    if focal is None:
        focal = mx.where(degenerate, 1.0, optim_focal)
    return focal.reshape(batch_shape), shift


def intrinsics_from_focal_center(fx, fy, cx, cy) -> mx.array:
    """OpenCV intrinsics matrix (..., 3, 3) from focal lengths and center."""
    zeros, ones = mx.zeros_like(fx), mx.ones_like(fx)
    rows = [
        mx.stack([fx, zeros, cx], axis=-1),
        mx.stack([zeros, fy, cy], axis=-1),
        mx.stack([zeros, zeros, ones], axis=-1),
    ]
    return mx.stack(rows, axis=-2)


def _inverse_3x3(m: mx.array) -> mx.array:
    """Closed-form inverse of (..., 3, 3) via the adjugate; runs on the GPU."""
    c0, c1, c2 = m[..., :, 0], m[..., :, 1], m[..., :, 2]
    r0 = mx.linalg.cross(c1, c2)
    r1 = mx.linalg.cross(c2, c0)
    r2 = mx.linalg.cross(c0, c1)
    det = (c0 * r0).sum(axis=-1, keepdims=True)[..., None]
    return mx.stack([r0, r1, r2], axis=-2) / det


def depth_map_to_point_map(depth: mx.array, intrinsics: mx.array) -> mx.array:
    """Unproject depth (..., H, W) to camera-space points (..., H, W, 3)."""
    height, width = depth.shape[-2], depth.shape[-1]
    # (H, W, 3) pixel-center coordinates (u, v, 1) normalized to [0, 1].
    u = (mx.arange(width, dtype=mx.float32) + 0.5) / width
    v = (mx.arange(height, dtype=mx.float32) + 0.5) / height
    u, v = mx.meshgrid(u, v, indexing="xy")
    hom = mx.stack([u, v, mx.ones_like(u)], axis=-1)
    rays = mx.einsum("...ij,hwj->...hwi", _inverse_3x3(intrinsics), hom)
    return rays * depth[..., None]
