"""MLX image/mask/point-map preprocessing and layout decoding."""

import mlx.core as mx

from ..interpolate import resize_bilinear_nhwc, resize_nearest_nhwc
from .sparse import nonzero


def _square(x, fill=0):
    h, w = x.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return mx.pad(
        x,
        [(top, side - h - top), (left, side - w - left), (0, 0)],
        constant_values=fill,
    )


def _crop(x, box):
    left, top, right, bottom = box
    h, w = x.shape[:2]
    pads = [
        (max(0, -top), max(0, bottom - h)),
        (max(0, -left), max(0, right - w)),
        (0, 0),
    ]
    x = mx.pad(x, pads)
    y, z = max(top, 0), max(left, 0)
    return x[y : y + bottom - top, z : z + right - left]


def _median(x):
    x = x.reshape(-1)
    valid = ~mx.isnan(x)
    count = int(valid.sum().item())
    if not count:
        return mx.array(float("nan"))
    return mx.sort(mx.where(valid, x, float("inf")))[(count - 1) // 2]


def float_rgb(image):
    """Drop any alpha channel; return float32 RGB in [0, 1] (uint8 is scaled)."""
    image = image[..., :3]
    if image.dtype == mx.uint8:
        return image.astype(mx.float32) / 255
    image = image.astype(mx.float32)
    if not bool(mx.all(mx.isfinite(image) & (image >= 0) & (image <= 1)).item()):
        raise ValueError("Floating-point image values must be finite and in [0, 1]")
    return image


def prepare_inputs(image, mask, pointmap=None, *, size=518):
    """Accept HWC RGB/RGBA and HW masks; floating RGB must be in [0, 1].

    Point maps use the upstream PyTorch3D camera convention (+X left,
    +Y up, +Z forward). Omitted maps use the trained point-map dropout.
    """
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("Expected an HWC RGB or RGBA image")
    if mask is None:
        if image.shape[-1] != 4:
            raise ValueError("Provide an object mask or an RGBA image")
        mask = image[..., 3]
    image = float_rgb(image)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != image.shape[:2]:
        raise ValueError("The mask must have the same height and width as the image")
    threshold = 127 if mask.dtype == mx.uint8 and int(mask.max().item()) > 1 else 0.5
    mask = (mask > threshold).astype(mx.float32)[..., None]
    ys, xs = nonzero(mx.any(mask[..., 0] > 0, axis=1)), nonzero(
        mx.any(mask[..., 0] > 0, axis=0)
    )
    if ys.size < 2 or xs.size < 2:
        raise ValueError("The object mask must span at least two rows and columns")
    x0, x1, y0, y1 = (int(v.item()) for v in (xs[0], xs[-1], ys[0], ys[-1]))
    side = int(max(x1 - x0, y1 - y0, 2) * 1.2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (
        int(cx - side // 2),
        int(cy - side // 2),
        int(cx + side // 2),
        int(cy + side // 2),
    )
    result = {}
    for name, value in (("image", image), ("mask", mask)):
        for output, raw in (
            (name, _crop(value, box)),
            ("rgb_image" if name == "image" else "rgb_image_mask", value),
        ):
            square = _square(raw)[None]
            result[output] = (
                resize_nearest_nhwc(square, (size, size))
                if name == "mask"
                else resize_bilinear_nhwc(square, (size, size), antialias=True)
            )
    shift, scale = mx.zeros(3), mx.ones(3)
    if pointmap is not None:
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            raise ValueError("Expected an HWC point map with XYZ channels")
        pointmap = resize_nearest_nhwc(
            pointmap[None].astype(mx.float32), image.shape[:2]
        )[0]
        selected = pointmap.reshape(-1, 3)[nonzero(mask[..., 0] > 0)]
        if not bool(mx.any(mx.isfinite(selected)).item()):
            raise ValueError("The object mask contains no finite point-map values")
        shift = mx.stack([_median(selected[:, c]) for c in range(3)])
        scale_value = _median(mx.max(mx.abs(pointmap - shift), axis=-1))
        if (
            not bool(mx.all(mx.isfinite(shift)).item())
            or not bool(mx.isfinite(scale_value).item())
            or float(scale_value.item()) <= 0
        ):
            raise ValueError("The point map has invalid or zero spatial scale")
        scale = mx.broadcast_to(scale_value, (3,))
        normalized = (pointmap - shift) / scale
        result["pointmap"] = resize_nearest_nhwc(
            _square(_crop(normalized, box), float("nan"))[None], (size, size)
        )
        result["rgb_pointmap"] = resize_nearest_nhwc(
            _square(normalized)[None], (size, size)
        )
    else:
        result["pointmap"] = result["rgb_pointmap"] = None
    return result, {
        "pointmap_scale": scale,
        "pointmap_shift": shift,
        "pointmap_conditioned": pointmap is not None,
    }


def _normalize(x):
    return x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), 1e-12)


def _cross(a, b):
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def decode_pose(latents, metadata, downsample_factor=1):
    mean = mx.array(
        [
            -0.06366084883674913,
            0.008438224692279752,
            0.00017084786438302483,
            0.0007126610473540038,
            -0.0030916726538816417,
            0.5166093753457688,
        ]
    )
    std = mx.array(
        [
            0.6656971967514863,
            0.6787012271867754,
            0.30345010594844524,
            0.4394504420678794,
            0.39817973931717104,
            0.6176286868761914,
        ]
    )
    rot = latents["6drotation_normalized"].astype(mx.float32).reshape(6) * std + mean
    a = _normalize(rot[:3])
    b = _normalize(rot[3:] - mx.sum(a * rot[3:]) * a)
    matrix = mx.stack([a, b, _cross(a, b)], axis=-1)
    m = matrix
    qabs = mx.sqrt(
        mx.maximum(
            mx.array([1.0, 1.0, 1.0, 1.0])
            + mx.stack(
                [
                    m[0, 0] + m[1, 1] + m[2, 2],
                    m[0, 0] - m[1, 1] - m[2, 2],
                    -m[0, 0] + m[1, 1] - m[2, 2],
                    -m[0, 0] - m[1, 1] + m[2, 2],
                ]
            ),
            0,
        )
    )
    qw, qx, qy, qz = qabs * qabs
    candidates = mx.stack(
        [
            mx.stack([qw, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]),
            mx.stack([m[2, 1] - m[1, 2], qx, m[1, 0] + m[0, 1], m[0, 2] + m[2, 0]]),
            mx.stack([m[0, 2] - m[2, 0], m[1, 0] + m[0, 1], qy, m[2, 1] + m[1, 2]]),
            mx.stack([m[1, 0] - m[0, 1], m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], qz]),
        ]
    )
    quaternion = (candidates / (2 * mx.maximum(qabs[:, None], 0.1)))[mx.argmax(qabs)]
    quaternion = mx.where(quaternion[0] < 0, -quaternion, quaternion)
    scale = mx.exp(latents["scale"].reshape(3)) * metadata["pointmap_scale"]
    scale = mx.broadcast_to(mx.mean(scale) * downsample_factor, (3,))
    translation = (
        latents["translation"].reshape(3) * metadata["pointmap_scale"]
        + metadata["pointmap_shift"]
    )
    return {
        "rotation": quaternion,
        "rotation_matrix": matrix,
        "translation": translation,
        "scale": scale,
    }
