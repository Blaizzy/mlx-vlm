"""Batch preparation: crop to bounding box, resize, normalize."""

import numpy as np


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute 2x3 affine transform that maps bbox region to output_size.

    Args:
        center: (2,) bbox center (x, y)
        scale: (2,) bbox width, height
        output_size: (W, H) target size

    Returns:
        (2, 3) affine transform matrix
    """
    src_w, src_h = scale
    dst_w, dst_h = output_size

    # Make square crop (use max dimension)
    crop_size = max(src_w, src_h)
    # Add 20% padding
    crop_size = crop_size * 1.2

    # Source triangle: center, center+right, center+down
    src = np.array(
        [
            center,
            center + np.array([crop_size / 2.0, 0.0]),
            center + np.array([0.0, crop_size / 2.0]),
        ],
        dtype=np.float32,
    )

    # Destination triangle
    dst = np.array(
        [
            [dst_w / 2.0, dst_h / 2.0],
            [dst_w, dst_h / 2.0],
            [dst_w / 2.0, dst_h],
        ],
        dtype=np.float32,
    )

    # Solve for affine transform: dst = src @ M[:2,:2].T + M[:, 2]
    # Use cv2 if available, otherwise solve manually
    try:
        import cv2

        return cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
    except ImportError:
        return _solve_affine(src, dst)


def _solve_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve for 2x3 affine matrix from 3 point correspondences."""
    # src: (3, 2), dst: (3, 2)
    # [x' y'] = [x y 1] @ [[a c], [b d], [tx ty]]
    A = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros(6, dtype=np.float64)
    for i in range(3):
        A[2 * i, 0] = src[i, 0]
        A[2 * i, 1] = src[i, 1]
        A[2 * i, 2] = 1
        b[2 * i] = dst[i, 0]
        A[2 * i + 1, 3] = src[i, 0]
        A[2 * i + 1, 4] = src[i, 1]
        A[2 * i + 1, 5] = 1
        b[2 * i + 1] = dst[i, 1]
    params = np.linalg.solve(A, b)
    M = np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
        ],
        dtype=np.float64,
    )
    return M


def apply_affine_transform(
    image: np.ndarray,
    M: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Apply affine transform to image.

    Args:
        image: (H, W, 3) uint8
        M: (2, 3) affine matrix
        output_size: (W, H)

    Returns:
        (H_out, W_out, 3) uint8
    """
    try:
        import cv2

        return cv2.warpAffine(
            image,
            M,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    except ImportError:
        return _warp_affine_numpy(image, M, output_size)


def _warp_affine_numpy(
    image: np.ndarray,
    M: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Pure numpy fallback for affine warp (bilinear interpolation)."""
    W_out, H_out = output_size
    out = np.zeros((H_out, W_out, image.shape[2]), dtype=image.dtype)
    H_in, W_in = image.shape[:2]

    # Invert affine to map output -> input
    M_full = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_full)[:2]

    # Build output coordinate grid
    gy, gx = np.mgrid[0:H_out, 0:W_out].astype(np.float64)
    coords = np.stack([gx.ravel(), gy.ravel(), np.ones(H_out * W_out)], axis=0)
    src_coords = M_inv @ coords  # (2, N)
    sx = src_coords[0].reshape(H_out, W_out)
    sy = src_coords[1].reshape(H_out, W_out)

    # Bilinear interpolation
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wa = (x1 - sx) * (y1 - sy)
    wb = (sx - x0) * (y1 - sy)
    wc = (x1 - sx) * (sy - y0)
    wd = (sx - x0) * (sy - y0)

    mask = (x0 >= 0) & (x1 < W_in) & (y0 >= 0) & (y1 < H_in)
    x0c = np.clip(x0, 0, W_in - 1)
    y0c = np.clip(y0, 0, H_in - 1)
    x1c = np.clip(x1, 0, W_in - 1)
    y1c = np.clip(y1, 0, H_in - 1)

    for c in range(image.shape[2]):
        val = (
            wa * image[y0c, x0c, c]
            + wb * image[y0c, x1c, c]
            + wc * image[y1c, x0c, c]
            + wd * image[y1c, x1c, c]
        )
        out[:, :, c] = np.where(mask, val, 0).astype(image.dtype)

    return out


def prepare_image(
    image: np.ndarray,
    bbox: list | np.ndarray,
    image_size: tuple[int, int] = (512, 384),
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Crop, resize, and normalize an image for SAM 3D Body inference.

    Args:
        image: (H, W, 3) RGB uint8
        bbox: [x1, y1, x2, y2]
        image_size: target (H, W)
        mean: ImageNet channel means
        std: ImageNet channel stds

    Returns:
        (1, H, W, 3) float32 normalized array
    """
    bbox = np.array(bbox, dtype=np.float32)
    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
    scale = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])

    target_h, target_w = image_size
    M = get_affine_transform(center, scale, (target_w, target_h))
    cropped = apply_affine_transform(image, M, (target_w, target_h))

    # Normalize: uint8 -> float32 [0,1] -> ImageNet
    img = cropped.astype(np.float32) / 255.0
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

    return img[None]  # (1, H, W, 3)


def get_cliff_condition(
    bbox: list | np.ndarray,
    image_shape: tuple[int, int],
    focal_length: float | None = None,
) -> np.ndarray:
    """Compute CLIFF conditioning vector from bbox and image shape.

    Matches PyTorch: condition = [(cx - W/2) / f, (cy - H/2) / f, bbox_scale / f]

    Args:
        bbox: [x1, y1, x2, y2]
        image_shape: (H, W) of original image
        focal_length: scalar focal length. If None, estimated from image height
            assuming 60° vertical FOV.

    Returns:
        (3,) float32: [cx_norm, cy_norm, scale_norm]
    """
    import math

    bbox = np.array(bbox, dtype=np.float32)
    H, W = image_shape

    if focal_length is None:
        # PyTorch default: image diagonal
        focal_length = math.sqrt(H**2 + W**2)

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bw = bbox[2] - bbox[0]
    # PyTorch uses bbox_width * padding (1.25), not max(w,h) * 1.2
    bbox_scale = bw * 1.25

    cx_norm = (cx - W / 2.0) / focal_length
    cy_norm = (cy - H / 2.0) / focal_length
    scale_norm = bbox_scale / focal_length

    return np.array([cx_norm, cy_norm, scale_norm], dtype=np.float32)
