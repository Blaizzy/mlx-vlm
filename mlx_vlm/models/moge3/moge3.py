"""MoGe-3 model (MLX port of moge.model.v3.MoGeModel).

Monocular geometry estimation: DINOv2 encoder + convolutional neck and
heads + sparse 3D UNet refiner. Inputs are channel-last RGB images in
[0, 1]; outputs are point maps, depth maps, masks, normals, and camera
intrinsics.
"""

import math
import re
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..interpolate import resize_bilinear_nhwc
from .config import ModelConfig
from .conv_stack import ConvStack, make_scale_head, run_sequential
from .geometry import (
    depth_map_to_point_map,
    intrinsics_from_focal_center,
    normalized_view_plane_uv,
    recover_focal_shift,
)
from .refiner import Sparse3DUNet
from .vision import MoGe3Encoder

DEFAULT_REFINE_STEPS = 3


def resize(x: mx.array, size) -> mx.array:
    """Channel-last bilinear resize matching ``F.interpolate`` (no antialias)."""
    return resize_bilinear_nhwc(x, size)


# ``<stack>.resamplers.<i>.0.weight`` is always the ConvTranspose2d of a
# "conv_transpose" resampler (bilinear/nearest resamplers have no ``.0`` weight).
_CONV_TRANSPOSE_KEY = re.compile(r"\.resamplers\.\d+\.0\.weight$")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = MoGe3Encoder(config.encoder)
        self.neck = ConvStack(config.neck)
        if config.points_head is not None:
            self.points_head = ConvStack(config.points_head)
        if config.mask_head is not None:
            self.mask_head = ConvStack(config.mask_head)
        if config.normal_head is not None:
            self.normal_head = ConvStack(config.normal_head)
        if config.scale_head is not None:
            self.scale_head = make_scale_head(config.scale_head)
        if config.refiner is not None:
            self.refiner = Sparse3DUNet(config.refiner)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Relayout ConvTranspose2d weights from torch (in, out, 2, 2) to MLX
        (out, 2, 2, in). Idempotent: a weight whose shape already matches the
        module's parameter is left alone (shape sniffing alone cannot tell the
        two layouts apart when a channel count equals the kernel size)."""
        targets = {
            k: v.shape
            for k, v in tree_flatten(self.parameters())
            if _CONV_TRANSPOSE_KEY.search(k)
        }
        sanitized = {}
        for k, v in weights.items():
            target = targets.get(k)
            if target is not None and v.ndim == 4 and v.shape != target:
                v = v.transpose(1, 2, 3, 0)
            sanitized[k] = v
        return sanitized

    def _has(self, name: str) -> bool:
        """Whether an optional submodule exists and was not disabled by the
        loader (which sets modules without checkpoint weights to ``None``)."""
        return getattr(self, name, None) is not None

    # ---- refiner helpers -------------------------------------------------

    def _voxelize(self, point_coord: mx.array):
        """Dense (B, H, W, 3) (x/z, y/z, logz) -> sparse feats/coords/shape/logz.

        Binning is quantized at 1/refiner_depth_resolution and done in fp32.
        Coords come out in raster order (b, i, j), one voxel per pixel, which
        the sparse ops rely on. The depth extent of ``shape`` is a lazy 0-d
        array so voxelization does not force a host sync.
        """
        if point_coord.ndim != 4 or point_coord.shape[-1] != 3:
            raise ValueError(
                f"point_coord must be [B, H, W, 3], got {point_coord.shape}"
            )
        point_coord = point_coord.astype(mx.float32)
        bsz, height, width, _ = point_coord.shape

        logz = point_coord[..., 2]
        zq = mx.round(logz * self.config.refiner_depth_resolution).astype(mx.int64)
        z_offset = mx.min(zq, axis=(1, 2), keepdims=True)
        z_idx = (zq - z_offset).astype(mx.int32)
        z_extent = z_idx.max() + 1

        b, i, j = mx.meshgrid(
            mx.arange(bsz, dtype=mx.int32),
            mx.arange(height, dtype=mx.int32),
            mx.arange(width, dtype=mx.int32),
            indexing="ij",
        )
        coords = mx.stack([b, i, j, z_idx], axis=-1).reshape(-1, 4)
        feats = point_coord.reshape(-1, 3)
        shape = (bsz, height, width, z_extent)
        return feats, coords, shape, logz

    def _refine_logz(
        self, point_coord: mx.array, encoder_feature: mx.array
    ) -> mx.array:
        bsz, height, width, _ = point_coord.shape
        feats, coords, shape, logz = self._voxelize(point_coord)
        out = self.refiner(feats, coords, shape, encoder_feature)
        return logz + out.astype(mx.float32).reshape(bsz, height, width)

    # ---- forward ----------------------------------------------------------

    def _remap_points(self, points: mx.array) -> mx.array:
        xy, z = mx.split(points, [2], axis=-1)
        z = mx.exp(z)
        return mx.concatenate([xy * z, z], axis=-1)

    def __call__(
        self,
        image: mx.array,
        num_tokens: int,
        refine_steps: int = DEFAULT_REFINE_STEPS,
        return_per_step: bool = False,
    ) -> Dict[str, mx.array]:
        """image: (B, H, W, 3) float in [0, 1]."""
        if refine_steps > 0 and not self._has("refiner"):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        batch_size, img_h, img_w, _ = image.shape

        aspect_ratio = img_w / img_h
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        # Backbone encoding
        features, cls_token = self.encoder(image, base_h, base_w)

        # Each pyramid level gets a UV map for the aspect ratio; the first
        # level concatenates it to the backbone features.
        uvs = [
            mx.broadcast_to(uv[None], (batch_size, *uv.shape))
            for uv in (
                normalized_view_plane_uv(
                    base_w * 2**level, base_h * 2**level, aspect_ratio
                )
                for level in range(len(self.neck.res_blocks))
            )
        ]
        features = [mx.concatenate([features, uvs[0]], axis=-1), *uvs[1:]]

        # Shared neck
        neck_features = self.neck(features)

        # Heads decoding
        raw_coord = (
            self.points_head(neck_features)[-1] if self._has("points_head") else None
        )
        normal = (
            self.normal_head(neck_features)[-1] if self._has("normal_head") else None
        )
        mask = self.mask_head(neck_features)[-1] if self._has("mask_head") else None
        metric_scale = (
            run_sequential(self.scale_head, cls_token)
            if self._has("scale_head")
            else None
        )

        # Refine the point map in factorized coordinate space (x/z, y/z, logz)
        points: Optional[mx.array] = None
        points_per_step: Optional[List[mx.array]] = None
        if raw_coord is not None:
            current_coord = raw_coord.astype(mx.float32)
            coord_per_step = [current_coord]
            refiner_feature = features[0]
            for _ in range(refine_steps):
                refined_logz = self._refine_logz(
                    mx.stop_gradient(current_coord),
                    mx.stop_gradient(refiner_feature),
                )
                current_coord = mx.concatenate(
                    [current_coord[..., :2], refined_logz[..., None]], axis=-1
                )
                coord_per_step.append(current_coord)
            if not return_per_step:
                coord_per_step = coord_per_step[-1:]

            num_point_steps = len(coord_per_step)
            coords = resize(mx.concatenate(coord_per_step, axis=0), (img_h, img_w))
            points_all = self._remap_points(
                coords.reshape(num_point_steps, batch_size, img_h, img_w, 3)
            )
            points = points_all[-1]
            if return_per_step:
                points_per_step = list(points_all)

        if normal is not None:
            normal = resize(normal, (img_h, img_w))
            normal = normal / mx.maximum(
                mx.linalg.norm(normal, axis=-1, keepdims=True), 1e-12
            )
        if mask is not None:
            mask = mx.sigmoid(resize(mask, (img_h, img_w)).squeeze(-1))
        if metric_scale is not None:
            metric_scale = mx.exp(metric_scale.squeeze(-1))

        return_dict = {
            "points": points,
            "points_per_step": points_per_step,
            "normal": normal,
            "mask": mask,
            "metric_scale": metric_scale,
        }
        return {k: v for k, v in return_dict.items() if v is not None}

    # ---- user-facing inference ---------------------------------------------

    def infer(
        self,
        image: mx.array,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[float, mx.array]] = None,
        refine_steps: int = DEFAULT_REFINE_STEPS,
        return_per_step: bool = False,
    ) -> Dict[str, mx.array]:
        """Run the full inference pipeline.

        - ``image``: (B, H, W, 3) or (H, W, 3) RGB float in [0, 1].
        - ``num_tokens``: base ViT token count; derived from
          ``resolution_level`` (0-9) when None.
        - ``force_projection``: recompute the point map from depth and
          intrinsics.
        - ``apply_mask``: mask invalid points/depths with the predicted mask.
        - ``fov_x``: horizontal field of view in degrees, or None to infer.
        - ``refine_steps``: number of sparse 3D refinement updates.

        Returns a dict with ``points`` (B, H, W, 3), ``intrinsics`` (B, 3, 3),
        ``depth`` (B, H, W), ``mask`` (B, H, W), ``normal`` (B, H, W, 3) and
        per-step variants when ``return_per_step`` is True.
        """
        omit_batch_dim = image.ndim == 3
        if omit_batch_dim:
            image = image[None]

        batch_size, original_height, original_width, _ = image.shape
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.config.num_tokens_range
            num_tokens = int(
                min_tokens + (resolution_level / 9) * (max_tokens - min_tokens)
            )

        output = self(
            image,
            num_tokens=num_tokens,
            refine_steps=refine_steps,
            return_per_step=return_per_step,
        )
        affine_points_per_step = output.get("points_per_step")
        if affine_points_per_step is None and "points" in output:
            affine_points_per_step = [output["points"]]

        # Post-process in fp32
        normal, mask, metric_scale = (
            output[k].astype(mx.float32) if k in output else None
            for k in ("normal", "mask", "metric_scale")
        )
        mask_binary = mask > 0.5 if mask is not None else None

        points = depth = intrinsics = None
        points_per_step = depth_per_step = intrinsics_per_step = mask_per_step = None
        if affine_points_per_step is not None:
            # Per-step (focal, shift) recovery: refinement modifies logz which
            # changes the 3D shape, so each step gets its own alignment.
            focal_fixed = None
            if fov_x is not None:
                fov = mx.array(fov_x, dtype=mx.float32)
                focal_fixed = mx.broadcast_to(
                    aspect_ratio
                    / (1 + aspect_ratio**2) ** 0.5
                    / mx.tan(fov * (math.pi / 360)),
                    (batch_size,),
                )

            points_per_step, depth_per_step, intrinsics_per_step = [], [], []
            for affine_points in affine_points_per_step:
                affine_points = affine_points.astype(mx.float32)
                focal_i, shift_i = recover_focal_shift(
                    affine_points, mask_binary, focal=focal_fixed
                )
                fx_i = focal_i / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
                fy_i = focal_i / 2 * (1 + aspect_ratio**2) ** 0.5
                half = mx.full(focal_i.shape, 0.5)
                intrinsics_i = intrinsics_from_focal_center(fx_i, fy_i, half, half)

                depth_i = affine_points[..., 2] + shift_i[..., None, None]
                if force_projection:
                    points_i = depth_map_to_point_map(depth_i, intrinsics=intrinsics_i)
                else:
                    points_i = mx.concatenate(
                        [affine_points[..., :2], depth_i[..., None]], axis=-1
                    )

                if metric_scale is not None:
                    points_i = points_i * metric_scale[:, None, None, None]
                    depth_i = depth_i * metric_scale[:, None, None]

                points_per_step.append(points_i)
                depth_per_step.append(depth_i)
                intrinsics_per_step.append(intrinsics_i)

            intrinsics = intrinsics_per_step[-1]

            if mask_binary is not None:
                mask_per_step = [mask_binary & (d > 0) for d in depth_per_step]
                mask_binary = mask_per_step[-1]

            points, depth = points_per_step[-1], depth_per_step[-1]

        if apply_mask:
            if mask_per_step is not None:
                inf = mx.array(float("inf"), mx.float32)
                points_per_step = [
                    mx.where(m[..., None], p, inf)
                    for p, m in zip(points_per_step, mask_per_step)
                ]
                depth_per_step = [
                    mx.where(m, d, inf) for d, m in zip(depth_per_step, mask_per_step)
                ]
                points, depth = points_per_step[-1], depth_per_step[-1]
            if mask_binary is not None and normal is not None:
                normal = mx.where(mask_binary[..., None], normal, mx.zeros_like(normal))

        if not return_per_step:
            points_per_step = depth_per_step = intrinsics_per_step = None

        return_dict = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "normal": normal,
            "points_per_step": points_per_step,
            "intrinsics_per_step": intrinsics_per_step,
            "depth_per_step": depth_per_step,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {
                k: [item[0] for item in v] if isinstance(v, list) else v[0]
                for k, v in return_dict.items()
            }
        return return_dict
