"""MapAnything: feed-forward metric 3D reconstruction from images plus any
optional calibration, depth and pose inputs (MLX port of
``mapanything.models.MapAnything``, inference only).

Arrays are channel-last and every step, input conversion and post-processing
included, is a lazy MLX op: ``infer`` returns unevaluated arrays.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import heads as adaptors
from .confidence import multiview_depth_confidence
from .config import ModelConfig
from .encoders import (
    DenseRepresentationEncoder,
    GlobalRepresentationEncoder,
    ImageEncoder,
)
from .geometry import (
    _apply,
    depth_edge,
    depth_z_to_depth_along_ray,
    intrinsics_from_rays,
    log_scale,
    normalize_depth,
    normalize_translations,
    normals_edge,
    points_to_normals,
    pose_matrix,
    quaternion_to_rotation_matrix,
    rays_from_intrinsics,
    relative_pose,
    rotation_matrix_to_quaternion,
)
from .heads import DPTHead, PoseHead, ScaleHead
from .info_sharing import AlternatingAttentionTransformer
from .layers import layer_norm

IMAGE_NORMALIZATIONS = {
    "dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "identity": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}

ALLOWED_VIEW_KEYS = {
    "img",
    "data_norm_type",
    "depth_z",
    "ray_directions",
    "intrinsics",
    "camera_poses",
    "is_metric_scale",
    "true_shape",
    "idx",
    "instance",
}

# Modules loaded in ``config.head_dtype``.
_FLOAT_MODULES = (
    "ray_dirs_encoder.",
    "depth_encoder.",
    "depth_scale_encoder.",
    "cam_rot_encoder.",
    "cam_trans_encoder.",
    "cam_trans_scale_encoder.",
    "fusion_norm_layer.",
    "dense_head.",
    "pose_head.",
    "scale_head.",
)
# Parameters of the bf16 trunk that stay float32: norms, LayerScale and embeddings.
_FLOAT_PARAMS = (
    "norm",
    ".ls1.",
    ".ls2.",
    "pos_embed",
    "cls_token",
    "view_pos_table",
    "scale_token",
)


def _as_array(x, dtype=mx.float32) -> mx.array:
    return (x if isinstance(x, mx.array) else mx.array(np.asarray(x))).astype(dtype)


def _norm_type(view: Dict[str, Any]) -> str:
    norm = view["data_norm_type"]
    return norm[0] if isinstance(norm, (list, tuple)) else norm


def _per_view(values: mx.array, present: List[int], num_views: int) -> mx.array:
    """Rows of the ``present`` views -> rows of all views, zeros elsewhere."""
    if len(present) == num_views:
        return values
    chunks = dict(zip(present, mx.split(values, len(present))))
    zeros = mx.zeros_like(chunks[present[0]])
    return mx.concatenate([chunks.get(v, zeros) for v in range(num_views)])


def _quantile(x: mx.array, q: float) -> mx.array:
    """Linear-interpolation quantile over the last axis (``torch.quantile``)."""
    x = mx.sort(x, axis=-1)
    pos = q * (x.shape[-1] - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[..., lo] + (x[..., hi] - x[..., lo]) * (pos - lo)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        enc, info = config.encoder_config, config.info_sharing_config
        geo, pred = config.geometric_input_config, config.pred_head_config
        dim, patch = enc.embed_dim, enc.patch_size

        self.encoder = ImageEncoder(enc)
        dense_dims = geo.dense_intermediate_dims
        self.ray_dirs_encoder = DenseRepresentationEncoder(3, dim, patch, dense_dims)
        self.depth_encoder = DenseRepresentationEncoder(1, dim, patch, dense_dims)
        global_dims = geo.global_intermediate_dims
        self.depth_scale_encoder = GlobalRepresentationEncoder(1, dim, global_dims)
        self.cam_rot_encoder = GlobalRepresentationEncoder(4, dim, global_dims)
        self.cam_trans_encoder = GlobalRepresentationEncoder(3, dim, global_dims)
        self.cam_trans_scale_encoder = GlobalRepresentationEncoder(1, dim, global_dims)
        self.fusion_norm_layer = nn.LayerNorm(dim, eps=1e-6)
        self.scale_token = mx.zeros((dim,))

        self.info_sharing = AlternatingAttentionTransformer(
            info, dim, config.info_sharing_mlp_layer_str
        )
        self.dense_head = DPTHead(config.dpt_input_dims, pred)
        self.pose_head = PoseHead(info.dim, config.pose_head_dim, pred)
        self.scale_head = ScaleHead(info.dim, pred)

    # ---- forward ------------------------------------------------------------

    def _encode_geometry(
        self,
        views: List[Dict[str, mx.array]],
        features: mx.array,
        use_calibration: bool,
        use_depth: bool,
        use_pose: bool,
        use_depth_scale: bool,
        use_pose_scale: bool,
    ) -> mx.array:
        """Add the encoded optional inputs to the (V * B, h, w, D) image
        features, then apply the fusion norm, in float32."""
        num = len(views)
        batch = features.shape[0] // num
        features = features.astype(mx.float32)

        def encode(encoder, x):
            return encoder(x.astype(encoder.norm_layer.weight.dtype)).astype(mx.float32)

        def rows(key, present):
            return mx.concatenate([views[v][key] for v in present])

        def metric(present):
            not_metric = mx.zeros((batch,), dtype=mx.bool_)
            return mx.concatenate(
                [views[v].get("is_metric_scale", not_metric) for v in present]
            )

        rays = [v for v in range(num) if "ray_directions_cam" in views[v]]
        if use_calibration and rays:
            encoded = encode(self.ray_dirs_encoder, rows("ray_directions_cam", rays))
            features = features + _per_view(encoded, rays, num)

        depths = [v for v in range(num) if "depth_along_ray" in views[v]]
        if use_depth and depths:
            depth, factor = normalize_depth(
                rows("depth_along_ray", depths).astype(mx.float32)
            )
            encoded = encode(self.depth_encoder, log_scale(depth))
            features = features + _per_view(encoded, depths, num)
            scale = encode(self.depth_scale_encoder, mx.log(factor + 1e-8)[:, None])
            scale = scale * (metric(depths) & use_depth_scale)[:, None]
            features = features + _per_view(scale, depths, num)[:, None, None]

        posed = [v for v in range(num) if "camera_pose_quats" in views[v]]
        if use_pose and posed:
            if posed[0] != 0:
                raise ValueError("The first view needs a camera pose when others do")
            ref_quats = views[0]["camera_pose_quats"].astype(mx.float32)
            ref_trans = views[0]["camera_pose_trans"].astype(mx.float32)
            identity = mx.broadcast_to(mx.array([0.0, 0.0, 0.0, 1.0]), (batch, 4))
            quats, trans = [], []
            for v in range(num):
                if v in posed:
                    q, t = relative_pose(
                        ref_quats,
                        ref_trans,
                        views[v]["camera_pose_quats"].astype(mx.float32),
                        views[v]["camera_pose_trans"].astype(mx.float32),
                    )
                else:
                    q, t = identity, mx.zeros((batch, 3))
                quats.append(q)
                trans.append(t)
            quats = mx.concatenate(quats)
            trans, factor = normalize_translations(mx.stack(trans, axis=1))
            trans = trans.transpose(1, 0, 2).reshape(num * batch, 3)
            cam = mx.array([[v in posed] for v in range(num) for _ in range(batch)])
            log_factor = mx.tile(mx.log(factor + 1e-8)[:, None], (num, 1))
            rotation = encode(self.cam_rot_encoder, quats) * cam
            translation = encode(self.cam_trans_encoder, trans) * cam
            scale = encode(self.cam_trans_scale_encoder, log_factor) * cam
            scale = scale * (metric(range(num)) & use_pose_scale)[:, None]
            features = features + rotation[:, None, None]
            features = features + translation[:, None, None]
            features = features + scale[:, None, None]

        return layer_norm(self.fusion_norm_layer, features)

    def _dense_head(self, inputs: List[mx.array], size, chunk: int) -> mx.array:
        step = chunk or inputs[0].shape[0]
        outputs = [
            self.dense_head([x[i : i + step] for x in inputs], size)
            for i in range(0, inputs[0].shape[0], step)
        ]
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs)

    def __call__(
        self,
        views: Sequence[Dict[str, Any]],
        use_calibration: bool = True,
        use_depth: bool = True,
        use_pose: bool = True,
        use_depth_scale: bool = True,
        use_pose_scale: bool = True,
        dense_head_chunk_size: Optional[int] = None,
    ) -> List[Dict[str, mx.array]]:
        """Raw predictions for preprocessed views.

        Each view holds ``img`` (B, H, W, 3), normalized, and optionally
        ``ray_directions_cam`` (B, H, W, 3) unit rays, ``depth_along_ray``
        (B, H, W, 1), ``camera_pose_quats`` (B, 4) / ``camera_pose_trans``
        (B, 3) cam2world, and ``is_metric_scale`` (B,) bool. ``infer``
        builds these from user inputs. ``dense_head_chunk_size`` overrides
        the config value (images per dense-head pass, 0 for all).
        """
        num = len(views)
        batch, height, width, _ = views[0]["img"].shape
        patch = self.config.encoder_config.patch_size
        h, w = height // patch, width // patch
        images = mx.concatenate([view["img"] for view in views])
        patches, prefix = self.encoder(images)
        features = self._encode_geometry(
            views,
            patches.reshape(num * batch, h, w, -1),
            use_calibration,
            use_depth,
            use_pose,
            use_depth_scale,
            use_pose_scale,
        )

        dim = features.shape[-1]
        tokens = features.reshape(num, batch, h * w, dim)
        if self.config.use_register_tokens_from_encoder:
            prefix = prefix.reshape(num, batch, -1, dim).astype(mx.float32)
            tokens = mx.concatenate([tokens, prefix], axis=2)
        scale_token = mx.broadcast_to(self.scale_token, (batch, 1, dim))
        layers, scale_token = self.info_sharing(
            tokens.transpose(1, 0, 2, 3), scale_token
        )

        def grid(x):
            x = x[:, :, : h * w].transpose(1, 0, 2, 3)
            return x.reshape(num * batch, h, w, -1)

        head_dtype = self.dense_head.conv1.weight.dtype
        dense_inputs = [features] + [grid(x) for x in layers]
        if dense_head_chunk_size is None:
            dense_head_chunk_size = self.config.dense_head_chunk_size
        dense = self._dense_head(
            [x.astype(head_dtype) for x in dense_inputs],
            (height, width),
            dense_head_chunk_size,
        ).astype(mx.float32)
        pose = self.pose_head(dense_inputs[-1].astype(head_dtype)).astype(mx.float32)
        scale = self.scale_head(scale_token[:, 0].astype(head_dtype)).astype(mx.float32)
        return self._outputs(dense, pose, scale, num)

    def _outputs(self, dense, pose, scale, num) -> List[Dict[str, mx.array]]:
        pred = self.config.pred_head_config
        dpt = pred.dpt_adaptor
        rays = adaptors.ray_directions(dense[..., :3], dpt)
        depth = adaptors.scale_value(
            dense[..., 3:4],
            dpt.get("depth_mode", "exp"),
            dpt.get("depth_vmin", 0.0),
            dpt.get("depth_vmax", math.inf),
        )
        trans = adaptors.camera_translation(pose[:, :3], pred.pose_adaptor)
        quats = adaptors.quaternions(pose[:, 3:], pred.pose_adaptor)
        sa = pred.scale_adaptor
        scale = adaptors.scale_value(
            scale, sa.get("mode", "exp"), sa.get("vmin", 1e-8), sa.get("vmax", math.inf)
        )

        points_cam = rays * depth
        rotation = quaternion_to_rotation_matrix(quats)[:, None, None]
        points = _apply(rotation, points_cam) + trans[:, None, None]
        outputs = {
            "pts3d": points,
            "pts3d_cam": points_cam,
            "ray_directions": rays,
            "depth_along_ray": depth,
            "cam_trans": trans,
            "cam_quats": quats,
        }
        if pred.has_confidence:
            outputs["conf"] = adaptors.confidence(dense[..., 4], dpt)
        if pred.has_mask:
            logits = dense[..., -1]
            outputs["non_ambiguous_mask"] = mx.sigmoid(logits) > 0.5
            outputs["non_ambiguous_mask_logits"] = logits

        scaled = {"pts3d", "pts3d_cam", "depth_along_ray"}
        per_view = []
        for chunk in zip(*(mx.split(v, num) for v in outputs.values())):
            view = dict(zip(outputs, chunk))
            for key in scaled:
                view[key] = view[key] * scale[:, :, None, None]
            view["cam_trans"] = view["cam_trans"] * scale
            view["metric_scaling_factor"] = scale
            per_view.append(view)
        return per_view

    # ---- inference API ------------------------------------------------------

    @staticmethod
    def validate_views(views: Sequence[Dict[str, Any]]) -> None:
        """The input rules of the reference ``infer`` (keys, conflicts, and a
        posed first view whenever any view is posed)."""
        if not views:
            raise ValueError("At least one view must be provided")
        posed = []
        for i, view in enumerate(views):
            keys = set(view)
            if keys - ALLOWED_VIEW_KEYS:
                raise ValueError(
                    f"View {i} contains invalid keys: {keys - ALLOWED_VIEW_KEYS}. "
                    f"Allowed keys are: {sorted(ALLOWED_VIEW_KEYS)}"
                )
            missing = {"img", "data_norm_type"} - keys
            if missing:
                raise ValueError(f"View {i} missing required keys: {missing}")
            if {"intrinsics", "ray_directions"} <= keys:
                raise ValueError(
                    f"View {i} contains conflicting keys: provide 'intrinsics' or "
                    "'ray_directions', not both."
                )
            if "depth_z" in keys and not keys & {"intrinsics", "ray_directions"}:
                raise ValueError(
                    f"View {i}: 'depth_z' needs 'intrinsics' or 'ray_directions'."
                )
            if "camera_poses" in keys:
                posed.append(i)
        if posed and posed[0] != 0:
            raise ValueError(
                f"Views {posed} have camera_poses but view 0 (the reference view) "
                "does not."
            )

    def prepare_views(self, views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """User views -> the internal inputs of ``__call__`` (rays, depth along
        ray, pose quaternions, metric flags), as lazy MLX arrays."""
        expected = self.config.encoder_config.data_norm_type
        prepared = []
        for i, view in enumerate(views):
            if _norm_type(view) != expected:
                raise ValueError(
                    f"View {i}: images must be normalized with {expected!r}, got "
                    f"{_norm_type(view)!r}"
                )
            image = view["img"]
            image = image if isinstance(image, mx.array) else mx.array(image)
            if image.ndim == 3:
                image = image[None]
            batch, height, width, _ = image.shape
            out = {"img": image, "data_norm_type": expected}
            if "intrinsics" in view:
                intrinsics = _as_array(view["intrinsics"])
                intrinsics = mx.broadcast_to(intrinsics, (batch, 3, 3))
                out["ray_directions_cam"] = rays_from_intrinsics(
                    intrinsics, height, width
                )
            elif "ray_directions" in view:
                rays = _as_array(view["ray_directions"])
                rays = rays / (mx.linalg.norm(rays, axis=-1, keepdims=True) + 1e-8)
                out["ray_directions_cam"] = mx.broadcast_to(
                    rays, (batch, height, width, 3)
                )
            if "depth_z" in view:
                depth = _as_array(view["depth_z"])
                if depth.shape[-1] == 1 and depth.shape[-3:-1] == (height, width):
                    depth = depth[..., 0]
                depth = mx.broadcast_to(depth, (batch, height, width))
                out["depth_along_ray"] = depth_z_to_depth_along_ray(
                    depth, out["ray_directions_cam"]
                )
            if "camera_poses" in view:
                poses = view["camera_poses"]
                if isinstance(poses, (tuple, list)) and len(poses) == 2:
                    quats, trans = (_as_array(p) for p in poses)
                else:
                    poses = _as_array(poses)
                    if poses.shape[-2:] != (4, 4):
                        raise ValueError(
                            f"View {i}: camera_poses must be (quats, trans) or "
                            "(B, 4, 4) matrices."
                        )
                    quats = rotation_matrix_to_quaternion(poses[..., :3, :3])
                    trans = poses[..., :3, 3]
                out["camera_pose_quats"] = mx.broadcast_to(quats, (batch, 4))
                out["camera_pose_trans"] = mx.broadcast_to(trans, (batch, 3))
            metric = view.get("is_metric_scale", True)
            out["is_metric_scale"] = mx.broadcast_to(
                _as_array(metric, mx.bool_).reshape(-1), (batch,)
            )
            prepared.append(out)
        return prepared

    def infer(
        self,
        views: Sequence[Dict[str, Any]],
        memory_efficient_inference: bool = True,
        minibatch_size: Optional[int] = None,
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        ignore_calibration_inputs: bool = False,
        ignore_depth_inputs: bool = False,
        ignore_pose_inputs: bool = False,
        ignore_depth_scale_inputs: bool = False,
        ignore_pose_scale_inputs: bool = False,
        use_multiview_confidence: bool = False,
        multiview_conf_depth_abs_thresh: float = 0.02,
        multiview_conf_depth_rel_thresh: float = 0.02,
    ) -> List[Dict[str, mx.array]]:
        """Predict metric geometry for a list of views (see README.md).

        Each view is a dict with ``img`` (B, H, W, 3) or (H, W, 3) normalized
        with ``data_norm_type``, and optionally ``intrinsics`` (B, 3, 3) *or*
        ``ray_directions`` (B, H, W, 3), ``depth_z`` (B, H, W) (needs
        calibration), ``camera_poses`` (B, 4, 4) or ``(quats, trans)``
        OpenCV cam2world (needs a posed first view) and ``is_metric_scale``
        (default True). Returns one dict of lazy arrays per view.

        ``memory_efficient_inference`` runs the dense head ``minibatch_size``
        images at a time (default ``config.dense_head_chunk_size``); off, all
        at once. The trunk precision is the weight dtype chosen at conversion.
        """
        self.validate_views(views)
        prepared = self.prepare_views(views)
        outputs = self(
            prepared,
            use_calibration=not ignore_calibration_inputs,
            use_depth=not ignore_depth_inputs,
            use_pose=not ignore_pose_inputs,
            use_depth_scale=not ignore_depth_scale_inputs,
            use_pose_scale=not ignore_pose_scale_inputs,
            dense_head_chunk_size=minibatch_size if memory_efficient_inference else 0,
        )
        return self.postprocess(
            outputs,
            prepared,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            use_multiview_confidence=use_multiview_confidence,
            multiview_conf_depth_abs_thresh=multiview_conf_depth_abs_thresh,
            multiview_conf_depth_rel_thresh=multiview_conf_depth_rel_thresh,
        )

    @staticmethod
    def postprocess(
        outputs: List[Dict[str, mx.array]],
        views: List[Dict[str, Any]],
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        use_multiview_confidence: bool = False,
        multiview_conf_depth_abs_thresh: float = 0.02,
        multiview_conf_depth_rel_thresh: float = 0.02,
    ) -> List[Dict[str, mx.array]]:
        """Add ``img_no_norm``, ``depth_z``, ``intrinsics`` and
        ``camera_poses``; optionally swap in multi-view confidence; mask the
        dense geometry (non-ambiguous, confidence and edge masks)."""
        results = []
        for raw, view in zip(outputs, views):
            out = dict(raw)
            mean, std = IMAGE_NORMALIZATIONS[_norm_type(view)]
            out["img_no_norm"] = mx.clip(
                view["img"].astype(mx.float32) * mx.array(std) + mx.array(mean), 0, 1
            )
            out["depth_z"] = out["pts3d_cam"][..., 2:3]
            out["intrinsics"] = intrinsics_from_rays(out["ray_directions"])
            out["camera_poses"] = pose_matrix(out["cam_quats"], out["cam_trans"])
            results.append(out)

        if use_multiview_confidence and "conf" in results[0]:
            masks = [
                r.get("non_ambiguous_mask", mx.ones(r["conf"].shape, dtype=mx.bool_))
                for r in results
            ]
            confidences = multiview_depth_confidence(
                [r["depth_z"] for r in results],
                [r["intrinsics"] for r in results],
                [r["camera_poses"] for r in results],
                masks,
                multiview_conf_depth_abs_thresh,
                multiview_conf_depth_rel_thresh,
            )
            for r, c in zip(results, confidences):
                r["conf"] = c

        if not apply_mask:
            return results
        for out in results:
            mask = out.get("non_ambiguous_mask")
            if apply_confidence_mask and "conf" in out:
                conf = out["conf"]
                threshold = _quantile(
                    conf.reshape(conf.shape[0], -1), confidence_percentile / 100.0
                )
                conf_mask = conf > threshold[:, None, None]
                mask = conf_mask if mask is None else mask & conf_mask
            if mask is None:
                continue
            if mask_edges:
                normals, normals_mask = points_to_normals(out["pts3d"], mask)
                edges = normals_edge(normals, edge_normal_threshold, normals_mask)
                edges = edges & depth_edge(
                    out["depth_z"][..., 0], edge_depth_threshold, mask
                )
                mask = mask & ~edges
            mask = mask[..., None]
            for key in ("pts3d", "pts3d_cam", "depth_along_ray", "depth_z"):
                out[key] = out[key] * mask
            out["mask"] = mask
        return results

    # ---- weights --------------------------------------------------------------

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Cast the geometric encoders and heads to ``config.head_dtype``, and
        the trunk's norms, LayerScales and embeddings to float32. Official
        checkpoints go through ``convert.py`` first."""
        head_dtype = self.config.head_dtype and getattr(mx, self.config.head_dtype)
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(_FLOAT_MODULES):
                if head_dtype is not None:
                    value = value.astype(head_dtype)
            elif any(p in key for p in _FLOAT_PARAMS):
                value = value.astype(mx.float32)
            sanitized[key] = value
        return sanitized
