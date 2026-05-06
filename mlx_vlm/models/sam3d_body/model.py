"""SAM 3D Body top-level model: backbone -> decoder -> MHR head."""

import math

import mlx.core as mx
import mlx.nn as nn

from .backbone import DINOv3Backbone
from .config import SAM3DConfig
from .decoder import PromptableDecoder
from .mhr_head import MHRHead
from .prompt_encoder import PromptEncoder
from .transformer import DecoderFFN


class CameraHead(nn.Module):
    """2-layer MLP predicting weak-perspective camera (s, tx, ty).

    Weight keys:
        proj.layers.0.0.{weight,bias}: (1024, 1024)
        proj.layers.1.{weight,bias}: (3, 1024)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 3):
        super().__init__()
        self.proj = DecoderFFN(input_dim, input_dim)
        self.proj.layers[1] = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array, init_estimate: mx.array = None) -> mx.array:
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate
        return pred


class RayConditionEmbedding(nn.Module):
    """Adds camera ray conditioning to image features.

    Weight keys:
        conv.weight: (1280, 1, 1, 1379) — 1x1 conv from 1379 ray channels
        norm.{weight,bias}: (1280,)
    """

    def __init__(self, embed_dim: int = 1280, ray_channels: int = 1379):
        super().__init__()
        self.conv = nn.Conv2d(
            ray_channels, embed_dim, kernel_size=1, stride=1, bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, image_features: mx.array, ray_map: mx.array) -> mx.array:
        ray_embed = self.conv(ray_map)
        ray_embed = self.norm(ray_embed)
        return image_features + ray_embed


class MLP2Layer(nn.Module):
    """2-layer MLP with ReLU, matching weight key pattern layers.0.0 / layers.1.

    Used for keypoint_posemb_linear, keypoint3d_posemb_linear.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = [
            [nn.Linear(input_dim, hidden_dim)],
            nn.Linear(hidden_dim, output_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layers[0][0](x))
        return self.layers[1](x)


class MLP3Layer(nn.Module):
    """3-layer MLP for bbox prediction.

    Weight keys: layers.{0,1,2}.{weight,bias}
    """

    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.layers = [
            nn.Linear(dim, dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, output_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layers[0](x))
        x = nn.relu(self.layers[1](x))
        return self.layers[2](x)


def fourier_encode(pos, num_bands=16, max_resolution=64):
    """Generate Fourier position encoding from 3D positions.

    pos: (..., 3) positions
    Returns: (..., 99) = 3 (raw) + 3*16*2 (sin/cos)
    """
    freq_bands = mx.linspace(1.0, max_resolution / 2, num_bands)  # (16,)

    # pos shape: (..., 3), expand for broadcasting
    # pos[..., None] * freq_bands -> (..., 3, 16)
    features = pos[..., None] * freq_bands  # (..., 3, 16)
    orig_shape = pos.shape[:-1]
    features = features.reshape(*orig_shape, 48)  # (..., 48)

    encoded = mx.concatenate(
        [
            mx.sin(math.pi * features),
            mx.cos(math.pi * features),
        ],
        axis=-1,
    )  # (..., 96)

    return mx.concatenate([pos, encoded], axis=-1)  # (..., 99)


def grid_sample_2d(features, coords):
    """Bilinear sample from (B, H, W, C) at coords (B, N, 2) in [-1, 1] range.

    Returns (B, N, C).
    """
    B, H, W, C = features.shape
    N = coords.shape[1]

    # Convert from [-1, 1] to pixel coords
    x = (coords[:, :, 0] + 1) * (W - 1) / 2
    y = (coords[:, :, 1] + 1) * (H - 1) / 2

    x0 = mx.floor(x).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp
    x0c = mx.clip(x0, 0, W - 1)
    x1c = mx.clip(x1, 0, W - 1)
    y0c = mx.clip(y0, 0, H - 1)
    y1c = mx.clip(y1, 0, H - 1)

    # Weights
    wa = (x1.astype(mx.float32) - x) * (y1.astype(mx.float32) - y)
    wb = (x - x0.astype(mx.float32)) * (y1.astype(mx.float32) - y)
    wc = (x1.astype(mx.float32) - x) * (y - y0.astype(mx.float32))
    wd = (x - x0.astype(mx.float32)) * (y - y0.astype(mx.float32))

    # Gather corners — loop over batch (batch is usually 1)
    results = []
    for b in range(B):
        f = features[b]  # (H, W, C)
        vals = (
            wa[b, :, None] * f[y0c[b], x0c[b]]
            + wb[b, :, None] * f[y0c[b], x1c[b]]
            + wc[b, :, None] * f[y1c[b], x0c[b]]
            + wd[b, :, None] * f[y1c[b], x1c[b]]
        )
        results.append(vals)

    return mx.stack(results, axis=0)  # (B, N, C)


class SAM3DBody(nn.Module):
    """Full SAM 3D Body model wiring backbone, decoder, and MHR head.

    Forward: image (B, H, W, 3) NHWC -> body mesh + camera params.
    """

    def __init__(self, config: SAM3DConfig = None):
        super().__init__()
        if config is None:
            config = SAM3DConfig()
        self.config = config

        # Sub-modules
        self.backbone = DINOv3Backbone(config)
        self.decoder = PromptableDecoder(
            dims=config.decoder_dim,
            context_dims=config.embed_dim,
            depth=config.decoder_depth,
            num_heads=config.decoder_heads,
            head_dims=config.decoder_head_dim,
            mlp_dims=config.decoder_mlp_dim,
        )
        self.head_pose = MHRHead(input_dim=config.decoder_dim, config=config)
        self.head_camera = CameraHead(
            input_dim=config.decoder_dim, output_dim=config.camera_output_dim
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=config.prompt_embed_dim,
            num_point_embeddings=config.num_point_embeddings,
        )

        # Projections
        # init_to_token: 525 = 519 pose + 3 cam + 3 CLIFF condition
        self.init_to_token_mhr = nn.Linear(
            config.pose_output_dim + config.camera_output_dim + 3, config.decoder_dim
        )
        # prev_to_token: 522 = 519 pose + 3 cam
        self.prev_to_token_mhr = nn.Linear(
            config.pose_output_dim + config.camera_output_dim, config.decoder_dim
        )
        # prompt projection: 1280 -> 1024
        self.prompt_to_token = nn.Linear(config.prompt_embed_dim, config.decoder_dim)

        # Learnable initial estimates (loaded from weights)
        self.init_pose = mx.zeros((1, config.pose_output_dim))
        self.init_camera = mx.zeros((1, config.camera_output_dim))

        # Keypoint embeddings and projections (already 1024-dim in weights)
        self.keypoint_embedding = mx.zeros(
            (config.num_point_embeddings, config.decoder_dim)
        )
        self.keypoint_feat_linear = nn.Linear(config.embed_dim, config.decoder_dim)
        # 2D keypoint pos: input 2 -> 1024
        self.keypoint_posemb_linear = MLP2Layer(
            2, config.decoder_dim, config.decoder_dim
        )
        # 3D keypoint pos: input 3 -> 1024
        self.keypoint3d_embedding = mx.zeros(
            (config.num_point_embeddings, config.decoder_dim)
        )
        self.keypoint3d_posemb_linear = MLP2Layer(
            3, config.decoder_dim, config.decoder_dim
        )

        # Hand detection tokens
        self.hand_box_embedding = mx.zeros((2, config.decoder_dim))
        self.hand_cls_embed = nn.Linear(config.decoder_dim, 2)
        self.bbox_embed = MLP3Layer(config.decoder_dim, 4)

        # Hand PE layer (gaussian positional encoding for hand boxes)
        from .prompt_encoder import PositionalEncodingGaussian

        self.hand_pe_layer = PositionalEncodingGaussian(
            num_feats=config.prompt_embed_dim // 2
        )

        # Ray conditioning
        self.ray_cond_emb = RayConditionEmbedding(config.embed_dim, 1379)

    def compute_ray_map(self, bbox, img_size, cam_int):
        """Compute camera ray map for the cropped image region.

        bbox: [x1, y1, x2, y2]
        img_size: (img_h, img_w) original image size
        cam_int: (3, 3) intrinsic matrix as mx.array
        Returns: (1, H, W, 2) ray directions
        """
        H, W = self.config.image_size  # 512, 384

        # Create pixel grid in crop space
        ys = mx.arange(H).astype(mx.float32)
        xs = mx.arange(W).astype(mx.float32)
        # meshgrid: ij indexing
        grid_y = mx.broadcast_to(ys[:, None], (H, W))
        grid_x = mx.broadcast_to(xs[None, :], (H, W))

        # Map crop pixels back to original image space
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        scale_x = bbox_w / W
        scale_y = bbox_h / H

        grid_orig_x = grid_x * scale_x + bbox[0]
        grid_orig_y = grid_y * scale_y + bbox[1]

        # Camera rays: (pixel - principal_point) / focal_length
        fx = cam_int[0, 0]
        fy = cam_int[1, 1]
        cx = cam_int[0, 2]
        cy = cam_int[1, 2]

        ray_x = (grid_orig_x - cx) / fx
        ray_y = (grid_orig_y - cy) / fy

        rays = mx.stack([ray_x, ray_y], axis=-1)  # (H, W, 2)
        return rays[None]  # (1, H, W, 2)

    def apply_ray_conditioning(self, image_features, rays):
        """Condition image features on camera rays.

        image_features: (B, H_p, W_p, 1280) — backbone output at patch resolution
        rays: (B, H_img, W_img, 2) — camera rays at full image resolution
        Returns: (B, H_p, W_p, 1280)
        """
        B, H_p, W_p, C = image_features.shape

        # Downsample rays to patch resolution via area-averaging (matches antialias)
        # Image is (512, 384), patches are (32, 24) = 16x downsample
        patch_size = self.config.patch_size
        rays_down = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)(
            rays
        )  # (B, H_p, W_p, 2)

        # Append z=1 to get 3D ray directions
        ones = mx.ones((*rays_down.shape[:-1], 1))
        rays_3d = mx.concatenate([rays_down, ones], axis=-1)  # (B, H_p, W_p, 3)

        # Fourier encode each ray direction
        rays_flat = rays_3d.reshape(B, -1, 3)  # (B, H_p*W_p, 3)
        rays_encoded = fourier_encode(rays_flat)  # (B, H_p*W_p, 99)
        rays_encoded = rays_encoded.reshape(B, H_p, W_p, 99)

        # Concatenate with image features -> 1280 + 99 = 1379 channels
        combined = mx.concatenate([image_features, rays_encoded], axis=-1)

        # 1x1 conv + norm via RayConditionEmbedding
        # But we need to call conv directly on combined, not as additive
        result = self.ray_cond_emb.conv(combined)  # (B, H_p, W_p, 1280)
        result = self.ray_cond_emb.norm(result)

        return result

    def _perspective_projection(self, kp3d, pred_cam, bbox, img_size, cam_int=None):
        """Project 3D keypoints to 2D crop-normalized coordinates.

        kp3d: (B, N, 3) 3D keypoints
        pred_cam: (B, 3) camera params [scale, tx, ty]
        bbox: [x1, y1, x2, y2]
        img_size: (img_h, img_w)
        cam_int: (3, 3) camera intrinsics, or None for default

        Returns: (B, N, 2) keypoints in [-1, 1] range (grid_sample compatible)
        """
        B, N, _ = kp3d.shape
        img_h, img_w = img_size

        # Flip scale and ty
        s = -pred_cam[:, 0:1]  # (B, 1)
        tx = pred_cam[:, 1:2]
        ty = -pred_cam[:, 2:3]

        if cam_int is not None:
            focal_length = float(cam_int[0, 0])
        else:
            focal_length = math.sqrt(img_h**2 + img_w**2)

        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        bs = bbox_size * s + 1e-8  # (B, 1)
        tz = 2 * focal_length / bs  # (B, 1)
        cx_offset = 2 * (bbox_center_x - img_w / 2) / bs
        cy_offset = 2 * (bbox_center_y - img_h / 2) / bs

        cam_t = mx.concatenate(
            [
                tx + cx_offset,
                ty + cy_offset,
                tz,
            ],
            axis=1,
        )  # (B, 3)

        # Translate to camera space
        j3d_cam = kp3d + cam_t[:, None, :]  # (B, N, 3)

        # Perspective divide
        j3d_norm = j3d_cam / (j3d_cam[:, :, 2:3] + 1e-8)

        # Project to pixel coords
        kp_2d_x = focal_length * j3d_norm[:, :, 0] + img_w / 2
        kp_2d_y = focal_length * j3d_norm[:, :, 1] + img_h / 2

        # Convert to crop-normalized coords: map pixel to crop-relative
        # Crop covers bbox with 1.2x padding, centered at bbox_center
        crop_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2
        H_out, W_out = self.config.image_size

        # Pixel in crop: (pixel_orig - bbox_center) / crop_size * output_size + output_center
        # Invert to get [-1, 1] coords for grid_sample
        crop_x = (kp_2d_x - bbox_center_x) / (crop_size / 2)  # [-1, 1]
        crop_y = (kp_2d_y - bbox_center_y) / (crop_size / 2)

        return mx.stack([crop_x, crop_y], axis=-1)  # (B, N, 2)

    def __call__(
        self,
        image: mx.array,
        cliff_condition: mx.array = None,
        bbox: list = None,
        img_size: tuple = None,
        cam_int: mx.array = None,
    ) -> tuple[dict, mx.array]:
        """Forward pass for body-only inference.

        Args:
            image: (B, H, W, 3) NHWC float32, ImageNet-normalized
            cliff_condition: (B, 3) optional CLIFF conditioning vector
            bbox: [x1, y1, x2, y2] bounding box in original image coords
            img_size: (img_h, img_w) original image dimensions
            cam_int: (3, 3) camera intrinsic matrix (optional, for ray conditioning)

        Returns:
            (body_output_dict, pred_camera)
        """
        B = image.shape[0]
        H, W = image.shape[1], image.shape[2]
        H_p = H // self.config.patch_size
        W_p = W // self.config.patch_size

        # 1. Backbone
        image_features = self.backbone(image)  # (B, H_p, W_p, 1280)

        # 2. Ray conditioning (if camera intrinsics provided)
        if cam_int is not None and bbox is not None and img_size is not None:
            rays = self.compute_ray_map(bbox, img_size, cam_int)
            image_features = self.apply_ray_conditioning(image_features, rays)

        # 3. Build initial estimates
        init_pose = mx.broadcast_to(self.init_pose, (B, self.config.pose_output_dim))
        init_cam = mx.broadcast_to(self.init_camera, (B, self.config.camera_output_dim))
        init_estimate = mx.concatenate([init_pose, init_cam], axis=1)  # (B, 522)

        # 4. Build init token (with CLIFF condition or zeros)
        if cliff_condition is None:
            cliff_condition = mx.zeros((B, 3))
        init_input = mx.concatenate(
            [
                cliff_condition,
                init_estimate,
            ],
            axis=1,
        )  # (B, 525)
        init_token = self.init_to_token_mhr(
            init_input.reshape(B, 1, -1)
        )  # (B, 1, 1024)

        # 5. Previous estimate token (same as init on first pass)
        prev_token = self.prev_to_token_mhr(
            init_estimate.reshape(B, 1, -1)
        )  # (B, 1, 1024)

        # 6. Prompt token (dummy - all invalid)
        dummy_points = mx.zeros((B, 1, 2))
        dummy_labels = mx.full((B, 1), -1, dtype=mx.int32)
        prompt_embed, _ = self.prompt_encoder.encode_points(dummy_points, dummy_labels)
        prompt_token = self.prompt_to_token(prompt_embed)  # (B, 1, 1024)

        # 7. Hand detection tokens
        hand_tokens = mx.broadcast_to(
            self.hand_box_embedding[None, :, :], (B, 2, self.config.decoder_dim)
        )  # (B, 2, 1024)

        # 8. 2D Keypoint tokens
        kp_tokens = mx.broadcast_to(
            self.keypoint_embedding[None, :, :],
            (B, self.config.num_point_embeddings, self.config.decoder_dim),
        )  # (B, 70, 1024)

        # 9. 3D Keypoint tokens
        kp3d_tokens = mx.broadcast_to(
            self.keypoint3d_embedding[None, :, :],
            (B, self.config.num_point_embeddings, self.config.decoder_dim),
        )  # (B, 70, 1024)

        # 10. Assemble all tokens:
        # [init, prev, prompt, hand_det_0, hand_det_1, kp2d_0..69, kp3d_0..69]
        # Total: 1 + 1 + 1 + 2 + 70 + 70 = 145
        tokens = mx.concatenate(
            [init_token, prev_token, prompt_token, hand_tokens, kp_tokens, kp3d_tokens],
            axis=1,
        )  # (B, 145, 1024)

        # Token index layout
        KP2D_START = 5
        KP2D_END = 5 + self.config.num_point_embeddings  # 75
        KP3D_START = KP2D_END  # 75
        KP3D_END = KP3D_START + self.config.num_point_embeddings  # 145

        # 11. Build positional embeddings for tokens
        init_pe = mx.zeros((B, 1, self.config.decoder_dim))
        prev_pe = prev_token  # reuse as PE
        prompt_pe = prompt_token  # reuse as PE
        hand_pe = mx.zeros((B, 2, self.config.decoder_dim))
        kp_pe = mx.zeros((B, self.config.num_point_embeddings, self.config.decoder_dim))
        kp3d_pe = mx.zeros(
            (B, self.config.num_point_embeddings, self.config.decoder_dim)
        )
        token_pe = mx.concatenate(
            [init_pe, prev_pe, prompt_pe, hand_pe, kp_pe, kp3d_pe], axis=1
        )

        # 12. Image positional encoding
        image_pe = self.prompt_encoder.get_dense_pe(H_p, W_p)  # (1, H_p, W_p, 1280)

        # 13. Intermediate prediction callback
        def token_to_pose_fn(normed_tokens, layer_idx):
            pose_token = normed_tokens[:, 0, :]  # (B, 1024)
            body_output = self.head_pose(pose_token, init_estimate=init_pose)
            pred_cam = self.head_camera(pose_token, init_estimate=init_cam)
            return {
                "body_output": body_output,
                "pred_cam": pred_cam,
            }

        # 14. Keypoint update callback
        def kp_update_fn(tokens, token_pe, pose_output, layer_idx, img_feats):
            body_output = pose_output["body_output"]
            pred_cam = pose_output["pred_cam"]
            kp3d = body_output["pred_keypoints_3d"]  # (B, 70, 3)

            # --- Update 2D keypoint tokens ---
            if bbox is not None and img_size is not None:
                # Project 3D keypoints to 2D crop coordinates
                kp2d_norm = self._perspective_projection(
                    kp3d, pred_cam, bbox, img_size, cam_int=cam_int
                )  # (B, 70, 2)

                # New position embeddings from predicted 2D coords
                new_kp_pe = self.keypoint_posemb_linear(kp2d_norm)  # (B, 70, 1024)

                # Sample image features at predicted 2D locations
                sampled_feats = grid_sample_2d(img_feats, kp2d_norm)  # (B, 70, 1280)
                sampled_proj = self.keypoint_feat_linear(sampled_feats)  # (B, 70, 1024)

                # Update 2D kp tokens: add sampled features
                new_kp_tokens = tokens[:, KP2D_START:KP2D_END, :] + sampled_proj
                tokens = mx.concatenate(
                    [
                        tokens[:, :KP2D_START, :],
                        new_kp_tokens,
                        tokens[:, KP2D_END:, :],
                    ],
                    axis=1,
                )

                # Update 2D kp position embeddings
                token_pe = mx.concatenate(
                    [
                        token_pe[:, :KP2D_START, :],
                        new_kp_pe,
                        token_pe[:, KP2D_END:, :],
                    ],
                    axis=1,
                )

            # --- Update 3D keypoint tokens ---
            # Pelvis-normalize: average of left_hip (9) and right_hip (10)
            pelvis = (kp3d[:, 9:10, :] + kp3d[:, 10:11, :]) / 2  # (B, 1, 3)
            kp3d_centered = kp3d - pelvis  # (B, 70, 3)

            # New position embeddings from predicted 3D coords
            new_kp3d_pe = self.keypoint3d_posemb_linear(kp3d_centered)  # (B, 70, 1024)

            # Update 3D kp position embeddings
            token_pe = mx.concatenate(
                [
                    token_pe[:, :KP3D_START, :],
                    new_kp3d_pe,
                    token_pe[:, KP3D_END:, :],
                ],
                axis=1,
            )

            return tokens, token_pe

        # 15. Run decoder with intermediate predictions
        output, all_outputs = self.decoder(
            tokens,
            image_features,
            token_pe,
            image_pe,
            token_to_pose_fn=token_to_pose_fn,
            kp_update_fn=kp_update_fn,
        )  # output: (B, 145, 1024)

        # 16. Use the final layer's output (last in all_outputs)
        if all_outputs:
            final_output = all_outputs[-1]
            body_output = final_output["body_output"]
            pred_cam = final_output["pred_cam"]
        else:
            # Fallback if no intermediate predictions (shouldn't happen normally)
            pose_token = output[:, 0, :]
            body_output = self.head_pose(pose_token, init_estimate=init_pose)
            pred_cam = self.head_camera(pose_token, init_estimate=init_cam)

        return body_output, pred_cam

    def load_all_weights(self, weights_dir: str):
        """Load all model weights from safetensors in weights_dir."""
        from pathlib import Path

        from safetensors import safe_open

        weights_dir = Path(weights_dir)

        # Find safetensors file(s)
        safetensors_path = weights_dir / "model.safetensors"
        index_path = weights_dir / "model.safetensors.index.json"

        if index_path.exists():
            import json

            with open(index_path) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            files = [weights_dir / s for s in shard_files]
        else:
            files = [safetensors_path]

        # Collect all tensors
        all_tensors = {}
        for fpath in files:
            with safe_open(str(fpath), framework="numpy") as f:
                for key in f.keys():
                    all_tensors[key] = mx.array(f.get_tensor(key))

        # Build weight list for self.load_weights()
        weights = []

        # Skip keys that belong to hand-specific modules
        hand_prefixes = (
            "decoder_hand.",
            "head_pose_hand.",
            "head_camera_hand.",
            "init_pose_hand.",
            "init_camera_hand.",
            "init_to_token_mhr_hand.",
            "prev_to_token_mhr_hand.",
            "keypoint_embedding_hand.",
            "keypoint3d_embedding_hand.",
            "keypoint_posemb_linear_hand.",
            "keypoint3d_posemb_linear_hand.",
            "keypoint_feat_linear_hand.",
            "ray_cond_emb_hand.",
        )

        # Bare array parameters stored with ".weight" suffix in safetensors
        # but are plain attributes on the model (not nn.Module submodules)
        bare_param_keys = {
            "init_pose.weight": "init_pose",
            "init_camera.weight": "init_camera",
            "keypoint_embedding.weight": "keypoint_embedding",
            "keypoint3d_embedding.weight": "keypoint3d_embedding",
            "hand_box_embedding.weight": "hand_box_embedding",
        }

        # mask_downscaling is in the prompt_encoder but not modeled (unused for inference)
        skip_prefixes = ("prompt_encoder.mask_downscaling.",)

        for key, tensor in all_tensors.items():
            # Skip hand variants
            if any(key.startswith(p) for p in hand_prefixes):
                continue

            # MHR body model weights go through head_pose.load_all_weights
            if key.startswith("mhr."):
                continue

            # Skip mask_downscaling (not used in body-only inference)
            if any(key.startswith(p) for p in skip_prefixes):
                continue

            # Backbone weights: already prefixed correctly
            if key.startswith("backbone."):
                # Skip bias_mask keys (not used in MLX)
                if "bias_mask" in key:
                    continue
                # Skip k_proj.bias (K bias is masked to zero)
                if "k_proj.bias" in key:
                    continue
                weights.append((key, tensor))
                continue

            # Remap bare param keys
            if key in bare_param_keys:
                weights.append((bare_param_keys[key], tensor))
                continue

            # All other keys map directly
            weights.append((key, tensor))

        self.load_weights(weights, strict=False)

        # Load MHR body model weights separately (they need key remapping)
        self.head_pose.load_all_weights(str(safetensors_path))

    @staticmethod
    def sanitize(weights):
        """Remap weight keys for mlx-vlm compatibility.

        This is the single source of truth for key naming. Handles both:
          * Raw PyTorch checkpoint keys (post-`convert_weights` output with
            `--raw`): QKV splitting, backbone `encoder.*` prefix rewriting,
            Conv2d (O,I,H,W)→(O,H,W,I) transposition, MHR JIT prefix mapping.
          * Already-sanitized keys (legacy convert_weights output): passes
            through the rename pipeline untouched.

        Detection is by canary: presence of `backbone.encoder.cls_token` or
        `character_torch.*` keys means the input is raw and needs full remap.
        Also drops hand-only modules (not implemented yet — see README) and
        renames bare array params that sit on the model as attributes.
        """
        is_raw = ("backbone.encoder.cls_token" in weights) or any(
            k.startswith("character_torch.") for k in weights
        )
        if is_raw:
            weights = SAM3DBody._remap_raw_pytorch_keys(weights)

        hand_prefixes = (
            "decoder_hand.",
            "head_pose_hand.",
            "head_camera_hand.",
            "init_pose_hand.",
            "init_camera_hand.",
            "init_to_token_mhr_hand.",
            "prev_to_token_mhr_hand.",
            "keypoint_embedding_hand.",
            "keypoint3d_embedding_hand.",
            "keypoint_posemb_linear_hand.",
            "keypoint3d_posemb_linear_hand.",
            "keypoint_feat_linear_hand.",
            "ray_cond_emb_hand.",
        )
        bare_param_keys = {
            "init_pose.weight": "init_pose",
            "init_camera.weight": "init_camera",
            "keypoint_embedding.weight": "keypoint_embedding",
            "keypoint3d_embedding.weight": "keypoint3d_embedding",
            "hand_box_embedding.weight": "hand_box_embedding",
        }
        skip_prefixes = ("prompt_encoder.mask_downscaling.",)

        sanitized = {}
        for key, tensor in weights.items():
            if any(key.startswith(p) for p in hand_prefixes):
                continue
            if any(key.startswith(p) for p in skip_prefixes):
                continue
            if key.startswith("backbone.") and (
                "bias_mask" in key or "k_proj.bias" in key
            ):
                continue
            if key in bare_param_keys:
                sanitized[bare_param_keys[key]] = tensor
                continue
            sanitized[key] = tensor
        return sanitized

    @staticmethod
    def _remap_raw_pytorch_keys(weights):
        """Convert raw PyTorch checkpoint keys to MLX-native module paths.

        Pulled out of convert_weights.py so key naming lives with the model.
        The inverse transformations (dtype conversion, JIT model extraction,
        safetensors sharding) stay in convert_weights.py — only key names and
        Conv2d layouts belong to the model architecture.
        """
        import re

        qkv_pattern = re.compile(
            r"backbone\.encoder\.blocks\.(\d+)\.attn\.qkv\.(weight|bias|bias_mask)"
        )
        backbone_block_pattern = re.compile(r"backbone\.encoder\.blocks\.(\d+)\.(.+)")
        backbone_simple = {
            "backbone.encoder.cls_token": "backbone.cls_token",
            "backbone.encoder.storage_tokens": "backbone.storage_tokens",
            "backbone.encoder.patch_embed.proj.weight": "backbone.patch_embed.projection.weight",
            "backbone.encoder.patch_embed.proj.bias": "backbone.patch_embed.projection.bias",
            "backbone.encoder.rope_embed.periods": "backbone.rope_embed.periods",
            "backbone.encoder.norm.weight": "backbone.norm.weight",
            "backbone.encoder.norm.bias": "backbone.norm.bias",
        }

        result = {}
        for key, value in weights.items():
            # Fused QKV -> split q/k/v
            m = qkv_pattern.match(key)
            if m:
                block_idx, ptype = m.group(1), m.group(2)
                dim = value.shape[0] // 3
                q, k, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]
                prefix = f"backbone.blocks.{block_idx}.attention"
                if ptype == "bias_mask":
                    result[f"{prefix}.q_bias_mask"] = q
                    result[f"{prefix}.k_bias_mask"] = k
                    result[f"{prefix}.v_bias_mask"] = v
                else:
                    result[f"{prefix}.q_proj.{ptype}"] = q
                    result[f"{prefix}.k_proj.{ptype}"] = k
                    result[f"{prefix}.v_proj.{ptype}"] = v
                continue

            # Simple backbone prefix renames (+ Conv2d transpose for patch_embed)
            if key in backbone_simple:
                new_key = backbone_simple[key]
                if value.ndim == 4 and "patch_embed" in key:
                    value = value.transpose(0, 2, 3, 1)
                result[new_key] = value
                continue

            # Backbone block pattern (non-QKV): encoder.blocks.N.X -> blocks.N.X
            m = backbone_block_pattern.match(key)
            if m:
                block_idx, rest = m.group(1), m.group(2)
                if rest.startswith("attn.proj."):
                    new_key = (
                        f"backbone.blocks.{block_idx}.attention.o_proj."
                        + rest[len("attn.proj.") :]
                    )
                else:
                    new_key = f"backbone.blocks.{block_idx}.{rest}"
                result[new_key] = value
                continue

            # Conv2d (O,I,H,W) -> (O,H,W,I) for mask_downscaling and ray_cond_emb
            if "mask_downscaling" in key and value.ndim == 4:
                result[key] = value.transpose(0, 2, 3, 1)
                continue
            if (
                "ray_cond_emb" in key
                and key.endswith("conv.weight")
                and value.ndim == 4
            ):
                result[key] = value.transpose(0, 2, 3, 1)
                continue

            # MHR JIT prefix renames
            new_key = key
            new_key = new_key.replace("character_torch.", "mhr.character.")
            new_key = new_key.replace(
                "face_expressions_model.", "mhr.face_expressions."
            )
            new_key = new_key.replace(
                "pose_correctives_model.", "mhr.pose_correctives."
            )
            result[new_key] = value

        return result


# mlx-vlm convention alias
Model = SAM3DBody
