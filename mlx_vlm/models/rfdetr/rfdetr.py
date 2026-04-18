"""RF-DETR main model with weight sanitization."""

from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .segmentation import SegmentationHead
from .transformer import MLP, Transformer, inverse_sigmoid
from .vision import DINOv2Backbone, MultiScaleProjector


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Vision backbone
        self.backbone = DINOv2Backbone(config.backbone_config)
        self.backbone.num_windows = config.num_windows
        self.backbone.embeddings.num_windows = config.num_windows

        # Feature projector
        self.projector = MultiScaleProjector(config._projector_config)

        # Transformer (encoder selection + decoder)
        self.transformer = Transformer(config.transformer_config)

        # Detection heads
        d = config.transformer_config.hidden_dim
        num_classes = config.transformer_config.num_classes
        self.class_embed = nn.Linear(d, num_classes)
        self.bbox_embed = MLP(d, d, 4, num_layers=3)

        # Learnable queries and reference points
        total_queries = (
            config.transformer_config.num_queries * config.transformer_config.group_detr
        )
        self.query_feat = nn.Embedding(total_queries, d)
        self.refpoint_embed = nn.Embedding(total_queries, 4)

        # Optional segmentation head
        if config.segmentation and config.segmentation_config is not None:
            sc = config.segmentation_config
            self.segmentation_head = SegmentationHead(
                in_dim=sc.in_dim,
                num_blocks=sc.num_blocks,
                bottleneck_ratio=sc.bottleneck_ratio,
                downsample_ratio=sc.downsample_ratio,
            )
        else:
            self.segmentation_head = None

    def __call__(self, pixel_values: mx.array) -> Dict[str, mx.array]:
        """
        Args:
            pixel_values: (B, H, W, 3) channel-last image
        Returns:
            dict with pred_logits (B, Q, num_classes) and pred_boxes (B, Q, 4)
        """
        B, H, W, _ = pixel_values.shape

        # 1. Backbone: extract multi-scale features
        features = self.backbone(pixel_values)

        # 2. Projector: merge features into single scale
        memory = self.projector(features)  # (B, h, w, D)
        h, w = memory.shape[1], memory.shape[2]
        memory_flat = memory.reshape(B, h * w, -1)  # (B, HW, D)

        # 3. Transformer: two-stage selection + decoder
        hs, ref_points = self.transformer(
            memory_flat,
            spatial_shape=(h, w),
            query_feat=self.query_feat.weight,
            refpoint_embed=self.refpoint_embed.weight,
            bbox_embed=self.bbox_embed,
        )

        # 4. Detection heads on final decoder output
        pred_logits = self.class_embed(hs)  # (B, Q, num_classes)

        # ref_points are in [0, 1] coordinate space from decoder (for bbox_reparam)
        if self.config.transformer_config.bbox_reparam:
            delta = self.bbox_embed(hs)  # (B, Q, 4)
            # Parametric: delta_cxcy * ref_wh + ref_cxcy, exp(delta_wh) * ref_wh
            pred_cxcy = delta[..., :2] * ref_points[..., 2:] + ref_points[..., :2]
            pred_wh = mx.exp(delta[..., 2:]) * ref_points[..., 2:]
            pred_boxes = mx.concatenate([pred_cxcy, pred_wh], axis=-1)
        else:
            pred_boxes = mx.sigmoid(self.bbox_embed(hs) + inverse_sigmoid(ref_points))

        result = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

        # Optional segmentation
        if self.segmentation_head is not None:
            pred_masks = self.segmentation_head(memory, hs, (H, W))
            result["pred_masks"] = pred_masks

        return result

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert HuggingFace PyTorch weights to MLX format."""
        sanitized = {}

        for k, v in weights.items():
            new_k = k

            # 1. Strip model. prefix
            if new_k.startswith("model."):
                new_k = new_k[len("model.") :]

            # 2. Backbone key remapping
            new_k = new_k.replace(
                "backbone.0.encoder.encoder.embeddings.",
                "backbone.embeddings.",
            )
            new_k = new_k.replace(
                "backbone.0.encoder.encoder.encoder.layer.",
                "backbone.encoder.layers.",
            )
            new_k = new_k.replace(
                "backbone.0.encoder.encoder.layernorm.",
                "backbone.layernorm.",
            )

            # 3. Projector key remapping
            new_k = new_k.replace("backbone.0.projector.", "projector.")

            # 4. DINOv2 attention key remapping
            new_k = new_k.replace(".attention.attention.query.", ".attention.q_proj.")
            new_k = new_k.replace(".attention.attention.key.", ".attention.k_proj.")
            new_k = new_k.replace(".attention.attention.value.", ".attention.v_proj.")
            new_k = new_k.replace(".attention.output.dense.", ".attention.o_proj.")

            # 5. LayerScale: .layer_scale1.lambda1 -> .layer_scale1
            new_k = new_k.replace(".layer_scale1.lambda1", ".layer_scale1")
            new_k = new_k.replace(".layer_scale2.lambda1", ".layer_scale2")

            # 6. Skip mask_token (not needed for inference)
            if "mask_token" in new_k:
                continue

            # 7. Fused QKV split for decoder self-attention
            if "self_attn.in_proj_weight" in new_k:
                d = v.shape[1]  # 256
                base = new_k.replace("in_proj_weight", "")
                sanitized[base + "q_proj.weight"] = v[:d]
                sanitized[base + "k_proj.weight"] = v[d : 2 * d]
                sanitized[base + "v_proj.weight"] = v[2 * d :]
                continue
            if "self_attn.in_proj_bias" in new_k:
                d = v.shape[0] // 3
                base = new_k.replace("in_proj_bias", "")
                sanitized[base + "q_proj.bias"] = v[:d]
                sanitized[base + "k_proj.bias"] = v[d : 2 * d]
                sanitized[base + "v_proj.bias"] = v[2 * d :]
                continue

            # 8. Conv2d weight transposition: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
            if v.ndim == 4 and (
                "conv" in new_k.lower() or "spatial_features_proj" in new_k
            ):
                v = v.transpose(0, 2, 3, 1)
            elif v.ndim == 4 and "patch_embeddings.projection" in new_k:
                v = v.transpose(0, 2, 3, 1)

            sanitized[new_k] = v

        return sanitized

    @staticmethod
    def quant_predicate(path: str, module) -> bool:
        """Control which layers get quantized."""
        # Skip vision backbone (full precision for accuracy)
        if "backbone." in path:
            return False
        # Skip conv layers
        if "conv" in path:
            return False
        # Skip small embeddings
        if any(
            k in path for k in ["query_feat", "refpoint_embed", "position_embeddings"]
        ):
            return False
        # Skip layers with small dimensions
        if hasattr(module, "weight"):
            shape = module.weight.shape
            if any(d % 64 != 0 for d in shape):
                return False
        return isinstance(module, nn.Linear)
