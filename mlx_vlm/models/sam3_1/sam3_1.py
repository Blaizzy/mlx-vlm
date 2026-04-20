"""SAM 3.1 Main Model: Detector (DETR) + Multiplex Tracker.

Reuses DETR encoder/decoder, text encoder, geometry encoder from SAM 3.
New: TriViTDetNeck, MultiplexTrackerModel.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..sam3.decoder import DETRDecoder
from ..sam3.encoder import DETREncoder
from ..sam3.geometry import GeometryEncoder as _GeometryEncoder
from ..sam3.position import PositionEmbeddingSine
from ..sam3.segmentation import DotProductScoring, MaskDecoder
from ..sam3.text_encoder import TextEncoder
from .config import ModelConfig
from .tracker import MultiplexTrackerModel
from .vision import VisionEncoder


class GeometryEncoder(_GeometryEncoder):
    """SAM 3.1 geometry encoder — adds point prompt projections."""

    def __init__(self, config):
        super().__init__(config)
        d = config.hidden_size
        # SAM 3.1 adds point prompt projections (unused in detection-only mode)
        self.points_direct_project = nn.Linear(2, d)
        self.points_pool_project = nn.Linear(d, d)
        self.points_pos_enc_project = nn.Linear(d, d)


class DetectorModel(nn.Module):
    """SAM 3.1 detector — same DETR pipeline, TriViTDetNeck."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        det_cfg = config.detector_config

        self.vision_encoder = VisionEncoder(det_cfg.vision_config)
        self.text_encoder = TextEncoder(
            det_cfg.text_config, d_model=det_cfg.detr_encoder_config.hidden_size
        )
        self.text_projection = nn.Linear(
            det_cfg.text_config.hidden_size,
            det_cfg.detr_encoder_config.hidden_size,
        )
        self.detr_encoder = DETREncoder(det_cfg.detr_encoder_config)
        self.detr_decoder = DETRDecoder(det_cfg.detr_decoder_config)
        self.geometry_encoder = GeometryEncoder(det_cfg.geometry_encoder_config)
        self.mask_decoder = MaskDecoder(det_cfg.mask_decoder_config)
        self.dot_product_scoring = DotProductScoring(
            det_cfg.detr_encoder_config.hidden_size
        )
        self._pos_enc = PositionEmbeddingSine(
            det_cfg.detr_encoder_config.hidden_size // 2
        )

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        text_hidden = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_hidden)

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        boxes: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        B = pixel_values.shape[0]

        # Vision — only need detection FPN (not interactive/propagation)
        det_features, _, _ = self.vision_encoder(
            pixel_values, need_det=True, need_interactive=False, need_propagation=False
        )

        fpn_pos = [self._pos_enc(feat) for feat in det_features]

        # SAM 3.1: 3 scales, no trimming needed (no 0.5x level)
        encoder_feat = det_features[-1]  # 1x scale (72x72)
        encoder_pos = fpn_pos[-1]

        B, H, W, D = encoder_feat.shape
        src = encoder_feat.reshape(B, H * W, D)
        pos_flat = encoder_pos.reshape(B, H * W, D)

        # Text
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        prompt = inputs_embeds
        prompt_mask = attention_mask

        # DETR encoder
        encoded = self.detr_encoder(src, pos_flat, prompt, prompt_mask)

        # DETR decoder
        hs, ref_boxes, presence_logits = self.detr_decoder(
            vision_features=encoded,
            inputs_embeds=prompt,
            vision_pos_encoding=pos_flat,
            text_mask=prompt_mask,
            spatial_shape=(H, W),
        )

        # Box conversion cxcywh → xyxy
        pred_boxes_cxcywh = ref_boxes[-1]
        cx, cy, w, h = (
            pred_boxes_cxcywh[..., 0],
            pred_boxes_cxcywh[..., 1],
            pred_boxes_cxcywh[..., 2],
            pred_boxes_cxcywh[..., 3],
        )
        pred_boxes_xyxy = mx.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )

        # Scoring
        all_pred_logits = self.dot_product_scoring(hs, prompt, prompt_mask)
        pred_logits = all_pred_logits[-1].squeeze(-1)
        presence = presence_logits[-1]

        # Mask prediction
        last_hs = hs[-1]
        seg_out = self.mask_decoder(
            last_hs,
            list(det_features),
            encoder_hidden_states=encoded,
            prompt_features=prompt,
            prompt_mask=prompt_mask,
        )

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes_xyxy,
            "pred_masks": seg_out["pred_masks"],
            "presence_logits": presence,
            "semantic_seg": seg_out.get("semantic_seg"),
            "intermediate_hidden_states": hs,
            "encoder_hidden_states": encoded,
        }


class Model(nn.Module):
    """SAM 3.1 full model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        self.detector_model = DetectorModel(config)
        self.tracker_model = MultiplexTrackerModel(config.tracker_config)

    def _get_tracker_features(self, backbone_features: mx.array):
        """Get propagation FPN features from TriViTDetNeck for tracking."""
        _, _, prop_features = self.detector_model.vision_encoder.neck(
            backbone_features,
            need_det=False,
            need_interactive=False,
            need_propagation=True,
        )
        return prop_features

    def tracker_neck(self, backbone_features: mx.array):
        """Compat shim for Sam3VideoPredictor — returns propagation FPN features."""
        return self._get_tracker_features(backbone_features)

    def track_init(
        self,
        backbone_features: mx.array,
        detection_masks: mx.array,
    ) -> Dict[str, mx.array]:
        """Initialize tracker with detection results."""
        prop_fpn = self._get_tracker_features(backbone_features)
        features = prop_fpn[2]  # 1x scale (72x72)
        B, H, W, D = features.shape

        mask_input = detection_masks[:, :1].transpose(0, 2, 3, 1)  # (B, H, W, 1)
        memory = self.tracker_model.memory_encoder(features, mask_input)

        return {
            "memory": memory.reshape(B, -1, memory.shape[-1]),
            "features": features,
        }

    def track_step(
        self,
        backbone_features: mx.array,
        memory_bank: Optional[List[mx.array]] = None,
        prompt_points=None,
        prompt_boxes=None,
        prompt_masks=None,
        multimask_output: bool = False,
    ) -> Dict[str, mx.array]:
        """Run one tracking step using propagation FPN."""
        prop_fpn = self._get_tracker_features(backbone_features)
        features = prop_fpn[2]
        high_res = [prop_fpn[0], prop_fpn[1]] if len(prop_fpn) > 1 else None

        return self.tracker_model.track_step(
            current_features=features,
            memory_bank=memory_bank,
            prompt_points=prompt_points,
            prompt_boxes=prompt_boxes,
            prompt_masks=prompt_masks,
            multimask_output=multimask_output,
            high_res_features=high_res,
        )

    def detect(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        boxes: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        return self.detector_model(
            pixel_values,
            input_ids,
            attention_mask,
            boxes,
            inputs_embeds=inputs_embeds,
        )

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Alias for get_text_features (SAM 3 generate.py compat)."""
        return self.get_text_features(input_ids, attention_mask)

    def get_text_features(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        inputs_embeds = self.detector_model.get_input_embeddings(
            input_ids, attention_mask
        )
        mx.eval(inputs_embeds)
        return inputs_embeds, attention_mask

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> Dict[str, mx.array]:
        if input_ids is not None:
            return self.detect(
                pixel_values, input_ids, attention_mask, kwargs.get("boxes")
            )
        return {"features": self.detector_model.vision_encoder(pixel_values)}

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert weights to MLX format.

        Handles Conv2d/ConvTranspose2d transpositions and key remapping
        for SAM 3.1 architecture.
        """
        sanitized = {}

        conv_transpose_patterns = ["scale_layers.", "upscale_conv", "output_upscaling"]
        skip_patterns = ["memory_temporal_positional_encoding"]

        # Mask embed: sequential indices → named convs
        # 0 → conv1, 1 → layer_norm1, 3 → conv2, 4 → layer_norm2, 6 → conv3
        mask_embed_remap = {
            "mask_embed.0.": "mask_embed.conv1.",
            "mask_embed.1.": "mask_embed.layer_norm1.",
            "mask_embed.3.": "mask_embed.conv2.",
            "mask_embed.4.": "mask_embed.layer_norm2.",
            "mask_embed.6.": "mask_embed.conv3.",
        }

        for key, value in weights.items():
            # Remap mask_embed indexed keys to named keys
            for old, new in mask_embed_remap.items():
                if old in key:
                    key = key.replace(old, new)
                    break

            # Remap memory_fuser norm → layer_norm
            if "memory_fuser" in key and ".norm." in key:
                key = key.replace(".norm.", ".layer_norm.")

            # Remap mask_downsampler.layers.4.conv → final_conv
            if "mask_downsampler.layers.4.conv." in key:
                key = key.replace(
                    "mask_downsampler.layers.4.conv.",
                    "mask_downsampler.final_conv.",
                )

            if value.ndim == 4:
                if any(p in key for p in skip_patterns):
                    sanitized[key] = value
                    continue

                is_conv_transpose = any(p in key for p in conv_transpose_patterns)
                if is_conv_transpose:
                    value = value.transpose(1, 2, 3, 0)
                else:
                    value = value.transpose(0, 2, 3, 1)

            sanitized[key] = value

        return sanitized

    @staticmethod
    def quant_predicate(path: str, module) -> bool:
        """Control quantization — same logic as SAM 3."""
        if any(
            k in path
            for k in [
                "conv",
                "depthwise",
                "mask_downsample",
                "pixel_decoder",
                "instance_projection",
                "semantic_projection",
                "fpn_layers",
                "patch_embeddings",
            ]
        ):
            return False
        if any(
            k in path
            for k in [
                "query_embed",
                "reference_points",
                "presence_token",
                "label_embed",
                "cls_embed",
                "point_embed",
                "not_a_point",
                "no_mask_embed",
                "no_memory",
                "no_object",
                "iou_token",
                "mask_tokens",
                "obj_score_token",
                "shared_embedding",
                "shared_image_embedding",
                "occlusion_spatial",
                "memory_temporal",
                "position_embedding",
                "output_valid_embed",
                "output_invalid_embed",
                "no_obj_embed_spatial",
                "image_pe_layer",
                "interactivity_no_mem",
            ]
        ):
            return False
        if hasattr(module, "weight"):
            shape = module.weight.shape
            if any(d % 64 != 0 for d in shape):
                return False
        return True
