"""SAM3 Main Model: Combines detector (DETR-based) + tracker (SAM2-based).

Weight keys:
    detector_model.*  -> self.detector_model.*
    tracker_model.*   -> self.tracker_model.*
    tracker_neck.*    -> self.tracker_neck.*
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .decoder import DETRDecoder
from .encoder import DETREncoder
from .geometry import GeometryEncoder
from .position import PositionEmbeddingSine
from .segmentation import DotProductScoring, MaskDecoder
from .text_encoder import TextEncoder
from .tracker import TrackerModel
from .vision import FPNNeck, VisionEncoder

# ---------------------------------------------------------------------------
# Detector Model
# ---------------------------------------------------------------------------


class DetectorModel(nn.Module):
    """SAM3 detection model: vision + text -> DETR -> masks.

    Weight keys: detector_model.*
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        det_cfg = config.detector_config

        # Vision encoder (backbone + FPN neck)
        self.vision_encoder = VisionEncoder(det_cfg.vision_config)

        # Text encoder (CLIP)
        self.text_encoder = TextEncoder(
            det_cfg.text_config, d_model=det_cfg.detr_encoder_config.hidden_size
        )

        # Text projection: CLIP hidden -> DETR d_model
        self.text_projection = nn.Linear(
            det_cfg.text_config.hidden_size,
            det_cfg.detr_encoder_config.hidden_size,
        )

        # DETR encoder + decoder
        self.detr_encoder = DETREncoder(det_cfg.detr_encoder_config)
        self.detr_decoder = DETRDecoder(det_cfg.detr_decoder_config)

        # Geometry encoder (for box/point prompts)
        self.geometry_encoder = GeometryEncoder(det_cfg.geometry_encoder_config)

        # Segmentation head
        self.mask_decoder = MaskDecoder(det_cfg.mask_decoder_config)

        # Scoring
        self.dot_product_scoring = DotProductScoring(
            det_cfg.detr_encoder_config.hidden_size
        )

        # Position encoding
        self._pos_enc = PositionEmbeddingSine(
            det_cfg.detr_encoder_config.hidden_size // 2
        )

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode text separately (cacheable across frames).

        Returns:
            inputs_embeds: (B, T, D) projected text features
        """
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
        """Matching HF Sam3Model.forward.

        Either pass (input_ids, attention_mask) to encode text on the fly,
        or pass pre-computed inputs_embeds to skip text encoding.
        """
        B = pixel_values.shape[0]

        # 1. Vision backbone -> multi-scale FPN features + position encodings
        fpn_features = self.vision_encoder(pixel_values)

        # Generate position encoding for each FPN level
        fpn_pos = [self._pos_enc(feat) for feat in fpn_features]

        # Drop coarsest level (scalp=1): [288, 144, 72, 36] -> [288, 144, 72]
        fpn_features_trimmed = fpn_features[:-1]
        fpn_pos_trimmed = fpn_pos[:-1]

        # 2. Text encoding (use cache if provided)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        # 3. Combine text + optional geometry prompts
        prompt = inputs_embeds
        prompt_mask = attention_mask

        # 4. DETR Encoder: uses LAST level of trimmed FPN (72x72)
        encoder_feat = fpn_features_trimmed[-1]  # (B, 72, 72, D)
        encoder_pos = fpn_pos_trimmed[-1]  # (B, 72, 72, D)

        B, H, W, D = encoder_feat.shape
        # Flatten: (B, H, W, D) -> (B, HW, D)
        src = encoder_feat.reshape(B, H * W, D)
        pos_flat = encoder_pos.reshape(B, H * W, D)

        encoded = self.detr_encoder(src, pos_flat, prompt, prompt_mask)

        # 5. DETR Decoder
        hs, ref_boxes, presence_logits = self.detr_decoder(
            vision_features=encoded,
            inputs_embeds=prompt,
            vision_pos_encoding=pos_flat,
            text_mask=prompt_mask,
            spatial_shape=(H, W),
        )

        # 6. Box refinement (done inside decoder now, but HF also applies box_head
        #    on intermediate_hidden_states outside - the decoder already does this)
        # Convert cxcywh to xyxy for final output
        pred_boxes_cxcywh = ref_boxes[-1]  # (B, Q, 4)
        cx, cy, w, h = (
            pred_boxes_cxcywh[..., 0],
            pred_boxes_cxcywh[..., 1],
            pred_boxes_cxcywh[..., 2],
            pred_boxes_cxcywh[..., 3],
        )
        pred_boxes_xyxy = mx.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )

        # 7. Classification: dot-product scoring on all intermediate hidden states
        all_pred_logits = self.dot_product_scoring(hs, prompt, prompt_mask)

        pred_logits = all_pred_logits[-1]  # (B, Q, 1) last layer
        presence = presence_logits[-1]  # (B, 1)

        # 8. Mask prediction (pass encoder hidden states to replace finest backbone feature)
        last_hs = hs[-1]
        seg_out = self.mask_decoder(
            last_hs,
            list(fpn_features_trimmed),
            encoder_hidden_states=encoded,
            prompt_features=prompt,
            prompt_mask=prompt_mask,
        )

        return {
            "pred_logits": pred_logits.squeeze(-1),  # (B, Q)
            "pred_boxes": pred_boxes_xyxy,  # (B, Q, 4) xyxy
            "pred_masks": seg_out["pred_masks"],
            "presence_logits": presence,  # (B, 1)
            "semantic_seg": seg_out.get("semantic_seg"),
            "intermediate_hidden_states": hs,
            "encoder_hidden_states": encoded,
        }


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """SAM3 full model: detector + tracker.

    Weight keys:
        detector_model.*  -> self.detector_model.*
        tracker_model.*   -> self.tracker_model.*
        tracker_neck.*    -> self.tracker_neck.*
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Detector (image segmentation)
        self.detector_model = DetectorModel(config)

        # Tracker (video tracking)
        self.tracker_model = TrackerModel(config.tracker_config)

        # Tracker FPN neck (separate from detector's neck)
        self.tracker_neck = FPNNeck(config.tracker_config.vision_config)

    @staticmethod
    def quant_predicate(path: str, module) -> bool:
        """Control which layers get quantized.

        Skip:
        - Vision encoder (keep full precision for accuracy)
        - Small embeddings (query_embed, reference_points, presence_token, etc.)
        - Conv layers (not supported by quantization)
        - Layers with dimensions not divisible by 64
        """
        # Skip conv layers (not supported by QuantizedLinear)
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
        # Skip small/structural embeddings
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
            ]
        ):
            return False
        # Skip layers with weight dims not divisible by 64
        if hasattr(module, "weight"):
            shape = module.weight.shape
            if any(d % 64 != 0 for d in shape):
                return False
        return True

    def detect(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        boxes: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        """Run detection on a single image.

        Pass inputs_embeds (from get_input_embeddings) to skip text encoding.
        """
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
        """Encode text once, reuse across frames.

        Returns:
            (inputs_embeds, attention_mask) tuple for passing to detect().
        """
        inputs_embeds = self.detector_model.get_input_embeddings(
            input_ids, attention_mask
        )
        mx.eval(inputs_embeds)
        return inputs_embeds, attention_mask

    def track_init(
        self,
        backbone_features: mx.array,
        detection_masks: mx.array,
    ) -> Dict[str, mx.array]:
        """Initialize tracker with detection results."""
        tracker_fpn = self.tracker_neck(backbone_features)
        features = tracker_fpn[2]  # 1x scale
        B, H, W, D = features.shape

        # Encode initial memory from detection masks
        mask_input = detection_masks[:, :1].transpose(0, 2, 3, 1)  # (B, H, W, 1)
        memory = self.tracker_model.memory_encoder(features, mask_input)

        return {
            "memory": memory.reshape(B, -1, memory.shape[-1]),
            "features": features,
        }

    def track_step(
        self,
        backbone_features: mx.array,
        memory_bank: List[mx.array],
        prompt_points: Optional[Tuple[mx.array, mx.array]] = None,
        prompt_boxes: Optional[mx.array] = None,
        prompt_masks: Optional[mx.array] = None,
        multimask_output: bool = False,
    ) -> Dict[str, mx.array]:
        """Run one tracking step."""
        tracker_fpn = self.tracker_neck(backbone_features)
        features = tracker_fpn[2]

        # Get high-res features for skip connections
        high_res = [tracker_fpn[0], tracker_fpn[1]] if len(tracker_fpn) > 1 else None

        return self.tracker_model.track_step(
            current_features=features,
            memory_bank=memory_bank,
            prompt_points=prompt_points,
            prompt_boxes=prompt_boxes,
            prompt_masks=prompt_masks,
            multimask_output=multimask_output,
            high_res_features=high_res,
        )

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> Dict[str, mx.array]:
        """Default forward: run detection."""
        if input_ids is not None:
            return self.detect(
                pixel_values, input_ids, attention_mask, kwargs.get("boxes")
            )
        # If no text, return backbone features (for tracker use)
        return {"features": self.detector_model.vision_encoder(pixel_values)}

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert HuggingFace PyTorch weights to MLX format.

        Main conversions:
        1. Conv2d: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
        2. ConvTranspose2d: PyTorch [in, out, H, W] -> MLX [out, H, W, in]
        """
        sanitized = {}

        # Patterns for ConvTranspose2d weights (need transpose(1,2,3,0))
        conv_transpose_patterns = [
            "scale_layers.",  # FPN upsampling layers
            "upscale_conv",  # Tracker mask decoder upscaling
        ]

        # Patterns for special 4D weights that are NOT ConvTranspose2d
        # (regular Conv2d: transpose(0,2,3,1))
        conv2d_patterns = [
            "projection.weight",  # Patch embedding
            "proj1.weight",  # FPN 1x1 conv
            "proj2.weight",  # FPN 3x3 conv
            ".conv.",  # Generic conv layers
            "conv_layers.",  # Pixel decoder convs
            "instance_projection.",  # 1x1 conv
            "semantic_projection.",  # 1x1 conv
            "feature_projection.",  # Memory encoder 1x1
            "final_conv.",  # Mask downsampler final
            "conv_s0.",  # Tracker skip conv
            "conv_s1.",  # Tracker skip conv
            "depthwise_conv.",  # CXBlock depthwise
            "mask_downsample.",  # Mask downsampling
            "conv1.",  # Mask embed convs
            "conv2.",  # Mask embed convs
            "conv3.",  # Mask embed convs
            "boxes_pool_project.",  # Geometry encoder Conv2d
        ]

        # 4D parameters that are NOT convolution weights (skip transposition)
        skip_transpose_patterns = [
            "memory_temporal_positional_encoding",
        ]

        for key, value in weights.items():
            if value.ndim == 4:
                # Skip non-conv 4D parameters
                if any(p in key for p in skip_transpose_patterns):
                    sanitized[key] = value
                    continue

                # 4D tensor: either Conv2d or ConvTranspose2d
                is_conv_transpose = any(p in key for p in conv_transpose_patterns)

                if is_conv_transpose:
                    # PyTorch ConvTranspose2d: (in_ch, out_ch, kH, kW)
                    # MLX ConvTranspose2d: (out_ch, kH, kW, in_ch)
                    value = value.transpose(1, 2, 3, 0)
                else:
                    # PyTorch Conv2d: (out_ch, in_ch, kH, kW)
                    # MLX Conv2d: (out_ch, kH, kW, in_ch)
                    value = value.transpose(0, 2, 3, 1)

            sanitized[key] = value

        return sanitized
