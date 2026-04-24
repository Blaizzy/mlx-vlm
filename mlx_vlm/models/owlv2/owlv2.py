"""OWLv2 open-vocabulary object detection model."""

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .text_encoder import TextTransformer
from .vision import VisionTransformer


class ClassPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, projection_dim)
        # Predict per-token shift and scale from image embeddings
        self.logit_shift = nn.Linear(hidden_size, 1)
        self.logit_scale = nn.Linear(hidden_size, 1)
        self.elu = nn.ELU()

    def __call__(self, image_embeds: mx.array, query_embeds: mx.array) -> mx.array:
        # image_embeds: (B, N, hidden_size)
        # query_embeds: (B, num_queries, projection_dim) - already projected
        image_class_embeds = self.dense0(image_embeds)  # (B, N, projection_dim)

        # Normalize
        image_class_embeds = image_class_embeds / (
            mx.linalg.norm(image_class_embeds, axis=-1, keepdims=True) + 1e-6
        )
        query_embeds = query_embeds / (
            mx.linalg.norm(query_embeds, axis=-1, keepdims=True) + 1e-6
        )

        # Compute similarity logits
        # (B, N, projection_dim) @ (B, projection_dim, num_queries) -> (B, N, num_queries)
        pred_logits = mx.matmul(image_class_embeds, query_embeds.transpose(0, 2, 1))

        # Apply predicted per-token shift and scale
        logit_shift = self.logit_shift(image_embeds)  # (B, N, 1)
        logit_scale = self.elu(self.logit_scale(image_embeds)) + 1  # (B, N, 1)
        pred_logits = (pred_logits + logit_shift) * logit_scale

        return pred_logits


class BoxPredictionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, 4)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        # Returns raw box predictions (no sigmoid)
        out = self.gelu(self.dense0(image_embeds))
        out = self.gelu(self.dense1(out))
        return self.dense2(out)


class ObjectnessHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, 1)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        # (B, N, D) -> (B, N, 1)
        out = self.gelu(self.dense0(image_embeds))
        out = self.gelu(self.dense1(out))
        return self.dense2(out)


class Model(nn.Module):
    ModelConfig = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        text_config = config.text_config

        self.vision_model = VisionTransformer(vision_config)
        self.text_model = TextTransformer(text_config)

        self.visual_projection = nn.Linear(
            vision_config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            text_config.hidden_size, config.projection_dim, bias=False
        )

        self.logit_scale = mx.array(0.0)

        # Detection heads
        self.num_patches_per_side = vision_config.image_size // vision_config.patch_size
        self.class_head = ClassPredictionHead(
            vision_config.hidden_size, config.projection_dim
        )
        self.box_head = BoxPredictionHead(vision_config.hidden_size)
        self.objectness_head = ObjectnessHead(vision_config.hidden_size)

        self.layer_norm = nn.LayerNorm(
            vision_config.hidden_size, eps=vision_config.layer_norm_eps
        )

        # Precompute box bias (inverse-sigmoid of grid positions)
        self._box_bias = self._compute_box_bias(
            self.num_patches_per_side, self.num_patches_per_side
        )

    @staticmethod
    def _compute_box_bias(num_patches_h: int, num_patches_w: int) -> mx.array:
        # Grid coordinates: (1, 2, ..., N) / N for each axis
        x_coords = mx.arange(1, num_patches_w + 1, dtype=mx.float32) / num_patches_w
        y_coords = mx.arange(1, num_patches_h + 1, dtype=mx.float32) / num_patches_h
        xx, yy = mx.meshgrid(x_coords, y_coords, indexing="xy")
        box_coords = mx.stack([xx, yy], axis=-1).reshape(-1, 2)
        box_coords = mx.clip(box_coords, 0.0, 1.0)

        # Inverse sigmoid of grid positions
        coord_bias = mx.log(box_coords + 1e-4) - mx.log(1.0 - box_coords + 1e-4)

        # Box size bias: each patch covers 1/N of the image
        box_size = mx.ones_like(box_coords)
        box_size = box_size.at[..., 0].add(0)  # already 1.0
        box_size = mx.stack(
            [
                mx.full((num_patches_h * num_patches_w,), 1.0 / num_patches_w),
                mx.full((num_patches_h * num_patches_w,), 1.0 / num_patches_h),
            ],
            axis=-1,
        )
        size_bias = mx.log(box_size + 1e-4) - mx.log(1.0 - box_size + 1e-4)

        return mx.concatenate([coord_bias, size_bias], axis=-1)  # (N, 4)

    def encode_text(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        text_output = self.text_model(input_ids, attention_mask=attention_mask)
        # Pool at end-of-text (argmax of input_ids gives EOS position)
        eos_idx = mx.argmax(input_ids, axis=-1)  # (B,)
        pooled = text_output[mx.arange(text_output.shape[0]), eos_idx]  # (B, D_text)
        return self.text_projection(pooled)

    def encode_image(self, pixel_values: mx.array) -> mx.array:
        vision_output = self.vision_model(pixel_values)  # (B, 1+N, D_vision)

        # Broadcast CLS token and multiply with patch tokens
        cls_token = vision_output[:, :1, :]  # (B, 1, D)
        patch_tokens = vision_output[:, 1:, :]  # (B, N, D)
        cls_broadcast = mx.broadcast_to(cls_token, patch_tokens.shape)
        image_embeds = patch_tokens * cls_broadcast

        image_embeds = self.layer_norm(image_embeds)
        return image_embeds

    def __call__(
        self,
        pixel_values: mx.array,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        # Encode text queries
        query_embeds = self.encode_text(
            input_ids, attention_mask
        )  # (nq, projection_dim)
        if query_embeds.ndim == 2:
            query_embeds = query_embeds[None]  # (1, nq, projection_dim)

        # Encode image
        image_embeds = self.encode_image(pixel_values)  # (B, N, D_vision)

        # Detection heads
        pred_logits = self.class_head(image_embeds, query_embeds)  # (B, N, nq)

        # Box prediction: raw output + bias -> sigmoid
        pred_boxes = self.box_head(image_embeds)  # (B, N, 4) raw
        pred_boxes = mx.sigmoid(
            pred_boxes + self._box_bias
        )  # (B, N, 4) cxcywh in [0,1]

        # Objectness (detached in HF, doesn't matter for inference)
        objectness_logits = self.objectness_head(image_embeds)  # (B, N, 1)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "objectness_logits": objectness_logits,
        }

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for k, v in weights.items():
            # Remove "owlv2." prefix if present
            if k.startswith("owlv2."):
                k = k[len("owlv2.") :]

            # Skip position_ids (buffers, not parameters)
            if "position_ids" in k:
                continue

            # Transpose Conv2d weights: PyTorch (out, in, kH, kW) -> MLX (out, kH, kW, in)
            if "patch_embedding.weight" in k and v.ndim == 4:
                v = v.transpose(0, 2, 3, 1)

            sanitized[k] = v
        return sanitized

    @staticmethod
    def quant_predicate(path: str, module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if any(
            s in path
            for s in [
                "embeddings",
                "class_head",
                "box_head",
                "objectness_head",
                "projection",
                "layer_norm",
            ]
        ):
            return False
        dims = getattr(module, "weight", None)
        if dims is not None and all(d % 64 == 0 for d in dims.shape):
            return True
        return False
