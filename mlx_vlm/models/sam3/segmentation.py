"""SAM3 Segmentation Head and Dot Product Scoring.

Weight keys:
    detector_model.mask_decoder.*
    detector_model.dot_product_scoring.*
"""

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import DetectorMaskDecoderConfig
from .encoder import MultiheadAttention


class PixelDecoder(nn.Module):
    """FPN-like upsampling decoder matching HF Sam3PixelDecoder.

    Iterates from coarsest to finest, upsampling to match each level's size.
    """

    def __init__(self, hidden_size: int, num_upsampling_stages: int = 3):
        super().__init__()
        self.conv_layers = [
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
            for _ in range(num_upsampling_stages)
        ]
        self.norms = [
            nn.GroupNorm(8, hidden_size) for _ in range(num_upsampling_stages)
        ]

    def __call__(self, features: List[mx.array]) -> mx.array:
        """
        Args:
            features: list of (B, H_i, W_i, D) from LOW to HIGH resolution
                      e.g. [72x72, 144x144, 288x288] (coarsest first when reversed from FPN)
                      Actually: features[-1] is coarsest, features[0] is finest
        Returns:
            (B, H_finest, W_finest, D)
        """
        # Start from coarsest (last in list)
        x = features[-1]

        # Iterate from coarse to fine (skip the coarsest which we started with)
        for i, backbone_feat in enumerate(reversed(features[:-1])):
            target_h, target_w = backbone_feat.shape[1], backbone_feat.shape[2]
            # Upsample to match backbone feature size using nearest interpolation
            x = self._interpolate(x, target_h, target_w)
            # Add skip connection
            x = x + backbone_feat
            # Conv + GroupNorm + ReLU
            x = self.conv_layers[i](x)
            x = self.norms[i](x)
            x = nn.relu(x)

        return x

    def _interpolate(self, x: mx.array, target_h: int, target_w: int) -> mx.array:
        """Nearest-neighbor interpolation to exact target size."""
        B, H, W, C = x.shape
        if H == target_h and W == target_w:
            return x
        up = nn.Upsample(scale_factor=(target_h / H, target_w / W), mode="nearest")
        return up(x)


class MaskEmbedder(nn.Module):
    """3-layer MLP that projects query features to mask embedding space."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


class MaskDecoder(nn.Module):
    """Segmentation head matching HF Sam3MaskDecoder."""

    def __init__(self, config: DetectorMaskDecoderConfig):
        super().__init__()
        d = config.hidden_size

        self.pixel_decoder = PixelDecoder(d, config.num_upsampling_stages)
        self.mask_embedder = MaskEmbedder(d)

        self.prompt_cross_attn = MultiheadAttention(d, config.num_attention_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.semantic_projection = nn.Conv2d(d, 1, kernel_size=1)
        self.instance_projection = nn.Conv2d(d, d, kernel_size=1)

    def __call__(
        self,
        obj_queries: mx.array,
        backbone_features: List[mx.array],
        encoder_hidden_states: Optional[mx.array] = None,
        prompt_features: Optional[mx.array] = None,
        prompt_mask: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        """
        Args:
            obj_queries: (B, Q, D) decoder output queries
            backbone_features: list of (B, H_i, W_i, D) multi-scale FPN features
            encoder_hidden_states: (B, HW, D) DETR encoder output
            prompt_features: (B, P, D) for cross-attention
            prompt_mask: (B, P)
        """
        # Cross-attention: encoder states attend to prompt (pre-norm style)
        if prompt_features is not None and encoder_hidden_states is not None:
            residual = encoder_hidden_states
            normed = self.prompt_cross_attn_norm(encoder_hidden_states)
            cross_mask = None
            if prompt_mask is not None:
                cross_mask = (
                    1 - prompt_mask[:, None, None, :].astype(mx.float32)
                ) * -1e9
            attn_out = self.prompt_cross_attn(
                normed, prompt_features, prompt_features, mask=cross_mask
            )
            encoder_hidden_states = residual + attn_out

        # Build pixel embeddings: replace finest backbone feature with encoder output
        feats_for_fpn = list(backbone_features)
        if encoder_hidden_states is not None:
            # Reshape encoder output to spatial format matching finest backbone feature
            finest = feats_for_fpn[-1]  # (B, H, W, D) - the 72x72 level
            B, H, W, D = finest.shape
            spatial_dim = H * W
            encoder_visual = encoder_hidden_states[:, :spatial_dim, :]  # (B, HW, D)
            encoder_visual = encoder_visual.reshape(B, H, W, D)
            feats_for_fpn[-1] = encoder_visual

        # Process through pixel decoder
        pixel_embed = self.pixel_decoder(feats_for_fpn)  # (B, H_out, W_out, D)

        # Instance projection (1x1 conv)
        instance_embed = self.instance_projection(pixel_embed)  # (B, H, W, D)

        # Predict masks: dot product of query embeddings with pixel embeddings
        # einsum("bqc,bhwc->bqhw")
        mask_embeddings = self.mask_embedder(obj_queries)  # (B, Q, D)
        B, H, W, D = instance_embed.shape
        instance_flat = instance_embed.reshape(B, H * W, D)  # (B, HW, D)
        pred_masks = mx.matmul(
            mask_embeddings, instance_flat.transpose(0, 2, 1)
        )  # (B, Q, HW)
        pred_masks = pred_masks.reshape(B, -1, H, W)

        # Semantic segmentation
        semantic_seg = self.semantic_projection(pixel_embed)  # (B, H, W, 1)
        semantic_seg = semantic_seg.transpose(0, 3, 1, 2)  # (B, 1, H, W)

        return {
            "pred_masks": pred_masks,
            "semantic_seg": semantic_seg,
        }


class TextScoringMLP(nn.Module):
    """2-layer MLP for text scoring (residual done externally)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size * 8)
        self.layer2 = nn.Linear(hidden_size * 8, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer2(nn.relu(self.layer1(x)))


class DotProductScoring(nn.Module):
    """Dot-product classifier matching HF Sam3DotProductScoring."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.text_mlp = TextScoringMLP(hidden_size)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size)
        self.scale = 1.0 / (hidden_size**0.5)
        self.clamp_max_val = 12.0

    def __call__(
        self,
        hs: mx.array,
        inputs_embeds: mx.array,
        text_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            hs: (L, B, Q, D) intermediate hidden states from decoder
            inputs_embeds: (B, T, D)
            text_mask: (B, T) where 1=valid, 0=padding
        Returns:
            scores: (L, B, Q, 1)
        """
        orig_text = inputs_embeds
        text_processed = self.text_mlp(inputs_embeds) + orig_text
        text_processed = self.text_mlp_out_norm(text_processed)

        if text_mask is not None:
            is_valid = text_mask[..., None].astype(mx.float32)
            num_valid = mx.maximum(is_valid.sum(axis=1), 1.0)
            pooled_text = (text_processed * is_valid).sum(axis=1) / num_valid
        else:
            pooled_text = text_processed.mean(axis=1)

        proj_text = self.text_proj(pooled_text)  # (B, D)
        proj_queries = self.query_proj(hs)  # (L, B, Q, D)

        scores = mx.matmul(proj_queries, proj_text[None, :, :, None])
        scores = scores * self.scale
        scores = mx.clip(scores, -self.clamp_max_val, self.clamp_max_val)

        return scores
