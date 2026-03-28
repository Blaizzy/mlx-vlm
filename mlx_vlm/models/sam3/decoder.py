"""SAM3 DETR Transformer Decoder matching HF Sam3DetrDecoder.

Weight keys: detector_model.detr_decoder.*
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import DETRDecoderConfig
from .encoder import MLP, MultiheadAttention


class DETRDecoderLayer(nn.Module):
    """Single DETR decoder layer: post-norm with 3 attention stages + MLP.

    Order (matching HF Sam3DetrDecoderLayer):
    1. Self-attn(q=hidden+pos, k=hidden+pos, v=hidden) → residual → LayerNorm
    2. Text cross-attn(q=hidden+pos, k=text, v=text) → residual → LayerNorm
    3. Vision cross-attn(q=hidden+pos, k=vis+vis_pos, v=vis) → residual → LayerNorm
    4. MLP → residual → LayerNorm
    """

    def __init__(self, config: DETRDecoderConfig):
        super().__init__()
        d = config.hidden_size

        self.self_attn = MultiheadAttention(d, config.num_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.text_cross_attn = MultiheadAttention(d, config.num_attention_heads)
        self.text_cross_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.vision_cross_attn = MultiheadAttention(d, config.num_attention_heads)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.mlp = MLP(d, config.intermediate_size, config.hidden_act)
        self.mlp_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        query_pos: mx.array,
        inputs_embeds: mx.array,
        vision_features: mx.array,
        vision_pos_encoding: mx.array,
        text_cross_attn_mask: Optional[mx.array] = None,
        vision_cross_attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: (B, Q+1, D) queries + presence token at index 0
            query_pos: (B, Q+1, D) position embeddings (0 for presence token)
            inputs_embeds: (B, T, D)
            vision_features: (B, HW, D)
            vision_pos_encoding: (B, HW, D)
            text_cross_attn_mask: (B, 1, Q+1, T) cross-attention mask
            vision_cross_attn_mask: (B, H, Q+1, HW) RPB bias
        """
        # 1. Self-attention
        residual = hidden_states
        qk = hidden_states + query_pos
        attn_out = self.self_attn(qk, qk, hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + attn_out)

        # 2. Text cross-attention
        residual = hidden_states
        q_with_pos = hidden_states + query_pos
        attn_out = self.text_cross_attn(
            q_with_pos, inputs_embeds, inputs_embeds, mask=text_cross_attn_mask
        )
        hidden_states = self.text_cross_attn_layer_norm(residual + attn_out)

        # 3. Vision cross-attention (with RPB bias)
        residual = hidden_states
        q_with_pos = hidden_states + query_pos
        k_with_pos = vision_features + vision_pos_encoding
        attn_out = self.vision_cross_attn(
            q_with_pos, k_with_pos, vision_features, mask=vision_cross_attn_mask
        )
        hidden_states = self.vision_cross_attn_layer_norm(residual + attn_out)

        # 4. MLP
        residual = hidden_states
        mlp_out = self.mlp(hidden_states)
        hidden_states = self.mlp_layer_norm(residual + mlp_out)

        return hidden_states


class BoxHead(nn.Module):
    """3-layer MLP for box regression."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 4)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        return self.layer3(x)


class PresenceHead(nn.Module):
    """3-layer MLP for presence score."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        return self.layer3(x)


class RefPointHead(nn.Module):
    """Maps sine-encoded reference points to query position encoding."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # Input: 4 * num_pos_feats (from encode_boxes) = 4 * 128 = 512
        self.layer1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.layer2(nn.relu(self.layer1(x))))


class BoxRPBEmbed(nn.Module):
    """Box Relative Position Bias embedding."""

    def __init__(self, num_heads: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_heads)

    def __call__(self, deltas: mx.array) -> mx.array:
        return self.layer2(nn.relu(self.layer1(deltas)))


class SinePositionEmbeddingForBoxes:
    """Sine position encoding for box coordinates (used in decoder conditioning)."""

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: float = 10000.0,
        scale: float = 2 * math.pi,
    ):
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def encode_boxes(self, boxes: mx.array) -> mx.array:
        """Encode 4D box coords (cx, cy, w, h) into sine/cos embeddings.

        Args:
            boxes: (B, Q, 4) box coordinates
        Returns:
            (B, Q, num_pos_feats*4) sine/cosine embeddings
        """
        dim_t = mx.arange(self.num_pos_feats).astype(mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Scale each coordinate
        x = boxes[..., 0:1] * self.scale  # (B, Q, 1)
        y = boxes[..., 1:2] * self.scale
        w = boxes[..., 2:3] * self.scale
        h = boxes[..., 3:4] * self.scale

        encodings = []
        for coord in [y, x, w, h]:
            pos = coord / dim_t  # (B, Q, D)
            sin_enc = mx.sin(pos[..., 0::2])
            cos_enc = mx.cos(pos[..., 1::2])
            enc = mx.stack([sin_enc, cos_enc], axis=-1)
            enc = enc.reshape(*enc.shape[:-2], -1)
            encodings.append(enc)

        return mx.concatenate(encodings, axis=-1)  # (B, Q, 4*D)


class DETRDecoder(nn.Module):
    """DETR Transformer Decoder matching HF Sam3DetrDecoder."""

    def __init__(self, config: DETRDecoderConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        num_q = config.num_queries
        num_heads = config.num_attention_heads

        self.layers = [DETRDecoderLayer(config) for _ in range(config.num_layers)]
        self.output_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        # Learned queries and reference points
        self.query_embed = nn.Embedding(num_q, d)
        self.reference_points = nn.Embedding(num_q, 4)

        # Presence token
        self.presence_token = nn.Embedding(1, d)
        self.presence_head = PresenceHead(d)
        self.presence_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.clamp_presence_logit_max_val = 10.0

        # Box refinement
        self.box_head = BoxHead(d)
        self.ref_point_head = RefPointHead(d)

        # Box RPB
        self.box_rpb_embed_x = BoxRPBEmbed(num_heads, d)
        self.box_rpb_embed_y = BoxRPBEmbed(num_heads, d)

        # Position encoding for sine box embedding
        self._pos_enc = SinePositionEmbeddingForBoxes(num_pos_feats=d // 2)

    def __call__(
        self,
        vision_features: mx.array,
        inputs_embeds: mx.array,
        vision_pos_encoding: mx.array,
        text_mask: Optional[mx.array] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, Optional[mx.array]]:
        """
        Returns:
            intermediate_hs: (L, B, Q, D) hidden states per layer
            intermediate_boxes: (L, B, Q, 4) reference boxes per layer
            presence_logits: (L, B, 1)
        """
        B = vision_features.shape[0]
        num_q = self.config.num_queries
        d = self.config.hidden_size

        query_embeds = mx.broadcast_to(self.query_embed.weight[None], (B, num_q, d))
        reference_boxes = mx.sigmoid(
            mx.broadcast_to(self.reference_points.weight[None], (B, num_q, 4))
        )
        presence_token = mx.broadcast_to(self.presence_token.weight[None], (B, 1, d))

        # Presence token at position 0
        hidden_states = mx.concatenate(
            [presence_token, query_embeds], axis=1
        )  # (B, Q+1, D)

        # Text cross-attention mask
        text_cross_mask = None
        if text_mask is not None:
            text_cross_mask = (
                1 - text_mask[:, None, None, :].astype(mx.float32)
            ) * -1e9

        intermediate_hs = []
        intermediate_boxes = []
        intermediate_presence = []

        for layer in self.layers:
            # Generate query position from sine-encoded reference boxes
            query_sine_embed = self._pos_enc.encode_boxes(
                reference_boxes
            )  # (B, Q, 4*D/2=2D)
            query_pos = self.ref_point_head(query_sine_embed)  # (B, Q, D)

            # Pad query_pos with zeros for presence token at position 0
            query_pos_padded = mx.concatenate(
                [mx.zeros((B, 1, d)), query_pos], axis=1
            )  # (B, Q+1, D)

            # Compute RPB
            vision_cross_mask = None
            if spatial_shape is not None:
                rpb = self._compute_rpb(reference_boxes, spatial_shape)  # (B, H, Q, HW)
                # Pad for presence token: (B, H, Q+1, HW)
                rpb_padded = mx.concatenate(
                    [mx.zeros((B, rpb.shape[1], 1, rpb.shape[3])), rpb], axis=2
                )
                vision_cross_mask = rpb_padded

            hidden_states = layer(
                hidden_states,
                query_pos=query_pos_padded,
                inputs_embeds=inputs_embeds,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_cross_attn_mask=text_cross_mask,
                vision_cross_attn_mask=vision_cross_mask,
            )

            # Extract query hidden states (skip presence token at index 0)
            query_hs = hidden_states[:, 1:]
            query_hs_normed = self.output_layer_norm(query_hs)

            # Box refinement
            delta = self.box_head(query_hs_normed)
            new_ref = mx.sigmoid(inverse_sigmoid(reference_boxes) + delta)
            reference_boxes = mx.stop_gradient(new_ref)

            intermediate_hs.append(query_hs_normed)
            intermediate_boxes.append(new_ref)

            # Presence logit from presence token (index 0)
            pres_hidden = hidden_states[:, :1]
            pres_logit = self.presence_head(
                self.presence_layer_norm(pres_hidden)
            ).squeeze(-1)
            pres_logit = mx.clip(
                pres_logit,
                -self.clamp_presence_logit_max_val,
                self.clamp_presence_logit_max_val,
            )
            intermediate_presence.append(pres_logit)

            mx.eval(hidden_states, reference_boxes)

        return (
            mx.stack(intermediate_hs),  # (L, B, Q, D)
            mx.stack(intermediate_boxes),  # (L, B, Q, 4)
            mx.stack(intermediate_presence),  # (L, B, 1)
        )

    def _compute_rpb(
        self,
        reference_boxes: mx.array,
        spatial_shape: Tuple[int, int],
    ) -> mx.array:
        """Compute Box Relative Position Bias (RPB) matrix.

        Args:
            reference_boxes: (B, Q, 4) in cxcywh sigmoid space
            spatial_shape: (height, width)
        Returns:
            (B, num_heads, Q, HW)
        """
        height, width = spatial_shape
        B, Q, _ = reference_boxes.shape

        # Convert cxcywh to xyxy
        cx, cy, w, h = (
            reference_boxes[..., 0],
            reference_boxes[..., 1],
            reference_boxes[..., 2],
            reference_boxes[..., 3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = mx.stack([x1, y1, x2, y2], axis=-1)  # (B, Q, 4)

        # Coordinate grids normalized to [0, 1]
        coords_h = (mx.arange(height).astype(mx.float32) + 0.5) / height
        coords_w = (mx.arange(width).astype(mx.float32) + 0.5) / width

        # Compute deltas: coords - box boundaries
        # y deltas: (B*Q, HW, 2) = coords_h - [y1, y2]
        y_bounds = boxes_xyxy[..., 1::2].reshape(B * Q, 1, 2)  # (B*Q, 1, 2) = [y1, y2]
        deltas_y = coords_h.reshape(1, -1, 1) - y_bounds  # (B*Q, H, 2)
        deltas_y = deltas_y.reshape(B, Q, height, 2)

        x_bounds = boxes_xyxy[..., 0::2].reshape(B * Q, 1, 2)  # (B*Q, 1, 2) = [x1, x2]
        deltas_x = coords_w.reshape(1, -1, 1) - x_bounds  # (B*Q, W, 2)
        deltas_x = deltas_x.reshape(B, Q, width, 2)

        # Log-scale encoding
        deltas_x_log = deltas_x * 8
        deltas_x_log = (
            mx.sign(deltas_x_log) * mx.log2(mx.abs(deltas_x_log) + 1.0) / math.log2(8)
        )
        deltas_y_log = deltas_y * 8
        deltas_y_log = (
            mx.sign(deltas_y_log) * mx.log2(mx.abs(deltas_y_log) + 1.0) / math.log2(8)
        )

        # Embed deltas -> (B, Q, H/W, num_heads)
        rpb_x = self.box_rpb_embed_x(deltas_x_log)  # (B, Q, W, num_heads)
        rpb_y = self.box_rpb_embed_y(deltas_y_log)  # (B, Q, H, num_heads)

        # Combine: (B, Q, H, W, num_heads)
        rpb = rpb_y[:, :, :, None, :] + rpb_x[:, :, None, :, :]
        # Flatten spatial: (B, Q, H*W, num_heads)
        rpb = rpb.reshape(B, Q, height * width, -1)
        # Permute to (B, num_heads, Q, HW)
        rpb = rpb.transpose(0, 3, 1, 2)

        return rpb


def inverse_sigmoid(x: mx.array, eps: float = 1e-5) -> mx.array:
    x = mx.clip(x, eps, 1 - eps)
    return mx.log(x / (1 - x))
