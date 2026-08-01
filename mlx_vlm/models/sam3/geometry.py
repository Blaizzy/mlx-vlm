"""SAM3 Geometry Encoder for box/point/mask visual prompts.

Weight keys: detector_model.geometry_encoder.*

Ported to match HF ``Sam3GeometryEncoder``: box prompts are encoded by
combining a direct linear projection of the box coordinates, ROI-aligned
image features pooled at the box location, and a sine positional encoding
of the box centre, plus a learned label embedding. A CLS token is appended
and the sequence is refined by transformer layers that cross-attend to the
image features.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import GeometryEncoderConfig
from .encoder import MLP, MultiheadAttention


def _sine_encode_1d(
    coords: mx.array, num_pos_feats: int, temperature: float = 10000.0
) -> mx.array:
    """1D sine/cosine positional encoding matching HF encode_1d_positions."""
    scale = 2 * math.pi
    embed = coords.astype(mx.float32) * scale
    dim_t = mx.arange(num_pos_feats).astype(mx.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos = embed[:, None] / dim_t
    pos = mx.stack([mx.sin(pos[:, 0::2]), mx.cos(pos[:, 1::2])], axis=2)
    return pos.reshape(pos.shape[0], -1)


def _bilinear_sample_grid(feat: mx.array, ys: mx.array, xs: mx.array) -> mx.array:
    """Bilinearly sample (H, W, C) features at the outer grid (ys, xs).

    Matches torchvision ``roi_align`` per-point interpolation (aligned=False):
    points outside [-1, H] x [-1, W] contribute zero.
    """
    H, W, _ = feat.shape

    valid_y = (ys >= -1.0) & (ys <= H)
    valid_x = (xs >= -1.0) & (xs <= W)

    y = mx.maximum(ys, 0.0)
    x = mx.maximum(xs, 0.0)

    yf = mx.floor(y).astype(mx.int32)
    xf = mx.floor(x).astype(mx.int32)

    at_edge_y = yf >= (H - 1)
    at_edge_x = xf >= (W - 1)

    y_low = mx.where(at_edge_y, H - 1, yf)
    x_low = mx.where(at_edge_x, W - 1, xf)
    y_high = mx.where(at_edge_y, H - 1, yf + 1)
    x_high = mx.where(at_edge_x, W - 1, xf + 1)

    ly = mx.where(at_edge_y, mx.zeros_like(y), y - yf.astype(mx.float32))
    lx = mx.where(at_edge_x, mx.zeros_like(x), x - xf.astype(mx.float32))
    hy = 1.0 - ly
    hx = 1.0 - lx

    f_ll = feat[y_low[:, None], x_low[None, :]]
    f_lh = feat[y_low[:, None], x_high[None, :]]
    f_hl = feat[y_high[:, None], x_low[None, :]]
    f_hh = feat[y_high[:, None], x_high[None, :]]

    w1 = (hy[:, None] * hx[None, :])[..., None]
    w2 = (hy[:, None] * lx[None, :])[..., None]
    w3 = (ly[:, None] * hx[None, :])[..., None]
    w4 = (ly[:, None] * lx[None, :])[..., None]

    val = w1 * f_ll + w2 * f_lh + w3 * f_hl + w4 * f_hh
    valid = (valid_y[:, None] & valid_x[None, :])[..., None]
    return val * valid.astype(val.dtype)


def roi_align(features: mx.array, boxes_xyxy: mx.array, output_size: int) -> mx.array:
    """torchvision-compatible ROI align (spatial_scale=1, aligned=False).

    Args:
        features:   (B, H, W, C) channel-last feature maps.
        boxes_xyxy: (B, N, 4) boxes in feature-map coordinates (x1, y1, x2, y2).
        output_size: pooled spatial size (roi_size x roi_size).
    Returns:
        (B, N, output_size, output_size, C) pooled features.
    """
    B, _, _, C = features.shape
    N = boxes_xyxy.shape[1]
    ph = pw = output_size

    boxes_np = np.array(boxes_xyxy, copy=False).astype(np.float32)

    out = []
    for b in range(B):
        row = []
        for i in range(N):
            x1, y1, x2, y2 = (float(v) for v in boxes_np[b, i])
            roi_w = max(x2 - x1, 1.0)
            roi_h = max(y2 - y1, 1.0)
            bin_w = roi_w / pw
            bin_h = roi_h / ph
            grid_w = max(int(math.ceil(bin_w)), 1)
            grid_h = max(int(math.ceil(bin_h)), 1)

            ph_arr = mx.arange(ph).astype(mx.float32)
            pw_arr = mx.arange(pw).astype(mx.float32)
            iy = mx.arange(grid_h).astype(mx.float32)
            ix = mx.arange(grid_w).astype(mx.float32)

            ys = y1 + ph_arr[:, None] * bin_h + (iy[None, :] + 0.5) * (bin_h / grid_h)
            xs = x1 + pw_arr[:, None] * bin_w + (ix[None, :] + 0.5) * (bin_w / grid_w)
            ys = ys.reshape(-1)
            xs = xs.reshape(-1)

            sampled = _bilinear_sample_grid(features[b], ys, xs)
            sampled = sampled.reshape(ph, grid_h, pw, grid_w, C)
            pooled = sampled.mean(axis=(1, 3))
            row.append(pooled)
        out.append(mx.stack(row, axis=0))
    return mx.stack(out, axis=0)


class GeometryEncoderLayer(nn.Module):
    """Pre-norm transformer layer matching HF Sam3GeometryEncoderLayer.

    Weight keys per layer:
        self_attn.{q,k,v,o}_proj.{weight,bias}
        cross_attn.{q,k,v,o}_proj.{weight,bias}
        layer_norm1.{weight,bias}
        layer_norm2.{weight,bias}
        layer_norm3.{weight,bias}
        mlp.fc1.{weight,bias}
        mlp.fc2.{weight,bias}
    """

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__()
        d = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.self_attn = MultiheadAttention(
            d, config.num_attention_heads, config.dropout
        )
        self.cross_attn = MultiheadAttention(
            d, config.num_attention_heads, config.dropout
        )
        self.layer_norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.mlp = MLP(d, config.intermediate_size, config.hidden_act)
        self.layer_norm3 = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def __call__(
        self,
        prompt_feats: mx.array,
        vision_feats: mx.array,
        vision_pos: mx.array,
        prompt_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = prompt_feats
        hidden = self.layer_norm1(prompt_feats)
        hidden = self.self_attn(hidden, hidden, hidden, mask=prompt_mask)
        prompt_feats = residual + hidden

        residual = prompt_feats
        hidden = self.layer_norm2(prompt_feats)
        key = vision_feats + vision_pos
        hidden = self.cross_attn(hidden, key, vision_feats)
        prompt_feats = residual + hidden

        residual = prompt_feats
        hidden = self.layer_norm3(prompt_feats)
        hidden = self.mlp(hidden)
        prompt_feats = residual + hidden

        return prompt_feats


class GeometryEncoder(nn.Module):
    """Encodes geometric prompts (boxes) into feature vectors.

    Weight keys: detector_model.geometry_encoder.*
    """

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__()
        d = config.hidden_size
        self.hidden_size = d
        self.roi_size = config.roi_size
        self.num_pos_feats = d // 2

        self.label_embed = nn.Embedding(2, d)
        self.cls_embed = nn.Embedding(1, d)

        self.boxes_direct_project = nn.Linear(4, d)
        self.boxes_pool_project = nn.Conv2d(
            d, d, kernel_size=config.roi_size, bias=True
        )
        self.boxes_pos_enc_project = nn.Linear(d + 2, d)

        self.vision_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.final_proj = nn.Linear(d, d)
        self.prompt_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.layers = [GeometryEncoderLayer(config) for _ in range(config.num_layers)]

        self.output_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def _encode_box_coordinates(
        self, cx: mx.array, cy: mx.array, w: mx.array, h: mx.array
    ) -> mx.array:
        pos_x = _sine_encode_1d(cx, self.num_pos_feats)
        pos_y = _sine_encode_1d(cy, self.num_pos_feats)
        return mx.concatenate([pos_y, pos_x, h[:, None], w[:, None]], axis=1)

    def _encode_boxes(
        self,
        boxes: mx.array,
        boxes_labels: mx.array,
        vision_features: mx.array,
    ) -> mx.array:
        B, N = boxes.shape[:2]
        H, W = vision_features.shape[1:3]
        dtype = vision_features.dtype
        boxes = boxes.astype(dtype)

        boxes_embed = self.boxes_direct_project(boxes)

        cx, cy, bw, bh = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1, y1 = cx - 0.5 * bw, cy - 0.5 * bh
        x2, y2 = cx + 0.5 * bw, cy + 0.5 * bh
        boxes_xyxy = mx.stack([x1 * W, y1 * H, x2 * W, y2 * H], axis=-1)
        sampled = roi_align(vision_features, boxes_xyxy, self.roi_size).astype(dtype)
        sampled = sampled.reshape(B * N, self.roi_size, self.roi_size, self.hidden_size)
        pooled = self.boxes_pool_project(sampled)
        pooled = pooled.reshape(B, N, self.hidden_size)
        boxes_embed = boxes_embed + pooled

        pos_enc = self._encode_box_coordinates(
            cx.reshape(-1), cy.reshape(-1), bw.reshape(-1), bh.reshape(-1)
        )
        pos_enc = pos_enc.reshape(B, N, -1).astype(dtype)
        boxes_embed = boxes_embed + self.boxes_pos_enc_project(pos_enc)

        label_embed = self.label_embed(boxes_labels.astype(mx.int32))
        return label_embed + boxes_embed

    def __call__(
        self,
        boxes: mx.array,
        box_labels: mx.array,
        box_mask: mx.array,
        img_feat_map: mx.array,
        img_pos_map: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            boxes:        (B, N, 4) normalized cxcywh box prompts
            box_labels:   (B, N) label ids (0=negative, 1=positive)
            box_mask:     (B, N) validity mask (True=valid)
            img_feat_map: (B, H, W, D) image features (finest DETR level)
            img_pos_map:  (B, H, W, D) image positional encoding
        Returns:
            (prompt_feats (B, N+1, D), prompt_mask (B, N+1))
        """
        B, H, W, D = img_feat_map.shape
        vision_feats_flat = img_feat_map.reshape(B, H * W, D)
        vision_pos_flat = img_pos_map.reshape(B, H * W, D)

        normalized_img_feats = self.vision_layer_norm(img_feat_map)

        prompt_embeds = self._encode_boxes(boxes, box_labels, normalized_img_feats)

        cls = mx.broadcast_to(self.cls_embed.weight[None], (B, 1, D))
        prompt_embeds = mx.concatenate([prompt_embeds, cls], axis=1)
        cls_mask = mx.ones((B, 1), dtype=box_mask.dtype)
        prompt_mask = mx.concatenate([box_mask, cls_mask], axis=1)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        valid = prompt_mask.astype(prompt_embeds.dtype)
        self_mask = (1.0 - valid[:, None, None, :]) * -1e9

        for layer in self.layers:
            prompt_embeds = layer(
                prompt_embeds, vision_feats_flat, vision_pos_flat, self_mask
            )

        prompt_embeds = self.output_layer_norm(prompt_embeds)
        return prompt_embeds, prompt_mask
