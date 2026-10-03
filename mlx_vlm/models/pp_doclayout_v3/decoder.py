"""PP-DocLayoutV3 decoder.

Deformable-attention decoder with iterative box refinement, a shared
class/box head (tied to the encoder query-selection heads), per-layer
mask predictions and the reading-order head (per-layer projection +
shared GlobalPointer). Mirrors transformers' PP-DocLayoutV3.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..kernels import grid_sample
from ..rt_detr_v2.transformer import inverse_sigmoid
from .config import LayoutConfig


class MLPHead(nn.Module):
    """ReLU MLP with ``layers.{i}`` keys (box/mask/query-pos heads)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.num_layers = num_layers
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = nn.relu(x)
        return x


class SelfAttention(nn.Module):
    """MHSA with position embeddings added to q,k (not v)."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x: mx.array, pos: Optional[mx.array]) -> mx.array:
        B, N, D = x.shape
        qk = x + pos if pos is not None else x
        q = (
            self.q_proj(qk)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(qk)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention (4D refs, plain 0.5/n_points scaling)."""

    def __init__(self, config: LayoutConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.decoder_attention_heads
        self.n_levels = config.num_feature_levels
        self.n_points = config.decoder_n_points
        self.head_dim = d // self.n_heads
        self.sampling_offsets = nn.Linear(
            d, self.n_heads * self.n_levels * self.n_points * 2
        )
        self.attention_weights = nn.Linear(
            d, self.n_heads * self.n_levels * self.n_points
        )
        self.value_proj = nn.Linear(d, d)
        self.output_proj = nn.Linear(d, d)

    def __call__(
        self,
        query: mx.array,
        reference_points: mx.array,
        value: mx.array,
        spatial_shapes: Tuple[Tuple[int, int], ...],
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if position_embeddings is not None:
            query = query + position_embeddings
        B, Q, D = query.shape
        n_heads, n_levels, n_points = self.n_heads, self.n_levels, self.n_points
        v = self.value_proj(value).reshape(B, value.shape[1], n_heads, self.head_dim)
        offsets = self.sampling_offsets(query).reshape(
            B, Q, n_heads, n_levels * n_points, 2
        )
        attn = mx.softmax(
            self.attention_weights(query).reshape(B, Q, n_heads, n_levels * n_points),
            axis=-1,
        )
        ref_xy = reference_points[:, :, None, :, :2]
        ref_wh = reference_points[:, :, None, :, 2:]
        loc = ref_xy + offsets / n_points * ref_wh * 0.5

        loc_levels = mx.split(loc, n_levels, axis=-2)
        sizes = [H * W for H, W in spatial_shapes]
        v_levels, start = [], 0
        for s in sizes:
            v_levels.append(v[:, start : start + s, :, :])
            start += s

        sampled = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            v_l = v_levels[lvl].reshape(B, H, W, n_heads, self.head_dim)
            v_l = v_l.transpose(0, 3, 1, 2, 4).reshape(B * n_heads, H, W, self.head_dim)
            samp = (
                loc_levels[lvl]
                .transpose(0, 2, 1, 3, 4)
                .reshape(B * n_heads, Q, n_points, 2)
            )
            out_l = grid_sample(v_l, 2.0 * samp - 1.0)
            sampled.append(out_l)
        sampled = mx.concatenate(sampled, axis=-2)
        w = attn.transpose(0, 2, 1, 3).reshape(B * n_heads, Q, n_levels * n_points)
        out = (sampled * w[..., None]).sum(axis=-2)
        out = (
            out.reshape(B, n_heads, Q, self.head_dim)
            .transpose(0, 2, 1, 3)
            .reshape(B, Q, D)
        )
        return self.output_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, config: LayoutConfig, relu) -> None:
        super().__init__()
        d = config.d_model
        self.self_attn = SelfAttention(d, config.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.encoder_attn = MSDeformableAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, d)
        self.activation = relu
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        pos: mx.array,
        memory: mx.array,
        ref: mx.array,
        spatial_shapes: Tuple[Tuple[int, int], ...],
    ) -> mx.array:
        r = x
        x = self.self_attn(x, pos)
        x = self.self_attn_layer_norm(r + x)
        r = x
        x = self.encoder_attn(x, ref, memory, spatial_shapes, pos)
        x = self.encoder_attn_layer_norm(r + x)
        r = x
        x = self.fc2(self.activation(self.fc1(x)))
        return self.final_layer_norm(r + x)


class GlobalPointer(nn.Module):
    """Pairwise reading-order affinity with causal (-1e4) lower mask."""

    def __init__(self, config: LayoutConfig) -> None:
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.dense = nn.Linear(config.d_model, self.head_size * 2)

    def __call__(self, x: mx.array) -> mx.array:
        B, Q, _ = x.shape
        qk = self.dense(x).reshape(B, Q, 2, self.head_size)
        queries, keys = qk[:, :, 0, :], qk[:, :, 1, :]
        scores = (queries @ keys.transpose(0, 2, 1)) / (self.head_size**0.5)
        mask = mx.tril(mx.ones((Q, Q), dtype=mx.bool_))
        return mx.where(mask[None, :, :], mx.array(-1e4, dtype=scores.dtype), scores)


class Decoder(nn.Module):
    """6-layer stack; shared heads/norm/mask-feed passed in per forward."""

    def __init__(self, config: LayoutConfig, relu) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.layers = [DecoderLayer(config, relu) for _ in range(config.decoder_layers)]
        self.query_pos_head = MLPHead(4, 2 * d, d, num_layers=2)

    def __call__(
        self,
        target: mx.array,
        ref_unact: mx.array,
        memory: mx.array,
        spatial_shapes: Tuple[Tuple[int, int], ...],
        bbox_embed: MLPHead,
        class_embed: nn.Linear,
        norm: nn.LayerNorm,
        order_heads: List[nn.Linear],
        global_pointer: GlobalPointer,
        mask_query_head: MLPHead,
        mask_feat: mx.array,
    ) -> dict:
        ref = mx.sigmoid(ref_unact)
        all_h, all_refs, all_logits, all_order, all_masks = [], [], [], [], []
        B = mask_feat.shape[0]
        mH, mW = mask_feat.shape[1], mask_feat.shape[2]
        mask_flat = mask_feat.reshape(B, mH * mW, mask_feat.shape[-1])
        for idx, layer in enumerate(self.layers):
            pos = self.query_pos_head(ref)
            hidden = layer(target, pos, memory, ref[:, :, None, :], spatial_shapes)
            target = hidden
            new_ref = mx.sigmoid(bbox_embed(hidden) + inverse_sigmoid(ref))
            ref = mx.stop_gradient(new_ref)
            out_q = norm(hidden)
            mq = mask_query_head(out_q)
            out_mask = (mq @ mask_flat.transpose(0, 2, 1)).reshape(B, -1, mH, mW)
            all_h.append(hidden)
            all_refs.append(new_ref)
            all_logits.append(class_embed(out_q))
            all_order.append(global_pointer(order_heads[idx](out_q)))
            all_masks.append(out_mask)
        return {
            "last_hidden_state": target,
            "intermediate_hidden_states": mx.stack(all_h, axis=1),
            "intermediate_reference_points": mx.stack(all_refs, axis=1),
            "intermediate_logits": mx.stack(all_logits, axis=1),
            "out_order_logits": mx.stack(all_order, axis=1),
            "out_masks": mx.stack(all_masks, axis=1),
        }


def mask_to_box_coordinate(mask: mx.array) -> mx.array:
    """Tight box per query mask as normalized cxcywh (empty mask -> zeros)."""
    B, Q, H, W = mask.shape
    m = (mask > 0).reshape(B, Q, H * W)
    xs = mx.tile(mx.arange(W, dtype=mx.float32)[None, None, :], (B, Q, H))
    ys = mx.broadcast_to(
        mx.repeat(mx.arange(H, dtype=mx.float32)[None, :], W, axis=-1), (B, Q, H * W)
    )
    big = mx.array(float(2**24), dtype=mx.float32)
    x_max = (xs * m).reshape(B, Q, -1).max(axis=-1) + 1
    x_min = mx.where(m, xs, big).reshape(B, Q, -1).min(axis=-1)
    y_max = (ys * m).reshape(B, Q, -1).max(axis=-1) + 1
    y_min = mx.where(m, ys, big).reshape(B, Q, -1).min(axis=-1)
    nonempty = m.reshape(B, Q, -1).any(axis=-1)[..., None].astype(mx.float32)
    box = mx.stack([x_min, y_min, x_max, y_max], axis=-1) / mx.array(
        [W, H, W, H], dtype=mx.float32
    )
    box = box * nonempty
    x0, y0, x1, y1 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return mx.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=-1)


def decode_order(scores: mx.array) -> mx.array:
    """Reading sequence (first..last positions) from pairwise order scores."""
    sc = mx.sigmoid(scores)
    upper = mx.triu(sc, k=1).sum(axis=0)
    lower = 1.0 - sc.transpose(1, 0)
    lower = mx.tril(lower, k=-1).sum(axis=0)
    votes = upper + lower
    return mx.argsort(votes)
