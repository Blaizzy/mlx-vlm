"""RF-DETR Transformer: Two-stage encoder + Decoder with deformable attention."""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from ..kernels import grid_sample
from .config import TransformerConfig

# ─── Utility functions ───


def inverse_sigmoid(x: mx.array, eps: float = 1e-5) -> mx.array:
    x = mx.clip(x, eps, 1 - eps)
    return mx.log(x / (1 - x))


def gen_sineembed_for_position(pos: mx.array, d_model: int = 128) -> mx.array:
    """Generate sine positional embeddings from coordinates (matching PyTorch DETR).

    Args:
        pos: (..., 2 or 4) coordinates
        d_model: dimension PER COORDINATE (typically hidden_dim // 2 = 128)
    Returns:
        (..., d_model * num_coords) interleaved sine/cosine embeddings
        Order: (y, x, [w, h]) with interleaved sin/cos per frequency
    """
    temperature = 10000.0
    scale = 2 * math.pi
    num_coords = pos.shape[-1]

    dim_t = mx.arange(d_model, dtype=mx.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / d_model)

    def _embed_coord(coord):
        """Embed a single coordinate with interleaved sin/cos."""
        embed = coord[..., None] * scale / dim_t  # (..., d_model)
        # Interleave sin (even) and cos (odd): [sin, cos, sin, cos, ...]
        sin_part = mx.sin(embed[..., 0::2])  # (..., d_model // 2)
        cos_part = mx.cos(embed[..., 1::2])  # (..., d_model // 2)
        # Stack and flatten: (..., d_model//2, 2) -> (..., d_model)
        interleaved = mx.stack([sin_part, cos_part], axis=-1)
        return interleaved.reshape(*embed.shape[:-1], d_model)

    # PyTorch order: y, x, [w, h]
    if num_coords == 2:
        pos_x = _embed_coord(pos[..., 0])
        pos_y = _embed_coord(pos[..., 1])
        return mx.concatenate([pos_y, pos_x], axis=-1)
    elif num_coords == 4:
        pos_x = _embed_coord(pos[..., 0])
        pos_y = _embed_coord(pos[..., 1])
        pos_w = _embed_coord(pos[..., 2])
        pos_h = _embed_coord(pos[..., 3])
        return mx.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)
    else:
        embeds = [_embed_coord(pos[..., i]) for i in range(num_coords)]
        return mx.concatenate(embeds, axis=-1)


def _gen_encoder_output_proposals(H: int, W: int, scale: float = 0.05) -> mx.array:
    """Generate grid of anchor proposals in [0, 1] coordinate space.

    Args:
        H, W: spatial dimensions of the feature map
        scale: initial box size (fraction of image)
    Returns:
        (H*W, 4) proposals in [cx, cy, w, h] format, values in (0, 1)
    """
    # Grid centers in [0, 1]
    grid_y = (mx.arange(H, dtype=mx.float32) + 0.5) / H
    grid_x = (mx.arange(W, dtype=mx.float32) + 0.5) / W

    # Meshgrid: (H, W)
    yy = mx.broadcast_to(grid_y[:, None], (H, W))
    xx = mx.broadcast_to(grid_x[None, :], (H, W))

    # Box dimensions (constant scale)
    ww = mx.full((H, W), scale)
    hh = mx.full((H, W), scale)

    # Stack: (H, W, 4) -> (H*W, 4) in [0, 1]
    return mx.stack([xx, yy, ww, hh], axis=-1).reshape(-1, 4)


# ─── Multi-Scale Deformable Attention ───


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention using Metal grid_sample kernel."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 16,
        n_levels: int = 1,
        n_points: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def __call__(
        self,
        query: mx.array,
        reference_points: mx.array,
        value: mx.array,
        spatial_shape: Tuple[int, int],
    ) -> mx.array:
        """
        Args:
            query: (B, Q, D) query features
            reference_points: (B, Q, n_levels, 2) or (B, Q, n_levels, 4)
                For 2D: [cx, cy] in [0, 1] normalized coords
                For 4D: [cx, cy, w, h] (can be in logit space for bbox_reparam)
            value: (B, HW, D) flattened spatial features
            spatial_shape: (H, W) of the feature map
        Returns:
            (B, Q, D)
        """
        B, Q, _ = query.shape
        H, W = spatial_shape

        # Project values
        value = self.value_proj(value)  # (B, HW, D)

        # Compute sampling offsets
        offsets = self.sampling_offsets(
            query
        )  # (B, Q, n_heads * n_levels * n_points * 2)
        offsets = offsets.reshape(B, Q, self.n_heads, self.n_levels, self.n_points, 2)

        # Compute attention weights
        attn_weights = self.attention_weights(
            query
        )  # (B, Q, n_heads * n_levels * n_points)
        attn_weights = attn_weights.reshape(
            B, Q, self.n_heads, self.n_levels * self.n_points
        )
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.reshape(
            B, Q, self.n_heads, self.n_levels, self.n_points
        )

        # Compute sampling locations based on reference point dimensionality
        if reference_points.ndim == 3:
            # (B, Q, 2) -> add n_levels and n_heads dims
            ref = reference_points[:, :, None, None, None, :]  # (B, Q, 1, 1, 1, 2)
            offset_normalizer = mx.array([W, H], dtype=mx.float32)
            sampling_locations = ref + offsets / offset_normalizer
        elif reference_points.shape[-1] == 2:
            # (B, Q, n_levels, 2)
            ref = reference_points[:, :, None, :, None, :]  # (B, Q, 1, n_levels, 1, 2)
            offset_normalizer = mx.array([W, H], dtype=mx.float32)
            sampling_locations = ref + offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # 4D: offsets scaled by reference box size (PyTorch DETR formula)
            ref_center = reference_points[
                :, :, None, :, None, :2
            ]  # (B, Q, 1, n_levels, 1, 2)
            ref_wh = reference_points[
                :, :, None, :, None, 2:
            ]  # (B, Q, 1, n_levels, 1, 2)
            sampling_locations = ref_center + offsets / self.n_points * ref_wh * 0.5
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        # Reshape value for grid sampling: (B, HW, D) -> (B*n_heads, H, W, head_dim)
        value_spatial = value.reshape(B, H, W, self.n_heads, self.head_dim)
        value_spatial = value_spatial.transpose(
            0, 3, 1, 2, 4
        )  # (B, n_heads, H, W, head_dim)
        value_spatial = value_spatial.reshape(B * self.n_heads, H, W, self.head_dim)

        # For n_levels=1, squeeze level dim
        samp_loc = sampling_locations[:, :, :, 0, :, :]  # (B, Q, n_heads, n_points, 2)

        # Convert to grid_sample format: [-1, 1] from [0, 1]
        grid_coords = samp_loc * 2 - 1  # (B, Q, n_heads, n_points, 2)

        # Reshape for grid_sample: (B*n_heads, Q, n_points, 2)
        grid_coords = grid_coords.transpose(0, 2, 1, 3, 4).reshape(
            B * self.n_heads, Q, self.n_points, 2
        )

        # Grid sample: (B*n_heads, Q, n_points, head_dim)
        sampled = grid_sample(value_spatial, grid_coords)

        # Apply attention weights
        sampled = sampled.reshape(B, self.n_heads, Q, self.n_points, self.head_dim)
        weights = attn_weights[:, :, :, 0, :].transpose(0, 2, 1, 3)[..., None]

        # Weighted sum over points
        output = (sampled * weights).sum(axis=3)  # (B, n_heads, Q, head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(B, Q, self.d_model)

        return self.output_proj(output)


# ─── MLP (3-layer bbox regression) ───


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


# ─── Decoder Self-Attention ───


class DecoderSelfAttention(nn.Module):
    """Standard multi-head self-attention for decoder queries."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array, query_pos: mx.array) -> mx.array:
        """Self-attention with position added to q and k only (not v)."""
        B, N, D = x.shape
        qk_input = x + query_pos  # Position added to q,k
        q = (
            self.q_proj(qk_input)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(qk_input)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(x)


# ─── Decoder Layer ───


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        d = config.hidden_dim

        # Self-attention
        self.self_attn = DecoderSelfAttention(d, config.sa_nheads)
        self.norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)

        # Deformable cross-attention
        self.cross_attn = MSDeformableAttention(
            d_model=d,
            n_heads=config.ca_nheads,
            n_levels=config.n_levels,
            n_points=config.dec_n_points,
        )
        self.norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)

        # FFN
        self.linear1 = nn.Linear(d, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, d)
        self.norm3 = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        reference_points: mx.array,
        spatial_shape: Tuple[int, int],
        query_pos: mx.array = None,
    ) -> mx.array:
        """
        Args:
            tgt: (B, Q, D) query features
            memory: (B, HW, D) encoder features
            reference_points: (B, Q, n_levels, 4) reference boxes for deformable attn
            spatial_shape: (H, W)
            query_pos: (B, Q, D) position embedding
        """
        # Self-attention (query_pos added to q,k only)
        tgt = tgt + self.self_attn(tgt, query_pos)
        tgt = self.norm1(tgt)

        # Deformable cross-attention (query_pos added to query)
        cross_query = tgt + query_pos if query_pos is not None else tgt
        tgt = tgt + self.cross_attn(
            cross_query, reference_points, memory, spatial_shape
        )
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(nn.relu(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


# ─── Decoder ───


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = [DecoderLayer(config) for _ in range(config.dec_layers)]
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        # RefPointHead: maps sine-encoded reference points to position embeddings
        self.ref_point_head = MLP(
            input_dim=config.hidden_dim * 2,  # 512 (4 coords * 128 features)
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=2,
        )
        self.config = config

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        reference_points_unsigmoid: mx.array,
        spatial_shape: Tuple[int, int],
        bbox_embed: "MLP",
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            tgt: (B, Q, D) query features
            memory: (B, HW, D) encoder features
            reference_points_unsigmoid: (B, Q, 4) initial reference points (pre-sigmoid)
            spatial_shape: (H, W) of feature map
            bbox_embed: bbox regression head for iterative refinement
        Returns:
            hs: (B, Q, D) final hidden states
            reference_points: (B, Q, 4) refined reference points (sigmoid)
        """
        output = tgt
        d_half = self.config.hidden_dim // 2  # 128
        # reference_points_unsigmoid is actually in [0,1] space for bbox_reparam
        ref_coords = reference_points_unsigmoid

        # Compute query_pos ONCE from reference points (lite_refpoint_refine)
        ref_sine = gen_sineembed_for_position(ref_coords, d_half)
        query_pos = self.ref_point_head(ref_sine)  # (B, Q, D)

        for layer_idx, layer in enumerate(self.layers):
            # Pass 4D reference points for deformable cross-attention
            # Shape: (B, Q, n_levels=1, 4) in [0,1] coordinate space
            refpoints_input = ref_coords[:, :, None, :]  # (B, Q, 1, 4)

            # Decoder layer
            output = layer(
                output, memory, refpoints_input, spatial_shape, query_pos=query_pos
            )

        output = self.norm(output)
        return output, ref_coords


# ─── Transformer (Two-Stage Encoder + Decoder) ───


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # Two-stage encoder components (one per group_detr group)
        self.enc_output = [nn.Linear(d, d) for _ in range(config.group_detr)]
        self.enc_output_norm = [nn.LayerNorm(d) for _ in range(config.group_detr)]
        self.enc_out_class_embed = [
            nn.Linear(d, config.num_classes) for _ in range(config.group_detr)
        ]
        self.enc_out_bbox_embed = [
            MLP(d, d, 4, num_layers=3) for _ in range(config.group_detr)
        ]

        # Decoder
        self.decoder = Decoder(config)

    def two_stage_select(
        self,
        memory: mx.array,
        spatial_shape: Tuple[int, int],
        group_idx: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Select top-K queries from encoder memory (two-stage mechanism).

        Args:
            memory: (B, HW, D) encoder features (projected features)
            spatial_shape: (H, W) feature map dimensions
            group_idx: which group to use (0 for inference)
        Returns:
            refpoint_embed_ts: (B, num_queries, 4) selected reference points
            memory_ts: (B, num_queries, D) selected encoder features (for tgt init)
        """
        B = memory.shape[0]
        num_queries = self.config.num_queries
        H, W = spatial_shape

        # Generate grid proposals (in actual coordinate space, not logit)
        grid_proposals = _gen_encoder_output_proposals(
            H, W
        )  # (HW, 4) cx,cy,w,h in [0,1]

        # Project encoder features
        output = self.enc_output[group_idx](memory)
        output = self.enc_output_norm[group_idx](output)

        # Classify all positions
        cls_logits = self.enc_out_class_embed[group_idx](output)  # (B, HW, num_classes)

        # Predict box refinements
        bbox_delta = self.enc_out_bbox_embed[group_idx](output)  # (B, HW, 4)
        proposals = grid_proposals[None, :, :]  # (1, HW, 4)

        if self.config.bbox_reparam:
            # Parametric box encoding: delta * proposal_wh + proposal_center
            enc_cxcy = bbox_delta[..., :2] * proposals[..., 2:] + proposals[..., :2]
            enc_wh = mx.exp(bbox_delta[..., 2:]) * proposals[..., 2:]
            enc_outputs_coord = mx.concatenate([enc_cxcy, enc_wh], axis=-1)
        else:
            enc_outputs_coord = bbox_delta + inverse_sigmoid(proposals)

        # Top-K selection by max class score
        max_scores = cls_logits.max(axis=-1)  # (B, HW)
        topk_indices = mx.argpartition(-max_scores, kth=num_queries, axis=-1)[
            :, :num_queries
        ]
        topk_scores = mx.take_along_axis(max_scores, topk_indices, axis=-1)
        sort_idx = mx.argsort(-topk_scores, axis=-1)
        topk_indices = mx.take_along_axis(topk_indices, sort_idx, axis=-1)

        # Gather selected features and boxes
        topk_indices_expanded = topk_indices[:, :, None]
        selected_feat = mx.take_along_axis(
            output,
            mx.broadcast_to(topk_indices_expanded, (B, num_queries, output.shape[-1])),
            axis=1,
        )
        selected_boxes = mx.take_along_axis(
            enc_outputs_coord,
            mx.broadcast_to(topk_indices_expanded, (B, num_queries, 4)),
            axis=1,
        )

        # Detach for decoder input
        refpoint_embed_ts = mx.stop_gradient(selected_boxes)  # (B, nq, 4)
        memory_ts = selected_feat  # NOT detached (used for tgt init but not directly)

        return refpoint_embed_ts, memory_ts

    def __call__(
        self,
        memory: mx.array,
        spatial_shape: Tuple[int, int],
        query_feat: mx.array,
        refpoint_embed: mx.array,
        bbox_embed: "MLP",
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            memory: (B, HW, D) projected spatial features
            spatial_shape: (H, W) of the feature map
            query_feat: (num_queries * group_detr, D) all query features
            refpoint_embed: (num_queries * group_detr, 4) all reference points
            bbox_embed: bbox regression MLP for iterative refinement
        Returns:
            hs: (B, Q, D) decoder output
            ref_points: (B, Q, 4) refined reference points
        """
        B = memory.shape[0]
        nq = self.config.num_queries
        d = self.config.hidden_dim

        # At inference, use only group 0
        qf = query_feat[:nq]  # (num_queries, D)
        rp = refpoint_embed[:nq]  # (num_queries, 4)

        # Two-stage query selection
        refpoint_embed_ts, memory_ts = self.two_stage_select(
            memory, spatial_shape, group_idx=0
        )

        # Combine learnable refpoint_embed with two-stage proposals
        if self.config.bbox_reparam:
            # Parametric combination: rp acts as delta relative to ts proposals
            ref_cxcy = (
                rp[None, :, :2] * refpoint_embed_ts[..., 2:]
                + refpoint_embed_ts[..., :2]
            )
            ref_wh = mx.exp(rp[None, :, 2:]) * refpoint_embed_ts[..., 2:]
            combined_refpoints = mx.concatenate([ref_cxcy, ref_wh], axis=-1)
        else:
            combined_refpoints = rp[None, :, :] + refpoint_embed_ts

        # tgt is JUST the query features (NOT enriched with encoder features)
        tgt = mx.broadcast_to(qf[None, :, :], (B, nq, d))

        # Decoder
        hs, ref_unsig = self.decoder(
            tgt, memory, combined_refpoints, spatial_shape, bbox_embed
        )

        return hs, ref_unsig
