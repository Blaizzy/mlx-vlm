"""RT-DETRv2 transformer: deformable-attention decoder and helpers.

Multi-scale deformable attention runs once per feature level via the
shared `grid_sample` Metal kernel and sums across levels with the
softmaxed attention weights. `n_points_scale` is carried as a
non-learnable `mx.array` buffer per layer to remain faithful to
checkpoints that use non-uniform `n_points_list`.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..kernels import grid_sample
from .config import RTDetrV2TransformerConfig

# ─── helpers ───


def inverse_sigmoid(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Numerically-stable inverse sigmoid."""
    x = mx.clip(x, 0.0, 1.0)
    x1 = mx.clip(x, eps, 1.0)
    x2 = mx.clip(1.0 - x, eps, 1.0)
    return mx.log(x1 / x2)


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU between layers.

    Field `layers` is a list of `nn.Linear` matching HF saved keys
    `.layers.{i}.{weight,bias}` of `RTDetrV2MLPPredictionHead`.
    """

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


# ─── Multi-Scale Deformable Attention ───


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention for RT-DETRv2.

    Reference points are 4D `(cx, cy, w, h)` normalized to `[0, 1]`.
    Sampling offsets are predicted per `(n_heads, n_levels, n_points)`
    and scaled by `1/n_points * ref_wh * offset_scale` before being added
    to the reference center. Sampling itself is done by `grid_sample`
    (bilinear, padding=zeros, align_corners=False), once per level, and
    the outputs are concatenated and weighted-summed by the softmaxed
    attention weights.
    """

    def __init__(self, config: RTDetrV2TransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        n_heads = config.decoder_attention_heads
        if d % n_heads != 0:
            raise ValueError(
                f"d_model ({d}) must be divisible by num_heads ({n_heads})"
            )

        self.d_model = d
        self.n_heads = n_heads
        self.n_levels = config.decoder_n_levels
        self.n_points = config.decoder_n_points
        self.head_dim = d // n_heads
        self.offset_scale = config.decoder_offset_scale
        # HF supports "default" (sampling locations in [0,1] then mapped to
        # [-1,1] for grid_sample) and "discrete" (locations used as-is).
        if config.decoder_method not in ("default", "discrete"):
            raise ValueError(
                f"Unsupported decoder_method {config.decoder_method!r}; "
                "expected 'default' or 'discrete'"
            )
        self.method = config.decoder_method

        self.sampling_offsets = nn.Linear(
            d, n_heads * self.n_levels * self.n_points * 2
        )
        self.attention_weights = nn.Linear(d, n_heads * self.n_levels * self.n_points)
        self.value_proj = nn.Linear(d, d)
        self.output_proj = nn.Linear(d, d)

        # Per-(level, point) scale = 1/n_points. Stored as a non-learnable
        # buffer so non-uniform n_points_list configurations (where each
        # level can have a different number of sampling points) also work.
        self.n_points_scale = mx.array(
            [1.0 / self.n_points] * (self.n_levels * self.n_points), dtype=mx.float32
        )

    def __call__(
        self,
        query: mx.array,
        reference_points: mx.array,
        value: mx.array,
        spatial_shapes: Tuple[Tuple[int, int], ...],
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            query: (B, Q, D) decoder hidden states.
            reference_points: (B, Q, 1, 4) center+wh, broadcast across levels.
            value: (B, sum_HW, D) flattened multi-scale encoder features.
            spatial_shapes: tuple of (H, W) per level.
            position_embeddings: optional (B, Q, D) added to query.
        Returns:
            (B, Q, D)
        """
        if position_embeddings is not None:
            query = query + position_embeddings

        B, Q, D = query.shape
        n_heads = self.n_heads
        head_dim = self.head_dim

        v = self.value_proj(value).reshape(B, value.shape[1], n_heads, head_dim)
        offsets = self.sampling_offsets(query).reshape(
            B, Q, n_heads, self.n_levels * self.n_points, 2
        )
        attn = self.attention_weights(query).reshape(
            B, Q, n_heads, self.n_levels * self.n_points
        )
        attn = mx.softmax(attn, axis=-1)

        if reference_points.shape[-1] != 4:
            raise ValueError(
                f"Expected 4D reference points (cx,cy,w,h), got last dim "
                f"{reference_points.shape[-1]}"
            )
        n_pts_scale = self.n_points_scale.astype(query.dtype)[None, None, None, :, None]
        ref_wh = reference_points[:, :, None, :, 2:]
        ref_xy = reference_points[:, :, None, :, :2]
        loc = ref_xy + offsets * n_pts_scale * ref_wh * self.offset_scale

        # Per-level grid_sample calls.
        loc_levels = mx.split(loc, self.n_levels, axis=-2)

        level_sizes = [H * W for H, W in spatial_shapes]
        offsets_running = 0
        v_levels = []
        for s in level_sizes:
            v_levels.append(v[:, offsets_running : offsets_running + s, :, :])
            offsets_running += s

        sampled_per_level = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            v_l = v_levels[lvl].reshape(B, H, W, n_heads, head_dim)
            v_l = v_l.transpose(0, 3, 1, 2, 4).reshape(B * n_heads, H, W, head_dim)
            samp = loc_levels[lvl]
            samp = samp.transpose(0, 2, 1, 3, 4).reshape(
                B * n_heads, Q, self.n_points, 2
            )
            if self.method == "default":
                samp = 2.0 * samp - 1.0
            out_l = grid_sample(v_l, samp)
            sampled_per_level.append(out_l)

        sampled = mx.concatenate(sampled_per_level, axis=-2)
        w = attn.transpose(0, 2, 1, 3).reshape(
            B * n_heads, Q, self.n_levels * self.n_points
        )
        out = (sampled * w[..., None]).sum(axis=-2)
        out = (
            out.reshape(B, n_heads, Q, head_dim).transpose(0, 2, 1, 3).reshape(B, Q, D)
        )
        return self.output_proj(out)


# ─── Self-attention ───


class SelfAttention(nn.Module):
    """Decoder self-attention with position embeddings added to q,k (not v)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        B, N, D = x.shape
        qk = x + position_embeddings if position_embeddings is not None else x
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
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


# ─── Decoder ───


class DecoderLayer(nn.Module):
    """One decoder block: self-attn -> norm -> deformable cross-attn -> norm
    -> FFN -> norm. Field names match the saved state-dict."""

    def __init__(self, config: RTDetrV2TransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        self.self_attn = SelfAttention(d, config.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.encoder_attn = MSDeformableAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.activation = _resolve_activation(config.decoder_activation_function)

    def __call__(
        self,
        x: mx.array,
        object_queries_position_embeddings: mx.array,
        encoder_hidden_states: mx.array,
        reference_points: mx.array,
        spatial_shapes: Tuple[Tuple[int, int], ...],
    ) -> mx.array:
        residual = x
        x = self.self_attn(x, object_queries_position_embeddings)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.encoder_attn(
            query=x,
            reference_points=reference_points,
            value=encoder_hidden_states,
            spatial_shapes=spatial_shapes,
            position_embeddings=object_queries_position_embeddings,
        )
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.fc2(self.activation(self.fc1(x)))
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class Decoder(nn.Module):
    """Decoder stack with iterative bbox refinement.

    Per-layer `bbox_embed` (3-layer MLP) and `class_embed` (Linear) heads
    are attached here, matching the saved keys
    `decoder.bbox_embed.{L}.layers.{i}.{weight,bias}` and
    `decoder.class_embed.{L}.{weight,bias}`.
    """

    def __init__(self, config: RTDetrV2TransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.layers = [DecoderLayer(config) for _ in range(config.decoder_layers)]

        # 2-layer MLP that converts 4D reference points to D-dim position
        # embeddings added to queries in each decoder layer.
        self.query_pos_head = MLP(4, 2 * d, d, num_layers=2)

        self.bbox_embed = [
            MLP(d, d, 4, num_layers=3) for _ in range(config.decoder_layers)
        ]
        self.class_embed = [
            nn.Linear(d, config.num_labels) for _ in range(config.decoder_layers)
        ]

    def __call__(
        self,
        target: mx.array,
        reference_points_unact: mx.array,
        encoder_hidden_states: mx.array,
        spatial_shapes: Tuple[Tuple[int, int], ...],
    ):
        """
        Args:
            target: (B, Q, D) initial query content.
            reference_points_unact: (B, Q, 4) initial reference boxes in logit
                space (pre-sigmoid). Decoder takes sigmoid as first step.
            encoder_hidden_states: (B, sum_HW, D) flattened encoder features.
            spatial_shapes: per-level (H, W) tuples.
        Returns:
            dict with `last_hidden_state`, `intermediate_hidden_states`,
            `intermediate_reference_points`, `intermediate_logits`.
        """
        hidden = target
        ref_points = mx.sigmoid(reference_points_unact)

        all_hidden = []
        all_refs = []
        all_logits = []

        for idx, layer in enumerate(self.layers):
            # (B, Q, 1, 4) broadcasts across feature levels in cross-attn.
            ref_input = ref_points[:, :, None, :]
            pos_embed = self.query_pos_head(ref_points)
            hidden = layer(
                x=hidden,
                object_queries_position_embeddings=pos_embed,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=ref_input,
                spatial_shapes=spatial_shapes,
            )

            predicted_corners = self.bbox_embed[idx](hidden)
            new_refs = mx.sigmoid(predicted_corners + inverse_sigmoid(ref_points))
            ref_points = mx.stop_gradient(new_refs)

            all_hidden.append(hidden)
            all_refs.append(new_refs)
            all_logits.append(self.class_embed[idx](hidden))

        return {
            "last_hidden_state": hidden,
            "intermediate_hidden_states": mx.stack(all_hidden, axis=1),
            "intermediate_reference_points": mx.stack(all_refs, axis=1),
            "intermediate_logits": mx.stack(all_logits, axis=1),
        }


# ─── Anchor priors for encoder query selection ───


def generate_anchors(
    spatial_shapes: Tuple[Tuple[int, int], ...],
    grid_size: float = 0.05,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array]:
    """Multi-scale anchor priors used by encoder query selection.

    Returns `(anchors_logit, valid_mask)` where anchors are
    `(1, sum_HW, 4)` in logit space and the mask marks positions whose
    sigmoid would fall in `[eps, 1-eps]`.
    """
    anchors_per_level = []
    eps = 1e-2
    for level, (h, w) in enumerate(spatial_shapes):
        gy, gx = mx.meshgrid(
            mx.arange(h, dtype=dtype), mx.arange(w, dtype=dtype), indexing="ij"
        )
        grid_xy = mx.stack([gx, gy], axis=-1)[None, ...] + 0.5
        grid_xy = grid_xy / mx.array([w, h], dtype=dtype)[None, None, None, :]
        wh = mx.ones_like(grid_xy) * grid_size * (2.0**level)
        anchors_per_level.append(
            mx.concatenate([grid_xy, wh], axis=-1).reshape(1, h * w, 4)
        )
    anchors = mx.concatenate(anchors_per_level, axis=1)
    valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(axis=-1, keepdims=True)
    anchors_logit = mx.log(anchors / (1.0 - anchors))
    big = mx.array(mx.finfo(dtype).max, dtype=dtype)
    anchors_logit = mx.where(valid_mask, anchors_logit, big)
    return anchors_logit, valid_mask


def _resolve_activation(name: str):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")
