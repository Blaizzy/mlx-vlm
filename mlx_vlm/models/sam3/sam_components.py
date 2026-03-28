"""SAM2-style components for the tracker: PromptEncoder, MaskDecoder, TwoWayTransformer.

Weight keys: tracker_model.prompt_encoder.*, tracker_model.mask_decoder.*
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import PromptEncoderConfig, TrackerMaskDecoderConfig
from .position import apply_rotary_enc_1d, init_2d_freqs

# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """Simple 2-layer MLP (ReLU or GELU)."""

    def __init__(self, input_dim: int, hidden_dim: int, act: str = "relu"):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, input_dim)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_in(x)
        if self.act == "gelu":
            x = nn.gelu(x)
        else:
            x = nn.relu(x)
        return self.proj_out(x)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for spatial features (B, H, W, C)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - channel last in MLX
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# SAM Attention (for TwoWayTransformer)
# ---------------------------------------------------------------------------


class SAMAttention(nn.Module):
    """Multi-head attention with optional downsampling of internal dim."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        internal_dim = hidden_size // downsample_rate
        self.head_dim = internal_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, internal_dim)
        self.k_proj = nn.Linear(hidden_size, internal_dim)
        self.v_proj = nn.Linear(hidden_size, internal_dim)
        self.o_proj = nn.Linear(internal_dim, hidden_size)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        B, N_q, _ = q.shape
        N_k = k.shape[1]

        q = (
            self.q_proj(q)
            .reshape(B, N_q, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(k)
            .reshape(B, N_k, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(v)
            .reshape(B, N_k, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N_q, -1)
        return self.o_proj(out)


class RoPEAttention(nn.Module):
    """Attention with 2D Rotary Position Encoding (used in tracker memory attention).

    Weight keys: tracker_model.memory_attention.layers.*.{self_attn,cross_attn_image}.*
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        downsample_rate: int = 1,
        feat_sizes: Tuple[int, int] = (72, 72),
        rope_theta: float = 10000.0,
        kv_dim: Optional[int] = None,
        rope_k_repeat: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        internal_dim = hidden_size // downsample_rate
        self.head_dim = internal_dim // num_heads
        self.scale = self.head_dim**-0.5
        kv_dim = kv_dim if kv_dim is not None else hidden_size

        self.q_proj = nn.Linear(hidden_size, internal_dim)
        self.k_proj = nn.Linear(kv_dim, internal_dim)
        self.v_proj = nn.Linear(kv_dim, internal_dim)
        self.o_proj = nn.Linear(internal_dim, hidden_size)

        self.rope_k_repeat = rope_k_repeat

        # Precompute RoPE frequencies
        self._freqs_cos, self._freqs_sin = init_2d_freqs(
            internal_dim, feat_sizes[0], feat_sizes[1], theta=rope_theta
        )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        B, N_q, _ = q.shape
        N_k = k.shape[1]

        q = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim)
        k = self.k_proj(k).reshape(B, N_k, self.num_heads, self.head_dim)
        v = self.v_proj(v).reshape(B, N_k, self.num_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_enc_1d(
            q,
            k,
            self._freqs_cos,
            self._freqs_sin,
            repeat_freqs_k=self.rope_k_repeat,
        )

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N_q, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# TwoWayTransformer (SAM mask decoder's internal transformer)
# ---------------------------------------------------------------------------


class TwoWayAttentionBlock(nn.Module):
    """SAM two-way attention block.

    Weight keys per layer:
        self_attn.{q,k,v,o}_proj.{weight,bias}
        cross_attn_token_to_image.{q,k,v,o}_proj.{weight,bias}
        cross_attn_image_to_token.{q,k,v,o}_proj.{weight,bias}
        layer_norm1.{weight,bias}
        layer_norm2.{weight,bias}
        layer_norm3.{weight,bias}
        layer_norm4.{weight,bias}
        mlp.proj_in.{weight,bias}
        mlp.proj_out.{weight,bias}
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.self_attn = SAMAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.cross_attn_token_to_image = SAMAttention(
            hidden_size, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = MLPBlock(hidden_size, mlp_dim, act="relu")
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        self.cross_attn_image_to_token = SAMAttention(
            hidden_size, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_norm4 = nn.LayerNorm(hidden_size)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        query_pe: mx.array,
        key_pe: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        # Self-attention on queries
        q = queries + query_pe
        attn_out = self.self_attn(q, q, queries)
        queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross-attention: tokens to image
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.layer_norm2(queries)

        # MLP
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross-attention: image to tokens
        q = keys + key_pe
        k = queries + query_pe
        attn_out = self.cross_attn_image_to_token(q, k, queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    """SAM TwoWayTransformer for mask decoder.

    Weight keys: tracker_model.mask_decoder.transformer.*
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.layers = [
            TwoWayAttentionBlock(
                hidden_size, num_heads, mlp_dim, attention_downsample_rate
            )
            for _ in range(num_layers)
        ]

        self.final_attn_token_to_image = SAMAttention(
            hidden_size, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_norm_final_attn = nn.LayerNorm(hidden_size)

    def __call__(
        self,
        image_embedding: mx.array,
        image_pe: mx.array,
        point_embedding: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            image_embedding: (B, HW, D)
            image_pe: (B, HW, D)
            point_embedding: (B, N_tokens, D)
        Returns:
            queries: (B, N_tokens, D)
            keys: (B, HW, D)
        """
        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries,
                keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Final token->image attention
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        return queries, keys


# ---------------------------------------------------------------------------
# SAM Prompt Encoder
# ---------------------------------------------------------------------------


class SAMPromptEncoder(nn.Module):
    """SAM Prompt Encoder for points, boxes, and masks.

    Weight keys: tracker_model.prompt_encoder.*
    """

    def __init__(self, config: PromptEncoderConfig):
        super().__init__()
        d = config.hidden_size
        image_size = config.image_size
        patch_size = config.patch_size
        self.embed_dim = d
        self.image_embedding_size = (image_size // patch_size, image_size // patch_size)

        # Point embeddings
        self.point_embed = nn.Embedding(config.num_point_embeddings, d)
        self.not_a_point_embed = nn.Embedding(1, d)

        # Mask embedding (small conv network)
        self.mask_embed = MaskEmbedConvs(d, config.mask_input_channels)
        self.no_mask_embed = nn.Embedding(1, d)

        # Shared positional embedding
        self.shared_embedding = PositionalEmbedding(d // 2)

    def get_dense_pe(self) -> mx.array:
        """Get positional encoding for image-sized features."""
        H, W = self.image_embedding_size
        return self.shared_embedding((H, W))[None]  # (1, H*W, D)

    def __call__(
        self,
        points: Optional[Tuple[mx.array, mx.array]] = None,
        boxes: Optional[mx.array] = None,
        masks: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            points: (coords (B,N,2), labels (B,N)) point prompts
            boxes: (B, N_box, 4) box prompts
            masks: (B, 1, H, W) mask prompts
        Returns:
            sparse_embeddings: (B, N_tokens, D)
            dense_embeddings: (B, H*W, D)
        """
        B = 1
        sparse_embeddings = mx.zeros((B, 0, self.embed_dim))

        if points is not None:
            coords, labels = points
            B = coords.shape[0]
            point_emb = self._embed_points(coords, labels)
            sparse_embeddings = mx.concatenate(
                [mx.broadcast_to(sparse_embeddings, (B, 0, self.embed_dim)), point_emb],
                axis=1,
            )

        if boxes is not None:
            B = boxes.shape[0]
            box_emb = self._embed_boxes(boxes)
            sparse_embeddings = mx.concatenate([sparse_embeddings, box_emb], axis=1)

        if masks is not None:
            B = masks.shape[0]
            dense_embeddings = self.mask_embed(masks)
        else:
            H, W = self.image_embedding_size
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, self.embed_dim)
            dense_embeddings = mx.broadcast_to(
                dense_embeddings, (B, H * W, self.embed_dim)
            )

        return sparse_embeddings, dense_embeddings

    def _embed_points(self, coords: mx.array, labels: mx.array) -> mx.array:
        """Embed point prompts."""
        coords = coords + 0.5  # shift to center of pixel
        # Normalize to [0, 1] range
        coords = coords / mx.array(
            [self.image_embedding_size[1], self.image_embedding_size[0]],
            dtype=mx.float32,
        )
        point_emb = self.shared_embedding.forward_with_coords(coords)

        # Add label-specific embedding
        for i in range(labels.shape[-1]):
            label = labels[:, i : i + 1].astype(mx.int32)
            point_emb = point_emb.at[:, i : i + 1].add(self.point_embed(label))

        # Mark padding points
        padding_mask = labels == -1
        if padding_mask.any():
            not_a_point = self.not_a_point_embed.weight
            point_emb = mx.where(padding_mask[..., None], not_a_point, point_emb)

        return point_emb

    def _embed_boxes(self, boxes: mx.array) -> mx.array:
        """Embed box prompts as corner point pairs."""
        coords = boxes.reshape(-1, 2, 2)
        corner_emb = self.shared_embedding.forward_with_coords(coords)
        corner_emb = corner_emb.at[:, 0:1].add(self.point_embed(mx.array([[2]])))
        corner_emb = corner_emb.at[:, 1:2].add(self.point_embed(mx.array([[3]])))
        return corner_emb


class MaskEmbedConvs(nn.Module):
    """Conv layers for mask embedding.

    Weight keys: tracker_model.prompt_encoder.mask_embed.*
    """

    def __init__(self, embed_dim: int, mask_in_chans: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2
        )
        self.conv3 = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        self.layer_norm1 = LayerNorm2d(mask_in_chans // 4)
        self.layer_norm2 = LayerNorm2d(mask_in_chans)

    def __call__(self, masks: mx.array) -> mx.array:
        """
        Args:
            masks: (B, H, W, 1) mask input (MLX channel-last)
        Returns:
            (B, H', W', embed_dim) -> flattened to (B, H'*W', embed_dim)
        """
        x = self.conv1(masks)
        x = self.layer_norm1(x)
        x = nn.gelu(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = nn.gelu(x)
        x = self.conv3(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


class PositionalEmbedding(nn.Module):
    """Random spatial positional embedding for SAM prompt encoder.

    Weight keys: tracker_model.prompt_encoder.shared_embedding.positional_embedding
    """

    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.positional_embedding = mx.zeros((2, num_pos_feats))

    def __call__(self, size: Tuple[int, int]) -> mx.array:
        """Generate positional encoding for a given spatial size."""
        H, W = size
        grid_y = mx.arange(H).astype(mx.float32) / H
        grid_x = mx.arange(W).astype(mx.float32) / W

        # (H, W) grids
        gy, gx = mx.meshgrid(grid_y, grid_x, indexing="ij")
        coords = mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)  # (H*W, 2)

        return self.forward_with_coords(coords[None])[0]  # (H*W, D)

    def forward_with_coords(self, coords: mx.array) -> mx.array:
        """
        Args:
            coords: (B, N, 2) coordinates in [0, 1] range
        Returns:
            (B, N, D) positional encoding
        """
        # coords: (B, N, 2), positional_embedding: (2, D//2)
        coords = 2 * coords - 1  # to [-1, 1]
        coords = coords @ self.positional_embedding  # (B, N, D//2)
        coords = 2 * math.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)


# ---------------------------------------------------------------------------
# SAM Mask Decoder
# ---------------------------------------------------------------------------


class SAMMaskDecoder(nn.Module):
    """SAM-style mask decoder for the tracker.

    Weight keys: tracker_model.mask_decoder.*
    """

    def __init__(self, config: TrackerMaskDecoderConfig):
        super().__init__()
        d = config.hidden_size
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.transformer = TwoWayTransformer(
            hidden_size=d,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            mlp_dim=config.mlp_dim,
            attention_downsample_rate=config.attention_downsample_rate,
        )

        # Special tokens
        self.iou_token = nn.Embedding(1, d)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d)
        self.obj_score_token = nn.Embedding(1, d)

        # Output heads
        self.output_hypernetworks_mlps = [
            OutputMLP(d, d, d // 8) for _ in range(self.num_mask_tokens)
        ]
        self.iou_prediction_head = OutputMLP(d, d, self.num_mask_tokens)
        self.pred_obj_score_head = OutputMLP(d, d, 1)

        # Upscaling convolutions
        self.upscale_conv1 = nn.ConvTranspose2d(d, d // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(d // 4, d // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(d // 4)

        # Skip connection convolutions for high-res features
        self.conv_s0 = nn.Conv2d(d, d // 8, kernel_size=1, bias=True)
        self.conv_s1 = nn.Conv2d(d, d // 4, kernel_size=1, bias=True)

        self.dynamic_multimask_via_stability = config.dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = (
            config.dynamic_multimask_stability_delta
        )
        self.dynamic_multimask_stability_thresh = (
            config.dynamic_multimask_stability_thresh
        )

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        multimask_output: bool = True,
        high_res_features: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Args:
            image_embeddings: (B, HW, D)
            image_pe: (B, HW, D)
            sparse_prompt_embeddings: (B, N_sparse, D)
            dense_prompt_embeddings: (B, HW, D)
            high_res_features: optional list of high-res skip features
        Returns:
            masks: (B, N_masks, H, W)
            iou_pred: (B, N_masks)
            sam_tokens: (B, N_tokens, D)
            obj_score: (B, 1)
        """
        B = image_embeddings.shape[0]
        d = image_embeddings.shape[-1]

        # Concatenate special tokens with sparse embeddings
        tokens = mx.concatenate(
            [
                mx.broadcast_to(self.iou_token.weight[None], (B, 1, d)),
                mx.broadcast_to(
                    self.mask_tokens.weight[None], (B, self.num_mask_tokens, d)
                ),
                mx.broadcast_to(self.obj_score_token.weight[None], (B, 1, d)),
            ],
            axis=1,
        )
        tokens = mx.concatenate([tokens, sparse_prompt_embeddings], axis=1)

        # Add dense prompt to image embeddings
        src = image_embeddings + dense_prompt_embeddings

        # Run two-way transformer
        hs, src = self.transformer(src, image_pe, tokens)

        # Extract token outputs
        iou_token_out = hs[:, 0:1]
        mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens]
        obj_score_token_out = hs[:, 1 + self.num_mask_tokens : 2 + self.num_mask_tokens]

        # Upscale image features
        HW = src.shape[1]
        H = W = int(HW**0.5)
        src = src.reshape(B, H, W, d)

        upscaled = self.upscale_conv1(src)  # (B, 2H, 2W, D/4)
        upscaled = self.upscale_layer_norm(upscaled)
        upscaled = nn.gelu(upscaled)

        # Add s1 high-res skip (144x144 level, D/4=64 channels)
        if high_res_features is not None and len(high_res_features) >= 1:
            s1_feat = self.conv_s1(high_res_features[0])  # (B, 144, 144, 64)
            if s1_feat.shape[1:3] == upscaled.shape[1:3]:
                upscaled = upscaled + s1_feat

        upscaled = self.upscale_conv2(upscaled)  # (B, 4H, 4W, D/8)
        upscaled = nn.gelu(upscaled)

        # Add s0 high-res skip (288x288 level, D/8=32 channels)
        if high_res_features is not None and len(high_res_features) >= 2:
            s0_feat = self.conv_s0(high_res_features[1])  # (B, 288, 288, 32)
            if s0_feat.shape[1:3] == upscaled.shape[1:3]:
                upscaled = upscaled + s0_feat

        B, H_up, W_up, C_up = upscaled.shape
        upscaled_flat = upscaled.reshape(B, H_up * W_up, C_up)

        # Generate masks via hypernetworks
        masks = []
        for i in range(self.num_mask_tokens):
            hyper_out = self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, i]
            )  # (B, C_up)
            mask = (upscaled_flat * hyper_out[:, None, :]).sum(
                axis=-1
            )  # (B, H_up*W_up)
            masks.append(mask.reshape(B, 1, H_up, W_up))

        masks = mx.concatenate(masks, axis=1)  # (B, num_mask_tokens, H_up, W_up)

        # IoU prediction
        iou_pred = self.iou_prediction_head(iou_token_out.squeeze(1))

        # Object score
        obj_score = self.pred_obj_score_head(obj_score_token_out.squeeze(1))

        # Select masks based on multimask_output
        if multimask_output:
            out_masks = masks[:, 1:]  # skip first (low-res) mask
            out_iou = iou_pred[:, 1:]
        else:
            out_masks = masks[:, 0:1]
            out_iou = iou_pred[:, 0:1]

        return out_masks, out_iou, hs, obj_score


class OutputMLP(nn.Module):
    """3-layer MLP for output heads.

    Used for output_hypernetworks_mlps, iou_prediction_head, pred_obj_score_head.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim)]
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.proj_in(x))
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.proj_out(x)
