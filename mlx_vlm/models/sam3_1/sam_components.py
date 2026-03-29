"""SAM 3.1 tracker components: MultiplexMaskDecoder + Decoupled attention.

Reuses SAMPromptEncoder, TwoWayTransformer etc. from SAM 3.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..sam3.position import apply_rotary_enc_1d, init_2d_freqs
from ..sam3.sam_components import (  # noqa: F401
    LayerNorm2d,
    OutputMLP,
    PositionalEmbedding,
    SAMPromptEncoder,
    TwoWayTransformer,
)
from .config import TrackerMaskDecoderConfig


class MultiplexMaskDecoder(nn.Module):
    """SAM mask decoder that processes multiplex_count objects simultaneously.

    Key differences from SAMMaskDecoder:
    - iou_token: (multiplex_count, D) instead of (1, D)
    - mask_tokens: (multiplex_count * num_masks, D) instead of (num_masks, D)
    - obj_score_token: (multiplex_count, D) instead of (1, D)
    - Output: (B, multiplex_count, num_masks, H, W)

    Weight keys: tracker_model.sam_mask_decoder.*
    """

    def __init__(self, config: TrackerMaskDecoderConfig):
        super().__init__()
        d = config.hidden_size
        self.multiplex_count = config.multiplex_count
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs  # 3

        # Tokens sized for multiplex
        self.iou_token = nn.Embedding(config.multiplex_count, d)  # (16, 256)
        self.mask_tokens = nn.Embedding(
            config.multiplex_count * self.num_mask_tokens, d
        )  # (48, 256)
        self.obj_score_token = nn.Embedding(config.multiplex_count, d)  # (16, 256)

        # TwoWayTransformer (same architecture as SAM 3)
        self.transformer = TwoWayTransformer(
            hidden_size=d,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            mlp_dim=config.mlp_dim,
            attention_downsample_rate=config.attention_downsample_rate,
        )

        # Output MLPs — shared across multiplex slots (only num_mask_tokens MLPs)
        self.output_hypernetworks_mlps = [
            OutputMLP(d, d, d // 8) for _ in range(self.num_mask_tokens)
        ]
        self.iou_prediction_head = OutputMLP(d, d, self.num_mask_tokens)
        self.pred_obj_score_head = OutputMLP(d, d, 1)

        # Upscaling
        self.upscale_conv1 = nn.ConvTranspose2d(d, d // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(d // 4, d // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(d // 4)

        # High-res skip connections
        self.conv_s0 = nn.Conv2d(d, d // 8, kernel_size=1, bias=True)
        self.conv_s1 = nn.Conv2d(d, d // 4, kernel_size=1, bias=True)

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
        Returns:
            masks, iou_pred, sam_tokens, obj_score
        """
        B = image_embeddings.shape[0]
        d = image_embeddings.shape[-1]

        # Build token sequence: iou + mask + obj_score + sparse prompts
        tokens = mx.concatenate(
            [
                mx.broadcast_to(
                    self.iou_token.weight[None], (B, self.multiplex_count, d)
                ),
                mx.broadcast_to(
                    self.mask_tokens.weight[None],
                    (B, self.multiplex_count * self.num_mask_tokens, d),
                ),
                mx.broadcast_to(
                    self.obj_score_token.weight[None], (B, self.multiplex_count, d)
                ),
            ],
            axis=1,
        )
        tokens = mx.concatenate([tokens, sparse_prompt_embeddings], axis=1)

        src = image_embeddings + dense_prompt_embeddings
        hs, src = self.transformer(src, image_pe, tokens)

        # Extract outputs
        M = self.multiplex_count
        N_mask = self.num_mask_tokens
        iou_out = hs[:, :M]
        mask_out = hs[:, M : M + M * N_mask]
        obj_out = hs[:, M + M * N_mask : 2 * M + M * N_mask]

        # Upscale
        HW = src.shape[1]
        H = W = int(HW**0.5)
        src = src.reshape(B, H, W, d)

        upscaled = self.upscale_conv1(src)
        upscaled = self.upscale_layer_norm(upscaled)
        upscaled = nn.gelu(upscaled)

        if high_res_features is not None and len(high_res_features) >= 1:
            s1_feat = self.conv_s1(high_res_features[0])
            if s1_feat.shape[1:3] == upscaled.shape[1:3]:
                upscaled = upscaled + s1_feat

        upscaled = self.upscale_conv2(upscaled)
        upscaled = nn.gelu(upscaled)

        if high_res_features is not None and len(high_res_features) >= 2:
            s0_feat = self.conv_s0(high_res_features[1])
            if s0_feat.shape[1:3] == upscaled.shape[1:3]:
                upscaled = upscaled + s0_feat

        B, H_up, W_up, C_up = upscaled.shape
        upscaled_flat = upscaled.reshape(B, H_up * W_up, C_up)

        # Generate masks — MLPs are shared across multiplex slots
        masks = []
        for obj_i in range(M):
            for mask_j in range(N_mask):
                token_idx = obj_i * N_mask + mask_j
                hyper_out = self.output_hypernetworks_mlps[mask_j](
                    mask_out[:, token_idx]
                )
                mask = (upscaled_flat * hyper_out[:, None, :]).sum(axis=-1)
                masks.append(mask.reshape(B, 1, H_up, W_up))
        masks = mx.concatenate(masks, axis=1)  # (B, M*N_mask, H, W)
        masks = masks.reshape(B, M, N_mask, H_up, W_up)

        # IoU prediction — per multiplex slot
        iou_pred = mx.stack(
            [self.iou_prediction_head(iou_out[:, i]) for i in range(M)], axis=1
        )  # (B, M, N_mask)

        # Object score
        obj_score = mx.stack(
            [self.pred_obj_score_head(obj_out[:, i]) for i in range(M)], axis=1
        )  # (B, M, 1)

        if multimask_output:
            out_masks = masks
            out_iou = iou_pred
        else:
            out_masks = masks[:, :, 0:1]
            out_iou = iou_pred[:, :, 0:1]

        return out_masks, out_iou, hs, obj_score


class SimpleRoPEAttention(nn.Module):
    """RoPE attention without Q/K/V projections (caller handles them).

    Applies 2D rotary position encoding and scaled dot-product attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feat_sizes: Tuple[int, int] = (72, 72),
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rope_k_repeat = rope_k_repeat

        self._freqs_cos, self._freqs_sin = init_2d_freqs(
            hidden_size // num_heads, feat_sizes[0], feat_sizes[1], theta=rope_theta
        )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        """
        Args:
            q: (B, N_q, D) pre-projected queries
            k: (B, N_k, D) pre-projected keys
            v: (B, N_k, D) pre-projected values
            num_k_exclude_rope: exclude last N keys from RoPE (for object pointers)
        """
        B, N_q, _ = q.shape
        N_k = k.shape[1]

        q = q.reshape(B, N_q, self.num_heads, self.head_dim)
        k = k.reshape(B, N_k, self.num_heads, self.head_dim)
        v = v.reshape(B, N_k, self.num_heads, self.head_dim)

        # Apply RoPE (exclude last num_k_exclude_rope keys)
        if num_k_exclude_rope > 0:
            k_rope = k[:, :-num_k_exclude_rope]
            k_no_rope = k[:, -num_k_exclude_rope:]
        else:
            k_rope = k
            k_no_rope = None

        q, k_rope = apply_rotary_enc_1d(
            q,
            k_rope,
            self._freqs_cos,
            self._freqs_sin,
            repeat_freqs_k=self.rope_k_repeat,
        )

        if k_no_rope is not None:
            k = mx.concatenate([k_rope, k_no_rope], axis=1)
        else:
            k = k_rope

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return out.transpose(0, 2, 1, 3).reshape(B, N_q, -1)


class DecoupledMemoryAttentionLayer(nn.Module):
    """Decoupled transformer layer for SAM 3.1 memory attention.

    Has separate Q/K/V/O projections for self-attention and cross-attention,
    plus additional image_cross_attn projections.

    Weight keys per layer:
        self_attn_{q,k,v,out}_proj.{weight,bias}
        cross_attn_{q,k,v,out}_proj.{weight,bias}
        image_cross_attn_{q,k}_proj.{weight,bias}
        linear{1,2}.{weight,bias}
        norm{1,2,3}.{weight,bias}
    """

    def __init__(
        self,
        config,
        self_attn_rope: SimpleRoPEAttention,
        cross_attn_rope: SimpleRoPEAttention,
    ):
        super().__init__()
        d = config.memory_attention_hidden_size

        # Self-attention projections
        self.self_attn_q_proj = nn.Linear(d, d)
        self.self_attn_k_proj = nn.Linear(d, d)
        self.self_attn_v_proj = nn.Linear(d, d)
        self.self_attn_out_proj = nn.Linear(d, d)
        self.self_attention_rope = self_attn_rope

        # Cross-attention projections
        self.cross_attn_q_proj = nn.Linear(d, d)
        self.cross_attn_k_proj = nn.Linear(d, d)
        self.cross_attn_v_proj = nn.Linear(d, d)
        self.cross_attn_out_proj = nn.Linear(d, d)
        self.cross_attention_rope = cross_attn_rope

        # Image cross-attention (additional Q/K for image features)
        self.image_cross_attn_q_proj = nn.Linear(d, d)
        self.image_cross_attn_k_proj = nn.Linear(d, d)

        # FFN
        self.linear1 = nn.Linear(d, config.memory_attention_feed_forward_hidden_size)
        self.linear2 = nn.Linear(config.memory_attention_feed_forward_hidden_size, d)

        # Norms (pre-norm)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def __call__(
        self,
        src: mx.array,
        memory: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        """
        Args:
            src: (B, HW, D) current frame features
            memory: (B, N_mem, D) memory features
            num_k_exclude_rope: keys to exclude from RoPE
        """
        # 1. Self-attention with RoPE (pre-norm)
        residual = src
        src_normed = self.norm1(src)
        q = self.self_attn_q_proj(src_normed)
        k = self.self_attn_k_proj(src_normed)
        v = self.self_attn_v_proj(src_normed)
        src2 = self.self_attention_rope(q, k, v)
        src2 = self.self_attn_out_proj(src2)
        src = residual + src2

        # 2. Cross-attention to memory with RoPE (pre-norm)
        residual = src
        src_normed = self.norm2(src)
        q = self.cross_attn_q_proj(src_normed)
        k = self.cross_attn_k_proj(memory)
        v = self.cross_attn_v_proj(memory)

        # Add image cross-attention projections
        q = q + self.image_cross_attn_q_proj(src_normed)
        k = k + self.image_cross_attn_k_proj(memory)

        src2 = self.cross_attention_rope(q, k, v, num_k_exclude_rope=num_k_exclude_rope)
        src2 = self.cross_attn_out_proj(src2)
        src = residual + src2

        # 3. FFN (pre-norm)
        residual = src
        src2 = self.linear2(nn.relu(self.linear1(self.norm3(src))))
        src = residual + src2

        mx.eval(src)  # Free attention intermediates
        return src


class DecoupledMemoryAttention(nn.Module):
    """SAM 3.1 memory attention with decoupled projections.

    Weight keys: tracker_model.memory_attention.*
    """

    def __init__(self, config):
        super().__init__()
        d = config.memory_attention_hidden_size
        feat_sizes = tuple(config.memory_attention_rope_feat_sizes)
        theta = config.memory_attention_rope_theta

        self.layers = []
        for _ in range(config.memory_attention_num_layers):
            self_rope = SimpleRoPEAttention(
                d,
                config.memory_attention_num_attention_heads,
                feat_sizes=feat_sizes,
                rope_theta=theta,
            )
            cross_rope = SimpleRoPEAttention(
                d,
                config.memory_attention_num_attention_heads,
                feat_sizes=feat_sizes,
                rope_theta=theta,
                rope_k_repeat=True,
            )
            self.layers.append(
                DecoupledMemoryAttentionLayer(config, self_rope, cross_rope)
            )

        self.layer_norm = nn.LayerNorm(d)

    def __call__(
        self,
        src: mx.array,
        memory: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        for layer in self.layers:
            src = layer(src, memory, num_k_exclude_rope=num_k_exclude_rope)
        return self.layer_norm(src)
