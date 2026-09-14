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

    Port of sam3/model/multiplex_mask_decoder.py. Also covers the interactive
    (single-slot) MaskDecoder when multiplex_count=1 with sparse/dense prompts.

    - Token layout: [obj_score(M), iou(M), mask(M * num_per_object), sparse...]
    - multimask_outputs_only: no extra single-mask token (propagation decoder)
    - extra_per_object_embeddings: (B, M, D) added to every mask token
      (used by the output-suppression embeddings)

    Weight keys: tracker_model.sam_mask_decoder.* / tracker_model.interactive_sam_mask_decoder.*
    """

    def __init__(self, config: TrackerMaskDecoderConfig):
        super().__init__()
        d = config.hidden_size
        self.multiplex_count = config.multiplex_count
        self.num_multimask_outputs = config.num_multimask_outputs
        self.multimask_outputs_only = config.multimask_outputs_only
        self.use_multimask_token_for_obj_ptr = config.use_multimask_token_for_obj_ptr

        if self.multimask_outputs_only:
            self.num_mask_output_per_object = self.num_multimask_outputs
        else:
            # +1 for the single (best) mask token
            self.num_mask_output_per_object = self.num_multimask_outputs + 1
        self.num_mask_tokens = self.multiplex_count * self.num_mask_output_per_object

        # Tokens sized for multiplex
        self.iou_token = nn.Embedding(self.multiplex_count, d)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d)
        self.obj_score_token = nn.Embedding(self.multiplex_count, d)

        # TwoWayTransformer (same architecture as SAM 3)
        self.transformer = TwoWayTransformer(
            hidden_size=d,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            mlp_dim=config.mlp_dim,
            attention_downsample_rate=config.attention_downsample_rate,
        )

        # Output MLPs — one per mask token of an object (shared across slots)
        self.output_hypernetworks_mlps = [
            OutputMLP(d, d, d // 8) for _ in range(self.num_mask_output_per_object)
        ]
        self.iou_prediction_head = OutputMLP(d, d, self.num_mask_output_per_object)
        self.pred_obj_score_head = OutputMLP(d, d, 1)

        # Upscaling
        self.upscale_conv1 = nn.ConvTranspose2d(d, d // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(d // 4, d // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(d // 4)

        # 1x1 projections of the high-res FPN levels; applied by the caller
        # before decoding (as in the reference forward_image)
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
        multimask_output: bool,
        high_res_features: Optional[List[mx.array]] = None,
        extra_per_object_embeddings: Optional[mx.array] = None,
        sparse_prompt_embeddings: Optional[mx.array] = None,
        dense_prompt_embeddings: Optional[mx.array] = None,
    ) -> dict:
        """
        Args:
            image_embeddings: (B, HW, D) memory-conditioned image features
            image_pe: (1, HW, D) dense positional encoding
            multimask_output: return all mask tokens (else best/single token)
            high_res_features: [feat_s0, feat_s1] pre-projected by conv_s0/conv_s1
            extra_per_object_embeddings: (B, M, D) added to the mask tokens
            sparse/dense_prompt_embeddings: interactive prompts (multiplex_count=1)

        Returns:
            dict with masks (B, M, K, H, W), iou_pred (B, M, K),
            sam_tokens_out (B, M, K, D), object_score_logits (B, M, 1)
        """
        if self.multimask_outputs_only:
            assert (
                multimask_output
            ), "multimask_output must be True with multimask_outputs_only"

        out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
            extra_per_object_embeddings=extra_per_object_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        masks = out["masks"]  # (B, M, P, H, W)
        iou_pred = out["iou_pred"]  # (B, M, P)
        mask_tokens_out = out["mask_tokens_out"]  # (B, M, P, D)

        # Select the correct mask or masks for output
        if multimask_output:
            if not self.multimask_outputs_only:
                # drop the single-mask token, keep the multimask tokens
                masks = masks[:, :, 1:]
                iou_pred = iou_pred[:, :, 1:]
        elif self.dynamic_multimask_via_stability:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, :, 0:1]
            iou_pred = iou_pred[:, :, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            if self.multimask_outputs_only:
                sam_tokens_out = mask_tokens_out
            else:
                sam_tokens_out = mask_tokens_out[:, :, 1:]
        else:
            # Always take the single-mask token for the object pointer
            sam_tokens_out = mask_tokens_out[:, :, 0:1]

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "sam_tokens_out": sam_tokens_out,
            "object_score_logits": out["object_score_logits"],
        }

    def predict_masks(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        high_res_features: Optional[List[mx.array]] = None,
        extra_per_object_embeddings: Optional[mx.array] = None,
        sparse_prompt_embeddings: Optional[mx.array] = None,
        dense_prompt_embeddings: Optional[mx.array] = None,
    ) -> dict:
        B_img, HW, d = image_embeddings.shape
        M = self.multiplex_count
        P = self.num_mask_output_per_object

        if sparse_prompt_embeddings is not None:
            B = sparse_prompt_embeddings.shape[0]
        else:
            B = B_img

        # Repeat the image embeddings to the token batch size if needed
        if B_img != B:
            assert B_img == 1
            src = mx.broadcast_to(image_embeddings, (B, HW, d))
        else:
            src = image_embeddings
        if dense_prompt_embeddings is not None:
            src = src + dense_prompt_embeddings

        # Token layout: [obj_score(M), iou(M), mask(M * P), sparse...]
        tokens = [
            mx.broadcast_to(self.obj_score_token.weight[None], (B, M, d)),
            mx.broadcast_to(self.iou_token.weight[None], (B, M, d)),
        ]
        mask_tokens = self.mask_tokens.weight.reshape(1, M, P, d)
        if extra_per_object_embeddings is not None:
            mask_tokens = mask_tokens + extra_per_object_embeddings[:, :, None, :]
        else:
            mask_tokens = mx.broadcast_to(mask_tokens, (B, M, P, d))
        tokens.append(mask_tokens.reshape(B, M * P, d))
        if sparse_prompt_embeddings is not None:
            tokens.append(sparse_prompt_embeddings)
        tokens = mx.concatenate(tokens, axis=1)

        image_pe = mx.broadcast_to(image_pe, (B, HW, d))
        hs, src = self.transformer(src, image_pe, tokens)

        obj_score_token_out = hs[:, :M]  # (B, M, D)
        iou_token_out = hs[:, M : 2 * M]  # (B, M, D)
        mask_tokens_out = hs[:, 2 * M : 2 * M + M * P]  # (B, M * P, D)

        # Upscale image features (72 -> 144 -> 288) with high-res skip fusion
        H = W = int(HW**0.5)
        src = src.reshape(B, H, W, d)

        upscaled = self.upscale_conv1(src)
        if high_res_features is not None:
            feat_s0, feat_s1 = high_res_features
            upscaled = upscaled + feat_s1
        upscaled = self.upscale_layer_norm(upscaled)
        upscaled = nn.gelu(upscaled)

        upscaled = self.upscale_conv2(upscaled)
        if high_res_features is not None:
            upscaled = upscaled + feat_s0
        upscaled = nn.gelu(upscaled)

        B, H_up, W_up, C_up = upscaled.shape
        upscaled_flat = upscaled.reshape(B, H_up * W_up, C_up)

        # Hypernetwork projections of the mask tokens: (B, M, P, C_up)
        mask_tokens_out = mask_tokens_out.reshape(B, M, P, d)
        hyper_in = mx.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i])
                for i in range(P)
            ],
            axis=2,
        )

        # Generate masks: (B, M*P, C) @ (B, C, HW) -> (B, M, P, H, W)
        masks = (
            hyper_in.reshape(B, M * P, C_up) @ upscaled_flat.transpose(0, 2, 1)
        ).reshape(B, M, P, H_up, W_up)

        # Per-slot mask quality and object existence predictions
        iou_pred = self.iou_prediction_head(iou_token_out)  # (B, M, P)
        object_score_logits = self.pred_obj_score_head(obj_score_token_out)  # (B, M, 1)

        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens_out": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }

    def _get_stability_scores(self, mask_logits: mx.array) -> mx.array:
        """IoU between upper/lower thresholded masks, per mask."""
        mask_logits = mask_logits.reshape(*mask_logits.shape[:-2], -1)
        delta = self.dynamic_multimask_stability_delta
        area_i = (mask_logits > delta).sum(axis=-1).astype(mx.float32)
        area_u = (mask_logits > -delta).sum(axis=-1).astype(mx.float32)
        return mx.where(area_u > 0, area_i / area_u, mx.array(1.0, mx.float32))

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """Fall back to the best multimask output when the single-mask output
        has a low stability score."""
        B, M = all_mask_logits.shape[:2]
        all_mask_logits = all_mask_logits.reshape(-1, *all_mask_logits.shape[2:])
        all_iou_scores = all_iou_scores.reshape(-1, all_iou_scores.shape[-1])

        # Best mask among multimask output tokens (1..P-1)
        multimask_logits = all_mask_logits[:, 1:]
        multimask_iou = all_iou_scores[:, 1:]
        best_inds = mx.argmax(multimask_iou, axis=-1)  # (B*M,)
        best_multimask_logits = mx.take_along_axis(
            multimask_logits, best_inds[:, None, None, None], axis=1
        )
        best_multimask_iou = mx.take_along_axis(
            multimask_iou, best_inds[:, None], axis=1
        )

        # Single-mask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1]
        singlemask_iou = all_iou_scores[:, 0:1]
        stability = self._get_stability_scores(singlemask_logits)[:, :, None, None]
        is_stable = stability >= self.dynamic_multimask_stability_thresh

        mask_logits_out = mx.where(is_stable, singlemask_logits, best_multimask_logits)
        iou_scores_out = mx.where(
            is_stable[:, :, 0, 0], singlemask_iou, best_multimask_iou
        )

        mask_logits_out = mask_logits_out.reshape(B, M, *mask_logits_out.shape[1:])
        iou_scores_out = iou_scores_out.reshape(B, M, -1)
        return mask_logits_out, iou_scores_out


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
        image: mx.array,
        src: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        memory_image_pos: Optional[mx.array] = None,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        """
        Pre-norm decoupled layer (DecoupledTransformerDecoderLayerv2 port).

        Args:
            image: (1, HW, D) raw current-frame image features (not normed)
            src: (B, HW, D) current frame features (self-attention)
            memory_image: (1, N, D) image features of the memory frames
            memory: (B, N, D) mask-memory features (+ object pointers)
            memory_image_pos: (1, N, D) positional encodings for the memory keys
            num_k_exclude_rope: trailing keys excluded from RoPE (obj pointers)
        """
        # 1. Self-attention with RoPE (pre-norm, no pos enc at attention)
        residual = src
        src_normed = self.norm1(src)
        q = self.self_attn_q_proj(src_normed)
        k = self.self_attn_k_proj(src_normed)
        v = self.self_attn_v_proj(src_normed)
        src2 = self.self_attention_rope(q, k, v)
        src2 = self.self_attn_out_proj(src2)
        src = residual + src2

        # 2. Cross-attention to memory with RoPE (pre-norm)
        # q/k get additional projections of the (raw) image features
        residual = src
        src_normed = self.norm2(src)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(src_normed)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if memory_image_pos is not None:
            # pos enc at cross-attention keys only
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)

        src2 = self.cross_attention_rope(q, k, v, num_k_exclude_rope=num_k_exclude_rope)
        src2 = self.cross_attn_out_proj(src2)
        src = residual + src2

        # 3. FFN (pre-norm, gelu)
        residual = src
        src2 = self.linear2(nn.gelu(self.linear1(self.norm3(src))))
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
        image: mx.array,
        src: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        src_pos: Optional[mx.array] = None,
        memory_pos: Optional[mx.array] = None,
        memory_image_pos: Optional[mx.array] = None,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        """TransformerEncoderDecoupledCrossAttention port (batch-first).

        Args:
            image: (1, HW, D) raw current-frame image features
            src: (B, HW, D) current-frame features
            memory_image: (1, N_img, D) image features of memory frames
            memory: (B, N_mem, D) mask memories + object pointer tokens
            src_pos: (B, HW, D) pos enc added to src at input (scaled by 0.1)
            memory_pos: (B, N_mem, D) pos enc; only its object-pointer tail is
                used (to extend memory_image_pos)
            memory_image_pos: (1, N_img, D) pos enc for the memory image keys
            num_k_exclude_rope: number of trailing object-pointer tokens
        """
        # pos enc at input (scaled by 0.1 as in the reference)
        if src_pos is not None:
            src = src + 0.1 * src_pos

        # Pad the image memories with zeros for the object pointer tokens
        if memory_image.shape[1] != memory.shape[1]:
            pad = memory.shape[1] - memory_image.shape[1]
            assert pad == num_k_exclude_rope
            memory_image = mx.concatenate(
                [
                    memory_image,
                    mx.zeros((memory_image.shape[0], pad, memory_image.shape[2])),
                ],
                axis=1,
            )
            if memory_image_pos is not None and memory_pos is not None:
                memory_image_pos = mx.concatenate(
                    [memory_image_pos, memory_pos[0:1, -pad:]], axis=1
                )

        for layer in self.layers:
            src = layer(
                image,
                src,
                memory_image,
                memory,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
        # use_image_in_output=False: norm the output only
        return self.layer_norm(src)
