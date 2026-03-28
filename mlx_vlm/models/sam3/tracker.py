"""SAM3 Tracker: SAM2-style memory-based tracker for video segmentation.

Weight keys: tracker_model.*, tracker_neck.*
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import TrackerConfig
from .sam_components import LayerNorm2d, RoPEAttention, SAMMaskDecoder, SAMPromptEncoder

# ---------------------------------------------------------------------------
# Memory Components
# ---------------------------------------------------------------------------


class SimpleMaskDownSampler(nn.Module):
    """Progressive conv downsampling for mask encoding.

    Weight keys: tracker_model.memory_encoder.mask_downsampler.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        embed_dim = config.mask_downsampler_embed_dim
        kernel_size = config.mask_downsampler_kernel_size
        stride = config.mask_downsampler_stride
        padding = config.mask_downsampler_padding

        # 4 progressive downsampling layers: 1->4->16->64->256
        channels = [1, 4, 16, 64, embed_dim]
        self.layers = []
        for i in range(4):
            self.layers.append(
                DownsampleConvBlock(
                    channels[i], channels[i + 1], kernel_size, stride, padding
                )
            )

        self.final_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

    def __call__(self, masks: mx.array) -> mx.array:
        """
        Args:
            masks: (B, H, W, 1) binary masks
        Returns:
            (B, H', W', embed_dim) downsampled mask features
        """
        x = masks
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return x


class DownsampleConvBlock(nn.Module):
    """Single conv + layernorm + GELU block."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding
        )
        self.layer_norm = LayerNorm2d(out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.layer_norm(x)
        x = nn.gelu(x)
        return x


class CXBlock(nn.Module):
    """ConvNeXt-style block with depthwise conv.

    Weight keys: tracker_model.memory_encoder.memory_fuser.layers.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        dim = config.memory_fuser_embed_dim
        kernel_size = config.memory_fuser_kernel_size
        padding = config.memory_fuser_padding

        self.depthwise_conv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.layer_norm = LayerNorm2d(dim)
        self.pointwise_conv1 = nn.Linear(dim, config.memory_fuser_intermediate_dim)
        self.pointwise_conv2 = nn.Linear(config.memory_fuser_intermediate_dim, dim)
        self.scale = mx.ones((dim,)) * config.memory_fuser_layer_scale_init_value

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C)
        """
        residual = x
        x = self.depthwise_conv(x)
        x = self.layer_norm(x)
        B, H, W, C = x.shape
        x = x.reshape(B * H * W, C)
        x = self.pointwise_conv1(x)
        x = nn.gelu(x)
        x = self.pointwise_conv2(x)
        x = x.reshape(B, H, W, C)
        x = self.scale * x
        return residual + x


class MemoryFuser(nn.Module):
    """Stack of CXBlocks for fusing mask and image features.

    Weight keys: tracker_model.memory_encoder.memory_fuser.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.layers = [CXBlock(config) for _ in range(config.memory_fuser_num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    """Encodes image features + mask into compressed memory.

    Weight keys: tracker_model.memory_encoder.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        dim = config.memory_encoder_hidden_size
        out_dim = config.memory_encoder_output_channels

        self.mask_downsampler = SimpleMaskDownSampler(config)
        self.memory_fuser = MemoryFuser(config)
        self.feature_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.projection = nn.Conv2d(dim, out_dim, kernel_size=1, bias=True)

    def __call__(
        self,
        features: mx.array,
        masks: mx.array,
    ) -> mx.array:
        """
        Args:
            features: (B, H, W, D) backbone features
            masks: (B, H_mask, W_mask, 1) predicted masks
        Returns:
            (B, H', W', out_dim) compressed memory
        """
        mask_features = self.mask_downsampler(masks)

        # Resize mask features to match backbone features if needed
        features = self.feature_projection(features)
        fused = features + mask_features

        fused = self.memory_fuser(fused)
        memory = self.projection(fused)

        return memory


# ---------------------------------------------------------------------------
# Memory Attention
# ---------------------------------------------------------------------------


class MemoryAttentionLayer(nn.Module):
    """Single layer of memory attention: self-attn (RoPE) + cross-attn (RoPE) + FFN.

    Weight keys per layer:
        self_attn.{q,k,v,o}_proj.{weight,bias}
        cross_attn_image.{q,k,v,o}_proj.{weight,bias}
        layer_norm1.{weight,bias}
        layer_norm2.{weight,bias}
        layer_norm3.{weight,bias}
        linear1.{weight,bias}
        linear2.{weight,bias}
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        d = config.memory_attention_hidden_size

        self.self_attn = RoPEAttention(
            hidden_size=d,
            num_heads=config.memory_attention_num_attention_heads,
            downsample_rate=config.memory_attention_downsample_rate,
            feat_sizes=tuple(config.memory_attention_rope_feat_sizes),
            rope_theta=config.memory_attention_rope_theta,
        )
        self.cross_attn_image = RoPEAttention(
            hidden_size=d,
            num_heads=config.memory_attention_num_attention_heads,
            downsample_rate=config.memory_attention_downsample_rate,
            feat_sizes=tuple(config.memory_attention_rope_feat_sizes),
            rope_theta=config.memory_attention_rope_theta,
            kv_dim=config.memory_encoder_output_channels,
            rope_k_repeat=True,
        )

        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.layer_norm3 = nn.LayerNorm(d)

        self.linear1 = nn.Linear(d, config.memory_attention_feed_forward_hidden_size)
        self.linear2 = nn.Linear(config.memory_attention_feed_forward_hidden_size, d)

    def __call__(
        self,
        src: mx.array,
        memory: mx.array,
    ) -> mx.array:
        """
        Args:
            src: (B, HW, D) current frame features
            memory: (B, N_mem, mem_dim) memory features
        """
        # Self-attention with RoPE
        src2 = self.self_attn(src, src, src)
        src = src + src2
        src = self.layer_norm1(src)

        # Cross-attention to memory with RoPE
        src2 = self.cross_attn_image(src, memory, memory)
        src = src + src2
        src = self.layer_norm2(src)

        # FFN
        src2 = self.linear2(nn.relu(self.linear1(src)))
        src = src + src2
        src = self.layer_norm3(src)

        return src


class MemoryAttention(nn.Module):
    """Memory attention module with multiple layers.

    Weight keys: tracker_model.memory_attention.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.layers = [
            MemoryAttentionLayer(config)
            for _ in range(config.memory_attention_num_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.memory_attention_hidden_size)

    def __call__(
        self,
        src: mx.array,
        memory: mx.array,
    ) -> mx.array:
        """
        Args:
            src: (B, HW, D) current frame features
            memory: (B, N_mem, mem_dim) memory bank
        Returns:
            (B, HW, D) attended features
        """
        for layer in self.layers:
            src = layer(src, memory)
        src = self.layer_norm(src)
        return src


# ---------------------------------------------------------------------------
# Object Pointer
# ---------------------------------------------------------------------------


class ObjectPointerMLP(nn.Module):
    """Projects SAM output tokens to object pointers.

    Weight keys: tracker_model.object_pointer_proj.*
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_in = nn.Linear(hidden_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size)]
        self.proj_out = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.proj_in(x))
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.proj_out(x)


# ---------------------------------------------------------------------------
# Tracker Base
# ---------------------------------------------------------------------------


class TrackerModel(nn.Module):
    """SAM2-style memory-based tracker.

    Weight keys: tracker_model.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.config = config
        d = config.memory_attention_hidden_size
        mem_dim = config.memory_encoder_output_channels
        image_size = config.image_size
        feat_size = image_size // config.vision_config.backbone_config.patch_size

        # SAM components
        self.prompt_encoder = SAMPromptEncoder(config.prompt_encoder_config)
        self.mask_decoder = SAMMaskDecoder(config.mask_decoder_config)

        # Memory components
        self.memory_attention = MemoryAttention(config)
        self.memory_encoder = MemoryEncoder(config)

        # Learned embeddings
        self.no_memory_embedding = mx.zeros((1, 1, d))
        self.no_memory_positional_encoding = mx.zeros((1, 1, d))
        self.no_object_pointer = mx.zeros((1, d))

        # Memory temporal positional encoding
        self.memory_temporal_positional_encoding = mx.zeros(
            (config.num_maskmem, 1, 1, mem_dim)
        )

        # Object pointer
        self.object_pointer_proj = ObjectPointerMLP(d)

        # Mask downsampling (for comparing masks)
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4, bias=True)

        # Temporal positional encoding projection
        self.temporal_positional_encoding_projection_layer = nn.Linear(d, mem_dim)

        # Shared image embedding for positional encoding
        self.shared_image_embedding = SharedImageEmbedding(d // 2)

        # Occlusion embedding
        if config.enable_occlusion_spatial_embedding:
            self.occlusion_spatial_embedding_parameter = mx.zeros((1, mem_dim))

    def encode_image(self, backbone_features: mx.array) -> mx.array:
        """Flatten backbone features for memory attention."""
        B, H, W, C = backbone_features.shape
        return backbone_features.reshape(B, H * W, C)

    def track_step(
        self,
        current_features: mx.array,
        memory_bank: Optional[List[mx.array]] = None,
        memory_pos: Optional[List[mx.array]] = None,
        prompt_points: Optional[Tuple[mx.array, mx.array]] = None,
        prompt_boxes: Optional[mx.array] = None,
        prompt_masks: Optional[mx.array] = None,
        multimask_output: bool = False,
        high_res_features: Optional[List[mx.array]] = None,
    ) -> Dict[str, mx.array]:
        """Run one tracking step.

        Args:
            current_features: (B, H, W, D) backbone features for current frame
            memory_bank: list of (B, HW, mem_dim) past memory features
            memory_pos: list of temporal positional encodings
            prompt_*: optional prompt inputs for this frame
        Returns:
            dict with masks, iou_scores, obj_scores, object_pointer, memory
        """
        B, H, W, D = current_features.shape

        # Flatten current features
        src = current_features.reshape(B, H * W, D)

        # If we have memory, attend to it
        if memory_bank and len(memory_bank) > 0:
            memory = mx.concatenate(memory_bank, axis=1)
            src = self.memory_attention(src, memory)

        # Get image positional encoding
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, HW, D)
        image_pe = mx.broadcast_to(image_pe, (B, H * W, D))

        # Encode prompts
        sparse_emb, dense_emb = self.prompt_encoder(
            points=prompt_points,
            boxes=prompt_boxes,
            masks=prompt_masks,
        )

        # Predict masks
        masks, iou_pred, sam_tokens, obj_score = self.mask_decoder(
            image_embeddings=src,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
            high_res_features=high_res_features,
        )

        # Compute object pointer from SAM tokens
        obj_ptr = self.object_pointer_proj(sam_tokens[:, 0])

        # Encode memory from current frame
        mask_for_mem = masks[:, 0:1]  # (B, 1, H_mask, W_mask)
        mask_for_mem = mask_for_mem.transpose(0, 2, 3, 1)  # (B, H, W, 1)

        # Resize mask to 1152x1152 so downsampler (stride=16) produces 72x72
        target = 1152
        if mask_for_mem.shape[1] != target:
            up = nn.Upsample(
                scale_factor=(
                    target / mask_for_mem.shape[1],
                    target / mask_for_mem.shape[2],
                ),
                mode="nearest",
            )
            mask_for_mem = up(mask_for_mem)

        memory = self.memory_encoder(current_features, mask_for_mem)
        B_m, H_m, W_m, C_m = memory.shape
        memory = memory.reshape(B_m, H_m * W_m, C_m)

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "obj_scores": obj_score,
            "object_pointer": obj_ptr,
            "memory": memory,
        }


class SharedImageEmbedding(nn.Module):
    """Shared positional embedding for tracker.

    Weight keys: tracker_model.shared_image_embedding.positional_embedding
    """

    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.positional_embedding = mx.zeros((2, num_pos_feats))

    def __call__(self, size: Tuple[int, int]) -> mx.array:
        H, W = size
        grid_y = mx.arange(H).astype(mx.float32) / H
        grid_x = mx.arange(W).astype(mx.float32) / W
        gy, gx = mx.meshgrid(grid_y, grid_x, indexing="ij")
        coords = mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
        coords = 2 * coords - 1
        coords = coords @ self.positional_embedding
        coords = 2 * math.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)
