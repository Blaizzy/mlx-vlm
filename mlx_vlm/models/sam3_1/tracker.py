"""SAM 3.1 Multiplex Tracker: dual decoder, multiplex embeddings.

Weight keys: tracker_model.*
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..sam3.tracker import DownsampleConvBlock, MemoryFuser
from .config import TrackerConfig
from .sam_components import (
    DecoupledMemoryAttention,
    MultiplexMaskDecoder,
    PositionalEmbedding,
    SAMPromptEncoder,
)


class MultiplexMaskDownSampler(nn.Module):
    """Mask downsampler for multiplex: 32 input channels (16 objects × 2).

    Weight keys: tracker_model.memory_encoder.mask_downsampler.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        first_ch = config.mask_downsampler_first_channels  # 16
        kernel_size = config.mask_downsampler_kernel_size
        stride = config.mask_downsampler_stride
        padding = config.mask_downsampler_padding
        embed_dim = config.mask_downsampler_embed_dim

        # Progressive: 16 -> 64 -> 256 -> 1024 (from checkpoint)
        # Input channels = multiplex_count * 2 = 32
        channels = [first_ch, first_ch * 4, first_ch * 16, first_ch * 64]
        self.layers = []
        in_ch = first_ch * 2  # 32 input channels
        for out_ch in channels:
            self.layers.append(
                DownsampleConvBlock(in_ch, out_ch, kernel_size, stride, padding)
            )
            in_ch = out_ch

        self.final_conv = nn.Conv2d(channels[-1], embed_dim, kernel_size=1, bias=True)

    def __call__(self, masks: mx.array) -> mx.array:
        x = masks
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class MultiplexMemoryEncoder(nn.Module):
    """Memory encoder for SAM 3.1 multiplex.

    Weight keys: tracker_model.memory_encoder.*
    Note: SAM 3.1 uses dim=out_dim=256, so no separate output projection needed.
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        dim = config.memory_encoder_hidden_size

        self.mask_downsampler = MultiplexMaskDownSampler(config)
        self.memory_fuser = MemoryFuser(config)
        self.feature_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def __call__(self, features: mx.array, masks: mx.array) -> mx.array:
        mask_features = self.mask_downsampler(masks)
        features = self.feature_projection(features)
        fused = features + mask_features
        return self.memory_fuser(fused)


class ObjectPointerMLP(nn.Module):
    """Projects SAM output tokens to object pointers.

    Weight keys: tracker_model.obj_ptr_proj.* or tracker_model.interactive_obj_ptr_proj.*
    """

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


class MultiplexTrackerModel(nn.Module):
    """SAM 3.1 multiplex tracker with dual decoder.

    Weight keys: tracker_model.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.config = config
        d = config.memory_attention_hidden_size
        mem_dim = config.memory_encoder_output_channels
        M = config.multiplex_count

        # Interactive SAM components (for point/box prompts — single object, 4 mask outputs)
        from .config import TrackerMaskDecoderConfig

        self.interactive_sam_prompt_encoder = SAMPromptEncoder(
            config.prompt_encoder_config
        )
        interactive_cfg = TrackerMaskDecoderConfig(
            **{
                **config.mask_decoder_config.__dict__,
                "multiplex_count": 1,
                "num_multimask_outputs": 4,
            }  # interactive uses 4 mask outputs
        )
        self.interactive_sam_mask_decoder = MultiplexMaskDecoder(interactive_cfg)

        # Propagation SAM mask decoder (multiplex: 16 objects)
        self.sam_mask_decoder = MultiplexMaskDecoder(config.mask_decoder_config)

        # Memory components
        self.memory_attention = DecoupledMemoryAttention(config)
        self.memory_encoder = MultiplexMemoryEncoder(config)

        # Object pointer projections
        self.obj_ptr_proj = ObjectPointerMLP(d)
        self.interactive_obj_ptr_proj = ObjectPointerMLP(d)

        # Learned embeddings
        self.memory_temporal_positional_encoding = mx.zeros(
            (config.num_maskmem, 1, 1, d)
        )
        self.temporal_positional_encoding_projection_layer = nn.Linear(d, d)

        # Multiplex-specific embeddings
        self.output_valid_embed = mx.zeros((M, d))
        self.output_invalid_embed = mx.zeros((M, d))
        self.no_obj_embed_spatial = mx.zeros((M, d))
        self.no_obj_ptr_linear = nn.Linear(d, d)
        self.interactivity_no_mem_embed = mx.zeros((1, 1, d))

        # Image positional encoding
        self.image_pe_layer = PositionalEmbedding(d // 2)
        self.shared_image_embedding = PositionalEmbedding(d // 2)

        # Interactive mask downsample
        self.interactive_mask_downsample = nn.Conv2d(
            1, 1, kernel_size=4, stride=4, bias=True
        )

    def track_step(
        self,
        current_features: mx.array,
        memory_bank: Optional[List[mx.array]] = None,
        prompt_points: Optional[Tuple[mx.array, mx.array]] = None,
        prompt_boxes: Optional[mx.array] = None,
        prompt_masks: Optional[mx.array] = None,
        multimask_output: bool = False,
        high_res_features: Optional[List[mx.array]] = None,
    ) -> Dict[str, mx.array]:
        """Run one tracking step."""
        B, H, W, D = current_features.shape
        src = current_features.reshape(B, H * W, D)

        # Memory attention
        if memory_bank and len(memory_bank) > 0:
            memory = mx.concatenate(memory_bank, axis=1)
            src = self.memory_attention(src, memory)

        # Image positional encoding (resize if resolution differs from training)
        image_pe = self.interactive_sam_prompt_encoder.get_dense_pe()
        pe_len = image_pe.shape[1]
        if pe_len != H * W:
            pe_side = int(pe_len**0.5)
            image_pe = image_pe.reshape(1, pe_side, pe_side, D)
            image_pe = (
                mx.broadcast_to(image_pe[:, :H, :W, :], (B, H, W, D))
                if H <= pe_side
                else nn.Upsample(
                    scale_factor=(H / pe_side, W / pe_side), mode="nearest"
                )(image_pe)
            )
            image_pe = image_pe.reshape(B, H * W, D)
        else:
            image_pe = mx.broadcast_to(image_pe, (B, H * W, D))

        # Encode prompts
        sparse_emb, dense_emb = self.interactive_sam_prompt_encoder(
            points=prompt_points,
            boxes=prompt_boxes,
            masks=prompt_masks,
        )

        # Predict masks
        masks, iou_pred, sam_tokens, obj_score = self.sam_mask_decoder(
            image_embeddings=src,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
            high_res_features=high_res_features,
        )

        # Object pointer
        obj_ptr = self.obj_ptr_proj(sam_tokens[:, 0])

        # Simplified: return first multiplex slot for single-object tracking
        if masks.ndim == 5:
            masks = masks[:, 0]  # (B, num_masks, H, W)
            iou_pred = iou_pred[:, 0]
            obj_score = obj_score[:, 0]

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "obj_scores": obj_score,
            "object_pointer": obj_ptr,
        }
