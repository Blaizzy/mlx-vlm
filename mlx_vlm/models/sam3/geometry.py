"""SAM3 Geometry Encoder for box/point/mask visual prompts.

Weight keys: detector_model.geometry_encoder.*
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import GeometryEncoderConfig
from .encoder import MLP, MultiheadAttention


class GeometryEncoderLayer(nn.Module):
    """Transformer encoder layer for geometry features.

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

        self.self_attn = MultiheadAttention(
            d, config.num_attention_heads, config.dropout
        )
        self.cross_attn = MultiheadAttention(
            d, config.num_attention_heads, config.dropout
        )

        self.layer_norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.layer_norm3 = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.mlp = MLP(d, config.intermediate_size, config.hidden_act)

    def __call__(
        self,
        src: mx.array,
        memory: mx.array,
        memory_pos: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            src: (B, N, D) prompt features
            memory: (B, HW, D) image features for cross-attention
            memory_pos: (B, HW, D) positional encoding
        """
        # Self-attention
        src2 = self.self_attn(src, src, src)
        src = src + src2
        src = self.layer_norm1(src)

        # Cross-attention to image features
        key = memory + memory_pos if memory_pos is not None else memory
        src2 = self.cross_attn(src, key, memory)
        src = src + src2
        src = self.layer_norm2(src)

        # FFN
        src2 = self.mlp(src)
        src = src + src2
        src = self.layer_norm3(src)

        return src


class GeometryEncoder(nn.Module):
    """Encodes geometric prompts (boxes, points) into feature vectors.

    Weight keys: detector_model.geometry_encoder.*
    """

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__()
        d = config.hidden_size

        # Direct projections for points and boxes
        self.boxes_direct_project = nn.Linear(4, d)
        self.boxes_pool_project = nn.Conv2d(
            d, d, kernel_size=config.roi_size, bias=True
        )
        self.boxes_pos_enc_project = nn.Linear(d + 2, d)

        # Label and CLS embeddings
        self.label_embed = nn.Embedding(2, d)  # 2 labels (point/box)
        self.cls_embed = nn.Embedding(1, d)

        # Final projection + norms
        self.final_proj = nn.Linear(d, d)

        # Encoder layers
        self.layers = [GeometryEncoderLayer(config) for _ in range(config.num_layers)]

        self.output_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.prompt_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.vision_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)

    def encode_boxes(self, boxes: mx.array) -> mx.array:
        """Encode box prompts.

        Args:
            boxes: (B, N_box, 4) normalized box coordinates [cx, cy, w, h]
        Returns:
            (B, N_box, D) encoded box features
        """
        return self.boxes_direct_project(boxes)

    def __call__(
        self,
        boxes: Optional[mx.array] = None,
        memory: Optional[mx.array] = None,
        memory_pos: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            boxes: (B, N_box, 4) box prompts
            memory: (B, HW, D) image features
            memory_pos: (B, HW, D) positional encoding
        Returns:
            (B, N_prompt, D) encoded prompt features
        """
        B = memory.shape[0] if memory is not None else boxes.shape[0]
        d = self.cls_embed.weight.shape[-1]

        tokens = []

        # CLS token
        cls = mx.broadcast_to(self.cls_embed.weight[None], (B, 1, d))
        tokens.append(cls)

        # Box tokens
        if boxes is not None and boxes.shape[1] > 0:
            box_feats = self.encode_boxes(boxes)
            tokens.append(box_feats)

        src = mx.concatenate(tokens, axis=1)

        # Run through encoder layers with cross-attention to image
        for layer in self.layers:
            src = layer(src, memory, memory_pos)

        src = self.output_layer_norm(src)
        src = self.final_proj(src)

        return src
