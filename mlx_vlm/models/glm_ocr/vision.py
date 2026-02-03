"""GLM-OCR Vision Model.

Vision encoder for GLM-OCR 0.9B parameters.
"""

from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class GLMOCRVisionEmbeddings(nn.Module):
    """Vision embeddings for GLM-OCR."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        
    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Forward pass."""
        patch_embeds = self.patch_embedding(pixel_values)
        # Reshape from (B, C, H, W) to (B, H*W, C)
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.reshape(batch_size, self.embed_dim, -1).transpose(0, 2, 1)
        return patch_embeds


class GLMOCRVisionAttention(nn.Module):
    """Multi-head attention for GLM-OCR vision."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.attention_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.attention_bias)
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass."""
        batch_size, seq_length, _ = hidden_states.shape
        
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_weights = mx.matmul(query, key.transpose(0, 1, 3, 2)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, value)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        attn_output = self.proj(attn_output)
        
        return attn_output


class GLMOCRVisionMLP(nn.Module):
    """MLP for GLM-OCR vision."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation = nn.SiLU()
        
    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Forward pass."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GLMOCRVisionTransformerBlock(nn.Module):
    """Vision transformer block for GLM-OCR."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GLMOCRVisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GLMOCRVisionMLP(config)
        
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass."""
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class VisionModel(nn.Module):
    """Vision model for GLM-OCR."""
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = GLMOCRVisionEmbeddings(config)
        self.encoder = nn.Sequential(
            *[GLMOCRVisionTransformerBlock(config) for _ in range(config.depth)]
        )
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Projection to language model hidden size
        self.proj = nn.Linear(config.hidden_size, config.out_hidden_size, bias=True)
        
    def __call__(
        self,
        pixel_values: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> mx.array:
        """Forward pass."""
        hidden_states = self.embeddings(pixel_values)
        
        for block in self.encoder.layers:
            hidden_states = block(hidden_states, attention_mask)
        
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.proj(hidden_states)
        
        return hidden_states
