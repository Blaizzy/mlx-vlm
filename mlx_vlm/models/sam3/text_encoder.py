"""SAM3 CLIP Text Encoder.

Weight keys: detector_model.text_encoder.text_model.* and detector_model.text_projection.*
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TextEncoderConfig


class CLIPAttention(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, N, C = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask=mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.layers = [
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPTextModel(nn.Module):
    """CLIP text model.

    Weight keys: detector_model.text_encoder.text_model.*
    """

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) binary mask
        Returns:
            hidden_states: (B, seq_len, hidden_size)
        """
        x = self.embeddings(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        causal_mask = causal_mask.astype(x.dtype)

        if attention_mask is not None:
            # Combine causal mask with padding mask
            pad_mask = (1 - attention_mask[:, None, None, :].astype(x.dtype)) * -1e9
            causal_mask = causal_mask + pad_mask

        x = self.encoder(x, mask=causal_mask)
        x = self.final_layer_norm(x)
        return x


class TextEncoder(nn.Module):
    """Full text encoder with CLIP model + projection.

    Weight keys:
        detector_model.text_encoder.text_model.* -> self.text_model.*
        detector_model.text_encoder.text_projection.weight -> self.text_projection.weight
    """

    def __init__(self, config: TextEncoderConfig, d_model: int = 256):
        super().__init__()
        self.text_model = CLIPTextModel(config)
        # Projects from CLIP hidden to projection_dim (e.g. 1024 -> 512)
        self.text_projection = nn.Linear(
            config.hidden_size, config.projection_dim, bias=False
        )
        self.d_model = d_model

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            inputs_embeds: (B, seq_len, hidden_size) - raw hidden states
        """
        hidden_states = self.text_model(input_ids, attention_mask)
        return hidden_states


class LanguageModel(nn.Module):
    """Wrapper for mlx-vlm compatibility. SAM3's 'language model' is the CLIP text encoder."""

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.text_encoder = TextEncoder(config)

    def __call__(self, input_ids: mx.array, **kwargs) -> mx.array:
        return self.text_encoder(input_ids, **kwargs)

    @staticmethod
    def sanitize(weights):
        """No-op: all sanitization handled in Model.sanitize."""
        return weights
