"""Language tower for HunyuanOCR (hunyuan_vl).

Implements the HF `HunYuanVLForConditionalGeneration` text stack with:
- xdrope (4D rotary positional encoding)
- QK norm (RMSNorm on queries and keys after projection)
- Gated SiLU MLP
- GQA (grouped query attention)
"""

from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import TextConfig


class HunyuanRotaryEmbedding:
    """Rotary embedding with xdrope (4D positional encoding) support.

    xdrope splits the head dimension into 4 sections [16, 16, 16, 16] and applies
    different positional encodings to each section based on 4D position (base, t, h, w).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling or {}

        # Check for alpha scaling
        alpha = self.rope_scaling.get("alpha", 1.0)
        rope_type = self.rope_scaling.get("type", "default")

        # HF Logic for xdrope/dynamic with alpha:
        # base = theta * alpha ^ (dim / (dim - 2))
        if alpha != 1.0 and rope_type in ["xdrope", "dynamic"]:
            self.base = self.base * (alpha ** (self.dim / (self.dim - 2)))

        # xdrope section defines how head_dim is split for 4D positions
        self.xdrope_section = self.rope_scaling.get("xdrope_section", [16, 16, 16, 16])

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )

        self.inv_freq = inv_freq

    def apply_xdrope(self, freqs: mx.array, xdrope_section: List[int]) -> mx.array:
        """Apply xdrope to 4D rotary embeddings.

        Reorganizes frequency layout from 4D positions to interleaved format.
        Each dimension (base, t, h, w) contributes a slice of the head_dim.

        Args:
            freqs: (4, bs, seq_len, head_dim // 2) - frequencies for each dimension
            xdrope_section: [16, 16, 16, 16] - how to split head_dim (must sum to head_dim // 2)

        Returns:
            freqs_combined: (bs, seq_len, head_dim // 2)
        """
        # Take slices from each dimension according to xdrope_section
        # dim 0 gets indices [0:16], dim 1 gets [16:32], dim 2 gets [32:48], dim 3 gets [48:64]
        result_parts = []
        offset = 0
        for dim_idx, length in enumerate(xdrope_section):
            # Apply xdrope to split sections
            result_parts.append(freqs[dim_idx, ..., offset : offset + length])
            offset += length

        return mx.concatenate(result_parts, axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple:
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Input tensor for dtype reference
            position_ids: Position IDs, either:
                - 2D (bs, seq_len) for standard rope
                - 3D (bs, 4, seq_len) for xdrope with 4D positions (from processor)
                - 3D (4, bs, seq_len) for xdrope with 4D positions (internal)

        Returns:
            cos, sin: (bs, seq_len, head_dim) tensors
        """
        # Handle 2D position_ids (standard rope fallback for generation)
        if position_ids.ndim == 2:
            # Broadcast to 4D format for xdrope: (4, bs, seq_len)
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (4, position_ids.shape[0], position_ids.shape[1]),
            )
        elif position_ids.ndim == 3:
            # Check if shape is (bs, 4, seq_len) from processor or (4, bs, seq_len) from internal
            if position_ids.shape[1] == 4 and position_ids.shape[0] != 4:
                # Processor outputs (bs, 4, seq_len), transpose to (4, bs, seq_len)
                position_ids = position_ids.transpose(1, 0, 2)
            # else: already in (4, bs, seq_len) format from internal creation

        # position_ids shape: (4, bs, seq_len)
        # inv_freq shape: (dim // 2,)

        # Expand inv_freq for broadcasting: (4, 1, dim//2, 1)
        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None],
            (4, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )

        # Expand position_ids: (4, bs, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        # Compute frequencies: (4, bs, dim//2, seq_len)
        freqs = inv_freq_expanded @ position_ids_expanded

        # Transpose to (4, bs, seq_len, dim//2)
        freqs = mx.swapaxes(freqs, 2, 3)

        # Apply xdrope to combine 4D frequencies
        freqs = self.apply_xdrope(freqs, self.xdrope_section)

        # Concatenate for full head_dim: (bs, seq_len, head_dim)
        emb = mx.concatenate([freqs, freqs], axis=-1)

        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1
) -> tuple:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor (B, n_heads, L, head_dim)
        k: Key tensor (B, n_kv_heads, L, head_dim)
        cos: Cosine part (B, L, head_dim)
        sin: Sine part (B, L, head_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Rotated query and key tensors
    """
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with GQA, xdrope, and QK normalization."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # Projections (no bias for HunyuanOCR)
        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # QK normalization (RMSNorm applied to Q and K after projection)
        if config.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.query_layernorm = None
            self.key_layernorm = None

        # Rotary embeddings
        self.rotary_emb = HunyuanRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Project Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Handle position_ids for generation (fallback to standard rope)
        if position_ids is None:
            if cache is not None:
                offset = cache.offset
            else:
                offset = 0
            position_ids = mx.arange(offset, offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            # Broadcast to 4D for xdrope
            position_ids = mx.tile(position_ids[None, ...], (4, 1, 1))

        # Compute rotary embeddings
        cos, sin = self.rotary_emb(values, position_ids)

        # Apply rotary embeddings
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, unsqueeze_dim=1)

        # Apply QK normalization AFTER rotary embeddings (HunyuanOCR specific)
        if self.query_layernorm is not None:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        # Update cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Attention
        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Gated MLP with SiLU activation (gate * up -> down)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Transformer decoder layer with pre-norm."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with residual
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r

        # MLP with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r

        return out


class HunyuanModel(nn.Module):
    """Hunyuan language model (decoder-only transformer)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        # Get embeddings
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create attention mask if needed
        if mask is None:
            mask = create_attention_mask(h, cache)

        # Forward through layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids)

        return self.norm(h)


class LanguageModel(nn.Module):
    """Language model wrapper with optional lm_head (respects tie_word_embeddings)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = HunyuanModel(config)

        # Only create lm_head if embeddings are not tied
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )

        # Compute logits
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
