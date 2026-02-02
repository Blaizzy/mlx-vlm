from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen2EncoderConfig, VisionConfig


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class Qwen2RMSNorm(nn.Module):
    """RMSNorm for Qwen2 encoder."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    """Rotary position embeddings for Qwen2."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 1000000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Note: inv_freq is computed on-the-fly, not stored as a parameter

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        # Compute inv_freq on the fly (not stored as parameter)
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )

        # position_ids: [batch_size, seq_len]
        # inv_freq: [head_dim // 2]
        # We want freqs of shape [batch_size, seq_len, head_dim // 2]

        # Outer product: position_ids[:, :, None] * inv_freq[None, None, :]
        position_ids_float = position_ids[:, :, None].astype(mx.float32)  # [B, S, 1]
        inv_freq_expanded = inv_freq[None, None, :]  # [1, 1, D//2]
        freqs = position_ids_float * inv_freq_expanded  # [B, S, D//2]

        emb = mx.concatenate([freqs, freqs], axis=-1)  # [B, S, D]
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    """MLP for Qwen2 encoder."""

    def __init__(self, config: Qwen2EncoderConfig):
        super().__init__()
        self.hidden_size = config.dim
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """Multi-head attention for Qwen2 encoder with GQA support."""

    def __init__(self, config: Qwen2EncoderConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.dim
        self.num_heads = config.heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.kv_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048,
            base=config.rope_theta,
        )
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Repeat KV heads for GQA
        if self.num_key_value_groups > 1:
            key_states = mx.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = mx.repeat(value_states, self.num_key_value_groups, axis=1)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scale,
            mask=attention_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2DecoderLayer(nn.Module):
    """Transformer layer for Qwen2 encoder."""

    def __init__(self, config: Qwen2EncoderConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.dim
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.dim, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2Decoder2Encoder(nn.Module):
    """Qwen2-based decoder used as encoder for vision features.

    Takes SAM features and processes them through Qwen2 transformer layers
    using learnable queries to produce fixed-size output.
    """

    def __init__(self, config: Qwen2EncoderConfig):
        super().__init__()
        self.config = config

        # Learnable queries for cross-attention
        # query_1024: (256, dim) - for 1024x1024 images (SAM outputs 16x16=256 features)
        # query_768: (144, dim) - for 768x768 images (SAM outputs 12x12=144 features)
        # Initialized with zeros, will be loaded from weights
        self.query_1024 = mx.zeros((256, config.dim))
        self.query_768 = mx.zeros((144, config.dim))

        # Transformer layers
        self.layers = [
            Qwen2DecoderLayer(config, layer_idx=i) for i in range(config.layers)
        ]

        # Final layer norm
        self.norm = Qwen2RMSNorm(config.dim, eps=config.rms_norm_eps)

    def __call__(self, sam_features: mx.array) -> mx.array:
        """Process SAM features through Qwen2 encoder.

        Args:
            sam_features: SAM encoder output of shape (B, H, W, C) where C=896

        Returns:
            Encoded features of shape (B, 256, C)
        """
        batch_size = sam_features.shape[0]

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        sam_features_flat = sam_features.reshape(batch_size, -1, self.config.dim)
        num_image_tokens = sam_features_flat.shape[1]

        # Select appropriate query based on number of image tokens
        # 256 tokens -> use query_1024 (for 1024x1024 images, SAM outputs 16x16)
        # 144 tokens -> use query_768 (for 768x768 images, SAM outputs 12x12)
        if num_image_tokens == 256:
            query_embed = self.query_1024
            num_queries = 256
        elif num_image_tokens == 144:
            query_embed = self.query_768
            num_queries = 144
        else:
            # Default to query_1024 for unexpected sizes
            query_embed = self.query_1024
            num_queries = 256

        # Expand queries for batch: (num_queries, C) -> (B, num_queries, C)
        queries = mx.broadcast_to(
            query_embed[None, :, :], (batch_size, num_queries, self.config.dim)
        )

        # Concatenate: image tokens + query tokens
        # Shape: (B, num_image_tokens + num_queries, C)
        hidden_states = mx.concatenate([sam_features_flat, queries], axis=1)
        seq_len = hidden_states.shape[1]

        # Create mixed attention mask:
        # - Image tokens can attend to all image tokens (bidirectional)
        # - Image tokens CANNOT attend to query tokens (blocked)
        # - Query tokens can attend to all image tokens
        # - Query tokens use causal attention within queries (can attend to self + previous)
        # Shape: (1, 1, seq_len, seq_len) - will be broadcast across batch and heads
        mask_dtype = hidden_states.dtype

        # Start with all positions blocked (large negative value)
        mask = mx.full((seq_len, seq_len), -1e9, dtype=mx.float32)

        # 1. Image tokens can attend to all image tokens (bidirectional)
        # mask[0:num_image_tokens, 0:num_image_tokens] = 0
        image_to_image = mx.zeros(
            (num_image_tokens, num_image_tokens), dtype=mx.float32
        )
        mask = mx.concatenate(
            [
                mx.concatenate(
                    [image_to_image, mask[:num_image_tokens, num_image_tokens:]], axis=1
                ),
                mask[num_image_tokens:, :],
            ],
            axis=0,
        )

        # 2. Query tokens can attend to all image tokens
        # mask[num_image_tokens:, 0:num_image_tokens] = 0
        query_to_image = mx.zeros((num_queries, num_image_tokens), dtype=mx.float32)
        mask = mx.concatenate(
            [
                mask[:num_image_tokens, :],
                mx.concatenate(
                    [query_to_image, mask[num_image_tokens:, num_image_tokens:]], axis=1
                ),
            ],
            axis=0,
        )

        # 3. Query tokens use causal attention (can attend to self + previous queries)
        # Create lower triangular mask for query-query region
        query_causal = mx.tril(mx.zeros((num_queries, num_queries), dtype=mx.float32))
        query_causal = query_causal + mx.triu(
            mx.full((num_queries, num_queries), -1e9, dtype=mx.float32), k=1
        )

        # Update query-query region in mask
        mask = mx.concatenate(
            [
                mask[:, :num_image_tokens],
                mx.concatenate(
                    [mask[:num_image_tokens, num_image_tokens:], query_causal], axis=0
                ),
            ],
            axis=1,
        )

        # Cast to input dtype and reshape for attention: (1, 1, seq_len, seq_len)
        attention_mask = mask.astype(mask_dtype)[None, None, :, :]

        # Create position IDs
        position_ids = mx.broadcast_to(
            mx.arange(seq_len)[None, :], (batch_size, seq_len)
        )

        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Return only the query tokens (last num_queries tokens)
        return hidden_states[:, -num_queries:, :]


class VisionModel(nn.Module):
    """Vision model for DeepSeek-OCR-2 using Qwen2 encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        if self.model_type != "vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Get Qwen2 config from params
        qwen2_params = config.params.get("qwen2", {})
        qwen2_config = Qwen2EncoderConfig(
            dim=qwen2_params.get("dim", 896),
            layers=qwen2_params.get("layers", 24),
            heads=qwen2_params.get("heads", 14),
            kv_heads=qwen2_params.get("kv_heads", 2),
            intermediate_size=qwen2_params.get("intermediate_size", 4864),
            rms_norm_eps=qwen2_params.get("rms_norm_eps", 1e-6),
            rope_theta=qwen2_params.get("rope_theta", 1000000.0),
        )

        self.qwen2_encoder = Qwen2Decoder2Encoder(qwen2_config)

    def __call__(self, x: mx.array, sam_features: mx.array) -> mx.array:
        """Process vision input through Qwen2 encoder.

        Args:
            x: Original image tensor (not used, kept for API compatibility)
            sam_features: SAM encoder output of shape (B, H, W, C)

        Returns:
            Encoded features of shape (B, 256, C)
        """
        return self.qwen2_encoder(sam_features)

    def sanitize(self, weights):
        sanitized_weights = {}
        weight_keys = {
            "neck.0.weight",
            "neck.2.weight",
            "neck_hd.0.weight",
            "neck_hd.2.weight",
            "sam_model.net_2.weight",
            "sam_model.net_3.weight",
            "downsamples.0.weight",
            "downsamples.1.weight",
            "patch_embed.proj.weight",
            "embeddings.patch_embedding.weight",
        }
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue

            elif ".".join(k.split(".")[-3:]) in weight_keys:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)

            else:
                sanitized_weights[k] = v

        return sanitized_weights
