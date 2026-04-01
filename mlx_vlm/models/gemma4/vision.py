from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


class ClippableLinear(nn.Module):
    """Linear layer with optional input/output clamping.

    Matches PyTorch's Gemma4ClippableLinear: wraps nn.Linear, clamps input/output.
    Clip bounds are stored as buffers in the checkpoint (scalar tensors).
    Initialized to ±inf so clamping is a no-op until real values are loaded.
    When use_clipping=False, behaves as a standard nn.Linear (no clip params).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_clipping: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_clipping = use_clipping
        if use_clipping:
            self.input_min = mx.array(float("-inf"))
            self.input_max = mx.array(float("inf"))
            self.output_min = mx.array(float("-inf"))
            self.output_max = mx.array(float("inf"))

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_clipping:
            x = mx.clip(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = mx.clip(x, self.output_min, self.output_max)
        return x


def one_hot(indices: mx.array, num_classes: int) -> mx.array:
    """One-hot encoding."""
    return (mx.expand_dims(indices, -1) == mx.arange(num_classes)).astype(mx.float32)


class VisionRMSNorm(nn.Module):
    """RMS normalization with learned scale: normed * weight.

    Matches PyTorch Gemma4RMSNorm(with_scale=True): full float32 computation.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        normed = x_float * mx.rsqrt(var + self.eps)
        result = normed * self.weight.astype(mx.float32)
        return result.astype(x.dtype)


class VisionRMSNormNoScale(nn.Module):
    """RMS normalization without learnable scale (parameter-free).

    Matches PyTorch Gemma4RMSNorm(with_scale=False): full float32 computation.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        return (x_float * mx.rsqrt(var + self.eps)).astype(x.dtype)


class RMSNorm(nn.Module):
    """Standard Gemma4 RMSNorm: weight applied directly."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _rotate_half(x):
    """Rotate half: [-x2, x1] matching PyTorch's rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multidimensional_rope(inputs, positions, base_frequency=100.0):
    """Apply multidimensional RoPE matching PT's apply_multidimensional_rope.

    Splits the head dimension into ndim parts and applies rotate_half
    independently to each part (one per spatial dimension). This is critical:
    rotate_half must NOT mix features across spatial dimensions.

    inputs: [B, L, N, H], positions: [B, L, 2] or [B, L].
    """
    head_dim = inputs.shape[-1]

    if positions.ndim == 2:
        # 1D fallback - standard rotary embedding
        half = head_dim // 2
        freq_exponents = (2.0 / head_dim) * mx.arange(0, half).astype(mx.float32)
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid_inp = positions[..., None].astype(mx.float32) / timescale
        cos_val = mx.cos(sinusoid_inp)
        sin_val = mx.sin(sinusoid_inp)
        cos_val = mx.concatenate([cos_val, cos_val], axis=-1).astype(inputs.dtype)
        sin_val = mx.concatenate([sin_val, sin_val], axis=-1).astype(inputs.dtype)
        cos_val = mx.expand_dims(cos_val, axis=2)
        sin_val = mx.expand_dims(sin_val, axis=2)
        return inputs * cos_val + _rotate_half(inputs) * sin_val

    ndim = positions.shape[-1]
    channels_per_dim = 2 * (head_dim // (2 * ndim))  # 32 for 2D with head_dim=64
    half_per_dim = channels_per_dim // 2  # 16

    # Split input into per-dimension parts, apply RoPE independently, concatenate
    result_parts = []
    for d in range(ndim):
        # Extract this dimension's slice of the head
        x_part = inputs[..., d * channels_per_dim : (d + 1) * channels_per_dim]

        # Compute frequencies for this dimension
        freq_exponents = (2.0 / channels_per_dim) * mx.arange(0, half_per_dim).astype(
            mx.float32
        )
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid_inp = (
            positions[..., d : d + 1].astype(mx.float32) / timescale
        )  # [B, L, half_per_dim]
        cos_d = mx.cos(sinusoid_inp)
        sin_d = mx.sin(sinusoid_inp)
        # Duplicate: [B, L, half] -> [B, L, channels_per_dim]
        cos_d = mx.concatenate([cos_d, cos_d], axis=-1).astype(inputs.dtype)
        sin_d = mx.concatenate([sin_d, sin_d], axis=-1).astype(inputs.dtype)
        cos_d = mx.expand_dims(cos_d, axis=2)  # [B, L, 1, channels_per_dim]
        sin_d = mx.expand_dims(sin_d, axis=2)

        # Apply rotate_half WITHIN this dimension's partition only
        y_part = x_part * cos_d + _rotate_half(x_part) * sin_d
        result_parts.append(y_part)

    return mx.concatenate(result_parts, axis=-1)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rope_base_frequency = config.rope_parameters["rope_theta"]

        clip = getattr(config, "use_clipped_linears", False)
        self.q_proj = ClippableLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            use_clipping=clip,
        )
        self.k_proj = ClippableLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            use_clipping=clip,
        )
        self.v_proj = ClippableLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            use_clipping=clip,
        )
        self.o_proj = ClippableLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            use_clipping=clip,
        )

        self.q_norm = VisionRMSNorm(self.head_dim)
        self.k_norm = VisionRMSNorm(self.head_dim)
        self._v_norm = VisionRMSNormNoScale()

    def __call__(
        self, x: mx.array, positions: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self._v_norm(v)

        # Apply 2D RoPE
        q = apply_multidimensional_rope(q, positions, self.rope_base_frequency)
        k = apply_multidimensional_rope(k, positions, self.rope_base_frequency)

        # Transpose to [B, H, L, D] for SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Pad head_dim to fused SDPA-supported size (64, 80, 128) to avoid NaN
        # from all-masked padding rows in non-fused fallback path
        from ..base import ensure_fused_sdpa

        attn_output = ensure_fused_sdpa(q, k, v, scale=1.0, mask=mask)

        # [B, H, L, D] -> [B, L, H*D]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        clip = getattr(config, "use_clipped_linears", False)
        self.gate_proj = ClippableLinear(
            config.hidden_size, config.intermediate_size, bias=False, use_clipping=clip
        )
        self.up_proj = ClippableLinear(
            config.hidden_size, config.intermediate_size, bias=False, use_clipping=clip
        )
        self.down_proj = ClippableLinear(
            config.intermediate_size, config.hidden_size, bias=False, use_clipping=clip
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class VisionTransformerBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self, x: mx.array, positions: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed, positions, mask)
        attn_out = self.post_attention_layernorm(attn_out)
        h = x + attn_out

        normed_h = self.pre_feedforward_layernorm(h)
        ffw_out = self.mlp(normed_h)
        ffw_out = self.post_feedforward_layernorm(ffw_out)
        return h + ffw_out


class VisionPatchEmbedder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(
            3 * self.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = mx.ones(
            (2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, patch_positions: mx.array, padding_positions: mx.array
    ) -> mx.array:
        oh = one_hot(patch_positions, self.position_embedding_size)
        # [B, num_patches, 2, pos_size] -> [B, 2, num_patches, pos_size]
        oh = oh.transpose(0, 2, 1, 3).astype(self.position_embedding_table.dtype)
        position_embeddings = oh @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(axis=1)
        position_embeddings = mx.where(
            mx.expand_dims(padding_positions, -1), 0.0, position_embeddings
        )
        return position_embeddings

    def _patchify(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: [B, C, H, W] (channel-first from processor)
        B, C, H, W = pixel_values.shape
        p = self.patch_size
        pH = H // p
        pW = W // p

        # Reshape: [B, C, pH, p, pW, p] -> permute to [B, pH, pW, p, p, C] -> [B, pH*pW, p*p*C]
        patches = pixel_values.reshape(B, C, pH, p, pW, p)
        patches = patches.transpose(0, 2, 4, 3, 5, 1)  # [B, pH, pW, p, p, C]
        patches = patches.reshape(B, pH * pW, C * p * p)
        patches = 2 * (patches - 0.5)
        return self.input_proj(patches.astype(self.input_proj.weight.dtype))

    def __call__(
        self,
        pixel_values: mx.array,
        patch_positions: mx.array,
        padding_positions: mx.array,
    ) -> mx.array:
        hidden_states = self._patchify(pixel_values)
        position_embeddings = self._position_embeddings(
            patch_positions, padding_positions
        )
        return hidden_states + position_embeddings


class VisionPooler(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.default_output_length = config.default_output_length
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(self, x, patch_positions, length):
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2

        clamped = mx.clip(patch_positions, 0, None)
        max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel_idxs = mx.floor(clamped.astype(mx.float32) / k).astype(mx.int32)
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = one_hot(kernel_idxs, length).astype(mx.float32) / k_squared
        output = mx.einsum("bLl,bLd->bld", weights, x).astype(x.dtype)
        mask = mx.logical_not(mx.all(weights == 0, axis=1))
        return output, mask

    def __call__(
        self, hidden_states, patch_positions, padding_positions, output_length=None
    ):
        length = output_length or self.default_output_length
        if hidden_states.shape[1] == length:
            mask = padding_positions
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, mask


class VisionTransformerModel(nn.Module):
    """Holds just the transformer layers (maps to vision_tower.encoder in weights)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [
            VisionTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self, hidden_states: mx.array, positions: mx.array, mask: mx.array
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, mask)
        return hidden_states


class VisionModel(nn.Module):
    """Top-level vision encoder matching PyTorch's Gemma4VisionEncoder.

    Weight key structure:
      vision_tower.patch_embedder.*
      vision_tower.encoder.layers.*
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.patch_size = config.patch_size
        self.pooling_kernel_size = config.pooling_kernel_size
        self.default_output_length = config.default_output_length
        self.max_patches = self.default_output_length * self.pooling_kernel_size**2

        self.patch_embedder = VisionPatchEmbedder(config)
        self.encoder = VisionTransformerModel(config)
        self.pooler = VisionPooler(config)

        if config.standardize:
            self.std_bias = mx.zeros((config.hidden_size,))
            self.std_scale = mx.ones((config.hidden_size,))

    def _patch_positions(self, pixel_values):
        B, C, H, W = pixel_values.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        num_patches = pH * pW
        num_padding = self.max_patches - num_patches

        # Create position grid
        grid_x = np.arange(pW)
        grid_y = np.arange(pH)
        gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")
        real_positions = np.stack([gx.flatten(), gy.flatten()], axis=-1)
        real_positions = np.tile(real_positions[None], (B, 1, 1))

        if num_padding > 0:
            pad_positions = np.full((B, num_padding, 2), -1, dtype=np.int64)
            patch_positions = np.concatenate([real_positions, pad_positions], axis=1)
        else:
            patch_positions = real_positions

        padding_positions = np.zeros((B, self.max_patches), dtype=bool)
        if num_padding > 0:
            padding_positions[:, num_patches:] = True

        return mx.array(patch_positions.astype(np.int32)), mx.array(padding_positions)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(pixel_values, axis=0)

        B, C, H, W = pixel_values.shape
        num_real = (H // self.patch_size) * (W // self.patch_size)
        patch_positions, padding_positions = self._patch_positions(pixel_values)

        # Patchify and embed
        inputs_embeds = self.patch_embedder(
            pixel_values,
            patch_positions[:, :num_real],
            padding_positions[:, :num_real],
        )

        # Pad to max_patches
        num_padding = self.max_patches - num_real
        if num_padding > 0:
            pad_embeds = mx.zeros(
                (B, num_padding, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype
            )
            inputs_embeds = mx.concatenate([inputs_embeds, pad_embeds], axis=1)

        # Build bidirectional attention mask [B, 1, L, L] for SDPA
        valid_mask = ~padding_positions  # True = valid
        attn_mask = mx.expand_dims(valid_mask, 1) * mx.expand_dims(valid_mask, 2)
        neg_inf = mx.array(float("-inf"), dtype=inputs_embeds.dtype)
        attn_mask = mx.where(
            attn_mask, mx.array(0.0, dtype=inputs_embeds.dtype), neg_inf
        )
        attn_mask = mx.expand_dims(attn_mask, 1)  # [B, 1, L, L] for head broadcasting

        # Run transformer layers
        hidden_states = self.encoder(inputs_embeds, patch_positions, attn_mask)

        # Pool
        pooled, pool_mask = self.pooler(
            hidden_states, patch_positions, padding_positions
        )

        # Strip padding tokens using mask multiplication (MLX lacks boolean indexing)
        # Multiply by mask to zero out padding, then use compact representation
        if pool_mask.shape[1] == self.default_output_length:
            # From avg_pool_by_positions: mask is True=valid
            valid_mask = pool_mask
        else:
            valid_mask = ~pool_mask

        # For single batch (typical VLM case), count valid tokens and slice
        # Since pooling produces contiguous valid tokens followed by padding,
        # we can simply count valid tokens and take that many
        all_real = []
        for i in range(B):
            n_valid = int(valid_mask[i].astype(mx.int32).sum().item())
            all_real.append(pooled[i, :n_valid])

        hidden_states = mx.concatenate(all_real, axis=0)[None]  # [1, total_real, dim]

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states

    @staticmethod
    def sanitize(weights):
        sanitized = {}
        for k, v in weights.items():
            sanitized[k] = v
        return sanitized
