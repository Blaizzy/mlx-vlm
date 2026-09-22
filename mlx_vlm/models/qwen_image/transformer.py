"""Qwen-Image-2.1 single-stream DiT for generation and image editing."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.activations import swiglu

_ROPE_POS = 8192
_ROPE_NEG = 1024
_ROPE_ROWS = _ROPE_POS + _ROPE_NEG


def _sinusoidal_timesteps(
    timestep: mx.array,
    dim: int = 256,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = (time_factor * timestep.astype(mx.float32))[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class QwenImageTimestepEmbed(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        emb = _sinusoidal_timesteps(timestep).astype(dtype)
        return self.linear_2(nn.silu(self.linear_1(emb)))


class QwenImageZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored weight is zero-centered; effective scale is weight + 1."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.zeros(dim)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        rrms = mx.rsqrt(mx.mean(xf**2, axis=-1, keepdims=True) + self.eps)
        return (xf * rrms * (self.weight.astype(mx.float32) + 1)).astype(x.dtype)


class QwenImageTextProjection(nn.Module):
    def __init__(
        self, context_in_dim: int, hidden_size: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.text_norm = QwenImageZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.text_norm(x)
        return self.out_layer(nn.gelu_approx(self.in_layer(x)))


class QwenImageSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(swiglu(self.gate_layer(x), self.proj(x)))


def _select_modulation_rows(
    params: mx.array, target_token_mask: mx.array | None
) -> mx.array:
    """Broadcast per-sample modulation over tokens (see reference _select_modulation_rows)."""
    if target_token_mask is None:
        return params[:, None]
    real = params[:-1][:, None]
    zero = params[-1:][None]
    return mx.where(target_token_mask.reshape(1, -1, 1), real, zero)


class QwenImageAdaLayerNormContinuous(nn.Module):
    def __init__(
        self, embedding_dim: int, conditioning_dim: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(conditioning_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=False)

    def __call__(
        self, x: mx.array, conditioning: mx.array, target_token_mask: mx.array | None
    ) -> mx.array:
        scale = self.linear(nn.silu(conditioning).astype(x.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(x) * (1 + scale)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary embedding to ``x`` of shape [B, H, S, D]; cos/sin are [S, D/2]."""
    b, h, s, d = x.shape
    x = x.reshape(b, h, s, d // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    cos = cos[None, None]
    sin = sin[None, None]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return mx.stack([out0, out1], axis=-1).reshape(b, h, s, d).astype(x.dtype)


class QwenImageRope(nn.Module):
    """3-axis (frame, height, width) rotary embedding over the joint sequence."""

    def __init__(self, theta: int, axes_dim: tuple[int, int, int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos = mx.arange(_ROPE_POS, dtype=mx.float32)
        neg = mx.arange(_ROPE_NEG, dtype=mx.float32)[::-1] * -1 - 1
        index = mx.concatenate([pos, neg], axis=0)
        self._angles = [self._rope_params(index, dim) for dim in axes_dim]

    def _rope_params(self, index: mx.array, dim: int) -> mx.array:
        inv = 1.0 / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        return index[:, None] * inv[None]

    def _gather(self, angles: mx.array, idx: list[int]) -> mx.array:
        arr = mx.array(idx, dtype=mx.int32)
        arr = mx.where(arr < 0, arr + _ROPE_ROWS, arr)
        return mx.take(angles, arr, axis=0)

    def __call__(
        self, txt_len: int, img_shape: tuple[int, int, int]
    ) -> tuple[mx.array, mx.array]:
        _, height, width = img_shape
        return self.for_layout(txt_len + height * width, [(txt_len, height, width)])

    def for_layout(
        self, seq_len: int, image_blocks: list[tuple[int, int, int]]
    ) -> tuple[mx.array, mx.array]:
        """Build RoPE for interleaved text and image blocks (start, height, width)."""
        frame_index, height_index, width_index = [], [], []
        cursor = position = 0
        for start, height, width in image_blocks:
            text_positions = list(range(position, position + start - cursor))
            frame_index.extend(text_positions)
            height_index.extend(text_positions)
            width_index.extend(text_positions)
            position += start - cursor
            frame_index.extend([position] * (height * width))
            height_index.extend(
                h
                for h in range(-(height - height // 2), height // 2)
                for _ in range(width)
            )
            width_index.extend(
                w
                for _ in range(height)
                for w in range(-(width - width // 2), width // 2)
            )
            cursor = start + height * width
            position += max(height, width)
        tail = list(range(position, position + seq_len - cursor))
        frame_index.extend(tail)
        height_index.extend(tail)
        width_index.extend(tail)
        angles = mx.concatenate(
            [
                self._gather(self._angles[0], frame_index),
                self._gather(self._angles[1], height_index),
                self._gather(self._angles[2], width_index),
            ],
            axis=-1,
        )
        return mx.cos(angles), mx.sin(angles)


class QwenImageAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = [nn.Linear(inner, dim, bias=False)]
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None
    ) -> mx.array:
        b, s, _ = x.shape
        q = self.norm_q(self.to_q(x).reshape(b, s, self.heads, self.head_dim))
        k = self.norm_k(self.to_k(x).reshape(b, s, self.heads, self.head_dim))
        v = self.to_v(x).reshape(b, s, self.heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.heads * self.head_dim)
        return self.to_out[0](out)


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.attn = QwenImageAttention(dim, num_heads, head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.img_mlp = QwenImageSwiGLU(dim, dim * mlp_ratio)

    def _modulate(
        self, x: mx.array, mod: mx.array, target_token_mask: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        scale, gate = mx.split(mod, 2, axis=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return x * (1 + scale), gate

    def __call__(
        self,
        x: mx.array,
        modulation: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
        target_token_mask: mx.array | None,
    ) -> mx.array:
        mod1, mod2 = mx.split(modulation, 2, axis=-1)
        modulated, gate1 = self._modulate(self.img_norm1(x), mod1, target_token_mask)
        x = x + mx.tanh(gate1) * self.attn(modulated, cos, sin, mask)
        modulated2, gate2 = self._modulate(self.img_norm2(x), mod2, target_token_mask)
        x = x + mx.tanh(gate2) * self.img_mlp(modulated2)
        return x


class QwenImageTransformer(nn.Module):
    """Single-stream DiT with interleaved text, reference, and target tokens."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.causal_condition = causal_condition
        inner = num_attention_heads * attention_head_dim
        self.pos_embed = QwenImageRope(theta=10000, axes_dim=axes_dims_rope)
        self.time_text_embed = QwenImageTimestepEmbed(inner)
        self.txt_in = QwenImageTextProjection(context_in_dim, inner, eps=eps)
        self.img_in = nn.Linear(in_channels, inner, bias=False)
        self.modulation = [nn.Linear(inner, 4 * inner, bias=False)]
        self.transformer_blocks = [
            QwenImageTransformerBlock(
                inner, num_attention_heads, attention_head_dim, mlp_ratio, eps
            )
            for _ in range(num_layers)
        ]
        self.norm_out = QwenImageAdaLayerNormContinuous(inner, inner, eps=eps)
        self.proj_out = nn.Linear(inner, out_channels, bias=False)

    def _joint_inputs(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        img_shape: tuple[int, int, int],
        reference_image_shapes: list[tuple[int, int, int]],
        image_pad_mask: mx.array | None,
        text_valid: mx.array | None,
    ):
        """Replace each VLM image slot with four spatial VAE tokens."""
        batch, txt_len, _ = encoder_hidden_states.shape
        slots = (
            [False] * txt_len
            if image_pad_mask is None
            else image_pad_mask.reshape(-1).tolist()
        )
        if len(slots) != txt_len:
            raise ValueError("image_pad_mask must describe one prompt's image slots")
        shapes = [*reference_image_shapes, img_shape]
        if any(f != 1 for f, _, _ in shapes):
            raise ValueError("Qwen-Image supports single-frame images only")
        if hidden_states.shape[1] != sum(math.prod(shape) for shape in shapes):
            raise ValueError("Image latent count does not match image shapes")
        text = self.txt_in(encoder_hidden_states)
        images = self.img_in(hidden_states)
        if text_valid is None:
            text_valid = mx.ones((batch, txt_len), dtype=mx.bool_)
        else:
            text_valid = text_valid.reshape(batch, txt_len).astype(mx.bool_)
        pieces, valid, image_ids, blocks = [], [], [], []
        cursor = image_cursor = seq_len = 0
        for image_id, (_, height, width) in enumerate(shapes):
            target = image_id == len(reference_image_shapes)
            if target:
                start = txt_len
                if any(slots[cursor:]):
                    raise ValueError("Extra image slots without reference latents")
                end = start
            else:
                count = height * width
                if count % 4:
                    raise ValueError(
                        "Reference images must have dimensions divisible by 32"
                    )
                try:
                    start = slots.index(True, cursor)
                except ValueError as exc:
                    raise ValueError("Missing reference-image slots in prompt") from exc
                end = start + count // 4
                if end > txt_len or not all(slots[start:end]):
                    raise ValueError(
                        "Reference-image slots do not match VAE dimensions"
                    )
            if start > cursor:
                pieces.append(text[:, cursor:start])
                valid.append(text_valid[:, cursor:start])
                image_ids.extend([-1] * (start - cursor))
                seq_len += start - cursor
            count = height * width
            pieces.append(images[:, image_cursor : image_cursor + count])
            valid.append(mx.ones((batch, count), dtype=mx.bool_))
            image_ids.extend([image_id] * count)
            blocks.append((seq_len, height, width))
            seq_len += count
            image_cursor += count
            cursor = end
        return (
            mx.concatenate(pieces, axis=1),
            mx.array(image_ids, dtype=mx.int32),
            mx.concatenate(valid, axis=1),
            blocks,
        )

    @staticmethod
    def _block_causal_mask(image_ids: mx.array, key_valid: mx.array) -> mx.array:
        idx = mx.arange(image_ids.shape[0])
        causal = idx[:, None] >= idx[None, :]
        same_image = (image_ids[:, None] == image_ids[None, :]) & (
            image_ids[:, None] >= 0
        )
        return (causal | same_image)[None, None] & key_valid[:, None, None, :]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        img_shape: tuple[int, int, int],
        encoder_hidden_states_mask: mx.array | None = None,
        reference_image_shapes: list[tuple[int, int, int]] | None = None,
        image_pad_mask: mx.array | None = None,
    ) -> mx.array:
        reference_image_shapes = reference_image_shapes or []
        h, image_ids, key_valid, blocks = self._joint_inputs(
            hidden_states,
            encoder_hidden_states,
            img_shape,
            reference_image_shapes,
            image_pad_mask,
            encoder_hidden_states_mask,
        )
        cos, sin = self.pos_embed.for_layout(h.shape[1], blocks)
        target_token_mask = None
        timestep = timestep.astype(h.dtype)
        if self.causal_condition:
            timestep = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
            target_token_mask = image_ids == len(reference_image_shapes)
        temb = self.time_text_embed(timestep, h.dtype)
        modulation = self.modulation[0](nn.silu(temb))
        mask = self._block_causal_mask(image_ids, key_valid)
        for block in self.transformer_blocks:
            h = block(h, modulation, cos, sin, mask, target_token_mask)
        # The target is the last image block; only it is denoised.
        h = h[:, -math.prod(img_shape) :]
        target_mask = (
            None if target_token_mask is None else target_token_mask[-h.shape[1] :]
        )
        return self.proj_out(self.norm_out(h, temb, target_mask))


__all__ = ["QwenImageTransformer"]
