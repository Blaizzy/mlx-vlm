import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..kernels import bicubic_interpolate


@dataclass
class VisionConfig:
    model_type: str = "moonvit"
    depth: int = 27
    embed_dim: int = 1152
    hidden_size: int = 1152
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    intermediate_size: int = 4304
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    merge_kernel_size: list[int, int] = None

    def __post_init__(self):
        if self.merge_kernel_size is None:
            self.merge_kernel_size = (self.spatial_merge_size, self.spatial_merge_size)

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array:
    orig_dtype = tensor.dtype

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    cos = mx.expand_dims(cos, axis=1)  # Equivalent to unsqueeze(1)
    cos = mx.tile(cos, (1, 1, 2))  # Equivalent to repeat(1, 1, 2)
    cos = mx.expand_dims(cos, axis=0)  # Equivalent to [None, ...]

    sin = mx.expand_dims(sin, axis=1)  # Equivalent to unsqueeze(1)
    sin = mx.tile(sin, (1, 1, 2))  # Equivalent to repeat(1, 1, 2)
    sin = mx.expand_dims(sin, axis=0)  # Equivalent to [None, ...]

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen.tolist(), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = mx.ones((height, width, dim))

    def __call__(self, x: mx.array, grid_hws: mx.array) -> mx.array:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_axis=1))
            else:
                result = (
                    bicubic_interpolate(
                        mx.expand_dims(self.weight.transpose(2, 0, 1), axis=0),
                        size=shape,
                    )
                    .squeeze(0)
                    .transpose(1, 2, 0)
                    .flatten(end_axis=1)
                )

                pos_embs.append(result)

        out = x + mx.concatenate(pos_embs).astype(x.dtype)
        return out


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        num_channels: int = 3,
        embed_dim: int = 1152,
        init_pos_emb_height: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.init_pos_emb_height = init_pos_emb_height

        self.proj = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.pos_emb = Learnable2DInterpPosEmb(
            height=init_pos_emb_height, width=init_pos_emb_height, dim=embed_dim
        )

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states).swapaxes(1, 3)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw)
        return hidden_states


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == mx.complex64, freqs_cis.dtype


def view_as_complex(x):
    """
    Convert a tensor with shape (..., 2) to a complex tensor with shape (...).
    """
    # Get real and imaginary parts
    real, imag = x[..., 0], x[..., 1]
    # Create complex tensor
    return real + 1j * imag


def view_as_real(x):
    """
    Convert a complex tensor with shape (...) to a real tensor with shape (..., 2).
    """
    # Get real and imaginary parts
    real = mx.real(x)
    imag = mx.imag(x)
    # Combine into a tensor with last dimension 2
    return mx.stack([real, imag], axis=-1)


def apply_rope(
    q: mx.array, k: mx.array, freqs_cis: mx.array
) -> tuple[mx.array, mx.array]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        q: query, array of shape (..., num_heads, head_dim)
        k: key, array of shape (..., num_heads, head_dim)
        freqs_cis: array of shape (..., head_dim/2), dtype=mx.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: arrays of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(q, freqs_cis)
    _apply_rope_input_validation(k, freqs_cis)

    freqs_cis = mx.expand_dims(freqs_cis, axis=-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    q_ = view_as_complex(q.astype(mx.float32).reshape(*q.shape[:-1], -1, 2))
    k_ = view_as_complex(k.astype(mx.float32).reshape(*k.shape[:-1], -1, 2))
    q_out = view_as_real(q_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    k_out = view_as_real(k_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return q_out.astype(q.dtype), k_out.astype(k.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.wqkv = nn.Linear(dim, dim * 3, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def __call__(
        self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = self.wqkv(x)

        qkv_shape = qkv.shape[:-1] + (
            3,
            self.num_heads,
            self.head_dim,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        qkv = qkv.reshape(*qkv_shape)

        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1)
        k = k.squeeze(1)
        v = v.squeeze(1)

        q, k = apply_rope(q, k, rotary_pos_emb)

        attention_mask = mx.zeros((1, seq_length, seq_length), dtype=x.dtype)

        # Create attention mask for each sequence in the batch
        for i in range(1, len(cu_seqlens)):
            start = int(cu_seqlens[i - 1])
            end = int(cu_seqlens[i])
            attention_mask[..., start:end, start:end] = 1

        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)

        attn_weight = q @ k.swapaxes(-2, -1) / mx.sqrt(q.shape[-1])
        attn_weight += attention_mask
        attn_weight = mx.softmax(attn_weight, axis=-1).astype(q.dtype)

        attn_output = attn_weight @ v
        attn_output = attn_output.transpose(1, 0, 2)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.wo(attn_output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc0 = nn.Linear(dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc0(x))
        x = self.fc1(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm0 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)

        self.attn = Attention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.embed_dim, hidden_dim=config.intermediate_size)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm0(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self._freqs_cis = None

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self) -> mx.array:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex array of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = mx.arange(0, N, dtype=mx.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mx.arange(0, self.dim, 4)[: (self.dim // 4)].astype(
            mx.float32
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mx.outer(x_pos, freqs)  # N, C/4
        y_freqs = mx.outer(y_pos, freqs)  # N, C/4

        # Create complex numbers using cos and sin
        x_cos = mx.cos(x_freqs)
        x_sin = mx.sin(x_freqs)
        y_cos = mx.cos(y_freqs)
        y_sin = mx.sin(y_freqs)

        # Create complex numbers
        x_cis = x_cos + 1j * x_sin  # N, C/4
        y_cis = y_cos + 1j * y_sin  # N, C/4

        # N, C/4, 2
        freqs_cis = mx.stack([x_cis, y_cis], axis=-1)

        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: mx.array) -> mx.array:
        """
        Args:
            grid_hws (mx.array): grid height and width

        Returns:
            freqs_cis: array of shape (sum(t * height * width), dim//2)
        """
        if self._freqs_cis is None:
            self._freqs_cis = self._precompute_freqs_cis()

        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )

        freqs_cis_list = []
        for h, w in shapes:
            # Get the slice of precomputed frequencies for this shape
            shape_freqs = self._freqs_cis[:h, :w]
            # Reshape to flatten the spatial dimensions
            shape_freqs = shape_freqs.reshape(-1, self.dim // 2)
            freqs_cis_list.append(shape_freqs)

        freqs_cis = mx.concatenate(freqs_cis_list, axis=0)
        return freqs_cis


def patch_merger(
    x: mx.array,
    grid_hws: mx.array,
    merge_kernel_size: list[int, int] = (2, 2),
) -> List[mx.array]:
    d_model = x.shape[-1]

    outputs = []
    pre_sum = 0
    for x_shape in grid_hws.tolist():
        height, width = x_shape[0], x_shape[1]
        # Get the current sequence
        seq = x[pre_sum : pre_sum + height * width]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.reshape(
            new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = mx.transpose(reshaped_seq, (0, 2, 1, 3, 4))
        padded_seq = reshaped_seq.reshape(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)
        pre_sum += height * width

    return outputs


class VisionModel(nn.Module):

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type not in ["qwen2_vl", "moonvit"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size
        self.merge_kernel_size = config.merge_kernel_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.embed_dim,
            init_pos_emb_height=config.init_pos_emb_height,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.rope_pos_emb = Rope2DPosEmb(head_dim, 512, 512)

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        hidden_states = self.patch_embed(hidden_states, grid_thw)
        rotary_pos_emb = self.rope_pos_emb.get_freqs_cis(grid_thw)

        # Assuming grid_thw has shape (batch_size, 3)
        batch_size = grid_thw.shape[0]

        # Calculate cu_seqlens for each item in the batch
        lengths = mx.concatenate(
            (
                mx.zeros((1,), dtype=grid_thw.dtype),
                grid_thw[:, 0] * grid_thw[:, 1],
            )
        )
        cu_seqlens = mx.cumsum(lengths.astype(mx.int32), axis=0)

        encoder_states = (hidden_states,) if output_hidden_states else None

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.final_layernorm(hidden_states)

        hidden_states = patch_merger(
            hidden_states, grid_thw, merge_kernel_size=self.merge_kernel_size
        )

        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embed.proj.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)

            elif "vision_tower.blocks" in k:
                if "attn" not in k and ("wqkv" in k or "wo" in k):
                    new_key = k.replace("wqkv", "attn.wqkv").replace("wo", "attn.wo")
                    sanitized_weights[new_key] = v
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights
