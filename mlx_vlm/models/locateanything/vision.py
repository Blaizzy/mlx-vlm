from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..kernels import bicubic_interpolate
from .config import VisionConfig

LAYER_NORM_EPS = 1e-5


def _as_hw_shapes(grid_hws) -> List[Tuple[int, int]]:
    raw_shapes = grid_hws.tolist() if hasattr(grid_hws, "tolist") else grid_hws
    return [(int(shape[0]), int(shape[1])) for shape in raw_shapes]


def _resolve_hw_shapes(grid_hws, grid_shapes=None) -> List[Tuple[int, int]]:
    return _as_hw_shapes(grid_shapes if grid_shapes is not None else grid_hws)


def make_block_attention_mask(cu_seqlens: mx.array, seq_length: int) -> mx.array:
    pos = mx.arange(seq_length)
    block_id = mx.sum(pos[None, :] >= cu_seqlens[1:, None], axis=0)
    return block_id[:, None] == block_id[None, :]


def check_array_shape(arr):
    shape = arr.shape

    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = mx.ones((height, width, dim))

    def _get_pos_emb(self, shape: Tuple[int, int]) -> mx.array:
        if shape == self.weight.shape[:-1]:
            return self.weight.flatten(end_axis=1)

        return (
            bicubic_interpolate(
                mx.expand_dims(self.weight.transpose(2, 0, 1), axis=0),
                size=shape,
            )
            .squeeze(0)
            .transpose(1, 2, 0)
            .flatten(end_axis=1)
        )

    def __call__(self, x: mx.array, grid_hws: mx.array, grid_shapes=None) -> mx.array:
        shapes = _resolve_hw_shapes(grid_hws, grid_shapes)
        pos_embs = [self._get_pos_emb(shape) for shape in shapes]
        out = x + mx.concatenate(pos_embs, axis=0).astype(x.dtype)
        return out


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        num_channels: int = 3,
        embed_dim: int = 1152,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.pos_emb = Learnable2DInterpPosEmb(
            height=init_pos_emb_height, width=init_pos_emb_width, dim=embed_dim
        )

    def __call__(
        self, hidden_states: mx.array, grid_thw: mx.array, grid_shapes=None
    ) -> mx.array:
        hidden_states = self.proj(hidden_states).swapaxes(1, 3)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw, grid_shapes=grid_shapes)
        return hidden_states


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == mx.complex64, freqs_cis.dtype


def view_as_complex(x):
    real, imag = x[..., 0], x[..., 1]
    return real + 1j * imag


def view_as_real(x):
    real = mx.real(x)
    imag = mx.imag(x)
    return mx.stack([real, imag], axis=-1)


def apply_rope(
    q: mx.array, k: mx.array, freqs_cis: mx.array
) -> tuple[mx.array, mx.array]:
    _apply_rope_input_validation(q, freqs_cis)
    _apply_rope_input_validation(k, freqs_cis)

    freqs_cis = mx.expand_dims(freqs_cis, axis=-2)
    q_ = view_as_complex(q.astype(mx.float32).reshape(*q.shape[:-1], -1, 2))
    k_ = view_as_complex(k.astype(mx.float32).reshape(*k.shape[:-1], -1, 2))
    q_out = view_as_real(q_ * freqs_cis).flatten(-2)
    k_out = view_as_real(k_ * freqs_cis).flatten(-2)
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
        self,
        x: mx.array,
        cu_seqlens: Optional[mx.array] = None,
        rotary_pos_emb: mx.array = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = (
            self.wqkv(x)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        q, k = apply_rope(q, k, rotary_pos_emb)

        if attention_mask is None and cu_seqlens is None:
            raise ValueError("Either attention_mask or cu_seqlens must be provided.")
        if attention_mask is None and cu_seqlens.shape[0] > 2:
            attention_mask = make_block_attention_mask(cu_seqlens, seq_length)

        q = q.transpose(1, 0, 2)[None, ...]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.wo(attn_output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc0 = nn.Linear(dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc0(x))
        x = self.fc1(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm0 = nn.LayerNorm(config.embed_dim, eps=LAYER_NORM_EPS)
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=LAYER_NORM_EPS)

        self.attn = Attention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.embed_dim, hidden_dim=config.intermediate_size)

    def __call__(
        self, hidden_states, cu_seqlens, rotary_pos_emb, attention_mask: mx.array
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm0(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


class Rope2DPosEmb(nn.Module):
    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self) -> mx.array:
        N = self.max_height * self.max_width
        flat_pos = mx.arange(0, N, dtype=mx.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mx.arange(0, self.dim, 4)[: (self.dim // 4)].astype(mx.float32)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mx.outer(x_pos, freqs)
        y_freqs = mx.outer(y_pos, freqs)

        x_cis = mx.cos(x_freqs) + 1j * mx.sin(x_freqs)
        y_cis = mx.cos(y_freqs) + 1j * mx.sin(y_freqs)

        freqs_cis = mx.stack([x_cis, y_cis], axis=-1)

        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: mx.array, grid_shapes=None) -> mx.array:
        freqs_cis_full = self._precompute_freqs_cis()
        shapes = _resolve_hw_shapes(grid_hws, grid_shapes)
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )

        freqs_cis_list = []
        for h, w in shapes:
            freqs_cis_list.append(freqs_cis_full[:h, :w].reshape(-1, self.dim // 2))

        freqs_cis = mx.concatenate(freqs_cis_list, axis=0)
        return freqs_cis


def patch_merger(
    x: mx.array,
    grid_hws: Sequence[Tuple[int, int]],
    merge_kernel_size: list[int, int] = (2, 2),
    grid_shapes=None,
) -> List[mx.array]:
    d_model = x.shape[-1]
    kernel_height, kernel_width = merge_kernel_size
    shapes = _resolve_hw_shapes(grid_hws, grid_shapes)

    lengths = [h * w for h, w in shapes]
    split_points = []
    running = 0
    for length in lengths[:-1]:
        running += length
        split_points.append(running)

    sequences = mx.split(x, split_points, axis=0) if split_points else [x]
    outputs = []
    for seq, (height, width) in zip(sequences, shapes):
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.reshape(
            new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = mx.transpose(reshaped_seq, (0, 2, 1, 3, 4))
        padded_seq = reshaped_seq.reshape(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)

    return outputs


class VisionModel(nn.Module):

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "moonvit":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size
        self.merge_kernel_size = config.merge_kernel_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.embed_dim,
            init_pos_emb_height=config.init_pos_emb_height,
            init_pos_emb_width=config.init_pos_emb_width,
        )

        head_dim = config.embed_dim // config.num_heads

        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=LAYER_NORM_EPS)
        self.rope_pos_emb = Rope2DPosEmb(head_dim, 512, 512)

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
        grid_shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> mx.array:

        hidden_states = self.patch_embed(
            hidden_states, grid_thw, grid_shapes=grid_shapes
        )
        rotary_pos_emb = self.rope_pos_emb.get_freqs_cis(
            grid_thw, grid_shapes=grid_shapes
        )

        lengths = mx.concatenate(
            (
                mx.zeros((1,), dtype=grid_thw.dtype),
                grid_thw[:, 0] * grid_thw[:, 1],
            )
        )
        cu_seqlens = mx.cumsum(lengths.astype(mx.int32), axis=0)
        attention_mask = (
            make_block_attention_mask(cu_seqlens, hidden_states.shape[0])
            if grid_thw.shape[0] > 1
            else None
        )

        encoder_states = (hidden_states,) if output_hidden_states else None

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        hidden_states = self.final_layernorm(hidden_states)

        hidden_states = patch_merger(
            hidden_states,
            grid_thw,
            merge_kernel_size=self.merge_kernel_size,
            grid_shapes=grid_shapes,
        )

        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            elif "patch_embed.proj.weight" in k:
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
