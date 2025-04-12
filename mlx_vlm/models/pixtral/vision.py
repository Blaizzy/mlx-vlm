import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    head_dim: int = 64
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

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


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[0], patch.shape[1]
        h_grid, v_grid = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
        h_grid = h_grid.reshape(-1, 1)
        v_grid = v_grid.reshape(-1, 1)
        ids = h_grid * max_width + v_grid
        positions.append(ids.flatten())
    return mx.concatenate(positions)


def generate_block_attention_mask(patch_embeds_list, tensor):
    seq_len = tensor.shape[1]
    d_min = -1e9  # Using a large negative value as MLX doesn't have finfo

    causal_mask = mx.full((seq_len, seq_len), vals=d_min)

    block_end_idx = mx.cumsum(mx.array(patch_embeds_list))
    block_start_idx = mx.concatenate([mx.array([0]), mx.array(patch_embeds_list[:-1])])
    block_start_idx = mx.cumsum(block_start_idx)

    for start, end in zip(block_start_idx, block_end_idx):
        start, end = int(start), int(end)  # Convert to integers for indexing
        causal_mask[start:end, start:end] = 0

    causal_mask = mx.broadcast_to(
        causal_mask[None, None, :, :], (tensor.shape[0], 1, seq_len, seq_len)
    )
    return causal_mask


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.embed_dim = dims
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def __call__(self, queries, keys, values, position_embeddings, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, unsqueeze_dim=0)

        attn_weights = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        output = mx.matmul(attn_weights, values)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        hidden_dim = config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attention = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.attention_norm = nn.RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)
        self.ffn_norm = nn.RMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        position_embeddings: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        y = self.attention_norm(x)
        y = self.attention(y, y, y, position_embeddings, mask)
        x = x + y
        y = self.ffn_norm(x)
        y = self.feed_forward(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class PixtralRotaryEmbedding:
    def __init__(self, config):
        self.dim = config.head_dim
        self.base = config.rope_theta
        max_patches_per_side = config.image_size // config.patch_size
        freqs = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )

        h = mx.arange(max_patches_per_side)
        w = mx.arange(max_patches_per_side)

        freqs_h = mx.outer(h, freqs[::2]).astype(mx.float32)
        freqs_w = mx.outer(w, freqs[1::2]).astype(mx.float32)
        inv_freq = mx.concatenate(
            [
                mx.tile(freqs_h[:, None, :], (1, max_patches_per_side, 1)),
                mx.tile(freqs_w[None, :, :], (max_patches_per_side, 1, 1)),
            ],
            axis=-1,
        ).reshape(-1, self.dim // 2)

        self.inv_freq = mx.concatenate((inv_freq, inv_freq), axis=-1)

    def __call__(self, x, position_ids):
        freqs = self.inv_freq[position_ids]
        emb = freqs
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


class PixtralVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = nn.RMSNorm(config.hidden_size)
        self.transformer = Encoder(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def __call__(
        self,
        x: List[mx.array],
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        patch_embeds_list = self.patch_conv(x)
        patch_embeds = patch_embeds_list.reshape(1, -1, patch_embeds_list.shape[-1])

        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        )

        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)

        mask = generate_block_attention_mask(
            [p.shape[1] * p.shape[0] for p in patch_embeds_list], patch_embeds
        )

        encoder_states = (patch_embeds,) if output_hidden_states else None

        for l in self.transformer.layers:
            patch_embeds = l(
                patch_embeds, mask=mask, position_embeddings=position_embedding
            )
            if output_hidden_states:
                encoder_states = encoder_states + (patch_embeds,)

        return patch_embeds, encoder_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.model_type = config.model_type
        if self.model_type not in ["clip_vision_model", "pixtral"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = PixtralVisionModel(config)

    def __call__(
        self, x: List[mx.array], output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_conv.weight" in k:
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
