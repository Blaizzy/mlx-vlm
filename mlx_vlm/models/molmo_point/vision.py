import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class ViTMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        if hidden_act == "gelu_pytorch_tanh":
            self.act = nn.GELU(approx="tanh")
        else:
            self.act = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(self.act(self.w1(x)))


class ViTMultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        input_dim: Optional[int] = None,
        out_layer: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        input_dim = input_dim or hidden_size
        self.wq = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(
            input_dim, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.wv = nn.Linear(
            input_dim, self.num_key_value_heads * self.head_dim, bias=True
        )
        if out_layer:
            self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        else:
            self.wo = None
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        inputs_q: mx.array,
        inputs_kv: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        B = inputs_q.shape[0]
        L_q = inputs_q.shape[1]
        L_k = inputs_k.shape[1]

        xq = self.wq(inputs_q).reshape(B, L_q, self.num_heads, self.head_dim)
        xk = self.wk(inputs_k).reshape(B, L_k, self.num_key_value_heads, self.head_dim)
        xv = self.wv(inputs_v).reshape(B, L_k, self.num_key_value_heads, self.head_dim)

        # Transpose to (B, heads, L, head_dim)
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        if self.num_heads != self.num_key_value_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            xk = mx.repeat(xk, n_rep, axis=1)
            xv = mx.repeat(xv, n_rep, axis=1)

        attn_weights = (xq * self.scale) @ xk.transpose(0, 1, 3, 2)

        if attn_mask is not None:
            attn_weights = attn_weights + mx.where(attn_mask, 0.0, -1e9)

        attn_weights = mx.softmax(attn_weights.astype(mx.float32), axis=-1).astype(
            xq.dtype
        )
        attn_output = attn_weights @ xv

        # Back to (B, L, heads, head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L_q, -1)

        if self.wo is not None:
            attn_output = self.wo(attn_output)

        return attn_output


class Molmo2VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        self.feed_forward = ViTMLP(
            config.hidden_size, config.intermediate_size, config.hidden_act
        )
        self.attention_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.num_prefix_tokens = 0

        self.positional_embedding = mx.zeros((config.image_num_pos, config.hidden_size))
        self.patch_embedding = nn.Linear(
            config.image_patch_size * config.image_patch_size * 3,
            config.hidden_size,
            bias=True,
        )
        self.resblocks = [
            Molmo2VisionBlock(config) for _ in range(config.num_hidden_layers)
        ]

    def add_pos_emb(self, x: mx.array, patch_num) -> mx.array:
        pos_emb = self.positional_embedding
        side = int(math.sqrt(pos_emb.shape[0]))
        pos_emb = pos_emb.reshape(side, side, pos_emb.shape[1])

        patch_num_0, patch_num_1 = patch_num
        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Bicubic interpolation for different sizes
            from ..base import interpolate

            pos_emb_np = pos_emb.transpose(2, 0, 1)[None]  # (1, C, H, W)
            pos_emb_np = interpolate(
                pos_emb_np, (patch_num_0, patch_num_1), mode="bicubic"
            )
            pos_emb = pos_emb_np[0].transpose(1, 2, 0)  # (H, W, C)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :]
        return x

    def __call__(self, x: mx.array, patch_num=None) -> List[mx.array]:
        if patch_num is None:
            patch_num = self.config.image_num_patch

        x = self.patch_embedding(x)
        x = self.add_pos_emb(x, patch_num)

        hidden_states = []
        for block in self.resblocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states

    @staticmethod
    def sanitize(weights):
        sanitized = {}
        for k, v in weights.items():
            # Map transformer.resblocks -> resblocks
            k = k.replace("transformer.resblocks", "resblocks")
            sanitized[k] = v
        return sanitized
