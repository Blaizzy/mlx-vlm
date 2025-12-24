from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import interpolate
from .config import AdapterConfig, VisionConfig, VitConfig


def _gelu_from_name(name: str) -> nn.Module:
    if name == "gelu_pytorch_tanh":
        return nn.GELU(approx="fast")
    return nn.GELU(approx="fast")


class ViTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act = _gelu_from_name(hidden_act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(self.act(self.w1(x)))


class ViTMultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        input_dim: Optional[int] = None,
        use_bias: bool = True,
        float32_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = head_dim**-0.5
        self.float32_attention = float32_attention

        input_dim = input_dim or hidden_size
        self.wq = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=use_bias)
        self.wk = nn.Linear(
            input_dim, self.num_key_value_heads * self.head_dim, bias=use_bias
        )
        self.wv = nn.Linear(
            input_dim, self.num_key_value_heads * self.head_dim, bias=use_bias
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

    def __call__(
        self,
        inputs_q: mx.array,
        inputs_kv: Optional[mx.array] = None,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_kv is None:
            inputs_k = inputs_q
            inputs_v = inputs_q
        else:
            inputs_k = inputs_kv
            inputs_v = inputs_kv

        xq = self.wq(inputs_q)
        xk = self.wk(inputs_k)
        xv = self.wv(inputs_v)

        bsz, q_len, _ = xq.shape
        _, kv_len, _ = xk.shape

        xq = xq.reshape(bsz, q_len, self.num_heads, self.head_dim)
        xk = xk.reshape(bsz, kv_len, self.num_key_value_heads, self.head_dim)
        xv = xv.reshape(bsz, kv_len, self.num_key_value_heads, self.head_dim)

        if self.num_heads != self.num_key_value_heads:
            xk = mx.repeat(xk, self.num_key_value_groups, axis=2)
            xv = mx.repeat(xv, self.num_key_value_groups, axis=2)

        q = xq.transpose(0, 2, 1, 3)
        k = xk.transpose(0, 2, 1, 3)
        v = xv.transpose(0, 2, 1, 3)

        dtype = q.dtype
        if self.float32_attention:
            q = q.astype(mx.float32)
            k = k.astype(mx.float32)
            v = v.astype(mx.float32)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if attn_mask is not None:
            scores = mx.where(
                attn_mask,
                scores,
                mx.full(scores.shape, vals=-1e9, dtype=scores.dtype),
            )

        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, v).astype(dtype)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.wo(out)


class Molmo2VisionBlock(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            float32_attention=config.float32_attention,
            input_dim=config.hidden_size,
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


class Molmo2VisionTransformer(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config
        self.num_prefix_tokens = 0

        self.positional_embedding = mx.zeros((config.image_num_pos, config.hidden_size))
        patch_dim = config.image_patch_size * config.image_patch_size * 3
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=True)
        self.transformer = [
            Molmo2VisionBlock(config) for _ in range(config.num_hidden_layers)
        ]

    def add_pos_emb(self, x: mx.array, patch_num: Tuple[int, int]) -> mx.array:
        pos_emb = self.positional_embedding
        pos_emb_size = int(pos_emb.shape[0] ** 0.5)
        pos_emb = mx.reshape(pos_emb, (pos_emb_size, pos_emb_size, pos_emb.shape[1]))

        patch_h, patch_w = patch_num
        if pos_emb.shape[0] != patch_h or pos_emb.shape[1] != patch_w:
            pos_emb = mx.transpose(pos_emb[None, ...], (0, 3, 1, 2))
            pos_emb = interpolate(
                pos_emb, (patch_h, patch_w), mode="cubic", align_corners=False
            )
            pos_emb = mx.transpose(pos_emb, (0, 2, 3, 1))[0]

        pos_emb = mx.reshape(pos_emb, (-1, pos_emb.shape[-1]))
        return x + pos_emb[None, :, :].astype(x.dtype)

    def __call__(
        self,
        x: mx.array,
        patch_num: Optional[Tuple[int, int]] = None,
    ):
        if patch_num is None:
            patch_num = self.config.image_num_patch

        x = self.patch_embedding(x)
        x = self.add_pos_emb(x, patch_num)

        hidden_states = []
        for block in self.transformer:
            x = block(x)
            hidden_states.append(x)
        return hidden_states


class ImageProjectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = "molmo2"
        self.vit_config: VitConfig = config.vit_config
        self.adapter_config: AdapterConfig = config.adapter_config

        self.image_vit = Molmo2VisionTransformer(self.vit_config)

        self.vit_layers = []
        for layer in self.adapter_config.vit_layers:
            self.vit_layers.append(
                layer if layer >= 0 else layer + self.vit_config.num_hidden_layers
            )

        pool_dim = self.vit_config.hidden_size * len(self.vit_layers)

        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=self.adapter_config.hidden_size,
            num_heads=self.adapter_config.num_attention_heads,
            num_key_value_heads=self.adapter_config.num_key_value_heads,
            head_dim=self.adapter_config.head_dim,
            input_dim=pool_dim,
            float32_attention=self.adapter_config.float32_attention,
        )

        self.image_projector = ImageProjectorMLP(
            self.adapter_config.hidden_size,
            self.adapter_config.intermediate_size,
            self.adapter_config.text_hidden_size,
        )

    def encode_image(self, images: mx.array) -> mx.array:
        batch_size, num_crops, num_patch, patch_dim = images.shape
        images = images.reshape(batch_size * num_crops, num_patch, patch_dim)
        hidden_states = self.image_vit(images)

        features = [hidden_states[layer] for layer in self.vit_layers]
        image_features = mx.concatenate(features, axis=-1)
        image_features = image_features.reshape(batch_size, num_crops, num_patch, -1)
        return image_features

    def __call__(
        self,
        images: mx.array,
        pooled_patches_idx: mx.array,
    ) -> mx.array:
        batch_size, num_crops = images.shape[:2]

        image_features = self.encode_image(images)
        dim = image_features.shape[-1]

        valid = pooled_patches_idx >= 0
        valid_token = mx.any(valid, axis=-1)

        flat_features = image_features.reshape(batch_size, -1, dim)
        idx = mx.clip(pooled_patches_idx, 0, None)
        batch_idx = mx.arange(batch_size)[:, None, None]
        batch_idx = mx.broadcast_to(batch_idx, idx.shape)

        gathered = flat_features[mx.reshape(batch_idx, (-1,)), mx.reshape(idx, (-1,))]
        to_pool = gathered.reshape(
            pooled_patches_idx.shape[0],
            pooled_patches_idx.shape[1],
            pooled_patches_idx.shape[2],
            dim,
        )

        to_pool = to_pool * valid[..., None].astype(to_pool.dtype)
        to_pool = to_pool.reshape(-1, pooled_patches_idx.shape[-1], dim)

        if self.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape(-1, 1, 1, valid.shape[-1])
            denom = valid.reshape(-1, to_pool.shape[-2]).astype(mx.float32).sum(axis=-1)
            denom = mx.where(denom == 0, mx.ones_like(denom), denom)
            query = to_pool.sum(axis=-2, keepdims=True) / denom[:, None, None].astype(
                to_pool.dtype
            )
        else:
            attn_mask = None
            query = mx.mean(to_pool, axis=-2, keepdims=True)

        pooled = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled = pooled.reshape(batch_size, -1, pooled.shape[-1])
        pooled = self.image_projector(pooled)

        pooled = pooled.reshape(-1, pooled.shape[-1])

        # MLX doesn't support boolean indexing, so convert to integer indices
        valid_flat = np.array(valid_token).flatten()
        valid_indices = np.where(valid_flat)[0]
        return pooled[mx.array(valid_indices)]
