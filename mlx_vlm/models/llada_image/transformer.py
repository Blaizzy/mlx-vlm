# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted from inclusionAI/LLaDA-Image for MLX.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..z_image.transformer import (
    FeedForward,
    FinalLayer,
    MRoPE,
    TimestepEmbedder,
    ZImageTransformerBlock,
    apply_rotary,
)
from .conditioning import RMSNorm
from .config import LLaDAImageTransformerConfig

SEQUENCE_MULTIPLE = 32


class Attention(nn.Module):
    def __init__(self, config: LLaDAImageTransformerConfig):
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(config.dim, config.dim, bias=False)
        self.to_k = nn.Linear(config.dim, config.dim, bias=False)
        self.to_v = nn.Linear(config.dim, config.dim, bias=False)
        self.to_out = [nn.Linear(config.dim, config.dim, bias=False)]
        self.norm_q = RMSNorm(config.norm_eps) if config.qk_norm else nn.Identity()
        self.norm_k = RMSNorm(config.norm_eps) if config.qk_norm else nn.Identity()

    def __call__(self, x, cos, sin, mask=None):
        batch, length, dim = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        q = self.norm_q(self.to_q(x).reshape(shape))
        k = self.norm_k(self.to_k(x).reshape(shape))
        v = self.to_v(x).reshape(shape).transpose(0, 2, 1, 3)
        # RoPE is computed in float32, then rounded back before attention.
        q = apply_rotary(q.astype(mx.float32), cos, sin).astype(x.dtype)
        k = apply_rotary(k.astype(mx.float32), cos, sin).astype(x.dtype)
        x = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v,
            scale=self.scale,
            mask=mask,
        )
        return self.to_out[0](x.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class TransformerBlock(ZImageTransformerBlock):
    """The Z-Image residual/modulation equations with parameter-free norms."""

    def __init__(self, config: LLaDAImageTransformerConfig, modulation: bool):
        nn.Module.__init__(self)
        self.modulation = modulation
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config.dim, int(config.dim / 3 * 8))
        self.attention_norm1 = RMSNorm(config.norm_eps)
        self.attention_norm2 = RMSNorm(config.norm_eps)
        self.ffn_norm1 = RMSNorm(config.norm_eps)
        self.ffn_norm2 = RMSNorm(config.norm_eps)
        if modulation:
            self.adaLN_modulation = [nn.Linear(min(config.dim, 256), 4 * config.dim)]

    def __call__(
        self,
        x,
        cos,
        sin,
        adaln_input=None,
        mask=None,
        *,
        noise_mask=None,
        clean_time=None,
    ):
        if noise_mask is None:
            return super().__call__(x, cos, sin, adaln_input, mask)
        noisy = mx.split(self.adaLN_modulation[0](adaln_input), 4, axis=-1)
        clean = mx.split(self.adaLN_modulation[0](clean_time), 4, axis=-1)

        def select(noisy_value, clean_value):
            return mx.where(
                noise_mask[..., None], noisy_value[:, None], clean_value[:, None]
            )

        scale_msa = select(1 + noisy[0], 1 + clean[0])
        gate_msa = select(mx.tanh(noisy[1]), mx.tanh(clean[1]))
        scale_mlp = select(1 + noisy[2], 1 + clean[2])
        gate_mlp = select(mx.tanh(noisy[3]), mx.tanh(clean[3]))
        attended = self.attention(self.attention_norm1(x) * scale_msa, cos, sin, mask)
        x = x + gate_msa * self.attention_norm2(attended)
        return x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        )


def sequence_positions(length: int, start: int = 1) -> mx.array:
    ids = mx.arange(start, start + length, dtype=mx.float32)
    positions = mx.stack([ids, mx.zeros(length), mx.zeros(length)], axis=-1)
    return mx.pad(positions, [(0, (-length) % SEQUENCE_MULTIPLE), (0, 0)])[None]


def image_positions(height: int, width: int, start: int) -> mx.array:
    length = height * width
    positions = mx.stack(
        [
            mx.full((length,), start),
            mx.repeat(mx.arange(height), width),
            mx.tile(mx.arange(width), height),
        ],
        axis=-1,
    ).astype(mx.float32)
    return mx.pad(positions, [(0, (-length) % SEQUENCE_MULTIPLE), (0, 0)])[None]


def padded_positions(cap_length: int, height: int, width: int):
    caption = sequence_positions(cap_length)
    return caption, image_positions(height, width, caption.shape[1] + 1)


def pad_tokens(x: mx.array, token: mx.array) -> mx.array:
    padding = (-x.shape[1]) % SEQUENCE_MULTIPLE
    if not padding:
        return x
    return mx.concatenate(
        [x, mx.broadcast_to(token, (x.shape[0], padding, x.shape[-1]))], axis=1
    )


class LLaDAImageTransformer(nn.Module):
    """Text/VQ-conditioned generation and native editing on FLUX.2 latents."""

    def __init__(self, config: LLaDAImageTransformerConfig):
        super().__init__()
        self.config = config
        self.x_embedder = nn.Linear(config.in_channels, config.dim)
        self.t_embedder = TimestepEmbedder(min(config.dim, 256))
        self.cap_embedder = [
            RMSNorm(config.norm_eps),
            nn.Linear(config.cap_feat_dim, config.dim),
        ]
        self.semantic_embedder = [
            RMSNorm(config.norm_eps),
            nn.Linear(config.semantic_feat_dim, config.dim),
        ]
        self.sigvq_embedder = [
            RMSNorm(config.norm_eps),
            nn.Linear(config.semantic_feat_dim, config.dim),
        ]
        self.noise_refiner = [
            TransformerBlock(config, True) for _ in range(config.n_refiner_layers)
        ]
        self.context_refiner = [
            TransformerBlock(config, False) for _ in range(config.n_refiner_layers)
        ]
        self.sigvq_refiner = [
            TransformerBlock(config, False) for _ in range(config.n_refiner_layers)
        ]
        self.layers = [TransformerBlock(config, True) for _ in range(config.n_layers)]
        self.final_layer = FinalLayer(
            config.dim, config.in_channels, min(config.dim, 256)
        )
        self.x_pad_token = mx.zeros((1, config.dim))
        self.cap_pad_token = mx.zeros((1, config.dim))
        self.sigvq_pad_token = mx.zeros((1, config.dim))
        self.rope = MRoPE(config.axes_dims, config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        caption: mx.array,
        *,
        semantic=None,
        source_latents=None,
    ) -> mx.array:
        if source_latents is not None:
            if semantic is None:
                raise ValueError("Editing requires SigVQ features")
            return self._edit(x, t, caption, semantic, source_latents)
        batch, channels, height, width = x.shape
        image_length = height * width
        patches = x.transpose(0, 2, 3, 1).reshape(batch, image_length, channels)
        image_tokens = pad_tokens(self.x_embedder(patches), self.x_pad_token)
        caption_tokens = pad_tokens(
            self.cap_embedder[1](self.cap_embedder[0](caption)), self.cap_pad_token
        )
        cap_pos, image_pos = padded_positions(caption.shape[1], height, width)
        if semantic is not None:
            features = self.semantic_embedder[1](self.semantic_embedder[0](semantic))
            features = pad_tokens(features, self.cap_pad_token)
            semantic_pos = sequence_positions(
                semantic.shape[1], caption_tokens.shape[1] + 1
            )
            caption_tokens = mx.concatenate([caption_tokens, features], axis=1)
            cap_pos = mx.concatenate([cap_pos, semantic_pos], axis=1)
            image_pos = image_positions(height, width, caption_tokens.shape[1] + 1)
        cap_cos, cap_sin = self.rope.compute_freqs(cap_pos)
        image_cos, image_sin = self.rope.compute_freqs(image_pos)
        time = self.t_embedder(t * self.config.t_scale)
        for block in self.noise_refiner:
            image_tokens = block(image_tokens, image_cos, image_sin, time)
        for block in self.context_refiner:
            caption_tokens = block(caption_tokens, cap_cos, cap_sin)
        tokens = mx.concatenate([image_tokens, caption_tokens], axis=1)
        cos = mx.concatenate([image_cos, cap_cos], axis=1)
        sin = mx.concatenate([image_sin, cap_sin], axis=1)
        for block in self.layers:
            tokens = block(tokens, cos, sin, time)
        output = self.final_layer(tokens[:, :image_length], time)
        return output.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)

    def _edit(self, x, t, caption, semantic, source):
        batch, channels, height, width = x.shape
        if source.shape != x.shape:
            raise ValueError("Source and target latent shapes must match")
        time = self.t_embedder(mx.abs(t) * self.config.t_scale)
        clean_time = self.t_embedder(mx.zeros_like(t))
        captions = pad_tokens(
            self.cap_embedder[1](self.cap_embedder[0](caption)), self.cap_pad_token
        )
        cap_length = caption.shape[1]
        cap_pos = mx.concatenate(
            [
                sequence_positions(cap_length),
                sequence_positions(cap_length, cap_length + 3),
            ],
            axis=1,
        )
        cap_noise = mx.concatenate(
            [
                mx.zeros(captions.shape[:2], mx.bool_),
                mx.ones(captions.shape[:2], mx.bool_),
            ],
            axis=1,
        )
        captions = mx.concatenate([captions, captions], axis=1)
        images = [
            pad_tokens(
                self.x_embedder(
                    value.transpose(0, 2, 3, 1).reshape(batch, height * width, channels)
                ),
                self.x_pad_token,
            )
            for value in (source, x)
        ]
        source_length = images[0].shape[1]
        image_noise = mx.concatenate(
            [
                mx.zeros(images[0].shape[:2], mx.bool_),
                mx.ones(images[1].shape[:2], mx.bool_),
            ],
            axis=1,
        )
        images = mx.concatenate(images, axis=1)
        image_pos = mx.concatenate(
            [
                image_positions(height, width, cap_length + 1),
                image_positions(height, width, 2 * cap_length + 3),
            ],
            axis=1,
        )
        image_cos, image_sin = self.rope.compute_freqs(image_pos)
        for block in self.noise_refiner:
            images = block(
                images,
                image_cos,
                image_sin,
                time,
                noise_mask=image_noise,
                clean_time=clean_time,
            )
        cap_cos, cap_sin = self.rope.compute_freqs(cap_pos)
        for block in self.context_refiner:
            captions = block(captions, cap_cos, cap_sin)
        tokens = [captions, images]
        positions = [cap_pos, image_pos]
        noise = [cap_noise, image_noise]
        # CFG drops the semantic sequence altogether on the unconditional branch.
        if semantic.shape[1]:
            features = pad_tokens(
                self.sigvq_embedder[1](self.sigvq_embedder[0](semantic)),
                self.sigvq_pad_token,
            )
            feature_pos = sequence_positions(
                semantic.shape[1], captions.shape[1] + images.shape[1] + 1
            )
            feature_cos, feature_sin = self.rope.compute_freqs(feature_pos)
            for block in self.sigvq_refiner:
                features = block(features, feature_cos, feature_sin)
            tokens.append(features)
            positions.append(feature_pos)
            noise.append(mx.zeros(features.shape[:2], mx.bool_))
        tokens = mx.concatenate(tokens, axis=1)
        noise = mx.concatenate(noise, axis=1)
        cos, sin = self.rope.compute_freqs(mx.concatenate(positions, axis=1))
        for block in self.layers:
            tokens = block(
                tokens, cos, sin, time, noise_mask=noise, clean_time=clean_time
            )
        # Only the noisy target image is decoded, so its final modulation is uniform.
        start = captions.shape[1] + source_length
        output = self.final_layer(tokens[:, start : start + height * width], time)
        return output.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)


def sanitize_transformer_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    result = {}
    for key, value in weights.items():
        key = key.replace("all_x_embedder.1-1.", "x_embedder.")
        key = key.replace("all_final_layer.1-1.", "final_layer.")
        key = key.replace(
            "final_layer.adaLN_modulation.1.", "final_layer.adaLN_modulation.0."
        )
        key = key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        key = key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        if ".attention.to_qkv." in key:
            for name, tensor in zip(
                ("to_q", "to_k", "to_v"), mx.split(value, 3, axis=0)
            ):
                result[key.replace(".to_qkv.", f".{name}.")] = tensor
            continue
        if ".feed_forward.w13." in key:
            for name, tensor in zip(("w1", "w3"), mx.split(value, 2, axis=0)):
                result[key.replace(".w13.", f".{name}.")] = tensor
            continue
        result[key] = value
    return result
