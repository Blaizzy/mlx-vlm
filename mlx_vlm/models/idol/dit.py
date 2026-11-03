# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
# The original license is as follows:
#
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
"""The denoiser: a rectified-flow transformer over the object latent (8192 tokens) and the input camera (1 token),
conditioned on the image tokens. Joint attention over [latent, image, camera] with learned rotary positions.
"""

import mlx.core as mx
import mlx.nn as nn

from .layers import (
    MLP,
    PcdAbsolutePositionEmbedder,
    RePo3DRotaryEmbedding,
    SinusoidalEmbedder,
    UnifiedTransformerBlock,
    sanitize_weights,
)


class LatentSeqMMFlowModel(nn.Module):
    def __init__(
        self,
        q_token_length: int,
        in_channels: int,
        cam_channels: int,
        model_channels: int,
        cond_channels: int,
        cond2_channels: int,
        num_blocks: int,
        num_refiner_blocks: int = 2,
        num_heads: int = 16,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
    ):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.dtype = mx.float16

        def blocks(count, modulation=True):
            return list(
                [
                    UnifiedTransformerBlock(
                        model_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        modulation=modulation,
                    )
                    for _ in range(count)
                ]
            )

        def rope_layers(count):
            return list(
                [
                    RePo3DRotaryEmbedding(
                        model_channels, num_heads=num_heads, head_dim=num_head_channels
                    )
                    for _ in range(count)
                ]
            )

        self.t_embedder = SinusoidalEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
        )
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)

        # positions of the latent tokens (a fixed scrambled Sobol sequence, stored with the weights)
        self.pos_pe = mx.zeros((1, q_token_length, 3))
        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)

        # each stream is refined on its own, then all tokens go through the joint blocks
        self.noise_refiner = blocks(num_refiner_blocks)
        self.context_refiner = blocks(num_refiner_blocks, modulation=False)
        self.cam_refiner = MLP(
            cam_channels,
            model_channels,
            model_channels,
            mlp_layer_num=num_refiner_blocks,
        )
        self.blocks = blocks(num_blocks)

        self.shift_table = mx.zeros((1, 2, model_channels))
        self.out_layer = nn.Linear(model_channels, in_channels)
        self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    def sanitize(self, weights):
        return sanitize_weights(self, weights, transformer_dtype=self.dtype)

    def __call__(self, x: dict, t: mx.array, cond: dict) -> dict:
        """
        Args:
            x: dict with 'latent' [B, 8192, 16] and 'camera' [B, 1, 5]
            t: [B] time in [0, 1000]
            cond: dict with 'feature1' [B, L_c, 1280] (DINOv3) and 'feature2' [B, L_c, 128] (FLUX.2 VAE)
        Returns:
            the velocity, a dict shaped like x
        """
        z, cam = x["latent"], x["camera"]
        h_x = self.input_layer(z).astype(self.dtype)
        h_cond = self.cond_embedder(cond["feature1"]) + self.cond_embedder2(
            cond["feature2"]
        )
        h_cond = h_cond.astype(self.dtype)

        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb).astype(self.dtype)

        h_x = h_x + self.pos_embedder(self.pos_pe).astype(self.dtype)
        for repo, block in zip(self.noise_repo_layers, self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=repo(h_x))
        for repo, block in zip(self.context_repo_layers, self.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=repo(h_cond))
        h_cam = self.cam_refiner(cam).astype(self.dtype)

        h = mx.concatenate([h_x, h_cond, h_cam], axis=1)
        for repo, block in zip(self.repo_layers, self.blocks):
            h = block(h, mod=t_mod, rotary_emb=repo(h))

        h_x = h[:, : z.shape[1]].astype(z.dtype)
        h_x = mx.fast.layer_norm(h_x, None, None, eps=1e-5)
        h_cam = h[:, -cam.shape[1] :].astype(cam.dtype)
        h_cam = mx.fast.layer_norm(h_cam, None, None, eps=1e-5)
        shift, scale = mx.split(self.shift_table + t_emb[:, None], 2, axis=1)
        h_x = h_x * (1 + scale) + shift
        h_cam = h_cam * (1 + scale) + shift

        out = {"latent": self.out_layer(h_x)}
        out["camera"] = self.cam_out_layer(h_cam)
        return out
