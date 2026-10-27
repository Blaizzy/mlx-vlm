"""Ming-Image NextDiT: a single-stream Lumina-2.0-style diffusion transformer."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.z_image.transformer import (
    FinalLayer,
    MRoPE,
    TimestepEmbedder,
    ZImageTransformerBlock,
    sanitize_transformer_weights,
)

from .config import MingImageDiTConfig

SEQ_MULTIPLE = 32


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class MingImageTransformer(nn.Module):
    def __init__(self, config: MingImageDiTConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        head_dim = dim // config.n_heads
        if config.n_kv_heads != config.n_heads:
            raise ValueError("Ming-Image DiT expects full multi-head attention")
        if sum(config.axes_dims) != head_dim:
            raise ValueError("axes_dims must sum to the attention head dimension")
        adaln_dim = min(dim, config.adaln_embed_dim)
        patch, f_patch = config.patch_size, config.f_patch_size
        self.patch_size = patch
        self.f_patch_size = f_patch
        self.in_channels = config.in_channels
        self.t_scale = config.t_scale

        def block(modulation: bool) -> ZImageTransformerBlock:
            return ZImageTransformerBlock(
                dim,
                config.n_heads,
                head_dim,
                config.intermediate_size,
                modulation=modulation,
                adaln_dim=adaln_dim,
            )

        self.t_embedder = TimestepEmbedder(adaln_dim)
        self.cap_embedder = [
            nn.RMSNorm(config.cap_feat_dim, eps=config.norm_eps),
            nn.Linear(config.cap_feat_dim, dim),
        ]
        self.x_embedder = nn.Linear(f_patch * patch * patch * config.in_channels, dim)
        self.noise_refiner = [block(True) for _ in range(config.n_refiner_layers)]
        self.context_refiner = [block(False) for _ in range(config.n_refiner_layers)]
        self.layers = [block(True) for _ in range(config.n_layers)]
        self.final_layer = FinalLayer(
            dim, patch * patch * f_patch * config.in_channels, adaln_dim
        )
        self.rope = MRoPE(config.axes_dims, config.rope_theta)

    def _patchify(self, x: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
        b, c, f, h, w = x.shape
        p, pf = self.patch_size, self.f_patch_size
        ft, ht, wt = f // pf, h // p, w // p
        x = x.reshape(b, c, ft, pf, ht, p, wt, p)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)
        return x.reshape(b, ft * ht * wt, pf * p * p * c), (ft, ht, wt)

    def _unpatchify(self, x: mx.array, shape: tuple[int, int, int]) -> mx.array:
        b = x.shape[0]
        ft, ht, wt = shape
        p, pf = self.patch_size, self.f_patch_size
        c = self.in_channels
        x = x.reshape(b, ft, ht, wt, pf, p, p, c)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(b, c, ft * pf, ht * p, wt * p)

    @staticmethod
    def _key_mask(valid_len: int, total_len: int) -> mx.array | None:
        if valid_len == total_len:
            return None
        return (mx.arange(total_len) < valid_len).reshape(1, 1, 1, total_len)

    def _positions(
        self, cap_len: int, image_shape: tuple[int, int, int]
    ) -> tuple[mx.array, mx.array]:
        ft, ht, wt = image_shape
        cap_t = mx.arange(1, cap_len + 1, dtype=mx.float32)
        zeros = mx.zeros(cap_len)
        cap_pos = mx.stack([cap_t, zeros, zeros], axis=-1)[None]
        f_ids = mx.arange(ft, dtype=mx.float32) + (cap_len + 1)
        h_ids = mx.arange(ht, dtype=mx.float32)
        w_ids = mx.arange(wt, dtype=mx.float32)
        img_pos = mx.stack(
            [
                mx.repeat(f_ids, ht * wt),
                mx.tile(mx.repeat(h_ids, wt), (ft,)),
                mx.tile(w_ids, (ft * ht,)),
            ],
            axis=-1,
        )[None]
        return cap_pos, img_pos

    @staticmethod
    def _pad_sequence(tokens: mx.array, target_len: int) -> mx.array:
        pad = target_len - tokens.shape[1]
        if pad == 0:
            return tokens
        b, _, dim = tokens.shape
        return mx.concatenate(
            [tokens, mx.zeros((b, pad, dim), dtype=tokens.dtype)], axis=1
        )

    def _unified_mask(
        self, image_len: int, padded_image_len: int, cap_len: int, padded_cap_len: int
    ) -> mx.array | None:
        if image_len == padded_image_len and cap_len == padded_cap_len:
            return None
        keep = mx.concatenate(
            [
                mx.arange(padded_image_len) < image_len,
                mx.arange(padded_cap_len) < cap_len,
            ],
            axis=0,
        )
        return keep.reshape(1, 1, 1, keep.shape[0])

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        cap_feats: mx.array,
        cap_feats_2: mx.array | None = None,
    ) -> mx.array:
        """One denoising forward; caption conditioning is shared across the batch."""
        patches, (ft, ht, wt) = self._patchify(x)
        image_len = patches.shape[1]
        padded_image_len = _round_up(image_len, SEQ_MULTIPLE)

        cap_tokens = self.cap_embedder[1](self.cap_embedder[0](cap_feats))
        if cap_feats_2 is not None:
            cap_tokens = mx.concatenate([cap_tokens, cap_feats_2], axis=1)
        cap_len = cap_tokens.shape[1]
        padded_cap_len = _round_up(cap_len, SEQ_MULTIPLE)
        cap_tokens = self._pad_sequence(cap_tokens, padded_cap_len)

        img_tokens = self._pad_sequence(self.x_embedder(patches), padded_image_len)
        t_emb = self.t_embedder(t * self.t_scale).astype(img_tokens.dtype)

        cap_pos, img_pos = self._positions(padded_cap_len, (ft, ht, wt))
        if padded_image_len != image_len:
            img_pos = mx.concatenate(
                [img_pos, mx.zeros((1, padded_image_len - image_len, 3))], axis=1
            )
        img_cos, img_sin = self.rope.compute_freqs(img_pos)
        cap_cos, cap_sin = self.rope.compute_freqs(cap_pos)

        img_mask = self._key_mask(image_len, padded_image_len)
        for blk in self.noise_refiner:
            img_tokens = blk(img_tokens, img_cos, img_sin, t_emb, img_mask)
        cap_mask = self._key_mask(cap_len, padded_cap_len)
        for blk in self.context_refiner:
            cap_tokens = blk(cap_tokens, cap_cos, cap_sin, None, cap_mask)

        if cap_tokens.shape[0] != img_tokens.shape[0]:
            cap_tokens = mx.broadcast_to(
                cap_tokens, (img_tokens.shape[0], *cap_tokens.shape[1:])
            )
        unified = mx.concatenate([img_tokens, cap_tokens], axis=1)
        cos = mx.concatenate([img_cos, cap_cos], axis=1)
        sin = mx.concatenate([img_sin, cap_sin], axis=1)
        mask = self._unified_mask(image_len, padded_image_len, cap_len, padded_cap_len)
        for blk in self.layers:
            unified = blk(unified, cos, sin, t_emb, mask)

        img_out = self.final_layer(unified[:, :image_len], t_emb)
        return self._unpatchify(img_out, (ft, ht, wt))


__all__ = ["MingImageTransformer", "sanitize_transformer_weights"]
