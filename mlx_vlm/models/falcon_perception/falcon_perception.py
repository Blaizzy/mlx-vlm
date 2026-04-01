import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig, VisionConfig
from .language import LanguageModel, compute_pos_hw, create_falcon_perception_mask


class FourierEncoder(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, feat_dim // 2, bias=False)
        self.transform = nn.Linear(feat_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        f = 2 * math.pi * self.embed(x)
        f = mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)
        return self.transform(f)


class BboxDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.relu(self.w1(x)) ** 2)


class SegmDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.layers = [nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)]
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.relu(layer(x)) ** 2
        return self.pixel_layer(x)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig = None):
        super().__init__()


class Model(nn.Module):
    no_chunked_prefill = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config, config)

        hidden_size = config.text_config.hidden_size

        self.coord_encoder = FourierEncoder(
            2, config.coord_enc_dim, hidden_size
        )
        self.coord_decoder = BboxDecoder(
            hidden_size, config.coord_dec_dim, config.coord_out_dim
        )
        self.size_encoder = FourierEncoder(
            2, config.size_enc_dim, hidden_size
        )
        self.size_decoder = BboxDecoder(
            hidden_size, config.size_dec_dim, config.size_out_dim
        )

        if config.do_segmentation:
            self.proj_segm = SegmDecoder(
                hidden_size, config.segm_out_dim, config.num_segm_layers
            )
            self.conv_segm = nn.Conv2d(
                hidden_size, config.segm_out_dim, kernel_size=3, padding=1
            )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_hw = kwargs.get("image_grid_hw", None)

        if pixel_values is None:
            self.language_model._rope_delta = None
            self.language_model._position_ids = None
            self.language_model._pos_hw = None
            self.language_model._full_attn_mask = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_states = self._patchify_and_project(pixel_values)

        final_embeds = self._merge_image_features(
            self.config.img_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        position_ids, pos_hw, delta = self._precompute_positions(
            input_ids, image_grid_hw
        )
        self.language_model._position_ids = position_ids
        self.language_model._pos_hw = pos_hw
        self.language_model._rope_delta = delta
        single_ids = input_ids[0:1] if input_ids.ndim == 2 else input_ids
        self.language_model._full_attn_mask = create_falcon_perception_mask(
            single_ids,
            self.config.image_cls_token_id,
            self.config.img_end_id,
        )

        return InputEmbeddingsFeatures(inputs_embeds=final_embeds)

    def _precompute_positions(self, input_ids, image_grid_hw):
        single_ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        ids = single_ids.reshape(-1).tolist()
        start_id = self.config.image_cls_token_id
        end_id = self.config.img_end_id

        pos_t = []
        in_image = False
        next_pos = 0
        for tok in ids:
            if tok == start_id and not in_image:
                in_image = True
            pos_t.append(next_pos)
            if not in_image:
                next_pos += 1
            if tok == end_id and in_image:
                in_image = False
                next_pos += 1

        position_ids = mx.array(pos_t, dtype=mx.int32)
        delta = int(mx.max(position_ids).item()) + 1 - len(ids)

        grid_hws = None
        if image_grid_hw is not None:
            if isinstance(image_grid_hw, mx.array):
                grid_hws = image_grid_hw.tolist()
            elif isinstance(image_grid_hw, list):
                grid_hws = image_grid_hw
            if grid_hws:
                grid_hws = [tuple(int(x) for x in g) for g in grid_hws]
                if input_ids.ndim == 2:
                    grid_hws = grid_hws[:1]

        pos_hw = compute_pos_hw(
            single_ids.reshape(-1),
            image_token_id=self.config.img_id,
            image_grid_hws=grid_hws,
        )

        return position_ids, pos_hw, delta

    def _patchify_and_project(self, pixel_values: mx.array) -> mx.array:
        ps = self.config.vision_config.spatial_patch_size
        pt = self.config.vision_config.temporal_patch_size

        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]

        N, H, W, C = pixel_values.shape
        h_patches = H // ps
        w_patches = W // ps

        patches = pixel_values.reshape(N, h_patches, ps, w_patches, ps, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(N * h_patches * w_patches, ps * ps * C * pt)

        return self.language_model.model.img_projector(
            patches.astype(self.language_model.model.img_projector.weight.dtype)
        )

    @staticmethod
    def _merge_image_features(
        image_token_id: int,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        image_positions = input_ids == image_token_id

        batch_outputs = []
        feature_start = 0

        for b in range(batch_size):
            mask = image_positions[b]
            n_pos = mx.sum(mask).item()

            if n_pos > 0:
                batch_feats = image_features[feature_start : feature_start + n_pos]
                if batch_feats.shape[0] != n_pos:
                    raise ValueError(
                        f"Image token positions ({n_pos}) does not match "
                        f"image features ({batch_feats.shape[0]}) for batch {b}"
                    )
                cumsum = mx.cumsum(mask.astype(mx.int32))
                feat_idx = mx.where(mask, cumsum - 1, 0)
                gathered = batch_feats[feat_idx]
                mask_exp = mx.expand_dims(mask, axis=-1)
                batch_out = mx.where(mask_exp, gathered, inputs_embeds[b])
                feature_start += n_pos
            else:
                batch_out = inputs_embeds[b]

            batch_outputs.append(batch_out)

        return mx.stack(batch_outputs, axis=0)

    def encode_coords_into_embeds(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        coord_xy: Optional[mx.array] = None,
    ) -> mx.array:
        if coord_xy is None:
            return inputs_embeds
        coord_mask = input_ids == self.config.coord_token_id
        if not mx.any(coord_mask).item():
            return inputs_embeds
        coord_tokens = self.coord_encoder(coord_xy.reshape(-1, 2))
        coord_tokens = coord_tokens.reshape(inputs_embeds.shape[0], -1, inputs_embeds.shape[-1])
        mask_exp = mx.expand_dims(coord_mask, axis=-1)
        return mx.where(mask_exp, coord_tokens, inputs_embeds)

    def encode_sizes_into_embeds(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        size_hw: Optional[mx.array] = None,
    ) -> mx.array:
        if size_hw is None:
            return inputs_embeds
        size_mask = input_ids == self.config.size_token_id
        if not mx.any(size_mask).item():
            return inputs_embeds
        size_tokens = self.size_encoder(size_hw.reshape(-1, 2))
        size_tokens = size_tokens.reshape(inputs_embeds.shape[0], -1, inputs_embeds.shape[-1])
        mask_exp = mx.expand_dims(size_mask, axis=-1)
        return mx.where(mask_exp, size_tokens, inputs_embeds)

    def decode_coords(self, hidden_state: mx.array) -> mx.array:
        logits = self.coord_decoder(hidden_state)
        half = self.config.coord_out_dim // 2
        return logits.reshape(-1, 2, half)

    def decode_sizes(self, hidden_state: mx.array) -> mx.array:
        logits = self.size_decoder(hidden_state)
        half = self.config.size_out_dim // 2
        return logits.reshape(-1, 2, half)

    @staticmethod
    def process_sizes(logits: mx.array) -> mx.array:
        num_bins = logits.shape[-1]
        pred = mx.argmax(logits, axis=-1).astype(mx.float32) / (num_bins - 1)
        min_size = math.log2(1.0 / num_bins)
        max_size = 0.0
        pred = pred * (max_size - min_size) + min_size
        return 2.0 ** pred

    @property
    def last_hidden_state(self) -> Optional[mx.array]:
        return self.language_model.model._last_hidden_state

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        coord_xy = kwargs.pop("coord_xy", None)
        size_hw = kwargs.pop("size_hw", None)

        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        embeds = features.inputs_embeds

        embeds = self.encode_coords_into_embeds(embeds, input_ids, coord_xy)
        embeds = self.encode_sizes_into_embeds(embeds, input_ids, size_hw)

        kwargs["pixel_values"] = pixel_values
        return self.language_model(
            input_ids,
            inputs_embeds=embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():
            # Skip itok_upsampler (AnyUp) weights - segmentation upsampler
            # not needed for text generation
            if k.startswith("itok_upsampler."):
                continue

            new_key = k

            if k.startswith("tok_embeddings."):
                new_key = k.replace(
                    "tok_embeddings.", "language_model.model.embed_tokens.", 1
                )
            elif k.startswith("img_projector."):
                new_key = k.replace(
                    "img_projector.", "language_model.model.img_projector.", 1
                )
            elif k.startswith("norm."):
                new_key = k.replace("norm.", "language_model.model.norm.", 1)
            elif k.startswith("output."):
                new_key = k.replace("output.", "language_model.lm_head.", 1)
            elif k == "freqs_cis_golden":
                new_key = "language_model.model.freqs_cis_golden"
            elif k.startswith("layers."):
                new_key = k.replace("layers.", "language_model.model.layers.", 1)
                new_key = new_key.replace(".attention.", ".self_attn.")
                new_key = new_key.replace(".feed_forward.", ".mlp.")

            if ".w13." in new_key:
                v = mx.concatenate([v[0::2], v[1::2]], axis=0)

            # Conv2d: transpose from PyTorch [O,I,H,W] to MLX [O,H,W,I]
            if "conv_segm.weight" in k:
                v = v.transpose(0, 2, 3, 1)

            new_weights[new_key] = v

        new_weights["language_model.model.cos_1d"] = self.language_model.model.cos_1d
        new_weights["language_model.model.sin_1d"] = self.language_model.model.sin_1d

        return new_weights
