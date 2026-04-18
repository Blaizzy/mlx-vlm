from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig, VisionConfig
from .language import LanguageModel, compute_pos_hw, create_falcon_ocr_mask


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig = None):
        super().__init__()


class Model(nn.Module):
    no_chunked_prefill = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config, config)

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

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
        else:
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
        self.language_model._full_attn_mask = create_falcon_ocr_mask(
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
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        kwargs["pixel_values"] = pixel_values
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        new_weights = {}
        for k, v in weights.items():

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

            new_weights[new_key] = v

        new_weights["language_model.model.cos_1d"] = self.language_model.model.cos_1d
        new_weights["language_model.model.sin_1d"] = self.language_model.model.sin_1d

        return new_weights
