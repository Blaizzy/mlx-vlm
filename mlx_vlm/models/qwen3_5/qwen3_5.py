from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..qwen3_vl import Model as Qwen3VLModel
from ..qwen3_vl.qwen3_vl import masked_scatter
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(Qwen3VLModel):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            # Reset position state for text-only generation
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states, _ = self.vision_tower(pixel_values, grid_thw)

        # Find the visual token boundary before merging (input_ids is still intact here).
        # mx.where() doesn't support the single-argument (nonzero) form, so use
        # (arange * mask).max() to locate the last image/video token position.
        flat_ids = input_ids.reshape(-1)
        image_mask = (flat_ids == self.config.image_token_index) | (
            flat_ids == self.config.video_token_index
        )
        has_image = bool(image_mask.any().item())
        if has_image:
            last_pos = int((mx.arange(flat_ids.shape[0]) * image_mask).max().item())
            image_end_index = last_pos + 1
        else:
            image_end_index = None

        # Insert special image tokens in the input_ids
        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        # Pre-calculate position_ids for chunked prefill
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            image_end_index=image_end_index,
        )

    def get_partial_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask,
        model_inputs: dict,
        partial_depth: int,
    ) -> mx.array:
        """
        Like get_input_embeddings but runs the vision tower only for images[partial_depth:].

        Returns inputs_embeds for the full sequence. Positions belonging to already-cached
        image blocks keep text embeddings — those tokens are not prefilled by the caller.
        """
        image_grid_thw = model_inputs.get("image_grid_thw")
        video_grid_thw = model_inputs.get("video_grid_thw")
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if grid_thw is None:
            raise ValueError(
                "image_grid_thw / video_grid_thw not available in model_inputs"
            )

        # Patches consumed by already-cached images
        thw = grid_thw.tolist()
        n_cached_patches = sum(t * h * w for t, h, w in thw[:partial_depth])

        # Vision tower for new images only
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        hidden_states_new, _ = self.vision_tower(
            pixel_values[n_cached_patches:].astype(dtype), grid_thw[partial_depth:]
        )

        # Text embeddings for the full sequence
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Find the start position of the first new image block (0-indexed: block partial_depth)
        img_tok = self.config.image_token_index
        vid_tok = self.config.video_token_index
        flat = input_ids[0].tolist()
        block, in_block, new_img_start = 0, False, len(flat)
        for i, tok in enumerate(flat):
            is_vis = tok == img_tok or tok == vid_tok
            if is_vis and not in_block:
                in_block = True
                if block == partial_depth:
                    new_img_start = i
                    break
            elif not is_vis and in_block:
                in_block = False
                block += 1

        # Overwrite only new image token positions using masked_scatter
        is_vis = (input_ids == img_tok) | (input_ids == vid_tok)
        is_new = mx.arange(input_ids.shape[1]) >= new_img_start
        new_img_mask = is_vis & is_new[None, :]
        new_img_mask_3d = mx.broadcast_to(new_img_mask[..., None], inputs_embeds.shape)
        inputs_embeds = masked_scatter(
            inputs_embeds, new_img_mask_3d, hidden_states_new
        )

        # Pre-calculate position_ids for multi-modal RoPE chunked prefill
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_image_mask = input_ids == image_token_index
        special_video_mask = input_ids == video_token_index
        special_image_mask = special_image_mask | special_video_mask
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds, special_image_mask, image_features
        )

        return inputs_embeds, special_image_mask

    def sanitize(self, weights):
        # ignore mtp weights
        weights = {key: value for key, value in weights.items() if "mtp." not in key}

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in norm_keys):
                if value.ndim == 1:
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
