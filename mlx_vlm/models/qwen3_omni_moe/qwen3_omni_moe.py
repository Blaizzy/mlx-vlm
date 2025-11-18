from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .audio import AudioModel
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(
    final_embedding: mx.array,
    modality_mask: mx.array,
    modality_features: mx.array,
):
    """
    MLX equivalent of ``torch.masked_scatter``.

    Args:
        final_embedding: Tensor containing the text embeddings.
        modality_mask: Boolean mask that selects the placeholder positions.
        modality_features: Tensor holding the multimodal features that should be
            inserted at the masked locations.
    """

    final_shape = final_embedding.shape
    mask_flat = modality_mask.reshape(-1)
    target = final_embedding.reshape(-1, final_shape[-1])
    source = modality_features.reshape(-1, final_shape[-1])

    if mx.sum(mask_flat).item() != source.shape[0]:
        raise ValueError(
            "Number of multimodal features does not match the amount of "
            "placeholder tokens in the prompt."
        )

    scatter_indices = mx.cumsum(mask_flat.astype(mx.int32)) - 1
    scatter_indices = mx.clip(
        scatter_indices, a_min=0, a_max=max(source.shape[0] - 1, 0)
    )
    gathered = source[scatter_indices]

    mask_broadcast = mx.broadcast_to(mask_flat[:, None], target.shape)
    updated = mx.where(mask_broadcast, gathered, target)
    return updated.reshape(final_shape)


class Model(nn.Module):
    """
    Minimal Thinker-only port of Qwen3-Omni-MoE.

    The architecture mirrors :mod:`mlx_vlm.models.qwen3_vl_moe` with the
    addition of the audio tower (currently stubbed) and the token ids required
    by the omni prompt template.  This enables loading Qwen3-Omni checkpoints in
    MLX for text+vision scenarios while we work on bringing the full audio
    encoder / talker stack online.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        self.audio_tower = AudioModel(config.audio_config)

    @property
    def layers(self):
        return self.language_model.model.layers

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[mx.array]]:
        """
        Compute the fused text + vision embeddings for the Thinker decoder.

        Returns the embeddings along with the DeepStack masks so that the
        language model can inject visual features in the early layers.
        """

        if input_ids is None:
            raise ValueError("Qwen3-Omni requires token ids for multimodal fusion")

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None or grid_thw is None:
            return inputs_embeds, None, None

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        hidden_states, deepstack_embeds = self.vision_tower(pixel_values, grid_thw)
        split_sizes = (
            grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2
        ).tolist()
        hidden_states = mx.split(hidden_states, split_sizes)
        hidden_states = mx.concatenate(hidden_states, axis=0).astype(
            hidden_states[0].dtype
        )

        inputs_embeds, image_mask = self.merge_with_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            (
                self.config.image_token_id,
                self.config.video_token_id,
            ),
        )

        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_embeds
        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    @staticmethod
    def merge_with_features(
        features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        placeholder_ids: Tuple[int, int],
    ) -> Tuple[mx.array, mx.array]:
        special_mask = mx.zeros_like(input_ids, dtype=mx.bool_)
        for token_id in placeholder_ids:
            special_mask = mx.logical_or(special_mask, input_ids == token_id)

        special_mask = mx.expand_dims(special_mask, -1)
        special_mask = mx.broadcast_to(special_mask, inputs_embeds.shape)

        if special_mask.sum() != features.size:
            raise ValueError(
                "Mismatch between multimodal placeholders and visual features"
            )

        fused = masked_scatter(inputs_embeds, special_mask, features)
        return fused, special_mask

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        input_features: Optional[mx.array] = None,
        **kwargs,
    ):
        if input_features is not None:
            raise NotImplementedError(
                "Audio inputs are not supported yet in the MLX port. "
                "Please omit `input_features` until the audio encoder is added."
            )

        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        pixels = pixel_values if pixel_values is not None else pixel_values_videos

        embeds, visual_pos_masks, deepstack_embeds = self.get_input_embeddings(
            input_ids,
            pixel_values=pixels,
            grid_thw=grid_thw,
        )

        logits = self.language_model(
            input_ids,
            embeds,
            mask=mask,
            cache=cache,
            pixel_values=pixels,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds,
            **kwargs,
        )
        return logits

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "language_model" in key:
                new_key = key.replace("model.language_model", "language_model.model")
                sanitized[new_key] = value
            elif "vision" in key:
                sanitized[key.replace("model.vision", "vision_tower")] = value
            else:
                sanitized[key.replace("model.", "")]
        return sanitized
