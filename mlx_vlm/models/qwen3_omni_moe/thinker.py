from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_omni_moe.audio import AudioModel

from .config import ThinkerConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)

    return final_embedding


class Thinker(nn.Module):
    def __init__(self, config: ThinkerConfig):
        super().__init__()
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)

        self.audio_tower = AudioModel(config.audio_config)

        self.language_model = LanguageModel(config.text_config, config)

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
        audio_feature_lengths: Optional[mx.array] = None,
    ):
        if feature_attention_mask is not None:
            input_features = mx.transpose(input_features, (0, 2, 1))
            audio_feature_lengths = mx.sum(feature_attention_mask, axis=1)
            mask_bool = feature_attention_mask.astype(mx.bool_)
            batch_size, seq_len, hidden_dim = input_features.shape
            input_features_flat = mx.reshape(input_features, (-1, hidden_dim))
            mask_flat = mx.reshape(mask_bool, (-1,))
            indices = mx.array(np.where(mask_flat)[0])
            input_features = mx.take(input_features_flat, indices, axis=0)
            input_features = mx.transpose(input_features, (1, 0))

        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else (
                mx.sum(feature_attention_mask, axis=-1)
                if feature_attention_mask is not None
                else None
            )
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        return audio_outputs

    def get_placeholder_mask(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        image_features: Optional[mx.array] = None,
        video_features: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
    ):
        if input_ids is None:
            image_token_embed = self.language_model.model.embed_tokens(
                mx.array([self.config.image_token_id], dtype=mx.int32)
            )
            video_token_embed = self.language_model.model.embed_tokens(
                mx.array([self.config.video_token_id], dtype=mx.int32)
            )
            audio_token_embed = self.language_model.model.embed_tokens(
                mx.array([self.config.audio_token_id], dtype=mx.int32)
            )

            special_image_mask = mx.all(inputs_embeds == image_token_embed, axis=-1)
            special_video_mask = mx.all(inputs_embeds == video_token_embed, axis=-1)
            special_audio_mask = mx.all(inputs_embeds == audio_token_embed, axis=-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = mx.sum(special_image_mask)
        special_image_mask = mx.expand_dims(special_image_mask, axis=-1)
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)
        if (
            image_features is not None
            and mx.sum(special_image_mask) != image_features.size
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = mx.sum(special_video_mask)
        special_video_mask = mx.expand_dims(special_video_mask, axis=-1)
        special_video_mask = mx.broadcast_to(special_video_mask, inputs_embeds.shape)
        if (
            video_features is not None
            and mx.sum(special_video_mask) != video_features.size
        ):
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = mx.expand_dims(special_audio_mask, axis=-1)
        special_audio_mask = mx.broadcast_to(special_audio_mask, inputs_embeds.shape)
        return special_image_mask, special_video_mask, special_audio_mask

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
        audio_feature_lengths: Optional[mx.array] = None,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.astype(inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds
            )
            if audio_features.ndim == 2:
                audio_mask_flat = mx.reshape(
                    audio_mask[..., 0] if audio_mask.ndim == 3 else audio_mask, (-1,)
                )
                num_audio_tokens = int(mx.sum(audio_mask_flat))
                if num_audio_tokens > 0:
                    if audio_features.shape[0] != num_audio_tokens:
                        if audio_features.shape[0] > num_audio_tokens:
                            audio_features = audio_features[:num_audio_tokens]
                        else:
                            padding = mx.zeros(
                                (
                                    num_audio_tokens - audio_features.shape[0],
                                    audio_features.shape[1],
                                ),
                                dtype=audio_features.dtype,
                            )
                            audio_features = mx.concatenate(
                                [audio_features, padding], axis=0
                            )
            inputs_embeds = masked_scatter(inputs_embeds, audio_mask, audio_features)

        if pixel_values is not None:
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            pixel_values = pixel_values.astype(dtype)
            vision_output = self.vision_tower(pixel_values, image_grid_thw)
            if isinstance(vision_output, tuple):
                image_embeds, image_embeds_multiscale = vision_output
            else:
                image_embeds = vision_output
                image_embeds_multiscale = None
            image_embeds = image_embeds.astype(inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = masked_scatter(inputs_embeds, image_mask, image_embeds)
            visual_pos_masks = (
                image_mask[..., 0] if image_mask.ndim == 3 else image_mask
            )
            visual_embeds_multiscale = image_embeds_multiscale

        if pixel_values_videos is not None:
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            pixel_values_videos = pixel_values_videos.astype(dtype)
            vision_output = self.vision_tower(pixel_values_videos, video_grid_thw)
            if isinstance(vision_output, tuple):
                video_embeds, video_embeds_multiscale = vision_output
            else:
                video_embeds = vision_output
                video_embeds_multiscale = None
            video_embeds = video_embeds.astype(inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = masked_scatter(inputs_embeds, video_mask, video_embeds)

            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale
                visual_pos_masks = (
                    video_mask[..., 0] if video_mask.ndim == 3 else video_mask
                )
            else:
                visual_pos_masks = (
                    (video_mask | image_mask)[..., 0]
                    if video_mask.ndim == 3
                    else (video_mask | image_mask)
                )
                visual_mask_flat = mx.reshape(visual_pos_masks, (-1,))
                image_mask_flat = mx.reshape(
                    image_mask[..., 0] if image_mask.ndim == 3 else image_mask, (-1,)
                )
                video_mask_flat = mx.reshape(
                    video_mask[..., 0] if video_mask.ndim == 3 else video_mask, (-1,)
                )
                visual_indices_flat = mx.where(visual_mask_flat)[0]
                image_mask_on_visual = mx.take(
                    image_mask_flat, visual_indices_flat, axis=0
                )
                video_mask_on_visual = mx.take(
                    video_mask_flat, visual_indices_flat, axis=0
                )
                image_indices = mx.where(image_mask_on_visual)[0]
                video_indices = mx.where(video_mask_on_visual)[0]

                visual_embeds_multiscale_joint = []
                for img_embed, vid_embed in zip(
                    visual_embeds_multiscale, video_embeds_multiscale
                ):
                    embed_joint = mx.zeros(
                        (len(visual_indices_flat), img_embed.shape[-1]),
                        dtype=img_embed.dtype,
                    )
                    if len(image_indices) > 0:
                        embed_joint = mx.scatter(
                            embed_joint,
                            image_indices,
                            mx.take(img_embed, image_indices, axis=0),
                            axis=0,
                        )
                    if len(video_indices) > 0:
                        embed_joint = mx.scatter(
                            embed_joint,
                            video_indices,
                            mx.take(vid_embed, video_indices, axis=0),
                            axis=0,
                        )
                    visual_embeds_multiscale_joint.append(embed_joint)
                visual_embeds_multiscale = tuple(visual_embeds_multiscale_joint)

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

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

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)

        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = (
            self.get_input_embeddings(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
        )

        kwargs = {
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            **kwargs,
        }

        logits = self.language_model(
            input_ids, inputs_embeds, mask=mask, cache=cache, **kwargs
        )

        return logits

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "thinker" in key:
                if "thinker.model" in key:
                    key = key.replace("thinker.model", "thinker.language_model.model")
                elif "thinker.visual" in key:
                    key = key.replace("thinker.visual", "thinker.vision_tower")
                    if "merger" in key:
                        key = (
                            key.replace("ln_q", "norm")
                            .replace("mlp.0", "linear_fc1")
                            .replace("mlp.2", "linear_fc2")
                        )
                    if "merger_list" in key:
                        key = key.replace("merger_list", "deepstack_merger_list")

                if "thinker.lm_head" in key:
                    key = key.replace(
                        "thinker.lm_head", "thinker.language_model.lm_head"
                    )

            sanitized_weights[key] = value

        return sanitized_weights
