from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class TokenType:
    """token type definition"""

    text = 0
    image = 1
    video = 2


class VariableResolutionResamplerModel(nn.Module):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_temporal_conv = config.use_temporal_conv

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress 3d conv(video) to 1d
        self.temporal_dim = (
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        self.spatial_linear = nn.Sequential(
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.GELU(),
            nn.Linear(self.spatial_dim, self.spatial_dim),
            nn.LayerNorm(self.spatial_dim, eps=1e-6),
        )

        if self.use_temporal_conv:
            self.temporal_linear = nn.Sequential(
                nn.Linear(self.temporal_dim, self.spatial_dim),
                nn.GELU(),
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.LayerNorm(self.spatial_dim, eps=1e-6),
            )

        self.mlp = nn.Linear(self.spatial_dim, self.out_dim)

        self.after_norm = nn.RMSNorm(self.out_dim)

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        reshape before linear to imitation conv
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def __call__(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x = self.spatial_linear(x)

            return x

        def fwd_placeholder(x, grid_thw):
            """
            x: [S, H]
            grid_thw: [S, 3]
                the second dimension: [t, h, w]
            """
            grid_thw = grid_thw.to_numpy()

            grid_t, grid_hw = grid_thw[:, 0], grid_thw[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(tokens_per_img_or_vid.size)
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = np.concatenate(slice_offsets, axis=-1)

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = np.concatenate(slice_offsets2, axis=-1)

            x_timestep_1 = x[slice_offsets, :]
            x_timestep_2 = x[slice_offsets2, :]
            x = mx.concat([x_timestep_1, x_timestep_2], axis=-1)
            return x

        def fwd_temporal(x):
            x = self.temporal_linear(x)
            return x

        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            return x

        x = fwd_spatial(x)
        if self.use_temporal_conv:
            x = fwd_placeholder(x, grid_thw)
            x = fwd_temporal(x)
        x = fwd_mlp(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.resampler_model = VariableResolutionResamplerModel(
            config.pixel_hidden_size,
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config=config,
        )

        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )
        return final_inputs_embeds

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merge image features into input embeddings at image token positions.

        Args:
            image_token_id: The token ID for image placeholders
            video_token_id: The token ID for video placeholders (fallback)
            image_features: Vision features from the vision tower [num_features, hidden_dim]
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            input_ids: Input token IDs [batch_size, seq_len]
            grid_thw: Grid dimensions for each image (optional, not used in simple case)

        Returns:
            Updated input embeddings with image features inserted
        """
        # Find positions of image tokens
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        if token_type_ids is None:
            token_type_ids = image_positions.to(mx.int64)
            token_type_ids_labels = mx.concat(
                [token_type_ids[:, 1:], token_type_ids[:, -1:]], axis=1
            )
        else:
            token_type_ids_labels = token_type_ids[..., 1:]

        token_type_ids_w_video = token_type_ids[..., :-1].clone()
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image

        image_features = self.resampler_model(
            image_features,
            image_mask,
            token_type_ids_w_video,
            image_type_ids,
            grid_thw,
        )

        # Get dimensions
        batch_size, seq_len = input_ids.shape

        # Process each batch item
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            # Get mask for this batch
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                # Extract features for this batch
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                # Validate we have the right number of features
                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                # Create indices for gathering
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)

                # Gather features
                gathered_features = batch_features[feature_indices]

                # Combine with original embeddings
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )

                feature_start_idx += num_positions
            else:
                # No image tokens in this batch item
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        # Stack all batch outputs
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
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values, grid_thw)

        kwargs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            **kwargs,
        }

        logits = self.language_model(
            input_ids, inputs_embeds, mask=mask, cache=cache, **kwargs
        )

        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" not in key:
                key = key.replace("visual", "vision_tower")
            if "language_model" not in key:
                if "model" in key:
                    key = key.replace("model", "language_model.model")
                elif "lm_head" in key:
                    key = key.replace("lm_head", "language_model.lm_head")
            return key

        return {transform_key(k): v for k, v in weights.items()}
