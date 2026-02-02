"""ERNIE 4.5 VL MoE model for MLX."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .processor import Ernie4_5_VLProcessor, Ernie4_5_VLTokenizer, ImageProcessor
from .vision import VisionModel

# Register custom processor classes for ernie4_5_moe_vl model type
MODEL_TYPE = "ernie4_5_moe_vl"
try:
    AutoImageProcessor.register(MODEL_TYPE, slow_image_processor_class=ImageProcessor)
    AutoTokenizer.register(MODEL_TYPE, slow_tokenizer_class=Ernie4_5_VLTokenizer)
    AutoProcessor.register(MODEL_TYPE, Ernie4_5_VLProcessor)
except Exception:
    pass  # Already registered or registration not needed


class TokenType:
    """Token type definition."""

    text = 0
    image = 1
    video = 2


class VariableResolutionResamplerModel(nn.Module):
    """Compresses vision features using spatial and temporal convolutions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_conv_size: int,
        temporal_conv_size: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_temporal_conv = config.use_temporal_conv

        self.spatial_dim = in_dim * spatial_conv_size * spatial_conv_size
        self.temporal_dim = (
            in_dim * spatial_conv_size * spatial_conv_size * temporal_conv_size
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

        self.mlp = nn.Linear(self.spatial_dim, out_dim)
        self.after_norm = nn.RMSNorm(out_dim)

    def spatial_conv_reshape(self, x: mx.array) -> mx.array:
        S, C = x.shape
        x = x.reshape(-1, C * (self.spatial_conv_size**2))
        return x

    def __call__(
        self,
        x: mx.array,
        grid_thw: mx.array,
    ) -> mx.array:
        def fwd_spatial(x):
            x = self.spatial_conv_reshape(x)
            x = self.spatial_linear(x)
            return x

        def fwd_placeholder(x, grid_thw):
            grid_thw_np = np.array(grid_thw.tolist(), dtype=np.int64)
            grid_t = grid_thw_np[:, 0]
            grid_hw = grid_thw_np[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_np.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(tokens_per_img_or_vid.size, dtype=np.int64)
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got: {self.temporal_conv_size}"

            slice_offsets = []
            for temporal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + temp_offset * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = np.concatenate(slice_offsets, axis=-1).astype(np.int32)

            slice_offsets2 = []
            for temporal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporal_size > 1 else 0, temporal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + temp_offset * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = np.concatenate(slice_offsets2, axis=-1).astype(np.int32)

            x_timestep_1 = x[mx.array(slice_offsets), :]
            x_timestep_2 = x[mx.array(slice_offsets2), :]
            x = mx.concatenate([x_timestep_1, x_timestep_2], axis=-1)
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
    """ERNIE 4.5 VL MoE model."""

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
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )
        image_features = self.resampler_model(hidden_states, image_grid_thw)
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        if mx.sum(image_positions) == 0:
            return inputs_embeds

        batch_size, seq_len = input_ids.shape
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = int(mx.sum(image_mask).item())

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(
                    image_mask, cumsum - 1, mx.zeros_like(cumsum)
                )
                gathered_features = batch_features[feature_indices]

                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )

                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

    @property
    def layers(self):
        return self.language_model.model.layers

    def _build_token_type_ids(
        self, input_ids: mx.array, pixel_values: Optional[mx.array] = None
    ) -> Optional[mx.array]:
        if pixel_values is None:
            return None

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        is_image = input_ids == image_token_id
        is_video = input_ids == video_token_id
        is_vision = is_image | is_video

        if mx.sum(is_vision) == 0:
            return None

        token_type_ids = mx.where(
            is_vision, mx.ones_like(input_ids), mx.zeros_like(input_ids)
        )
        return token_type_ids

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        token_type_ids = self._build_token_type_ids(input_ids, pixel_values)

        inputs_embeds_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )

        logits = self.language_model(
            input_ids,
            inputs_embeds_features.inputs_embeds,
            mask=mask,
            cache=cache,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        return logits

    def sanitize(self, weights):
        import re

        def transform_key(key):
            if "vision_tower" not in key and "vision_model" in key:
                key = key.replace("vision_model", "vision_tower")

            if "language_model" not in key:
                if (
                    "model.layers" in key
                    or "model.embed_tokens" in key
                    or "model.norm" in key
                ):
                    key = key.replace("model.", "language_model.model.")
                elif "lm_head" in key:
                    key = key.replace("lm_head", "language_model.lm_head")

            if "model.resampler_model" in key:
                key = key.replace("model.resampler_model", "resampler_model")

            key = re.sub(
                r"(spatial_linear|temporal_linear)\.(\d+)", r"\1.layers.\2", key
            )

            return key

        weights = {transform_key(k): v for k, v in weights.items()}
        weights = self.vision_tower.sanitize(weights)
        weights = self.language_model.sanitize(weights)

        return weights
