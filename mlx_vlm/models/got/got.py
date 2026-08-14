from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..qwen2.language import LanguageModel
from . import processing_got  # noqa: F401
from .config import ModelConfig
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.out_dim, config.text_config.hidden_size
        )
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Output from VisionModel is (B, H*W, 1024)
        hidden_states = self.vision_tower(pixel_values)

        # Project features
        image_features = self.multi_modal_projector(hidden_states)

        # In GOT-OCR, the image token is represented by im_patch_token.
        # We replace the embeddings at these positions with the image features.
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.im_patch_token,
            image_features,
            inputs_embeds,
            input_ids,
        )

        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        image_positions = input_ids == image_token_id

        batch_size, _ = input_ids.shape
        batch_outputs = []

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                # (B, 256, 1024): one image per sequence, image_token_len each.
                batch_features = image_features[batch_idx]

                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]

                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

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

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        logits = self.language_model(
            input_ids=input_ids,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
        )
        return logits

    def sanitize(self, weights):
        # Skip the conv transposes for checkpoints already in MLX layout, so
        # sanitize stays idempotent and does not double-transpose (see #1871).
        already_mlx = any(
            k.endswith("patch_embed.proj.weight")
            and v.ndim == 4
            and v.shape[-1] == 3
            and v.shape[1] != 3
            for k, v in weights.items()
        )

        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("model.vision_tower_high."):
                k_vis = k.replace("model.vision_tower_high.", "vision_tower.")
                if "neck.0." in k_vis:
                    sanitized_weights[k_vis.replace("neck.0.", "conv1.")] = v
                elif "neck.1." in k_vis:
                    sanitized_weights[k_vis.replace("neck.1.", "norm1.")] = v
                elif "neck.2." in k_vis:
                    sanitized_weights[k_vis.replace("neck.2.", "conv2.")] = v
                elif "neck.3." in k_vis:
                    sanitized_weights[k_vis.replace("neck.3.", "norm2.")] = v
                else:
                    sanitized_weights[k_vis] = v
            elif k.startswith("model.mm_projector_vary."):
                # Named so skip_multimodal_module() keeps it unquantized.
                sanitized_weights[
                    k.replace("model.mm_projector_vary.", "multi_modal_projector.")
                ] = v
            elif k.startswith("model."):
                new_k = k.replace("model.", "language_model.model.")
                sanitized_weights[new_k] = v
            elif k.startswith("lm_head."):
                sanitized_weights[k.replace("lm_head.", "language_model.lm_head.")] = v
            else:
                sanitized_weights[k] = v

        # Transpose PyTorch Conv2d weights (O, I, H, W) to MLX Conv2d weights (O, H, W, I)
        for k in list(sanitized_weights.keys()):
            if (
                "patch_embed.proj.weight" in k
                or "conv1.weight" in k
                or "conv2.weight" in k
                or "net_2.weight" in k
                or "net_3.weight" in k
            ):
                # PyTorch conv2d weight shape: (out_channels, in_channels, kH, kW)
                # MLX conv2d weight shape: (out_channels, kH, kW, in_channels)
                w = sanitized_weights[k]
                if w.ndim == 4 and not already_mlx:
                    sanitized_weights[k] = w.transpose(0, 2, 3, 1)

        if self.config.text_config.tie_word_embeddings:
            sanitized_weights.pop("language_model.lm_head.weight", None)

        return sanitized_weights
