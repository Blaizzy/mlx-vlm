import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import CallableModuleList, VisionModel


def build_vision_projector(config):
    hidden_size = config.text_config.hidden_size
    projector_type = getattr(config, "mm_projector_type", "mlp2x_gelu")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = CallableModuleList()
        modules.append(nn.Linear(config.mm_hidden_size, hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return modules
    raise ValueError(f"Unknown projector type: {projector_type}")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = build_vision_projector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # Cast pixel_values to model dtype for consistent dtype propagation
        model_dtype = self.language_model.model.embed_tokens.weight.dtype
        pixel_values = pixel_values.astype(model_dtype)

        _, image_features, _ = self.vision_tower(pixel_values.transpose(0, 2, 3, 1))
        B, H, W, C = image_features.shape
        image_features = image_features.reshape(B, H * W, C)
        image_features = self.mm_projector(image_features)

        final_inputs_embeds = self.prepare_inputs_for_multimodal(
            image_features, input_ids, mask
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    # Source: https://github.com/apple/ml-fastvlm/blob/592b4add3c1c8a518e77d95dc6248e76c1dd591f/llava/model/llava_arch.py#L146
    def prepare_inputs_for_multimodal(self, image_features, input_ids, mask):
        # Batch evaluation to avoid repeated sync points
        if mask is not None:
            starts = mx.argmax(mask, axis=-1)
            lengths = mask.sum(axis=-1)
            mx.eval(starts, lengths)
            starts = starts.tolist()
            lengths = lengths.tolist()
            input_ids = [
                cur_input_ids[start : start + length]
                for cur_input_ids, start, length in zip(input_ids, starts, lengths)
            ]

        # Pre-compute image token masks and counts for all batch items (single eval)
        image_token_index = self.config.image_token_index
        image_masks = [
            cur_input_ids == image_token_index for cur_input_ids in input_ids
        ]
        num_images_list = [m.sum() for m in image_masks]
        mx.eval(image_masks + num_images_list)
        num_images_list = [n.item() for n in num_images_list]

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = num_images_list[batch_idx]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.model.embed_tokens(
                    cur_input_ids
                )
                cur_input_embeds = mx.concatenate(
                    [cur_input_embeds_1, cur_image_features[0:0]], axis=0
                )
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            # Find image token indices - mask was already evaluated above
            image_token_mask = image_masks[batch_idx]
            image_token_indices = (
                [-1]
                + np.where(np.array(image_token_mask))[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
            split_sizes = image_token_indices[1:]
            cur_input_embeds = self.language_model.model.embed_tokens(
                mx.concatenate(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = mx.split(cur_input_embeds, split_sizes)

            cur_new_input_embeds = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
            cur_new_input_embeds = mx.concatenate(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)

        if self.config.tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[: self.config.tokenizer_model_max_length] for x in new_input_embeds
            ]

        max_len = max(x.shape[0] for x in new_input_embeds)
        new_input_embeds_padded = []
        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            padded = cur_new_embed
            if max_len > cur_len:
                if self.config.tokenizer_padding_side == "left":
                    padded = mx.concatenate(
                        (
                            mx.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                            ),
                            cur_new_embed,
                        ),
                        axis=0,
                    )
                else:
                    padded = mx.concatenate(
                        (
                            cur_new_embed,
                            mx.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                            ),
                        ),
                        axis=0,
                    )
            new_input_embeds_padded.append(padded)
        new_input_embeds = mx.stack(new_input_embeds_padded)
        return new_input_embeds

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )
        logits = self.language_model(
            input_ids,
            mask=mask,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" in key:
                if "model.vision_tower" in key:
                    key = key.replace(
                        "model.vision_tower.vision_tower.model",
                        "vision_tower.vision_model",
                    )
                    key = key.replace("patch_embed", "patch_embed.blocks")
                return key
            if "lm_head" in key:
                return key
            if "mm_projector" in key:
                return key.replace("model.", "")
            if "language_model" not in key:
                return "language_model." + key
            return key

        return {transform_key(k): v for k, v in weights.items()}
