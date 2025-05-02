import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .language import LanguageModel, RMSNorm, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mm_input_projection_weight = mx.ones(
            (config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.text_config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        b, _, l = x.shape

        reshaped_vision_outputs = x.transpose(0, 2, 1)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            b, l, self.patches_per_image, self.patches_per_image
        )
        print(f"{reshaped_vision_outputs.shape=} {b=}")

        # Transpose to place h, w in indices 1, 2
        reshaped_vision_outputs = reshaped_vision_outputs.transpose(0, 2, 3, 1)
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.transpose(0, 3, 1, 2).flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(0, 2, 1)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = mx.einsum(
            "btm,md->btd", normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.astype(x.dtype)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids), None

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        print(f"{pixel_values.shape=}, {pixel_values.transpose(0, 2, 3, 1).shape=}")
        hidden_state, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype),
            output_hidden_states=True,
        )
        print(f"{hidden_state.shape=}")

        image_features = hidden_state.astype(pixel_values.dtype)
        image_features = self.multi_modal_projector(image_features)

        final_inputs_embeds, final_attention_mask_4d = (
            self._prepare_inputs_for_multimodal(
                image_features, inputs_embeds, input_ids, mask
            )
        )
        return final_inputs_embeds, final_attention_mask_4d

    def _prepare_inputs_for_multimodal(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        _, _, embed_dim = image_features.shape

        batch_size, sequence_length = input_ids.shape
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        scaled_image_features = mx.flatten(
            scaled_image_features, start_axis=0, end_axis=1
        )
        final_embedding = mx.zeros((batch_size, sequence_length, embed_dim))

        pad_token_id = self.config.pad_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == pad_token_id

        # expand masks to match embedding dimension
        text_mask_expanded = mx.expand_dims(text_mask, -1)
        text_mask_expanded = mx.repeat(text_mask_expanded, embed_dim, axis=-1)
        pad_mask_expanded = mx.expand_dims(pad_mask, -1)
        pad_mask_expanded = mx.repeat(pad_mask_expanded, embed_dim, axis=-1)

        # insert padding and text token embeddings
        final_embedding = mx.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        )
        # pad_size = final_embedding.shape[1] - scaled_image_features.shape[1]
        # scaled_image_features = mx.pad(
        #     scaled_image_features, ((0, 0), (0, pad_size), (0, 0))
        # )
        # insert image embeddings - the image mask is always less or equal to the sentence in length
        # image_mask_expanded = mx.expand_dims(image_mask, -1)
        # image_mask_expanded = mx.repeat(image_mask_expanded, embed_dim, axis=-1)
        # final_embedding = mx.where(
        #     image_mask_expanded, scaled_image_features, final_embedding
        # )
        import numpy as np
        import torch

        img_mask_torch = torch.from_dlpack(np.array(image_mask.squeeze(), copy=True))
        final_embedding_torch = torch.from_dlpack(
            np.array(final_embedding.squeeze(), copy=True)
        )
        image_features_torch = torch.from_dlpack(
            np.array(scaled_image_features, copy=True)
        )
        print(
            f"{final_embedding_torch.shape=}, {image_features_torch.shape=}, {img_mask_torch.shape=}"
        )

        # Expand mask to match the dimensions of final_embedding_torch
        img_mask_expanded = img_mask_torch.unsqueeze(-1).expand_as(
            final_embedding_torch
        )

        final_embedding_torch = final_embedding_torch.masked_scatter(
            img_mask_expanded, image_features_torch
        )
        final_embedding = mx.array(final_embedding_torch.float().numpy()).astype(
            final_embedding.dtype
        )[None, :]

        # # Implement masked_scatter-like behavior for image embeddings
        # # We need to take features sequentially from scaled_image_features
        # # and place them at positions where image_mask is True
        # for b in range(batch_size):
        #     # Find positions of image tokens in this batch
        #     # Since mx.argwhere is not available, we'll manually find the indices
        #     img_indices = []
        #     for i in range(sequence_length):
        #         if image_mask[b, i]:
        #             img_indices.append(i)

        #     if len(img_indices) > 0:
        #         # Number of image tokens to process (limited by what's available)
        #         n_tokens = min(len(img_indices), scaled_image_features.shape[1])

        #         # For each image position, create a temporary mask and apply the feature
        #         for i in range(n_tokens):
        #             # Position in the sequence
        #             pos = img_indices[i]

        #             # Feature to insert (from the i-th position in image features)
        #             feature = scaled_image_features[b, i]

        #             # Create a mask for just this position in this batch
        #             # Since .at[].set() is not available, we'll create the mask differently
        #             batch_mask = mx.zeros((batch_size,), dtype=mx.bool_)
        #             batch_mask = mx.array([j == b for j in range(batch_size)])

        #             pos_mask = mx.zeros((sequence_length,), dtype=mx.bool_)
        #             pos_mask = mx.array([j == pos for j in range(sequence_length)])

        #             # Expand dimensions to match the embedding shape
        #             batch_mask_expanded = mx.expand_dims(batch_mask, -1)
        #             batch_mask_expanded = mx.expand_dims(batch_mask_expanded, -1)
        #             batch_mask_expanded = mx.repeat(batch_mask_expanded, sequence_length, axis=1)
        #             batch_mask_expanded = mx.repeat(batch_mask_expanded, embed_dim, axis=2)

        #             pos_mask_expanded = mx.expand_dims(pos_mask, -1)
        #             pos_mask_expanded = mx.expand_dims(pos_mask_expanded, 0)
        #             pos_mask_expanded = mx.repeat(pos_mask_expanded, batch_size, axis=0)
        #             pos_mask_expanded = mx.repeat(pos_mask_expanded, embed_dim, axis=2)

        #             # Combine masks to target just this position in this batch
        #             position_mask = mx.logical_and(batch_mask_expanded, pos_mask_expanded)

        #             # Apply the feature to just this position
        #             final_embedding = mx.where(position_mask, feature, final_embedding)

        # final_embedding = mx.where(
        #     pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        # )

        attention_mask_expanded_1 = mx.expand_dims(attention_mask, 1)
        attention_mask_expanded_2 = mx.expand_dims(attention_mask, 2)
        final_attention_mask_4d = attention_mask_expanded_1 * attention_mask_expanded_2
        final_attention_mask_4d = final_attention_mask_4d
        final_attention_mask_4d = mx.expand_dims(final_attention_mask_4d, 1)
        final_embedding = mx.array(final_embedding)
        return final_embedding, final_attention_mask_4d

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeddings, final_attention_mask_4d = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )
        print(f"{input_embeddings.shape=}")
        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            # mask=final_attention_mask_4d, # TODO: Fix mask
        )
        return logits

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model
