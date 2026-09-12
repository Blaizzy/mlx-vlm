from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_paligemma  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        output = self.linear(x)
        return output


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache, draft_model
        del draft_kind, prefill_kwargs
        # The prefix-LM mask is built across the whole prompt, so splitting the
        # prompt into causal chunks changes the attention pattern and with it
        # the generated text. Prefill this model in one pass.
        return not bool(
            getattr(self.config.text_config, "use_bidirectional_attention", False)
        )

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

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            hidden_state, _, _ = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype),
                output_hidden_states=True,
            )

            image_features = hidden_state.astype(pixel_values.dtype)
            image_features = self.multi_modal_projector(image_features)

        final_inputs_embeds, final_attention_mask_4d = (
            self._prepare_inputs_for_multimodal(
                image_features, inputs_embeds, input_ids, mask
            )
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=final_inputs_embeds, attention_mask_4d=final_attention_mask_4d
        )

    def _prepare_inputs_for_multimodal(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        embed_dim = image_features.shape[-1]
        batch_size, sequence_length = input_ids.shape
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        final_embedding = mx.zeros((batch_size, sequence_length, embed_dim))
        valid = (
            input_ids != self.config.pad_token_id
            if attention_mask is None
            else attention_mask.astype(mx.bool_)
        )
        final_embedding = mx.where(valid[..., None], inputs_embeds, final_embedding)
        rows, columns = np.where(np.asarray(input_ids) == self.config.image_token_index)
        features = scaled_image_features.reshape(-1, embed_dim)
        if len(rows) != features.shape[0]:
            raise ValueError("Image features do not match image placeholders")
        final_embedding[mx.array(rows), mx.array(columns)] = features
        allowed = valid[:, :, None] & valid[:, None, :]
        allowed = mx.where(
            valid[:, :, None], allowed, mx.eye(sequence_length, dtype=mx.bool_)[None]
        )
        final_attention_mask_4d = allowed[:, None]
        return final_embedding, final_attention_mask_4d

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )
        input_embeddings = input_embeddings_features.inputs_embeds
        final_attention_mask_4d = input_embeddings_features.attention_mask_4d

        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            mask=final_attention_mask_4d,
        )
        return logits
