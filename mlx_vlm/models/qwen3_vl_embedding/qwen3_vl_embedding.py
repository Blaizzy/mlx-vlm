from typing import Optional

import mlx.core as mx

from ..cache import create_causal_mask
from ..pooling import EmbeddingOutput, normalize_embeddings
from ..qwen3_vl.qwen3_vl import Model as Qwen3VLModel
from .config import ModelConfig


def last_non_padding_token(
    hidden_states: mx.array,
    attention_mask: Optional[mx.array],
) -> mx.array:
    if attention_mask is None:
        return hidden_states[:, -1]

    last_one_positions = mx.argmax(attention_mask[:, ::-1], axis=1)
    token_positions = attention_mask.shape[1] - last_one_positions - 1
    batch_positions = mx.arange(hidden_states.shape[0])
    return hidden_states[batch_positions, token_positions]


class Model(Qwen3VLModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.normalize = config.normalize
        self.language_model.lm_head = None

    def _last_hidden_state(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mask=attention_mask,
        )

        language_model = self.language_model
        if pixel_values is not None:
            language_model._rope_deltas = None
            language_model._position_ids = None

        rope_mask = attention_mask
        model_mask = attention_mask
        if attention_mask is not None and attention_mask.ndim == 2:
            seq_length = attention_mask.shape[-1]
            causal_mask = create_causal_mask(seq_length)
            valid_tokens = attention_mask.astype(mx.bool_)
            key_mask = mx.expand_dims(valid_tokens, axis=(1, 2))
            query_mask = mx.expand_dims(valid_tokens, axis=(1, 3))
            model_mask = causal_mask[None, None, :, :] & key_mask & query_mask
        if (
            attention_mask is not None
            and attention_mask.shape[-1] != input_ids.shape[-1]
        ):
            rope_mask = None

        position_ids = None
        if rope_mask is None or rope_mask.ndim == 2:
            if (
                language_model._position_ids is not None
                and language_model._position_ids.shape[-1] >= input_ids.shape[-1]
            ):
                position_ids = language_model._position_ids[..., : input_ids.shape[-1]]
            else:
                position_ids, rope_deltas = language_model.get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=rope_mask,
                )
                language_model._rope_deltas = rope_deltas
                language_model._position_ids = position_ids

        return language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=model_mask,
            cache=None,
            position_ids=position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        hidden_states = self._last_hidden_state(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        pooled = last_non_padding_token(hidden_states, attention_mask)
        text_embeds = normalize_embeddings(pooled) if self.normalize else pooled
        return EmbeddingOutput(last_hidden_state=hidden_states, text_embeds=text_embeds)
