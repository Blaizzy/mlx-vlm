from typing import Optional

import mlx.core as mx

from ..qwen3_5.language import LanguageModel as Qwen35LanguageModel


class LanguageModel(Qwen35LanguageModel):
    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        if image_grid_thw is not None or video_grid_thw is not None:
            raise ValueError(
                "MiniCPM-V 4.6 injects vision features through inputs_embeds and "
                "does not support Qwen3.5-VL image_grid_thw/video_grid_thw MRoPE "
                "inside its language model."
            )

        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(
                attention_mask == 0, mx.ones_like(position_ids), position_ids
            )
            max_position_ids = position_ids.max(axis=-1, keepdims=True)
            position_ids = mx.broadcast_to(
                position_ids[None, :, :], (3, *position_ids.shape)
            )
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(
                position_ids, (3, input_ids.shape[0], input_ids.shape[1])
            )
            mrope_position_deltas = mx.zeros(
                [input_ids.shape[0], 1],
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
