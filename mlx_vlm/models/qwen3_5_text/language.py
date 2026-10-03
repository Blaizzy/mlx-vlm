from typing import Optional

import mlx.core as mx

from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel


class LanguageModel(Qwen3_5LanguageModel):
    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        """Positions for a decoder-only checkpoint, which has no vision grids.

        The shared implementation reads vision_config.spatial_merge_size before
        branching, so it cannot run without a vision config.
        """
        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(
                attention_mask == 0, mx.ones_like(position_ids), position_ids
            )
            max_position_ids = position_ids.max(axis=-1, keepdims=True)
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(
                position_ids, (input_ids.shape[0], input_ids.shape[1])
            )
            mrope_position_deltas = mx.zeros(
                [input_ids.shape[0], 1], dtype=input_ids.dtype
            )
        return position_ids, mrope_position_deltas
