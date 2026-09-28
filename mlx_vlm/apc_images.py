"""Prefix-local image identity and suffix slicing for single-request Qwen3.5 APC.

Qwen4-Exp (Qwen3.8-Flash-Next) inherits the Qwen3.5 vision tower, image merge
and RoPE indexing, so the same identity applies to both model types.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .apc import media_token_spans, semantic_extra_hash
from .apc_prefix import PrefixContext

SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5", "qwen4_exp"})


@dataclass
class ImagePrefixContext(PrefixContext):
    pixel_offsets: list[int]
    pixel_values: Any
    image_grid_thw: Any

    @classmethod
    def prepare(cls, model, processor, token_ids, pixel_values, kwargs, tenant):
        config = model.config
        if config.model_type not in SUPPORTED_MODEL_TYPES:
            return None
        image_id = config.image_token_index
        if config.video_token_index in token_ids:
            return None
        spans = media_token_spans(token_ids, {image_id})
        grid = kwargs.get("image_grid_thw")
        if not spans:
            if pixel_values is not None or grid is not None:
                return None
            rows = []
        else:
            if (
                grid is None
                or pixel_values is None
                or grid.ndim != 2
                or grid.shape != (len(spans), 3)
                or pixel_values.ndim != 2
            ):
                return None
            rows = grid.tolist()
        merge = config.vision_config.spatial_merge_size
        offsets = [0]
        hashes = [
            semantic_extra_hash(
                tenant=tenant,
                model=model,
                processor=processor,
                media={"image_prefix_schema": f"{config.model_type}-v1"},
            )
        ]
        for i, ((start, end), row) in enumerate(zip(spans, rows)):
            if any(type(n) is not int or n <= 0 for n in row) or row[0] != 1:
                return None
            count = row[0] * row[1] * row[2]
            if (
                row[1] % merge
                or row[2] % merge
                or count // (merge * merge) != end - start
            ):
                return None
            offsets.append(offsets[-1] + count)
            if offsets[-1] > pixel_values.shape[0]:
                return None
            hashes.append(
                semantic_extra_hash(
                    image_hash=hashes[-1],
                    media={
                        "pixels": pixel_values[offsets[-2] : offsets[-1]],
                        "grid": grid[i],
                        "span": mx.array([start, end], dtype=mx.int32),
                    },
                )
            )
        if pixel_values is not None and offsets[-1] != pixel_values.shape[0]:
            return None
        return cls(list(token_ids), spans, hashes, offsets, pixel_values, grid)

    def suffix_inputs(self, prefix_len: int):
        self.prefix_hash(prefix_len)  # reject a split image before slicing
        completed = sum(end <= prefix_len for _, end in self.spans)
        if completed == len(self.spans):
            return None, None
        return (
            self.pixel_values[self.pixel_offsets[completed] :],
            self.image_grid_thw[completed:],
        )
