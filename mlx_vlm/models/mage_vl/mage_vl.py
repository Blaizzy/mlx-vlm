"""Mage-VL — codec-native streaming multimodal model (microsoft/Mage-VL, Apache-2.0).

Composition only: a Mage-ViT tower (`vision.py`, net-new) feeding a stock Qwen3-4B decoder
(`language.py`, reused verbatim). See those modules for the reproduced-upstream quirks.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig, VisionConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
) -> mx.array:
    shape = final_embedding.shape
    flat_embedding = mx.flatten(final_embedding)
    flat_embedding[mx.flatten(image_mask_expanded).astype(mx.bool_)] = mx.flatten(
        scaled_image_features
    )
    return mx.reshape(flat_embedding, shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        patch_positions = kwargs.get("patch_positions", None)
        grid_thw = kwargs.get("image_grid_thw", None)
        if grid_thw is None:
            grid_thw = kwargs.get("video_grid_thw", None)
        grid_thw = _as_grid_list(grid_thw)

        if patch_positions is None:
            if not grid_thw:
                raise ValueError(
                    "mage_vl needs patch_positions or a grid_thw to derive them"
                )
            patch_positions = _positions_from_grid(grid_thw, self.config.vision_config)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        hidden_states = self.vision_tower(
            pixel_values.astype(inputs_embeds.dtype),
            patch_positions,
            grid_thw,
            cu_seqlens=kwargs.get("cu_seqlens", None),
        )

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        is_image = (input_ids == image_token_id) | (input_ids == video_token_id)
        mask_expanded = mx.expand_dims(is_image, -1)
        mask_expanded = mx.repeat(mask_expanded, inputs_embeds.shape[-1], axis=-1)

        merged = masked_scatter(
            inputs_embeds, mask_expanded, hidden_states.astype(inputs_embeds.dtype)
        )
        # The generate dispatch consumes `.inputs_embeds` — a bare array is our internal shape,
        # InputEmbeddingsFeatures is the mlx-vlm interface.
        return InputEmbeddingsFeatures(inputs_embeds=merged)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def sanitize(self, weights):
        """Remap the hub's key namespace onto the mlx-vlm one.

        hub                             -> here
        model.language_model.*          -> language_model.model.*
        lm_head.*                       -> language_model.lm_head.*
        model.visual.*                  -> vision_tower.*
        """
        out = {}
        for k, v in weights.items():
            if k.startswith("model.language_model."):
                k = "language_model.model." + k[len("model.language_model.") :]
            elif k.startswith("model.visual."):
                k = "vision_tower." + k[len("model.visual.") :]
            elif k.startswith("lm_head."):
                k = "language_model.lm_head." + k[len("lm_head.") :]
            out[k] = v

        vision = {
            k[len("vision_tower.") :]: v
            for k, v in out.items()
            if k.startswith("vision_tower.")
        }
        vision = self.vision_tower.sanitize(vision)
        for k, v in vision.items():
            out["vision_tower." + k] = v
        return out


def _as_grid_list(grid_thw) -> List[Tuple[int, int, int]]:
    if grid_thw is None:
        return []
    rows = mx.array(grid_thw).tolist()
    if rows and not isinstance(rows[0], list):
        rows = [rows]  # a single (3,) grid rather than (N, 3)
    return [tuple(int(x) for x in row) for row in rows]


def _positions_from_grid(
    grid_thw: List[Tuple[int, int, int]], config: VisionConfig
) -> mx.array:
    """Derive (L, 3) [t, h, w] patch positions in the processor's 2x2 block order.

    The Qwen2VL image processor emits patches grouped so that each consecutive run of
    `spatial_merge_size**2` belongs to one merge block — the merger relies on that ordering, so
    the positions handed to RoPE have to follow it too. A plain row-major t/h/w walk would put
    the right count of patches in the wrong places: silent, and only visible as degraded output.
    """
    m = config.spatial_merge_size
    rows = []
    for t, h, w in grid_thw:
        for ti in range(t):
            for hb in range(h // m):
                for wb in range(w // m):
                    for dh in range(m):
                        for dw in range(m):
                            rows.append((ti, hb * m + dh, wb * m + dw))
    return mx.array(rows, dtype=mx.int32)
