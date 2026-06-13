"""MiniMax-M3 VL top-level model for mlx-vlm.

Wires the CLIP+3D-RoPE vision tower → two-stage GELU connector (per-patch projection then
spatial-merge fusion) → image-token scatter → MiniMax-M3 text tower. Text-only prompts skip the
vision path entirely, which is the strategist's use on bat-studio.
"""
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class MultiModalProjector(nn.Module):
    """Per-patch vision->text projection (checkpoint `multi_modal_projector`)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.projector_hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=True)

    def __call__(self, x):
        return self.linear_2(nn.gelu(self.linear_1(x)))


class PatchMerge(nn.Module):
    """Fuse spatial_merge_size**2 neighbouring patches into one text token (`patch_merge_mlp`)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        merged = config.text_config.hidden_size * (config.vision_config.spatial_merge_size ** 2)
        self.linear_1 = nn.Linear(merged, config.projector_hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=True)

    def __call__(self, x):
        return self.linear_2(nn.gelu(self.linear_1(x)))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.patch_merge_mlp = PatchMerge(config)
        self.language_model = LanguageModel(config.text_config, config)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

    def get_image_features(self, pixel_values, grid_thw):
        feats = self.vision_tower(pixel_values, grid_thw)        # [num_patches, vis_hidden]
        feats = self.multi_modal_projector(feats)                # [num_patches, text_hidden]
        m2 = self.spatial_merge_size ** 2
        feats = feats.reshape(feats.shape[0] // m2, -1)          # [num_patches/m^2, text_hidden*m^2]
        return self.patch_merge_mlp(feats)                       # [num_tokens, text_hidden]

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids))

        grid_thw = kwargs.get("image_grid_thw", kwargs.get("video_grid_thw"))
        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        image_features = self.get_image_features(pixel_values.astype(dtype), grid_thw)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        merged = self.merge_image_features(input_ids, inputs_embeds, image_features)
        return InputEmbeddingsFeatures(inputs_embeds=merged)

    def merge_image_features(self, input_ids, inputs_embeds, image_features):
        img_id = self.config.image_token_index
        vid_id = self.config.video_token_index
        positions = input_ids == img_id
        if mx.sum(positions) == 0:
            positions = input_ids == vid_id
        image_features = image_features.astype(inputs_embeds.dtype)

        outs = []
        start = 0
        for b in range(input_ids.shape[0]):
            mask = positions[b]
            n = int(mx.sum(mask).item())
            if n == 0:
                outs.append(inputs_embeds[b])
                continue
            feats = image_features[start:start + n]
            start += n
            idx = mx.where(mask, mx.cumsum(mask.astype(mx.int32)) - 1, 0)
            gathered = feats[idx]
            outs.append(mx.where(mask[:, None], gathered, inputs_embeds[b]))
        return mx.stack(outs, axis=0)

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(input_ids, feats.inputs_embeds, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model."):]
            if k.startswith("vision_tower.vision_model."):
                k = k.replace("vision_tower.vision_model.", "vision_tower.")
            if k.startswith("vision_tower.encoder.layers."):
                k = k.replace("vision_tower.encoder.layers.", "vision_tower.encoder_layers.")
            # Conv3d patch embed: checkpoint is torch layout (O,I,D,H,W); mlx wants (O,D,H,W,I).
            if k.endswith("patch_embedding.weight") and hasattr(v, "ndim") and v.ndim == 5:
                v = v.transpose(0, 2, 3, 4, 1)
            out[k] = v
        # hand text-tower weights to the language sanitize (drops indexer, stacks experts)
        text = {k: v for k, v in out.items() if k.startswith("language_model.")}
        rest = {k: v for k, v in out.items() if not k.startswith("language_model.")}
        text = self.language_model.sanitize(text)
        rest.update(text)
        return rest
