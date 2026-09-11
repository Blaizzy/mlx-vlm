from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .processing_deepseek_v41 import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
from .vision import Aligner, ViT


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        self.vision = ViT(config)
        self.aligner = Aligner(config)
        self.image_start = mx.zeros((config.hidden_size,))
        self.image_end = mx.zeros((config.hidden_size,))
        self.image_newline = mx.zeros((config.hidden_size,))

    def encode_image(self, patches: mx.array, n_vit_h: int, n_vit_w: int) -> mx.array:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: mx.array) -> mx.array:
        """Overwrite each image's token span with ViT features and span markers.

        `images` is a per-batch list of records with `start`, `patches`,
        `n_vit_h`, `n_vit_w`, and `types` (IMAGE_START/IMAGE/NEWLINE/END codes).
        IMAGE slots take aligner rows in reading order; delimiters take the
        learned span embeddings.
        """
        dtype = h.dtype
        marks = {
            IMAGE_START: self.image_start,
            IMAGE_END: self.image_end,
            IMAGE_NEW_LINE: self.image_newline,
        }
        rows = []
        for b, sample in enumerate(images):
            row = h[b]
            for img in sample or []:
                codes = mx.array(img.types)
                end = img.start + len(img.types)
                span = row[img.start : end]
                is_image = codes == IMAGE
                order = mx.cumsum(is_image.astype(mx.int32), axis=0) - 1
                embeds = self.encode_image(img.patches, img.n_vit_h, img.n_vit_w)
                gathered = embeds[mx.clip(order, 0, embeds.shape[0] - 1)]
                span = mx.where(is_image[:, None], gathered.astype(dtype), span)
                for code, mark in marks.items():
                    span = mx.where((codes == code)[:, None], mark.astype(dtype), span)
                row = mx.concatenate([row[: img.start], span, row[end:]], axis=0)
            rows.append(row)
        return mx.stack(rows, axis=0).astype(dtype)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[List] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None and input_ids.shape[1] != 1:
            inputs_embeds = self.merge_image_embeddings(pixel_values, inputs_embeds)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        def transform_key(key):
            if key.startswith("language_model."):
                return key
            if key.startswith("model.") or key.startswith("lm_head."):
                return f"language_model.{key}"
            if key.startswith("embed."):
                return f"language_model.embed_tokens.{key[len('embed.'):]}"
            if key.startswith("head."):
                return f"language_model.head.{key[len('head.'):]}"
            if key.startswith("norm."):
                return f"language_model.norm.{key[len('norm.'):]}"
            if key.startswith("layers."):
                return f"language_model.{key}"
            return key

        weights = {
            transform_key(k): v for k, v in weights.items() if not k.startswith("mtp.")
        }

        from .language import sanitize_moe_weights

        n_layers = self.config.num_hidden_layers
        n_routed = self.config.n_routed_experts
        for layer_idx in range(n_layers):
            weights = sanitize_moe_weights(
                weights, f"language_model.layers.{layer_idx}.ffn", n_routed
            )

        for layer_idx in range(n_layers):
            prefix = f"language_model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.config.o_groups, self.config.o_lora_rank, -1
                    )

        head_w = "language_model.head.weight"
        head_s = "language_model.head.scales"
        head_b = "language_model.head.biases"
        if head_s in weights and head_b in weights and head_w in weights:
            in_dim = weights[head_s].shape[-1] * 64
            bits = 32 * weights[head_w].shape[-1] // in_dim
            weights[head_w] = mx.dequantize(
                weights[head_w], weights[head_s], weights[head_b], 64, bits
            ).astype(mx.float32)
            del weights[head_s]
            del weights[head_b]

        return weights

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
