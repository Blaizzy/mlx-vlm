from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..qwen3.language import LanguageModel
from .config import ModelConfig
from .vision import VisionModel

NUM_INDICATORS = 4  # last rows of the visual vocabulary: image/video start/end


class VisionHead(nn.Module):
    """Visual tokenizer head: merged 2x2 patch features -> visual-vocab logits."""

    def __init__(self, in_dim: int, vocab: int):
        super().__init__()
        self.layers = [nn.Linear(in_dim, vocab, bias=False), nn.LayerNorm(vocab)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[1](self.layers[0](x))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        vc = config.vision_config
        self.vision_tower = VisionModel(vc)
        self.vision_head = VisionHead(
            vc.hidden_size * vc.hidden_stride**2,
            config.visual_vocab_size - NUM_INDICATORS,
        )
        # "vte": embedding table of the visual vocabulary (soft visual tokens).
        self.vision_embed = nn.Embedding(
            config.visual_vocab_size, config.text_config.hidden_size
        )
        self.language_model = LanguageModel(config.text_config)

    def encode_images(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        """Soft visual tokens: softmax over the visual vocabulary @ embedding table."""
        stride = self.config.vision_config.hidden_stride
        features = self.vision_tower(pixel_values, grid_thw)
        features = features.reshape(features.shape[0] // stride**2, -1)
        logits = self.vision_head(features)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(logits.dtype)
        # The four indicator columns are zero-padded in the reference: skip them.
        table = self.vision_embed.weight[
            : self.config.visual_vocab_size - NUM_INDICATORS
        ]
        return probs @ table

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        embed_tokens = self.language_model.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        grid_thw = kwargs.get("image_grid_thw")
        if grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is given")

        dtype = inputs_embeds.dtype
        visual = self.encode_images(pixel_values, grid_thw).astype(dtype)
        indicators = self.vision_embed.weight[
            self.config.visual_vocab_size - NUM_INDICATORS :
        ].astype(dtype)

        ids = np.array(input_ids)
        image_mask = ids == self.config.image_token_id
        if int(image_mask.sum()) != visual.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: {visual.shape[0]} "
                f"features and {int(image_mask.sum())} tokens."
            )

        # Image block = <|vision_start|> + N x <|image_pad|> + <|vision_end|>;
        # start/end take the image-start/end indicator embeddings (vte rows -4/-3).
        flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_ids = ids.reshape(-1)
        pad_pos = np.flatnonzero(flat_ids == self.config.image_token_id)
        start_pos = np.flatnonzero(flat_ids == self.config.vision_start_token_id)
        end_pos = np.flatnonzero(flat_ids == self.config.vision_end_token_id)
        flat[mx.array(pad_pos)] = visual
        if start_pos.size:
            flat[mx.array(start_pos)] = mx.broadcast_to(
                indicators[0], (start_pos.size, flat.shape[-1])
            )
        if end_pos.size:
            flat[mx.array(end_pos)] = mx.broadcast_to(
                indicators[1], (end_pos.size, flat.shape[-1])
            )
        return InputEmbeddingsFeatures(inputs_embeds=flat.reshape(inputs_embeds.shape))

    @property
    def layers(self):
        return self.language_model.model.layers

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
            input_ids, inputs_embeds=features.inputs_embeds, cache=cache
        )

    @property
    def quant_predicate(self):
        # Keep the visual tokenizer (ViT, head, visual embedding table) in full
        # precision: the soft-token product needs the dense vte matrix.
        def predicate(path, module):
            return not path.startswith(("vision_tower", "vision_head", "vision_embed"))

        return predicate

    def sanitize(self, weights):
        """Map the OXR checkpoint layout to this module tree (idempotent)."""
        tied = self.config.text_config.tie_word_embeddings
        out = {}
        for key, value in weights.items():
            if key.startswith("visual_tokenizer.vit.vision_model."):
                key = key[len("visual_tokenizer.vit.vision_model.") :]
                if key.startswith("post_layernorm."):
                    continue  # unused: OXR reads the last encoder layer
                key = "vision_tower." + key.replace("encoder.layers.", "layers.", 1)
            elif key.startswith("visual_tokenizer.head."):
                key = "vision_head.layers." + key[len("visual_tokenizer.head.") :]
            elif key.startswith("vte."):
                key = "vision_embed." + key[len("vte.") :]
            elif key.startswith("llm.lm_head."):
                if tied:
                    continue
                key = "language_model.lm_head." + key[len("llm.lm_head.") :]
            elif key.startswith("llm.model."):
                key = "language_model.model." + key[len("llm.model.") :]

            # Conv2d(kernel=stride=patch) weight -> Linear over flattened patches.
            if (
                key == "vision_tower.embeddings.patch_embedding.weight"
                and value.ndim == 4
            ):
                value = value.reshape(value.shape[0], -1)
            out[key] = value
        return out
