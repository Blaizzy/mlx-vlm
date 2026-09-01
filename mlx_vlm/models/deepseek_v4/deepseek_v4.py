import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import Aligner, ViT

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        if config.vision_n_layers > 0:
            self.vision = ViT(config)
            self.aligner = Aligner(config)
            self.image_start = mx.zeros((config.hidden_size,))
            self.image_end = mx.zeros((config.hidden_size,))
            self.image_newline = mx.zeros((config.hidden_size,))
            self.image_pad = mx.zeros((config.hidden_size,))

    def encode_image(self, patches: mx.array, n_vit_h: int, n_vit_w: int) -> mx.array:
        if self.config.vision_n_layers == 0:
            raise ValueError("This DeepSeek-V4 checkpoint has no vision tower")
        dtype = self.vision.patch_embed.proj.weight.dtype
        features = self.vision(patches.astype(dtype), n_vit_h, n_vit_w)
        return self.aligner(features, n_vit_h, n_vit_w)

    def _encode_images(
        self,
        pixel_values: mx.array,
        image_grid_hw: mx.array,
        image_permutations: mx.array,
    ) -> list[mx.array]:
        patch_offset = 0
        perm_offset = 0
        image_features = []
        ratio = self.config.vision_downsample_ratio
        for n_vit_h, n_vit_w in image_grid_hw.tolist():
            n_patches = int(n_vit_h * n_vit_w)
            n_features = math.ceil(n_vit_h / ratio) * math.ceil(n_vit_w / ratio)
            patches = pixel_values[patch_offset : patch_offset + n_patches]
            perm = image_permutations[perm_offset : perm_offset + n_features]
            encoded = self.encode_image(patches, n_vit_h, n_vit_w)
            image_features.append(encoded[perm])
            patch_offset += n_patches
            perm_offset += n_features

        if patch_offset != pixel_values.shape[0]:
            raise ValueError(
                "DeepSeek-V4 image grids do not account for all vision patches"
            )
        if perm_offset != image_permutations.shape[0]:
            raise ValueError(
                "DeepSeek-V4 image grids do not account for all permutations"
            )
        return image_features

    def encode_images(self, pixel_values: mx.array, **kwargs) -> list[mx.array]:
        """Encode packed images for the shared vision-feature cache."""
        image_grid_hw = kwargs.get("image_grid_hw")
        image_permutations = kwargs.get("image_permutations")
        if image_grid_hw is None or image_permutations is None:
            raise ValueError(
                "DeepSeek-V4 packed image caching requires image grids and "
                "permutations"
            )
        return self._encode_images(pixel_values, image_grid_hw, image_permutations)

    def _merge_image_embeddings(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        image_features: list[mx.array],
        image_sample_indices: mx.array,
        image_offsets: mx.array,
        image_types: mx.array,
        image_type_offsets: mx.array,
    ) -> mx.array:
        params = mx.stack(
            [
                self.image_start,
                self.image_pad,
                self.image_pad,
                self.image_newline,
                self.image_end,
            ]
        ).astype(inputs_embeds.dtype)

        for image_idx, features in enumerate(image_features):
            type_start = int(image_type_offsets[image_idx].item())
            type_end = int(image_type_offsets[image_idx + 1].item())
            types = image_types[type_start:type_end]
            block = params[types]
            image_mask = types == IMAGE
            if int(image_mask.sum().item()) != features.shape[0]:
                raise ValueError(
                    "DeepSeek-V4 image layout does not match aligned features"
                )
            block[image_mask] = features.astype(inputs_embeds.dtype)

            sample_idx = int(image_sample_indices[image_idx].item())
            prompt_offset = int(image_offsets[image_idx].item())
            prompt_end = prompt_offset + block.shape[0]
            expected_ids = self.config.vocab_size + types
            if not mx.array_equal(
                input_ids[sample_idx, prompt_offset:prompt_end], expected_ids
            ).item():
                raise ValueError(
                    "DeepSeek-V4 image metadata does not match prompt sentinels"
                )
            inputs_embeds[sample_idx, prompt_offset:prompt_end] = block
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        safe_ids = mx.where(
            (input_ids >= 0) & (input_ids < self.config.vocab_size), input_ids, 0
        )
        inputs_embeds = self.language_model.model.embed_tokens(safe_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)
        if self.config.vision_n_layers == 0:
            raise ValueError("Pixel values require a DeepSeek-V4 vision checkpoint")

        metadata_names = (
            "image_grid_hw",
            "image_sample_indices",
            "image_offsets",
            "image_types",
            "image_type_offsets",
            "image_permutations",
        )
        missing = [name for name in metadata_names if kwargs.get(name) is None]
        if missing:
            raise ValueError(
                "Missing DeepSeek-V4 image metadata: " + ", ".join(missing)
            )

        image_features = kwargs.get("cached_image_features")
        if image_features is None:
            image_features = self._encode_images(
                pixel_values,
                kwargs["image_grid_hw"],
                kwargs["image_permutations"],
            )
        inputs_embeds = self._merge_image_embeddings(
            input_ids,
            inputs_embeds,
            image_features,
            kwargs["image_sample_indices"],
            kwargs["image_offsets"],
            kwargs["image_types"],
            kwargs["image_type_offsets"],
        )
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            ).inputs_embeds
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=inputs_embeds, **kwargs
        )

    def sanitize(self, weights):
        weights = self.language_model.sanitize(weights)

        def transform_key(key):
            if key.startswith("language_model."):
                return key
            if key.startswith("model.") or key.startswith("lm_head."):
                return f"language_model.{key}"
            return key

        return {transform_key(k): v for k, v in weights.items()}

    @staticmethod
    def quantization_path_aliases(path: str) -> Tuple[str, ...]:
        """Return legacy checkpoint names for a loaded DeepSeek-V4 module path."""
        path = path.removeprefix("language_model.").removeprefix("model.")
        aliases = [path]

        if path == "embed_tokens":
            aliases.append("embed")
        elif path == "lm_head":
            aliases.append("head")

        for module_name, checkpoint_name in (
            ("gate_proj", "w1"),
            ("down_proj", "w2"),
            ("up_proj", "w3"),
        ):
            module_path = f".ffn.shared_experts.{module_name}"
            offset = path.find(module_path)
            if offset < 0:
                continue
            end = offset + len(module_path)
            if end == len(path) or path[end] == ".":
                aliases.append(
                    path[:offset]
                    + f".ffn.shared_experts.{checkpoint_name}"
                    + path[end:]
                )

        return tuple(dict.fromkeys(aliases))

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
