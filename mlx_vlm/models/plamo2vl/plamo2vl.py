from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel, RMSNorm
from .vision import VisionModel, check_array_shape


class Bias(nn.Module):
    """Upstream stores the parameter as `_bias`; MLX treats leading-underscore
    attributes as private, so it is renamed to `bias` in sanitize."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bias = mx.zeros((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.bias


class MLPImageProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        vision_config = config.vision_config
        text_config = config.text_config
        hidden_size = vision_config.image_proj_hidden_size

        self.norm0 = RMSNorm(
            vision_config.image_feature_size, eps=text_config.rms_norm_eps
        )
        self.bias0 = Bias(vision_config.image_feature_size)

        self.linear1 = nn.Linear(
            vision_config.image_feature_size, hidden_size, bias=False
        )
        self.bias1 = Bias(hidden_size)
        # torch nn.GELU() default is the exact (erf) form
        self.act1 = nn.GELU()

        self.linear2 = nn.Linear(hidden_size, text_config.hidden_size, bias=False)
        self.bias2 = Bias(text_config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm0(x)
        x = self.bias0(x)

        x = self.linear1(x)
        x = self.bias1(x)
        x = self.act1(x)

        x = self.linear2(x)
        x = self.bias2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.image_proj = MLPImageProjector(config)

        if config.image_token_index is None:
            config.image_token_index = config.vision_config.image_token_id

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = (
            self.vision_model.vision_encoder.model.embeddings.patch_embedding.weight.dtype
        )
        pixel_values = pixel_values.astype(dtype)

        if pixel_values.ndim == 5:
            # (1, total_num_tiles, 3, H, W) -> (total_num_tiles, 3, H, W)
            pixel_values = pixel_values[0]

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            # SigLIP: NCHW -> NHWC. No CLS token, nothing stripped from the front
            hidden_states, _, _ = self.vision_model(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=False
            )
            # (num_tiles, 729, 1152) -> (num_tiles * 729, 1152)
            image_features = self.image_proj(
                hidden_states.reshape(-1, hidden_states.shape[-1])
            )

        image_token_index = kwargs.get(
            "image_token_index", self.config.image_token_index
        )

        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, image_token_index
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, image_token_index=None
    ):
        B, N, C = inputs_embeds.shape
        if image_token_index is None:
            image_token_index = self.config.image_token_index

        image_positions = input_ids == image_token_index
        n_img = int(mx.sum(image_positions))
        image_features = image_features.reshape(-1, image_features.shape[-1])
        if n_img != image_features.shape[0]:
            raise ValueError(
                f"image token count ({n_img}) != vision feature count "
                f"({image_features.shape[0]})"
            )

        image_indices = np.where(image_positions)[1].tolist()
        inputs_embeds[:, image_indices, :] = image_features
        return inputs_embeds.reshape(B, N, C)

    @property
    def layers(self):
        return self.language_model.model.layers.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            # SigLIP attention pooling head is unused when only
            # last_hidden_state is consumed
            if ".vision_encoder.model.head." in k:
                continue
            # Checkpoint stores the language model at the top level; nest it
            # under language_model to follow the mlx-vlm layout
            if k.startswith("model."):
                k = "language_model." + k
            elif k.startswith("lm_head."):
                if self.config.text_config.tie_word_embeddings:
                    continue
                k = "language_model." + k
            # MLX cannot register leading-underscore parameters
            if k.endswith("._bias"):
                k = k[: -len("._bias")] + ".bias"
            if "patch_embedding.weight" in k:
                sanitized[k] = v if check_array_shape(v) else v.transpose(0, 2, 3, 1)
            else:
                sanitized[k] = v
        return sanitized

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            None, cache=cache, inputs_embeds=input_embeddings_features.inputs_embeds
        )
        return logits
