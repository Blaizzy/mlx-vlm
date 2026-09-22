from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if pixel_values is not None:
            raise NotImplementedError(
                "MiMo-V2.6 image input is not implemented yet; text-only for now"
            )
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        inputs_embeds = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        ).inputs_embeds
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=inputs_embeds, mask=mask, **kwargs
        )

    def sanitize(self, weights):
        # already-converted checkpoints carry the prefix, so re-prefixing them
        # on reload would produce language_model.language_model.*
        if any(k.startswith("language_model.") for k in weights):
            return weights

        vision, language = {}, {}
        for key, value in weights.items():
            # the audio tower and speech embeddings have no modules yet
            if key.startswith(("audio_encoder.", "speech_embeddings.")):
                continue
            if key.startswith("visual."):
                vision[key[len("visual.") :]] = value
            else:
                language[key] = value

        sanitized = {
            f"language_model.{k}": v
            for k, v in self.language_model.sanitize(language).items()
        }
        if self.vision_tower is not None:
            sanitized.update(
                {
                    f"vision_tower.{k}": v
                    for k, v in self.vision_tower.sanitize(vision).items()
                }
            )
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        """Quantize each half of the checkpoint on the grid it came from.

        The routed experts ship as MXFP4 with block 32, so re-using that grid
        is bit-exact; re-quantizing them affine costs ~12% per tensor. The
        dense projections come from block-FP8 instead, where 4 bits cost ~13%
        and 8-bit affine costs ~0.8%.
        """

        def predicate(path, module):
            if "switch_mlp" in path:
                return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            return {"group_size": 64, "bits": 8, "mode": "affine"}

        return predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self):
        return self.language_model.make_cache()
