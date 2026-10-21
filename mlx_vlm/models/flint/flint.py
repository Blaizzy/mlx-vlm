import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        return {
            f"language_model.{k}": v
            for k, v in self.language_model.sanitize(weights).items()
        }

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
