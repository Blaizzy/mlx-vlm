from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
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
        return self.language_model(input_ids, cache=cache, **kwargs)

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
