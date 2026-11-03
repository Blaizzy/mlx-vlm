from typing import Any, Optional

import mlx.core as mx

from ..nemotron_h.language import LanguageModel as NemotronHLanguageModel
from ..nemotron_h.language import Model as NemotronHForCausalLM


class LanguageModel(NemotronHLanguageModel):
    def get_input_embeddings(self, input_ids: mx.array) -> mx.array:
        return self.backbone.embeddings(input_ids)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        *,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        if output_hidden_states:
            if inputs_embeds is None:
                if inputs is None:
                    inputs = kwargs.get("input_ids")
                if inputs is None:
                    raise ValueError("Either inputs or inputs_embeds must be provided.")
                inputs_embeds = self.get_input_embeddings(inputs)
            # Omni returns each layer's input followed by the normalized output.
            kwargs.update(
                capture_layer_ids=list(range(len(self.layers) - 1)), return_hidden=True
            )
        output = super().__call__(
            inputs, inputs_embeds=inputs_embeds, cache=cache, **kwargs
        )
        if output_hidden_states:
            output.hidden_states.insert(0, inputs_embeds)
        return output

    def sanitize(self, weights):
        language_weights = {}
        passthrough = {}
        prefix = "language_model."
        for key, value in weights.items():
            if key.startswith(prefix):
                language_weights[key[len(prefix) :]] = value
            else:
                passthrough[key] = value

        if language_weights:
            language_weights = NemotronHForCausalLM.sanitize(self, language_weights)
            passthrough.update(
                {f"{prefix}{key}": value for key, value in language_weights.items()}
            )

        return passthrough

    @property
    def cast_predicate(self):
        def predicate(key):
            return "e_score_correction_bias" not in key and "A_log" not in key

        return predicate
