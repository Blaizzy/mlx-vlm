from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import Model as NemotronHForCausalLM
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHModel

from ..base import LanguageModelOutput


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches

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
        if inputs_embeds is None:
            if inputs is None:
                raise ValueError("Either inputs or inputs_embeds must be provided.")
            hidden_states = self.backbone.embeddings(inputs)
        else:
            hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        attn_cache = (
            cache[self.backbone.fa_idx] if self.backbone.fa_idx < len(cache) else None
        )
        ssm_cache = (
            cache[self.backbone.ssm_idx] if self.backbone.ssm_idx < len(cache) else None
        )
        attn_mask = create_attention_mask(hidden_states, attn_cache)
        ssm_mask = create_ssm_mask(hidden_states, ssm_cache)

        all_hidden_states = [] if output_hidden_states else None
        cache_counter = 0
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if layer.block_type in ("M", "*"):
                layer_cache = cache[cache_counter]
                cache_counter += 1
            else:
                layer_cache = None

            mask = attn_mask if layer.block_type == "*" else ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

        hidden_states = self.backbone.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return LanguageModelOutput(
            logits=self.lm_head(hidden_states),
            hidden_states=all_hidden_states,
        )

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
