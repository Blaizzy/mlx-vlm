from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from ....models.base import create_attention_mask
from ....models.cache import KVCache
from ....models.nemotron_h.language import Model as NemotronHForCausalLM
from ....models.nemotron_h.language import NemotronHBlock, NemotronHModel
from ..mtp_base import AutoregressiveMTPDraftModel
from .config import NemotronHMTPConfig


class NemotronHMTPDraftModel(AutoregressiveMTPDraftModel):
    """Native Nemotron MTP with shared target embeddings and output head.

    The checkpoint's single prediction module contains attention and MoE
    layers. Reusing that module recursively permits multiple draft tokens.
    """

    def __init__(self, config: NemotronHMTPConfig):
        super().__init__(config)
        text = config.text_config
        self.layers = [NemotronHBlock(text, t) for t in config.layer_pattern]
        first, last = self.layers[0], self.layers[-1]
        first.enorm = nn.RMSNorm(text.hidden_size, eps=text.layer_norm_epsilon)
        first.hnorm = nn.RMSNorm(text.hidden_size, eps=text.layer_norm_epsilon)
        first.eh_proj = nn.Linear(2 * text.hidden_size, text.hidden_size, bias=False)
        last.final_layernorm = nn.RMSNorm(text.hidden_size, eps=text.layer_norm_epsilon)

    def validate_target_compatibility(self, target_model):
        lm = getattr(target_model, "language_model", target_model)
        if not isinstance(getattr(lm, "backbone", None), NemotronHModel):
            raise ValueError("Nemotron MTP requires a Nemotron-H target.")
        for field in ("hidden_size", "vocab_size"):
            if getattr(lm.config, field) != getattr(self.config.text_config, field):
                raise ValueError(f"Nemotron MTP target {field} does not match.")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        lm = getattr(target_model, "language_model", target_model)
        self._input_embed = lm.backbone.embeddings
        self._lm_head_fn = lm.lm_head
        return self

    def make_cache(self):
        return [KVCache() for layer in self.layers if layer.block_type == "*"]

    def prefill_from_target_hidden(
        self,
        input_ids,
        hidden,
        bonus_token,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        if input_ids.shape[1] != hidden.shape[1]:
            raise ValueError(
                "Nemotron MTP requires one hidden state per prompt token. "
                "For video requests, set video_pruning_rate=0."
            )
        return super().prefill_from_target_hidden(
            input_ids, hidden, bonus_token, sampler, token_dtype, greedy
        )

    def _forward_hidden(self, token_embed, hidden, tokens, cache, **kwargs):
        del tokens, kwargs
        first = self.layers[0]
        hidden = first.eh_proj(
            mx.concatenate([first.enorm(token_embed), first.hnorm(hidden)], axis=-1)
        )
        mask = create_attention_mask(hidden, cache[0] if cache else None)
        cache_index = 0
        for layer in self.layers:
            layer_cache = None
            if layer.block_type == "*":
                layer_cache = cache[cache_index]
                cache_index += 1
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        hidden = self.layers[-1].final_layernorm(hidden)
        return hidden, hidden

    def sanitize(self, weights):
        # Reuse the target's expert stacking, with the MTP physical layer count.
        config = self.config
        args = replace(
            config.text_config,
            hybrid_override_pattern=list(config.layer_pattern),
        )
        weights = {
            "backbone." + key: value
            for key, value in weights.items()
            if not key.endswith((".k_scale", ".v_scale"))
        }
        weights = NemotronHForCausalLM.sanitize(SimpleNamespace(args=args), weights)
        return {key.removeprefix("backbone."): value for key, value in weights.items()}

    @property
    def cast_predicate(self):
        return lambda key: "e_score_correction_bias" not in key
