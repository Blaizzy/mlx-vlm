import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from ..gemma4.language import logit_softcap
from .config import ModelConfig
from .language import DiffusionGemma4Backbone


class _LanguageModelView:
    """Non-owning compatibility view used by mlx-vlm helpers."""

    def __init__(self, parent: "Model"):
        self._parent = parent
        self.model_type = parent.config.text_config.model_type

    @property
    def model(self):
        return self._parent.model.decoder

    @property
    def layers(self):
        return self._parent.layers

    def make_cache(self, max_size=None):
        return self._parent.make_cache(max_size=max_size)

    def __call__(
        self,
        inputs: mx.array = None,
        inputs_embeds: mx.array = None,
        cache=None,
        **kwargs,
    ):
        hidden_states, _ = self._parent.model(
            input_ids=inputs,
            cache=cache,
            canvas_ids=kwargs.get("canvas_ids"),
            self_conditioning_logits=kwargs.get("self_conditioning_logits"),
            decoder_attention_mask=kwargs.get("decoder_attention_mask"),
        )
        logits = self._parent.model.decoder.embed_tokens.as_linear(hidden_states)
        logits = logit_softcap(self._parent.final_logit_softcapping, logits)
        return LanguageModelOutput(logits=logits, hidden_states=[hidden_states])

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if "router" in path or path.endswith(
                ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = DiffusionGemma4Backbone(config)
        self.language_model = _LanguageModelView(self)
        self.final_logit_softcapping = config.text_config.final_logit_softcapping

    def __call__(
        self,
        input_ids: mx.array = None,
        attention_mask: mx.array = None,
        cache=None,
        past_key_values=None,
        canvas_ids: mx.array = None,
        self_conditioning_logits: mx.array = None,
        decoder_attention_mask: mx.array = None,
        **kwargs,
    ):
        if cache is None:
            cache = past_key_values

        hidden_states, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            canvas_ids=canvas_ids,
            self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = self.model.decoder.embed_tokens.as_linear(hidden_states)
        logits = logit_softcap(self.final_logit_softcapping, logits)
        return LanguageModelOutput(logits=logits, hidden_states=[hidden_states])

    @property
    def layers(self):
        return self.model.decoder.layers

    def make_cache(self, max_size=None):
        return self.model.encoder.make_cache(max_size=max_size)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if pixel_values is not None:
            raise ValueError("DiffusionGemma4 vision inputs are not supported yet.")
        if input_ids is None:
            raise ValueError("input_ids are required for DiffusionGemma4 embeddings.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.model.encoder._embed_inputs(input_ids)
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb" in key or key == "lm_head.weight":
                continue

            # Encoder text weights are tied to decoder weights; the checkpoint
            # only carries separate encoder layer scalars.
            if key.startswith("model.encoder.language_model."):
                if key.endswith(".layer_scalar"):
                    sanitized[key] = value
                continue

            if key.endswith(".experts.down_proj"):
                sanitized[
                    key.replace(
                        ".experts.down_proj",
                        ".experts.switch_glu.down_proj.weight",
                    )
                ] = value
                continue

            if key.endswith(".experts.gate_up_proj"):
                gate, up = mx.split(value, 2, axis=-2)
                sanitized[
                    key.replace(
                        ".experts.gate_up_proj",
                        ".experts.switch_glu.gate_proj.weight",
                    )
                ] = gate
                sanitized[
                    key.replace(
                        ".experts.gate_up_proj",
                        ".experts.switch_glu.up_proj.weight",
                    )
                ] = up
                continue

            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
