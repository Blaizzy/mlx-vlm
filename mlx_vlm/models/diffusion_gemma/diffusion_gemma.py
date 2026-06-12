import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import DiffusionGemma4Backbone, make_compiled_softcap
from .visualizer import make_unmasking_visualizer


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
            self_conditioning_embeddings=kwargs.get("self_conditioning_embeddings"),
            decoder_attention_mask=kwargs.get("decoder_attention_mask"),
        )
        logits = self._parent.model.decoder.embed_tokens.as_linear(hidden_states)
        logits = self._parent._softcap(logits)
        return LanguageModelOutput(logits=logits, hidden_states=[hidden_states])

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if (
                path.endswith(
                    ("embed_tokens", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
                )
                or ".self_attn." in path
                or "router" in path
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
        self._softcap = make_compiled_softcap(float(self.final_logit_softcapping))

    def __call__(
        self,
        input_ids: mx.array = None,
        attention_mask: mx.array = None,
        cache=None,
        past_key_values=None,
        canvas_ids: mx.array = None,
        self_conditioning_logits: mx.array = None,
        self_conditioning_embeddings: mx.array = None,
        decoder_attention_mask: mx.array = None,
        pixel_values: mx.array = None,
        mm_token_type_ids: mx.array = None,
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
            self_conditioning_embeddings=self_conditioning_embeddings,
            decoder_attention_mask=decoder_attention_mask,
            pixel_values=pixel_values,
            mm_token_type_ids=mm_token_type_ids,
        )
        logits = self.model.decoder.embed_tokens.as_linear(hidden_states)
        logits = self._softcap(logits)
        return LanguageModelOutput(logits=logits, hidden_states=[hidden_states])

    @property
    def layers(self):
        return self.model.decoder.layers

    def make_cache(self, max_size=None):
        return self.model.encoder.make_cache(max_size=max_size)

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache, draft_model, draft_kind
        return self.model.encoder.chunked_prefill_policy(prefill_kwargs=prefill_kwargs)

    @property
    def prefers_logits_self_conditioning(self) -> bool:
        return self.model.decoder.prefers_logits_self_conditioning

    def diffusion_prepare_self_conditioning(self):
        return self.model.decoder.diffusion_prepare_self_conditioning()

    def diffusion_self_conditioning(
        self,
        processed_logits: mx.array,
        embedding_weight: mx.array = None,
        *,
        token_probs: mx.array = None,
    ):
        return self.model.decoder.diffusion_self_conditioning(
            processed_logits,
            embedding_weight,
            token_probs=token_probs,
        )

    def diffusion_prefill_cache(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array = None,
        cache=None,
        pixel_values: mx.array = None,
        mm_token_type_ids: mx.array = None,
        prefill_step_size: int = None,
        chunk_prefill: bool = False,
    ):
        return self.model.diffusion_prefill_cache(
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
            pixel_values=pixel_values,
            mm_token_type_ids=mm_token_type_ids,
            prefill_step_size=prefill_step_size,
            chunk_prefill=chunk_prefill,
        )

    def diffusion_update_cache(self, input_ids: mx.array, *, cache):
        return self.model.diffusion_update_cache(input_ids, cache=cache)

    def diffusion_decoder_masks(
        self,
        canvas_ids: mx.array,
        cache,
        decoder_attention_mask: mx.array = None,
    ):
        return self.model.diffusion_decoder_masks(
            canvas_ids,
            cache,
            decoder_attention_mask,
        )

    def diffusion_decoder_logits(
        self,
        canvas_ids: mx.array,
        cache=None,
        self_conditioning: mx.array = None,
        decoder_attention_mask: mx.array = None,
    ):
        kwargs = (
            {"self_conditioning_logits": self_conditioning}
            if self.prefers_logits_self_conditioning
            else {"self_conditioning_embeddings": self_conditioning}
        )
        hidden_states = self.model.decoder(
            canvas_ids,
            cache=cache,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )
        logits = self.model.decoder.embed_tokens.as_linear(hidden_states)
        return self._softcap(logits)

    # Model-owned live unmasking view, like the nemotron/llada visualizers.
    make_unmasking_visualizer = staticmethod(make_unmasking_visualizer)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids are required for DiffusionGemma4 embeddings.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.model.encoder._embed_inputs(
                input_ids,
                pixel_values=pixel_values,
            )
        )

    def sanitize(self, weights):
        has_vision_tower = self.model.encoder.vision_tower is not None
        use_clipped = (
            getattr(self.config.vision_config, "use_clipped_linears", False)
            if self.config.vision_config is not None
            else False
        )
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb" in key or key == "lm_head.weight":
                continue
            if key.startswith("model.encoder.embed_vision.") or key.startswith(
                "model.encoder.vision_tower."
            ):
                if not has_vision_tower:
                    continue
                # Clipping calibration tensors are only used by clipped linears.
                if not use_clipped and any(
                    s in key
                    for s in ("input_max", "input_min", "output_max", "output_min")
                ):
                    continue
                sanitized[key] = value
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
                        ".experts.down_proj.weight",
                    )
                ] = value
                continue

            if key.endswith(".experts.gate_up_proj"):
                sanitized[
                    key.replace(
                        ".experts.gate_up_proj",
                        ".experts.gate_up_proj.weight",
                    )
                ] = value
                continue

            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
