import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..muse_glimmer.muse_glimmer import VisionAdapter, masked_scatter
from .config import ModelConfig
from .language import LanguageModel, RMSNormNoScale, residual_gate_coefficients
from .vision import VisionModel

_NESTED_VISION_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
)

_RESIDUAL_GATES = {
    "post_attention_residual_gate.gate": "add_attn",
    "post_feedforward_residual_gate.gate": "add_ffn",
}

# mlx-lm checkpoints store these norms as the gain 1 + w.
_GAIN_NORMS = (
    "input_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "mlp.pre_expert_norm.weight",
    "mlp.experts.post_expert_norm.weight",
)

_LAYER_RE = re.compile(r"^(language_model\.model\.layers\.\d+\.)(.+)$")


def _deinterleave_rope_rows(w: mx.array, head_dim: int) -> mx.array:
    """Undo mlx-lm's RoPE row order: pairs (2i, 2i + 1) back to (i, i + d / 2)."""
    shape = w.shape
    w = w.reshape(-1, head_dim // 2, 2, *shape[1:])
    return mx.swapaxes(w, 1, 2).reshape(shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)
        self.vision_adapter = VisionAdapter(config)
        self.vision_projection = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.perception_emb_norm = RMSNormNoScale(config.text_config.rms_norm_eps)

    def _encode_image(self, pixel_values, image_grid_thw=None):
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required to encode Muse Spark images.")
        dtype = self.vision_tower.patch_embedder.patch_embedding.weight.dtype
        features = self.vision_tower(pixel_values.astype(dtype), image_grid_thw)
        features = self.vision_adapter(features)
        features = self.vision_projection(features)
        return self.perception_emb_norm(features)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        language_model = self.language_model.model
        inputs_embeds = language_model.embed_norm(
            language_model.embed_tokens(input_ids)
        )
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos")
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        grid_thw = kwargs.get("image_grid_thw")
        if grid_thw is None:
            grid_thw = kwargs.get("video_grid_thw")
        cached = kwargs.get("cached_image_features")
        image_features = (
            cached
            if cached is not None
            else self._encode_image(pixel_values, image_grid_thw=grid_thw)
        ).astype(inputs_embeds.dtype)

        token_mask = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
        )
        if int(token_mask.sum().item()) != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={int(token_mask.sum().item())}, "
                f"features={image_features.shape[0]}"
            )
        expanded_mask = mx.broadcast_to(token_mask[..., None], inputs_embeds.shape)
        inputs_embeds = masked_scatter(inputs_embeds, expanded_mask, image_features)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        embeddings = self.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, **kwargs
        )
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=embeddings.inputs_embeds,
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        """As mlx-lm: quantize only the routed experts, except in the first and
        the last layer."""
        experts_re = re.compile(
            r"^language_model\.model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj$"
        )
        full_precision = {0, self.config.text_config.num_hidden_layers - 1}

        def predicate(path, module):
            match = experts_re.match(path)
            return match is not None and int(match.group(1)) not in full_precision

        return predicate

    @property
    def cast_predicate(self):
        """Keep the router and the residual gates in FP32, as Transformers does."""

        def predicate(key):
            return not key.endswith(
                ("mlp.gate.weight", "e_score_correction_bias", "_alpha", "_beta")
            )

        return predicate

    def sanitize(self, weights):
        text_config = self.config.text_config
        # Checkpoints from mlx_lm.convert store the gate weights and the norm gains.
        from_mlx_lm = any(key.endswith("_alpha") for key in weights)

        renamed = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key or key.startswith("model.audio_tower."):
                continue
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif key.startswith(_NESTED_VISION_PREFIXES):
                key = key[len("model.") :]
            elif key.startswith(("model.", "lm_head.")):
                key = "language_model." + key
            renamed[key] = value

        sanitized = {}
        for key, value in renamed.items():
            match = _LAYER_RE.match(key)
            if match is None:
                sanitized[key] = value
                continue
            prefix, suffix = match.groups()
            if suffix in _RESIDUAL_GATES:
                alpha, beta = residual_gate_coefficients(
                    value, text_config.residual_gate_temperature
                )
                sanitized[prefix + _RESIDUAL_GATES[suffix] + "_alpha"] = alpha
                sanitized[prefix + _RESIDUAL_GATES[suffix] + "_beta"] = beta
            elif from_mlx_lm and suffix in _GAIN_NORMS:
                # In FP32: 1 + w must give the stored gain back exactly.
                sanitized[key] = value.astype(mx.float32) - 1
            elif suffix == "mlp.experts.gate_up_proj":
                gate_proj, up_proj = mx.split(value, 2, axis=1)
                sanitized[prefix + "mlp.experts.gate_proj.weight"] = gate_proj
                sanitized[prefix + "mlp.experts.up_proj.weight"] = up_proj
            elif suffix == "mlp.experts.down_proj":
                sanitized[prefix + "mlp.experts.down_proj.weight"] = value
            elif "_weight_scale" in suffix or suffix.endswith("_input_scale"):
                raise ValueError(
                    "NVFP4 Muse Spark checkpoints are not supported. Convert the "
                    "BF16 checkpoint with mlx_vlm.convert instead."
                )
            else:
                sanitized[key] = value

        # mlx-lm fuses q, k, v and the attention gate into one projection.
        head_dim = text_config.head_dim
        q_dim = text_config.num_attention_heads * head_dim
        kv_dim = text_config.num_key_value_heads * head_dim
        for key in [k for k in sanitized if k.endswith("self_attn.qkvg_proj.weight")]:
            attn = key[: -len("qkvg_proj.weight")]
            for param in ("weight", "scales", "biases"):
                fused = sanitized.pop(f"{attn}qkvg_proj.{param}", None)
                if fused is None:
                    continue
                q, k, v, g = mx.split(
                    fused, [q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], axis=0
                )
                sanitized[f"{attn}q_proj.{param}"] = _deinterleave_rope_rows(
                    q, head_dim
                )
                sanitized[f"{attn}k_proj.{param}"] = _deinterleave_rope_rows(
                    k, head_dim
                )
                sanitized[f"{attn}v_proj.{param}"] = v
                sanitized[f"{attn}gate_proj.{param}"] = g

        vision_head_dim = (
            self.config.vision_config.hidden_size
            // self.config.vision_config.num_attention_heads
        )
        for key in [k for k in sanitized if k.endswith(".attn.qkv_proj.weight")]:
            attn = key[: -len("qkv_proj.weight")]
            for param in ("weight", "bias"):
                q, k, v = mx.split(sanitized.pop(f"{attn}qkv_proj.{param}"), 3, axis=0)
                sanitized[f"{attn}q_proj.{param}"] = _deinterleave_rope_rows(
                    q, vision_head_dim
                )
                sanitized[f"{attn}k_proj.{param}"] = _deinterleave_rope_rows(
                    k, vision_head_dim
                )
                sanitized[f"{attn}v_proj.{param}"] = v
        return sanitized
