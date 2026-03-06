import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures, LanguageModelOutput, create_attention_mask
from ..cache import KVCache
from .audio import AudioProjection, ConformerEncoder
from .config import AudioConfig, ModelConfig, VisionConfig
from .vision import VisionTower


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5

        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_dim = int(head_dim * config.partial_rotary_factor)
        self.rope = nn.RoPE(
            rope_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)
        query_pos = self.n_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_pos, query_pos + self.n_kv_heads * self.head_dim], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


def build_mm_projector(config: ModelConfig):
    """Build vision-to-language projector with bias (matching checkpoint)."""
    mm_hidden_size = config.mm_hidden_size
    hidden_size = config.hidden_size

    # MLP projector: Linear(mm_hidden, hidden) -> GELU -> Linear(hidden, hidden)
    # Checkpoint has img_projection.0 and img_projection.2 (with bias)
    return [
        nn.Linear(mm_hidden_size, hidden_size, bias=True),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size, bias=True),
    ]


class Phi4MM(nn.Module):
    """Phi4 backbone with vision tower, audio encoder, and projectors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vision_tower = VisionTower(config.vision_config)
        self.mm_projector = build_mm_projector(config)

        # Audio encoder + projection
        audio_config = getattr(config, "_audio_config", AudioConfig())
        self.audio_encoder = ConformerEncoder(audio_config)
        self.audio_projection = AudioProjection(
            audio_dim=audio_config.attention_dim,
            hidden_size=config.hidden_size,
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.model = Phi4MM(config)
        if not getattr(config, "tie_word_embeddings", True):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        pixel_values=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            input_embeddings_features = self.get_input_embeddings(
                inputs, pixel_values, **kwargs
            )
            inputs_embeds = input_embeddings_features.inputs_embeds

        out = self.model(inputs, inputs_embeds, mask=mask, cache=cache)

        if hasattr(self, "lm_head"):
            logits = self.lm_head(out)
        else:
            logits = self.model.embed_tokens.as_linear(out)

        return LanguageModelOutput(logits=logits)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        input_audio_embeds = kwargs.get("input_audio_embeds", None)
        audio_embed_sizes = kwargs.get("audio_embed_sizes", None)
        audio_attention_mask = kwargs.get("audio_attention_mask", None)

        has_images = pixel_values is not None
        has_audio = input_audio_embeds is not None and input_audio_embeds.size > 0

        # Auto-switch LoRA when multimodal inputs are provided
        if has_images or has_audio:
            self.set_modality(has_image=has_images, has_audio=has_audio)

        if not has_images and not has_audio:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.model.embed_tokens(input_ids)
            )

        # --- Process images ---
        image_features = None
        if has_images:
            pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
            spatial_shapes = kwargs.get("spatial_shapes", None)

            image_features = self.model.vision_tower(
                pixel_values, pixel_attention_mask, spatial_shapes
            )
            if isinstance(image_features, list):
                projected = []
                for feat in image_features:
                    x = feat
                    for layer in self.model.mm_projector:
                        x = layer(x)
                    projected.append(x)
                image_features = projected
            else:
                x = image_features
                for layer in self.model.mm_projector:
                    x = layer(x)
                image_features = x

        # --- Process audio ---
        audio_features = None
        if has_audio:
            # Run through conformer encoder
            encoded_audio, _ = self.model.audio_encoder(
                input_audio_embeds, audio_attention_mask
            )
            # Project to LM hidden size (speech mode)
            audio_features = self.model.audio_projection(encoded_audio, mode="speech")

        # --- Build safe input_ids for embedding lookup ---
        image_token_index = self.config.image_token_index
        audio_token_index = self.config.audio_token_index

        safe_input_ids = input_ids
        if has_images:
            safe_input_ids = mx.where(
                safe_input_ids == image_token_index, mx.array(0), safe_input_ids
            )
        if has_audio:
            safe_input_ids = mx.where(
                safe_input_ids == audio_token_index, mx.array(0), safe_input_ids
            )
        text_embeds = self.model.embed_tokens(safe_input_ids)

        # --- Merge multimodal features with text embeddings ---
        B = input_ids.shape[0]
        new_embeds = []
        img_idx = 0
        audio_offset = 0  # Tracks position in audio_features

        for b in range(B):
            cur_input_ids = input_ids[b].tolist()
            cur_text_embeds = text_embeds[b]

            # Check if this batch item has any special tokens
            has_special = False
            if has_images and any(x == image_token_index for x in cur_input_ids):
                has_special = True
            if has_audio and any(x == audio_token_index for x in cur_input_ids):
                has_special = True

            if not has_special:
                new_embeds.append(cur_text_embeds)
                continue

            # Build parts by splitting at special tokens
            parts = []
            prev = 0
            i = 0
            while i < len(cur_input_ids):
                token_id = cur_input_ids[i]

                if token_id == image_token_index and has_images:
                    # Text before this image token
                    if i > prev:
                        parts.append(cur_text_embeds[prev:i])
                    # Insert image features
                    if isinstance(image_features, list):
                        feat = image_features[img_idx].astype(cur_text_embeds.dtype)
                    else:
                        feat = image_features[img_idx].astype(cur_text_embeds.dtype)
                    parts.append(feat)
                    img_idx += 1
                    prev = i + 1

                elif token_id == audio_token_index and has_audio:
                    # Text before this audio token group
                    if i > prev:
                        parts.append(cur_text_embeds[prev:i])

                    # Count consecutive audio tokens
                    j = i
                    while j < len(cur_input_ids) and cur_input_ids[j] == audio_token_index:
                        j += 1
                    num_audio_tokens = j - i

                    # Get the audio features for this group
                    if audio_embed_sizes is not None:
                        # Use embed_sizes to slice from encoded audio
                        embed_size = int(audio_embed_sizes[audio_offset].item())
                        audio_feat = audio_features[audio_offset, :embed_size].astype(
                            cur_text_embeds.dtype
                        )
                        audio_offset += 1
                    else:
                        audio_feat = audio_features[0, :num_audio_tokens].astype(
                            cur_text_embeds.dtype
                        )

                    parts.append(audio_feat)
                    prev = j
                    i = j
                    continue

                i += 1

            # Remaining text
            if prev < len(cur_input_ids):
                parts.append(cur_text_embeds[prev:])

            if parts:
                merged = mx.concatenate(parts, axis=0)
            else:
                merged = cur_text_embeds
            new_embeds.append(merged)

        # Stack batch
        if B == 1:
            inputs_embeds = mx.expand_dims(new_embeds[0], axis=0)
        else:
            max_len = max(e.shape[0] for e in new_embeds)
            padded = []
            for e in new_embeds:
                if e.shape[0] < max_len:
                    pad = mx.zeros((max_len - e.shape[0], e.shape[-1]))
                    padded.append(mx.concatenate([e, pad], axis=0))
                else:
                    padded.append(e)
            inputs_embeds = mx.stack(padded)

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def language_model(self):
        return self

    @property
    def vision_model(self):
        return self.model.vision_tower

    def sanitize(self, weights):
        # Get LoRA configs
        vision_lora = getattr(self.config, "vision_lora", None)
        speech_lora = getattr(self.config, "speech_lora", None)

        vision_lora_scale = 1.0
        if vision_lora:
            r = vision_lora.get("r", 256)
            alpha = vision_lora.get("lora_alpha", 512)
            vision_lora_scale = alpha / r

        speech_lora_scale = 1.0
        if speech_lora:
            r = speech_lora.get("r", 320)
            alpha = speech_lora.get("lora_alpha", 640)
            speech_lora_scale = alpha / r

        # Collect LoRA weights for merging
        lora_a_vision = {}
        lora_b_vision = {}
        lora_a_speech = {}
        lora_b_speech = {}
        base_weights = {}
        audio_weights = {}
        sanitized = {}

        for k, v in weights.items():
            # Skip position_ids
            if "position_ids" in k:
                continue
            # Skip HD transform components (glb_GN, sub_GN, head)
            if "glb_GN" in k or "sub_GN" in k or "img_processor.head." in k:
                continue

            # Remap audio encoder weights
            if "embed_tokens_extend.audio_embed.encoder." in k:
                new_key = k.replace(
                    "model.embed_tokens_extend.audio_embed.encoder.",
                    "model.audio_encoder.",
                )
                audio_weights[new_key] = v
                continue

            # Remap audio projection weights
            if "embed_tokens_extend.audio_embed.audio_projection." in k:
                new_key = k.replace(
                    "model.embed_tokens_extend.audio_embed.audio_projection.",
                    "model.audio_projection.",
                )
                # Map sequential indices: speech.0 -> speech.proj_0, speech.2 -> speech.proj_2
                new_key = re.sub(r"(speech|vision)\.0\.", r"\1.proj_0.", new_key)
                new_key = re.sub(r"(speech|vision)\.2\.", r"\1.proj_2.", new_key)
                sanitized[new_key] = v
                continue

            # Remap vision tower keys
            if "embed_tokens_extend.image_embed.img_processor." in k:
                new_key = k.replace(
                    "model.embed_tokens_extend.image_embed.img_processor.",
                    "model.vision_tower.vision_tower.",
                )
                sanitized[new_key] = v
                continue

            # Remap projector keys
            if "embed_tokens_extend.image_embed.img_projection." in k:
                new_key = k.replace(
                    "model.embed_tokens_extend.image_embed.img_projection.",
                    "model.mm_projector.",
                )
                sanitized[new_key] = v
                continue

            # Handle LoRA weights
            if ".lora_A.vision." in k:
                base_key = k.replace(".lora_A.vision.", ".")
                lora_a_vision[base_key] = v
                continue
            if ".lora_B.vision." in k:
                base_key = k.replace(".lora_B.vision.", ".")
                lora_b_vision[base_key] = v
                continue
            if ".lora_A.speech." in k:
                base_key = k.replace(".lora_A.speech.", ".")
                lora_a_speech[base_key] = v
                continue
            if ".lora_B.speech." in k:
                base_key = k.replace(".lora_B.speech.", ".")
                lora_b_speech[base_key] = v
                continue

            # Handle base_layer weights
            if ".base_layer." in k:
                new_key = k.replace(".base_layer.", ".")
                base_weights[new_key] = v
                continue

            sanitized[k] = v

        # Merge vision LoRA into base weights (default for inference)
        for key, base_w in base_weights.items():
            if key in lora_a_vision and key in lora_b_vision:
                lora_a = lora_a_vision[key]
                lora_b = lora_b_vision[key]
                merged = base_w + vision_lora_scale * (lora_b @ lora_a)
                sanitized[key] = merged
            else:
                sanitized[key] = base_w

        # Store speech LoRA for runtime switching
        # For now, we don't merge speech LoRA by default (vision is default mode)
        # Speech LoRA can be applied on top when audio mode is needed
        self._speech_lora_a = lora_a_speech
        self._speech_lora_b = lora_b_speech
        self._speech_lora_scale = speech_lora_scale
        self._vision_lora_a = lora_a_vision
        self._vision_lora_b = lora_b_vision
        self._vision_lora_scale = vision_lora_scale
        self._base_weights = {k: v for k, v in base_weights.items()}
        self._active_lora = "vision"  # Vision LoRA merged by default

        # Sanitize audio encoder weights through audio encoder's sanitize
        if audio_weights:
            audio_sanitized = self.model.audio_encoder.sanitize(audio_weights)
            sanitized.update(audio_sanitized)

        return sanitized

    def apply_speech_lora(self):
        """Switch LLM weights from vision LoRA to speech LoRA for audio inference."""
        if not hasattr(self, "_speech_lora_a") or not self._speech_lora_a:
            return

        for key, base_w in self._base_weights.items():
            if key in self._speech_lora_a and key in self._speech_lora_b:
                lora_a = self._speech_lora_a[key]
                lora_b = self._speech_lora_b[key]
                merged = base_w + self._speech_lora_scale * (lora_b @ lora_a)

                # Set the weight on the model
                parts = key.split(".")
                obj = self
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                setattr(obj, parts[-1], merged)
        self._active_lora = "speech"

    def apply_vision_lora(self):
        """Switch LLM weights back to vision LoRA."""
        if not hasattr(self, "_vision_lora_a") or not self._vision_lora_a:
            return

        for key, base_w in self._base_weights.items():
            if key in self._vision_lora_a and key in self._vision_lora_b:
                lora_a = self._vision_lora_a[key]
                lora_b = self._vision_lora_b[key]
                merged = base_w + self._vision_lora_scale * (lora_b @ lora_a)

                parts = key.split(".")
                obj = self
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    else:
                        obj = getattr(obj, p)
                setattr(obj, parts[-1], merged)
        self._active_lora = "vision"

    def apply_both_loras(self):
        """Apply both vision and speech LoRA deltas for multimodal inference."""
        has_vision = hasattr(self, "_vision_lora_a") and self._vision_lora_a
        has_speech = hasattr(self, "_speech_lora_a") and self._speech_lora_a
        if not has_vision and not has_speech:
            return

        for key, base_w in self._base_weights.items():
            merged = base_w
            if (
                has_vision
                and key in self._vision_lora_a
                and key in self._vision_lora_b
            ):
                merged = merged + self._vision_lora_scale * (
                    self._vision_lora_b[key] @ self._vision_lora_a[key]
                )
            if (
                has_speech
                and key in self._speech_lora_a
                and key in self._speech_lora_b
            ):
                merged = merged + self._speech_lora_scale * (
                    self._speech_lora_b[key] @ self._speech_lora_a[key]
                )

            parts = key.split(".")
            obj = self
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            setattr(obj, parts[-1], merged)
        self._active_lora = "both"

    def apply_base_weights(self):
        """Reset LLM weights to base (no LoRA) for text-only inference."""
        if not hasattr(self, "_base_weights") or not self._base_weights:
            return

        for key, base_w in self._base_weights.items():
            parts = key.split(".")
            obj = self
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            setattr(obj, parts[-1], base_w)
        self._active_lora = None

    def set_modality(self, has_image: bool = False, has_audio: bool = False):
        """Automatically apply the correct LoRA based on input modality.

        Args:
            has_image: Whether the input contains images.
            has_audio: Whether the input contains audio.
        """
        if has_image and has_audio:
            target = "both"
        elif has_audio:
            target = "speech"
        elif has_image:
            target = "vision"
        else:
            target = None  # text-only: base weights

        current = getattr(self, "_active_lora", "vision")  # vision is default after load
        if current == target:
            return

        if target == "both":
            self.apply_both_loras()
        elif target == "speech":
            self.apply_speech_lora()
        elif target == "vision":
            self.apply_vision_lora()
        else:
            self.apply_base_weights()
