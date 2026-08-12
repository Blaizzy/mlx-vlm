import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .audio import AudioTokenizer
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionTokenizer, ensure_channels_last

_AUDIO_LSTM_KEY = re.compile(r"^(.*\.lstm)\.(weight|bias)_(ih|hh)_l(\d+)$")


def masked_scatter(input_tensor: mx.array, mask: mx.array, source: mx.array):
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(
        input_tensor.shape
    )


class Model(nn.Module):
    """Image/audio/text Apertus 1.5 inference.

    The initial port intentionally supports one prompt at a time. Custom
    attention masks are accepted for API compatibility but ignored; the
    Apertus backbone builds its own causal mask.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if config.tie_word_embeddings and not config.text_config.tie_word_embeddings:
            # utils.load_model rebuilds text_config from the raw nested dict
            # after ModelConfig.from_dict, discarding the hoisted top-level
            # flag; reapply it before the language model reads it.
            config.text_config.tie_word_embeddings = True
        self.language_model = LanguageModel(config.text_config)
        self.vision_tokenizer = VisionTokenizer(config.vision_tokenizer_config)
        self.audio_tokenizer = (
            AudioTokenizer(config.audio_tokenizer_config)
            if config.audio_tokenizer_config is not None
            else None
        )

    def get_image_tokens(
        self, pixel_values: mx.array, image_sizes: Optional[mx.array] = None
    ) -> mx.array:
        if pixel_values.ndim != 4 or pixel_values.shape[0] == 0:
            raise ValueError("pixel_values must contain at least one rank-4 image")
        pixel_values = ensure_channels_last(
            pixel_values, self.config.vision_tokenizer_config.in_channels
        )
        padded_height, padded_width = pixel_values.shape[1:3]
        if image_sizes is None:
            image_sizes = mx.array(
                [[padded_height, padded_width]] * pixel_values.shape[0]
            )
        elif image_sizes.shape[0] != pixel_values.shape[0]:
            raise ValueError(
                "The number of image sizes must match the number of images."
            )

        vocab_ids = []
        factor = self.config.vision_tokenizer_config.spatial_scale_factor
        for image_index in range(pixel_values.shape[0]):
            height, width = [int(value) for value in image_sizes[image_index].tolist()]
            if (
                height <= 0
                or width <= 0
                or height > padded_height
                or width > padded_width
                or height % factor
                or width % factor
            ):
                raise ValueError(
                    "Each image size must be positive, within the padded tensor, "
                    f"and divisible by spatial factor {factor}; got {height}x{width} "
                    f"inside {padded_height}x{padded_width}."
                )
            image = pixel_values[image_index : image_index + 1, :height, :width, :]
            codes = self.vision_tokenizer.encode(image)[0]
            vocab_ids.append(codes.flatten() + self.config.image_token_offset)
        return mx.concatenate(vocab_ids)

    def get_image_features(
        self, pixel_values: mx.array, image_sizes: Optional[mx.array] = None
    ) -> mx.array:
        vocab_ids = self.get_image_tokens(pixel_values, image_sizes)
        return self.language_model.model.embed_tokens(vocab_ids)

    def get_audio_tokens(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if self.audio_tokenizer is None:
            raise ValueError(
                "This checkpoint has no audio tokenizer configuration; audio "
                "inputs are not supported."
            )
        if input_features.ndim != 3 or input_features.shape[0] == 0:
            raise ValueError(
                "input_features must be a rank-3 (num_clips, 1, num_samples) "
                "array with at least one clip."
            )
        num_clips, _, padded_length = input_features.shape
        if feature_attention_mask is None:
            lengths = [padded_length] * num_clips
        else:
            if feature_attention_mask.shape[0] != num_clips:
                raise ValueError(
                    "The number of audio clips in input_features must match "
                    "the number of rows in feature_attention_mask."
                )
            # Clips are right-padded, so the valid length is the mask sum.
            lengths = [
                int(value)
                for value in mx.sum(
                    feature_attention_mask.astype(mx.int32), axis=-1
                ).tolist()
            ]
            if any(length == 0 for length in lengths):
                raise ValueError(
                    "feature_attention_mask marks an audio clip with zero "
                    "valid samples."
                )
            # Slicing ``[:length]`` below is only correct for right padding;
            # reject left-padded or gapped masks instead of mis-encoding.
            expected = mx.arange(padded_length)[None, :] < mx.array(lengths)[:, None]
            if not mx.array_equal(feature_attention_mask != 0, expected):
                raise ValueError(
                    "feature_attention_mask must be right-padded: each row "
                    "must be ones followed by zeros."
                )

        # Encode clip by clip so batch zero-padding never reaches the codec.
        vocab_ids = []
        for clip_index, length in enumerate(lengths):
            clip = input_features[clip_index : clip_index + 1, :, :length]
            codes = self.audio_tokenizer.encode(clip)[0]
            vocab_ids.append(codes.flatten() + self.config.audio_token_offset)
        return mx.concatenate(vocab_ids)

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        vocab_ids = self.get_audio_tokens(input_features, feature_attention_mask)
        return self.language_model.model.embed_tokens(vocab_ids)

    def _scatter_features(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        features: mx.array,
        token_id: int,
        modality: str,
    ) -> mx.array:
        placeholder_mask = input_ids == token_id
        num_placeholders = int(mx.sum(placeholder_mask).item())
        if num_placeholders != features.shape[0]:
            raise ValueError(
                f"{modality} features and {modality} placeholders do not match: "
                f"{features.shape[0]} features for {num_placeholders} placeholders."
            )
        expanded_mask = mx.broadcast_to(
            mx.expand_dims(placeholder_mask, -1), inputs_embeds.shape
        )
        return masked_scatter(inputs_embeds, expanded_mask, features)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_sizes: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_sizes)
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                image_features,
                self.config.image_token_id,
                "Image",
            )
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask
            )
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                audio_features,
                self.config.audio_token_id,
                "Audio",
            )
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        # Trainers pass the attention mask as the third positional argument
        # (see sft_trainer/orpo_trainer), so keep mask/cache in the shared
        # slots and the model-specific parameters keyword-friendly after.
        mask: Optional[mx.array] = None,
        cache=None,
        image_sizes: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if input_ids is not None and input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 array.")
        if inputs_embeds is not None and inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must be a rank-3 array.")
        if input_ids is not None and inputs_embeds is not None:
            if input_ids.shape != inputs_embeds.shape[:2]:
                raise ValueError(
                    "input_ids and inputs_embeds must have matching batch and "
                    "sequence dimensions."
                )
        batch_size = (
            inputs_embeds.shape[0]
            if inputs_embeds is not None
            else input_ids.shape[0] if input_ids is not None else None
        )
        if batch_size != 1:
            raise ValueError(
                "The initial Apertus 1.5 implementation supports batch size 1 only."
            )
        if inputs_embeds is None:
            features = self.get_input_embeddings(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )
            inputs_embeds = features.inputs_embeds
        elif pixel_values is not None or input_features is not None:
            raise ValueError(
                "Media inputs cannot be combined with precomputed inputs_embeds."
            )

        # The reused ApertusModel constructs its own causal mask from the cache.
        # Keep mask and extra generator kwargs in the public signature for API
        # compatibility, but do not forward unsupported arguments into the LM.
        return self.language_model(
            inputs=None,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )

    def _sanitize_audio(self, weights):
        """Convert reference WavTokenizer weights to this port's layout.

        Keeps the encoder and quantizer codebook only, fuses torch
        weight-normalization parametrizations into plain conv weights,
        transposes convs to MLX layout, and folds the two torch LSTM bias
        vectors into MLX's single one.
        """
        sanitized = {}
        norm_pairs = {}
        lstm_biases = {}
        for key, value in weights.items():
            if key.startswith("model.audio_tokenizer."):
                key = key[len("model.") :]
            elif not key.startswith("audio_tokenizer."):
                sanitized[key] = value
                continue
            if self.audio_tokenizer is None:
                continue
            rest = key[len("audio_tokenizer.") :]
            # The Vocos-style decoder is never used for understanding audio.
            if rest.startswith(("backbone.", "head.")):
                continue
            # EMA training buffers of the codebook; only `embed` is used.
            if rest.startswith("quantizer.codebook.") and not rest.endswith(".embed"):
                continue

            if key.endswith(".parametrizations.weight.original0"):
                base = key[: -len(".parametrizations.weight.original0")]
                norm_pairs.setdefault(base, {})["g"] = value
                continue
            if key.endswith(".parametrizations.weight.original1"):
                base = key[: -len(".parametrizations.weight.original1")]
                norm_pairs.setdefault(base, {})["v"] = value
                continue

            lstm_match = _AUDIO_LSTM_KEY.match(key)
            if lstm_match:
                prefix, kind, gate_input, layer = lstm_match.groups()
                target = f"{prefix}.{layer}"
                if kind == "weight":
                    name = "Wx" if gate_input == "ih" else "Wh"
                    sanitized[f"{target}.{name}"] = value
                else:
                    pending = lstm_biases.setdefault(target, {})
                    pending[gate_input] = value
                continue

            sanitized[key] = value

        for base, pair in norm_pairs.items():
            if "g" not in pair or "v" not in pair:
                raise ValueError(
                    f"Incomplete weight normalization pair for {base}.weight."
                )
            magnitude = pair["g"].astype(mx.float32)
            direction = pair["v"].astype(mx.float32)
            norm = mx.sqrt(mx.sum(direction * direction, axis=(1, 2), keepdims=True))
            fused = magnitude * direction / norm
            # torch (out, in, kernel) -> MLX (out, kernel, in)
            sanitized[f"{base}.weight"] = fused.transpose(0, 2, 1)

        for target, pending in lstm_biases.items():
            if "ih" not in pending or "hh" not in pending:
                raise ValueError(f"Incomplete LSTM bias pair for {target}.bias.")
            sanitized[f"{target}.bias"] = pending["ih"].astype(mx.float32) + pending[
                "hh"
            ].astype(mx.float32)

        return sanitized

    def sanitize(self, weights):
        sanitized = {}
        for key, value in self._sanitize_audio(weights).items():
            is_hf_vision_weight = key.startswith("model.vision_tokenizer.")
            if key.startswith("model.language_model."):
                suffix = key[len("model.language_model.") :]
                key = f"language_model.model.{suffix}"
            elif key.startswith("model."):
                key = key[len("model.") :]
            elif key.startswith("language_model.") and not key.startswith(
                ("language_model.model.", "language_model.lm_head.")
            ):
                # Also accept checkpoints where the outer ``model.`` prefix
                # was removed but the decoder has not yet gained its MLX
                # wrapper prefix. Already-converted keys are left untouched.
                suffix = key[len("language_model.") :]
                key = f"language_model.model.{suffix}"
            elif key == "lm_head.weight":
                key = "language_model.lm_head.weight"

            if is_hf_vision_weight and value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)
            if any(
                key.endswith(suffix)
                for suffix in ("alpha_p", "alpha_n", ".beta", ".eps")
            ):
                value = value.squeeze()
            if (
                self.config.text_config.tie_word_embeddings
                and key == "language_model.lm_head.weight"
            ):
                continue
            sanitized[key] = value
        return sanitized

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def cast_predicate(self):
        # Code assignment is an argmax over the codebooks. Casting the
        # tokenizer weights can flip otherwise deterministic code IDs.
        return lambda path: not path.startswith(
            ("vision_tokenizer.", "audio_tokenizer.")
        )

    @property
    def quant_predicate(self):
        # Keep the discrete tokenizers, especially their codebooks,
        # unquantized even when quantize_model() is called outside convert.py.
        return lambda path, module: not path.startswith(
            ("vision_tokenizer.", "audio_tokenizer.")
        )

    @property
    def layers(self):
        return self.language_model.layers
