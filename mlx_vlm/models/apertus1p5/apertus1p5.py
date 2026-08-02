"""Apertus 1.5 Omni.

An early-fusion, discrete-token multimodal model: images and audio are encoded
to code tokens that live in the same vocabulary as text, and the model emits
text. The language tower is a plain Apertus decoder, so the trunk is reused
from ``models/apertus``.

Fusion happens over token ids rather than projected features, which is what
sets this model apart from the rest of mlx-vlm. Each ``<|image|>`` /
``<|audio|>`` placeholder position receives the embedding of its code id,
shifted into the shared vocabulary by ``image_token_offset`` /
``audio_token_offset``. There is no projector.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .audio import AudioModel
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel

# Checkpoint prefixes, relative to the multimodal wrapper.
LANGUAGE_PREFIX = "model.language_model."
VISION_PREFIX = "model.vision_tokenizer."
AUDIO_PREFIX = "model.audio_tokenizer."
# Both tokenizers assign codes by argmax, which half precision perturbs: bf16
# flips roughly 8% of the image codes against the 131k codebook.
TOKENIZER_PREFIXES = ("vision_tokenizer.", "audio_tokenizer.")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vision_tokenizer = VisionModel(config.vision_tokenizer_config)
        self.audio_tokenizer = AudioModel(config.audio_tokenizer_config)

    def get_image_tokens(
        self, pixel_values: mx.array, image_sizes: mx.array
    ) -> Tuple[mx.array, List[int]]:
        """Encode images to vocabulary ids, one image at a time.

        Each image is cropped to its true size before encoding: the encoder has
        global attention, so batch padding would perturb the codes.

        Args:
            pixel_values: ``(num_images, H, W, C)`` in ``[-1, 1]``, padded to a
                common size.
            image_sizes: ``(num_images, 2)`` true ``(height, width)`` per image.

        Returns:
            The concatenated ids, and the code count per image.
        """
        factor = self.vision_tokenizer.spatial_scale_factor
        ids, counts = [], []
        for index in range(pixel_values.shape[0]):
            height, width = (int(side) for side in image_sizes[index])
            image = pixel_values[index : index + 1, :height, :width]
            codes = self.vision_tokenizer.encode(image)[0]
            ids.append(codes.reshape(-1) + self.config.image_token_offset)
            counts.append((height // factor) * (width // factor))
        return mx.concatenate(ids), counts

    def get_audio_tokens(
        self, input_features: mx.array, feature_attention_mask: mx.array
    ) -> Tuple[mx.array, List[int]]:
        """Encode audio clips to vocabulary ids, one clip at a time.

        Args:
            input_features: ``(num_clips, max_length, 1)`` mono 24 kHz samples,
                right-padded to a common length.
            feature_attention_mask: ``(num_clips, max_length)``, 1 for valid
                samples.

        Returns:
            The concatenated ids, and the code count per clip.
        """
        if input_features.ndim == 2:
            input_features = input_features[..., None]
        lengths = [int(row) for row in feature_attention_mask.sum(axis=-1)]
        ids, counts = [], []
        for index, length in enumerate(lengths):
            if length == 0:
                raise ValueError(
                    "`feature_attention_mask` marks an audio clip with zero "
                    "valid samples."
                )
            clip = input_features[index : index + 1, :length]
            codes = self.audio_tokenizer.encode(clip)[0]
            ids.append(codes.reshape(-1) + self.config.audio_token_offset)
            counts.append(self.audio_tokenizer.num_codes(length))
        return mx.concatenate(ids), counts

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_sizes: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        feature_attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        embed_tokens = self.language_model.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        if pixel_values is not None:
            if image_sizes is None:
                raise ValueError(
                    "`image_sizes` must be provided alongside `pixel_values` so "
                    "batch padding can be cropped away before encoding."
                )
            vocab_ids, counts = self.get_image_tokens(pixel_values, image_sizes)
            inputs_embeds = self._scatter_codes(
                inputs_embeds,
                input_ids,
                embed_tokens(vocab_ids),
                self.config.image_token_id,
                sum(counts),
                "image",
            )

        if input_features is not None:
            if feature_attention_mask is None:
                raise ValueError(
                    "`feature_attention_mask` must be provided alongside "
                    "`input_features` to mark the valid samples per clip."
                )
            vocab_ids, counts = self.get_audio_tokens(
                input_features, feature_attention_mask
            )
            inputs_embeds = self._scatter_codes(
                inputs_embeds,
                input_ids,
                embed_tokens(vocab_ids),
                self.config.audio_token_id,
                sum(counts),
                "audio",
            )

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @staticmethod
    def _scatter_codes(
        inputs_embeds: mx.array,
        input_ids: mx.array,
        code_embeds: mx.array,
        placeholder_id: int,
        num_codes: int,
        modality: str,
    ) -> mx.array:
        positions = mx.array(
            [
                index
                for index, token in enumerate(input_ids.reshape(-1).tolist())
                if token == placeholder_id
            ],
            dtype=mx.uint32,
        )
        if positions.size != num_codes:
            raise ValueError(
                f"The prompt has {positions.size} '{modality}' placeholder "
                f"tokens but the inputs produced {num_codes} codes."
            )
        if num_codes == 0:
            return inputs_embeds

        shape = inputs_embeds.shape
        flat = inputs_embeds.reshape(-1, shape[-1])
        flat[positions] = code_embeds.astype(flat.dtype)
        return flat.reshape(shape)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        features = self.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, **kwargs
        )
        return self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=features.inputs_embeds,
        )

    def sanitize(self, weights):
        if any(k.startswith("language_model.") for k in weights):
            return weights

        language, vision, audio, rest = {}, {}, {}, {}
        for k, v in weights.items():
            if k.startswith(LANGUAGE_PREFIX):
                language["model." + k[len(LANGUAGE_PREFIX) :]] = v
            elif k.startswith(VISION_PREFIX):
                vision[k[len(VISION_PREFIX) :]] = v
            elif k.startswith(AUDIO_PREFIX):
                audio[k[len(AUDIO_PREFIX) :]] = v
            else:
                # `lm_head.weight` sits at the top level of the checkpoint.
                rest[k] = v

        language = self.language_model.sanitize({**language, **rest})
        sanitized = {f"language_model.{k}": v for k, v in language.items()}
        # The tokenizers stay in float32 even when the language tower is
        # half precision, because their code argmax is not robust to it.
        for prefix, module, module_weights in (
            ("vision_tokenizer", self.vision_tokenizer, vision),
            ("audio_tokenizer", self.audio_tokenizer, audio),
        ):
            for k, v in module.sanitize(module_weights).items():
                sanitized[f"{prefix}.{k}"] = v.astype(mx.float32)
        return sanitized

    @staticmethod
    def quant_predicate(path: str, module: nn.Module):
        # Quantizing a codebook or its encoder would move code assignments.
        return not path.startswith(TOKENIZER_PREFIXES)

    @property
    def layers(self):
        return self.language_model.layers
