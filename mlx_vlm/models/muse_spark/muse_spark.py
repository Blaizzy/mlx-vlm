"""Muse Spark text, image, video and waveform audio inference."""

import re

import mlx.core as mx
from mlx import nn

from ..base import InputEmbeddingsFeatures
from .audio import AudioModel
from .config import ModelConfig
from .language import LanguageModel
from .sanitize import CheckpointMixin
from .vision import VisionAdapter, VisionModel, WeightlessRMSNorm


def splice_embeddings(embeddings, input_ids, features, token_id):
    mask = input_ids == token_id
    count = int(mask.sum().item())
    if count != features.shape[0]:
        raise ValueError(
            f"Modality tokens and features do not match: {count} tokens, "
            f"{features.shape[0]} features (token id {token_id})."
        )
    if count == 0:
        return embeddings
    indices = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    rows = features[mx.maximum(indices, 0)].astype(embeddings.dtype)
    return mx.where(mask[..., None], rows, embeddings)


class Model(CheckpointMixin, nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = self.args = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = self.vision_adapter = self.vision_projection = None
        self.audio_tower = None
        if config.vision_config is not None:
            self.vision_tower = VisionModel(config.vision_config)
            self.vision_adapter = VisionAdapter(
                config.out_hidden_size, config.projector_hidden_size
            )
            self.vision_projection = nn.Linear(
                config.projector_hidden_size, config.text_config.hidden_size, bias=False
            )
        if config.audio_config is not None:
            self.audio_tower = AudioModel(config.audio_config)
        self.perception_emb_norm = WeightlessRMSNorm(config.text_config.rms_norm_eps)

    def encode_image(self, pixel_values):
        if self.vision_tower is None:
            raise ValueError("This checkpoint does not contain a vision encoder.")
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        if not len(pixel_values) or any(p.ndim != 3 for p in pixel_values):
            raise ValueError("Images must be a nonempty sequence of (C, H, W) arrays.")
        dtype = self.vision_tower.patch_embedder.patch_embedding.weight.dtype
        features = mx.concatenate(
            [
                self.vision_adapter(self.vision_tower(p.astype(dtype)))
                for p in pixel_values
            ]
        )
        return self.perception_emb_norm(self.vision_projection(features))

    def encode_audio(self, input_values, input_values_mask=None, audio_lengths=None):
        if self.audio_tower is None:
            raise ValueError(
                "This checkpoint does not contain an audio encoder. Use an audio-inclusive checkpoint."
            )
        features = self.audio_tower(input_values)
        if audio_lengths is None and input_values_mask is not None:
            # One host transfer during encoding; decode never recomputes audio.
            audio_lengths = input_values_mask.sum(-1).tolist()
        if audio_lengths is not None:
            if isinstance(audio_lengths, mx.array):
                audio_lengths = audio_lengths.tolist()
            if len(audio_lengths) != input_values.shape[0]:
                raise ValueError("Expected one audio length per waveform.")
            samples = self.config.audio_config.samples_per_token
            if any(n <= 0 or n > input_values.shape[1] for n in audio_lengths):
                raise ValueError(
                    "Audio lengths must be positive and fit the padded waveforms."
                )
            features = mx.concatenate(
                [
                    features[idx, : (int(length) + samples - 1) // samples]
                    for idx, length in enumerate(audio_lengths)
                ]
            )
        else:
            features = features.reshape(-1, features.shape[-1])
        return self.perception_emb_norm(features)

    def encode_video(self, pixel_values_videos, video_grid_thw):
        if self.vision_tower is None:
            raise ValueError("This checkpoint does not contain a vision encoder.")
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is required for video input.")
        dtype = self.vision_tower.patch_embedder.patch_embedding.weight.dtype
        features = self.vision_tower(pixel_values_videos.astype(dtype), video_grid_thw)
        features = self.vision_projection(self.vision_adapter(features))
        return self.perception_emb_norm(features)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        embeddings = self.language_model.model.embed(input_ids)
        if pixel_values is not None:
            features = kwargs.get("cached_image_features")
            if features is None:
                features = self.encode_image(pixel_values)
            embeddings = splice_embeddings(
                embeddings, input_ids, features, self.config.image_token_id
            )
        if kwargs.get("input_values") is not None:
            features = self.encode_audio(
                kwargs["input_values"],
                kwargs.get("input_values_mask"),
                kwargs.get("audio_lengths"),
            )
            embeddings = splice_embeddings(
                embeddings, input_ids, features, self.config.audio_token_id
            )
        if kwargs.get("pixel_values_videos") is not None:
            features = self.encode_video(
                kwargs["pixel_values_videos"], kwargs.get("video_grid_thw")
            )
            embeddings = splice_embeddings(
                embeddings, input_ids, features, self.config.video_token_id
            )
        return InputEmbeddingsFeatures(inputs_embeds=embeddings)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        embeddings = kwargs.pop("inputs_embeds", None)
        if embeddings is None:
            embeddings = self.get_input_embeddings(
                input_ids, pixel_values, **kwargs
            ).inputs_embeds
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=embeddings,
            mask=mask,
            logits_to_keep=kwargs.get("logits_to_keep", 0),
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group=None):
        self.language_model.shard(group)

    def quantization_path_aliases(self, path):
        if path.startswith("language_model.model."):
            return ("model.language_model." + path[len("language_model.model.") :],)
        if path == "language_model.lm_head" or path.startswith(
            "language_model.lm_head."
        ):
            return (path[len("language_model.") :],)
        return ("model." + path,)

    @property
    def quant_predicate(self):
        experts = re.compile(
            r"^language_model\.model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj$"
        )
        full_precision = {0, self.config.text_config.num_hidden_layers - 1}

        def predicate(path, module):
            match = experts.match(path)
            return match is not None and int(match.group(1)) not in full_precision

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith(("_alpha", "_beta"))
