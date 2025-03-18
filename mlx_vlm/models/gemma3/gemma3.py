import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .language import LanguageModel, RMSNorm, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 257152
    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mm_input_projection_weight = mx.ones(
            (config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.text_config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        b, _, l = x.shape

        reshaped_vision_outputs = x.transpose(0, 2, 1)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            b, l, self.patches_per_image, self.patches_per_image
        )

        # Transpose to place h, w in indices 1, 2
        reshaped_vision_outputs = reshaped_vision_outputs.transpose(0, 2, 3, 1)
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.transpose(0, 3, 1, 2).flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(0, 2, 1)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = mx.einsum(
            "btm,md->btd", normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.astype(x.dtype)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids), None

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_state, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype),
            output_hidden_states=True,
        )

        image_features = hidden_state[None, :].astype(pixel_values.dtype)
        image_features = self.multi_modal_projector(image_features)

        final_inputs_embeds, final_attention_mask_4d = (
            self._prepare_inputs_for_multimodal(
                image_features, inputs_embeds, input_ids, mask
            )
        )
        return final_inputs_embeds, final_attention_mask_4d

    def _prepare_inputs_for_multimodal(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        _, _, embed_dim = image_features.shape

        batch_size, sequence_length = input_ids.shape
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        final_embedding = mx.zeros((batch_size, sequence_length, embed_dim))

        pad_token_id = self.config.pad_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == pad_token_id

        # expand masks to match embedding dimension
        text_mask_expanded = mx.expand_dims(text_mask, -1)
        text_mask_expanded = mx.repeat(text_mask_expanded, embed_dim, axis=-1)
        pad_mask_expanded = mx.expand_dims(pad_mask, -1)
        pad_mask_expanded = mx.repeat(pad_mask_expanded, embed_dim, axis=-1)

        # insert padding and text token embeddings
        final_embedding = mx.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        )
        pad_size = final_embedding.shape[1] - scaled_image_features.shape[1]
        scaled_image_features = mx.pad(
            scaled_image_features, ((0, 0), (0, pad_size), (0, 0))
        )
        # insert image embeddings - the image mask is always less or equal to the sentence in length
        image_mask_expanded = mx.expand_dims(image_mask, -1)
        image_mask_expanded = mx.repeat(image_mask_expanded, embed_dim, axis=-1)
        final_embedding = mx.where(
            image_mask_expanded, scaled_image_features, final_embedding
        )

        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        )

        attention_mask_expanded_1 = mx.expand_dims(attention_mask, 1)
        attention_mask_expanded_2 = mx.expand_dims(attention_mask, 2)
        final_attention_mask_4d = attention_mask_expanded_1 * attention_mask_expanded_2
        final_attention_mask_4d = final_attention_mask_4d
        final_attention_mask_4d = mx.expand_dims(final_attention_mask_4d, 1)
        final_embedding = mx.array(final_embedding)
        return final_embedding, final_attention_mask_4d

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeddings, final_attention_mask_4d = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )

        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            mask=final_attention_mask_4d,
        )
        return logits

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model
