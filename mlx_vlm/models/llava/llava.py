import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_topk_tokens(self, image_feature, attn, dominant_tokens_ratio=None):
        batch_size, seq_len = image_feature.shape[:2]

        k_tokens = (
            int(image_feature.shape[1] * dominant_tokens_ratio)
            if dominant_tokens_ratio is not None
            else None
        )  # keep 25% of the visual tokens
        if k_tokens is None:
            return image_feature
        cls_idx = 0  # self.config.image_token_index

        attn_rec = mx.sum(attn[:, :, cls_idx + 1 :, cls_idx], axis=1)

        topk_idx = mx.argsort(attn_rec, axis=1)[:, -k_tokens:]
        # use this to plot the dominant attention map
        # https://github.com/dvlab-research/VisionZip/blob/demo-chat/llava/model/multimodal_encoder/clip_encoder.py#L62
        # https://github.com/dvlab-research/VisionZip/blob/demo-chat/llava/serve/gradio_web_server.py#L424

        # Create CLS token indices array
        # Shape: (B, 1)
        cls_indices = mx.full((batch_size, 1), cls_idx, dtype=mx.int32)

        # Concat with CLS token index
        # Add 1 to account for the offset after CLS token
        dominant_idx = mx.concatenate([cls_indices, topk_idx + cls_idx + 1], axis=1)

        image_feature = mx.take(image_feature, dominant_idx, axis=1)[0]
        return image_feature

    def merge_similar_visual_tokens(self, image_feature, visual_token_ratio):
        # Skip CLS token (first token)
        tokens = image_feature[:, 1:]
        batch_size, num_tokens, hidden_dim = tokens.shape

        # Calculate target number of tokens
        target_tokens = max(1, int(num_tokens * visual_token_ratio))

        while num_tokens > target_tokens:
            # Calculate similarities between adjacent tokens
            tokens_a = tokens[:, :-1]  # all except last
            tokens_b = tokens[:, 1:]  # all except first

            # Calculate cosine similarity
            a_norm = mx.sqrt(mx.sum(tokens_a * tokens_a, axis=-1, keepdims=True))
            b_norm = mx.sqrt(mx.sum(tokens_b * tokens_b, axis=-1, keepdims=True))
            similarities = mx.sum(tokens_a * tokens_b, axis=-1)
            similarities = similarities / (a_norm.squeeze(-1) * b_norm.squeeze(-1))

            # Sort similarities and get indices of pairs to merge
            # We'll merge about 20% of remaining excess tokens in each iteration
            num_to_merge = max(1, int((num_tokens - target_tokens) * 0.2))
            merge_indices = mx.argsort(similarities, axis=-1)[:, -num_to_merge:]

            # Create a list to track which indices to merge
            to_merge = set(merge_indices[0].tolist())

            # Merge selected pairs
            new_tokens = []
            i = 0
            while i < num_tokens:
                if i < num_tokens - 1 and i in to_merge:
                    # Merge this token with the next one
                    merged = (tokens[:, i : i + 1] + tokens[:, i + 1 : i + 2]) / 2
                    new_tokens.append(merged)
                    i += 2
                elif i > 0 and (i - 1) in to_merge:
                    # Skip this token as it was merged in the previous step
                    i += 1
                else:
                    # Keep this token as is
                    new_tokens.append(tokens[:, i : i + 1])
                    i += 1

            # Update tokens
            tokens = mx.concatenate(new_tokens, axis=1)
            num_tokens = tokens.shape[1]

        # Reattach CLS token
        return mx.concatenate([image_feature[:, :1], tokens], axis=1)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        merge_similar_tokens_ratio: Optional[float] = 1,
        filter_topk_tokens_ratio: Optional[float] = 1,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        *_, hidden_states, all_attn = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
            output_attn=True,
        )
        # Get the attention from the desired layer
        all_attn = all_attn[self.vision_feature_layer]

        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        #  Select dominant tokens
        selected_image_feature = self.get_topk_tokens(
            selected_image_feature, all_attn, filter_topk_tokens_ratio
        )

        #  Merge similar tokens
        selected_image_feature = self.merge_similar_visual_tokens(
            selected_image_feature, merge_similar_tokens_ratio
        )

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                "Unexpected feature selection strategy: "
                f"{self.vision_feature_select_strategy}"
            )

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        merge_similar_tokens_ratio = kwargs.get("merge_similar_tokens_ratio", 1)
        filter_topk_tokens_ratio = kwargs.get("filter_topk_tokens_ratio", 1)
        input_embddings = self.get_input_embeddings(
            input_ids,
            pixel_values,
            merge_similar_tokens_ratio,
            filter_topk_tokens_ratio,
        )
        logits = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
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
            model_config = json.load(f)

        model_config = ModelConfig.from_dict(model_config)

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config.text_config)

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = VisionModel.sanitize(weights)
        weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model
