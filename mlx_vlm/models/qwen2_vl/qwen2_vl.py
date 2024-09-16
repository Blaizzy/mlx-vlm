import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    rope_scaling: dict
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


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        embed_std = 1 / mx.sqrt(config.text_config.hidden_size)
        # self.image_newline = (
        #     mx.random.normal((config.text_config.hidden_size,)) * embed_std
        # )
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0

            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]

                vision_start_indices = mx.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = mx.sum(vision_tokens == image_token_id)
                video_nums = mx.sum(vision_tokens == video_token_id)
                input_tokens = input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        mx.max(llm_pos_ids_list[-1]) + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        mx.broadcast_to(mx.arange(text_len).reshape(1, -1), (3, -1))
                        + st_idx
                    )

                    t_index = mx.broadcast_to(
                        mx.arange(llm_grid_t).reshape(-1, 1),
                        (-1, llm_grid_h * llm_grid_w),
                    ).flatten()
                    h_index = mx.broadcast_to(
                        mx.arange(llm_grid_h).reshape(1, -1, 1),
                        (llm_grid_t, -1, llm_grid_w),
                    ).flatten()
                    w_index = mx.broadcast_to(
                        mx.arange(llm_grid_w).reshape(1, 1, -1),
                        (llm_grid_t, llm_grid_h, -1),
                    ).flatten()
                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        mx.max(llm_pos_ids_list[-1]) + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        mx.broadcast_to(mx.arange(text_len).reshape(1, -1), (3, -1))
                        + st_idx
                    )

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids = position_ids.at[..., i, attention_mask[i] == 1].set(
                    llm_positions
                )
                mrope_position_deltas.append(
                    mx.max(llm_positions) + 1 - len(total_input_ids[i])
                )

            mrope_position_deltas = mx.array(mrope_position_deltas).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask, axis=-1) - 1
                position_ids = mx.where(attention_mask == 0, 1, position_ids)
                position_ids = mx.broadcast_to(
                    position_ids.reshape(1, *position_ids.shape), (3, -1, -1)
                )
                max_position_ids = mx.max(
                    mx.max(position_ids, axis=0, keepdims=False), axis=-1, keepdims=True
                )
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.broadcast_to(
                    mx.arange(input_ids.shape[1]).reshape(1, 1, -1),
                    (3, input_ids.shape[0], -1),
                )
                mrope_position_deltas = mx.zeros(
                    (input_ids.shape[0], 1), dtype=input_ids.dtype
                )

            return position_ids, mrope_position_deltas

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        print(pixel_values.shape)
        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=True
        )

        # Select the hidden states from the desired layer
        image_features = hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            image_features = image_features[:, 1:]
        else:
            raise ValueError(
                "Unexpected feature selection strategy: "
                f"{self.vision_feature_select_strategy}"
            )

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index

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
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        input_embddings = self.get_input_embeddings(
            input_ids, pixel_values, image_grid_thw
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
        model_config.text_config = TextConfig.from_dict(model_config)

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

    def sanitize(self, weights):
        weights = {
            k.replace("visual", "vision_tower").replace(
                "model", "language_model.model"
            ): v
            for k, v in weights.items()
        }
        return weights
