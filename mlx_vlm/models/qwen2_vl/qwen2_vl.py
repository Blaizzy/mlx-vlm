import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    image_token_index: int = 151655
    video_token_index: int = 151656
    vision_start_token_id: int = 151652
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level
        excluded_keys = {"vision_config"}
        params["text_config"] = dict(
            filter(lambda x: x[0] not in excluded_keys, params.items())
        )

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.rope_deltas = None

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
    ):

        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states = self.vision_tower(
            pixel_values, grid_thw, output_hidden_states=False
        )

        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            hidden_states, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        video_token_index = self.config.video_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = input_ids == image_token_index
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_index

        image_indices = np.where(image_positions)[1].tolist()

        inputs_embeds[:, image_indices, :] = image_features

        return inputs_embeds
    
    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        # Calculate RoPE index for image/video tokens
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))        
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = mx.where(attention_mask[i] == 1, input_ids, mx.zeros_like(input_ids))
                image_nums, video_nums = 0, 0
                vision_start_indices = mx.sum(mx.where(input_ids == vision_start_token_id, mx.arange(input_ids.shape[0]), mx.zeros_like(input_ids)))
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos =  image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) +1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) +1
                    if ed_image < ed_video:
                        t, h, w = (image_grid_thw[image_index][0], image_grid_thw[image_index][1], image_grid_thw[image_index][2])
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (video_grid_thw[video_index][0], video_grid_thw[video_index][1], video_grid_thw[video_index][2])
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t , llm_grid_h, llm_grid_w = (t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size)
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w))  # Equivalent to expand()
                    t_index = t_index.flatten()  # Flattens to 1D

                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)  # Equivalent to .view(1, -1)
                    h_index = mx.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w))  # Equivalent to expand()
                    h_index = h_index.flatten()  # Flattens to 1D

                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)  # Equivalent to .view(1, -1)
                    w_index = mx.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w))  # Equivalent to expand()
                    w_index = w_index.flatten()  # Flattens to 1D


                    llm_pos_ids_list.append(mx.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st

                    t_index = mx.arange(text_len).reshape(1, text_len)  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(t_index, (3, text_len))  # Equivalent to expand(3, -1)

                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3,1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(expanded_mask, expanded_positions, position_ids[:, i:i+1, :])
                updated_position_ids = mx.concatenate([
                    position_ids[:, :i, :],
                    new_positions,
                    position_ids[:, i+1:, :]
                ], axis=1)
                position_ids = updated_position_ids
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis= -1) - 1
                position_ids = mx.where(attention_mask == 0, mx.ones_like(position_ids), position_ids)
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:    
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (3, input_ids.shape[0], input_ids.shape[1]))
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas
    
    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        position_ids = kwargs.pop("position_ids", None)
        
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        # reset rope_deltas when processing a new image/video
        if pixel_values is not None:
            self.rope_deltas = None
        input_embddings = self.get_input_embeddings(input_ids, pixel_values, grid_thw)
        if position_ids is None and (mask is None or mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache is not None and cache[0] == 0)
                or self.rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, mask
                )
                self.rope_deltas = rope_deltas
            else:
                # Use the prev pre-calculated rope-deltas to get the correct position ids
                batch_size, seq_length = input_ids.shape
                delta = cache[-1].offset + self.rope_deltas if cache is not None else 0
                delta = delta[None][None]
                position_ids = mx.arange(seq_length).reshape(1, seq_length)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
                if cache is not None:
                    # Repeat delta for each batch
                    delta = mx.repeat(delta, batch_size // delta.shape[0], axis=0)
                position_ids = mx.add(position_ids, delta).reshape(position_ids.shape)
                position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_length))
        logits = self.language_model(None, cache=cache, inputs_embeds=input_embddings, position_ids=position_ids)
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
        def transform_key(key):
            if "vision_tower" not in key:
                key = key.replace("visual", "vision_tower")
            if "language_model" not in key:
                if "model" in key:
                    key = key.replace("model", "language_model.model")
                elif "lm_head" in key:
                    key = key.replace("lm_head", "language_model.lm_head")
            return key

        return {transform_key(k): v for k, v in weights.items()}
