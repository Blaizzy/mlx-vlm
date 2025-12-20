from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config, config)
        self.vision_tower = VisionModel(config.vision_config)

    @property
    def layers(self):
        return self.language_model.layers

    def build_batched_images(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        image_token_pooling: mx.array,
        image_grids: mx.array,
        image_num_crops: mx.array,
    ) -> tuple[mx.array, mx.array]:
        raw_counts = (input_ids == self.config.image_end_token_id).sum(axis=1)
        counts = raw_counts // 2
        batch_size = counts.shape[0]

        num_images = int(counts.sum().item())

        if image_grids.shape[0] != num_images:
            raise ValueError(
                f"Expected {num_images} image grids, got {image_grids.shape[0]}"
            )
        if image_num_crops.shape[0] != num_images:
            raise ValueError(
                f"Expected {num_images} image crop counts, got {image_num_crops.shape[0]}"
            )

        num_pooled_patches_per_image = (
            (image_grids[:, :2].prod(axis=1) + image_grids[:, 2:].prod(axis=1))
            .astype(image_num_crops.dtype)
            .reshape(-1)
        )

        n_crops, n_patches, pixels_per_patch = pixel_values.shape

        example_ids_for_image = mx.array(
            np.repeat(
                np.arange(batch_size), np.array(counts).astype(np.int32).tolist()
            ),
            dtype=mx.int32,
        )

        crops_per_example = mx.zeros((batch_size,), dtype=image_num_crops.dtype)
        pooled_per_example = mx.zeros(
            (batch_size,), dtype=num_pooled_patches_per_image.dtype
        )
        for image_idx in range(num_images):
            ex = int(example_ids_for_image[image_idx].item())
            crops_per_example[ex] = crops_per_example[ex] + image_num_crops[image_idx]
            pooled_per_example[ex] = (
                pooled_per_example[ex] + num_pooled_patches_per_image[image_idx]
            )

        total_crops = int(crops_per_example.sum().item())
        if total_crops != n_crops:
            raise ValueError(f"Expected {total_crops} crops, got {n_crops}")

        total_pooled = int(pooled_per_example.sum().item())
        if total_pooled != image_token_pooling.shape[0]:
            raise ValueError(
                f"Expected {total_pooled} pooled patches, got {image_token_pooling.shape[0]}"
            )

        max_crops = int(crops_per_example.max().item())
        images = mx.full(
            (batch_size, max_crops, n_patches, pixels_per_patch),
            vals=-1,
            dtype=pixel_values.dtype,
        )

        offset_crop = 0
        for i in range(batch_size):
            num = int(crops_per_example[i].item())
            images[i, :num] = pixel_values[offset_crop : offset_crop + num]
            offset_crop += num

        max_pooled = int(pooled_per_example.max().item())
        token_dim = image_token_pooling.shape[1]
        new_token_pooling = mx.full(
            (batch_size, max_pooled, token_dim),
            vals=-1,
            dtype=image_token_pooling.dtype,
        )

        patches_per_image = image_num_crops * n_patches
        counts_list = counts.tolist()
        image_idx = 0
        pooled_offset = 0
        patch_offset = 0
        for ex, c in enumerate(counts_list):
            num_pooled = int(pooled_per_example[ex].item())
            cur = mx.array(
                image_token_pooling[pooled_offset : pooled_offset + num_pooled]
            )

            per_img_patches = patches_per_image[image_idx : image_idx + c]
            index_offsets = [0] + np.cumsum(per_img_patches.tolist()).tolist()[:-1]
            per_img_pooled = num_pooled_patches_per_image[
                image_idx : image_idx + c
            ].tolist()

            offset = 0
            for j in range(c):
                n = int(per_img_pooled[j])
                idx_off = int(index_offsets[j])
                cur_slice = cur[offset : offset + n]
                cur[offset : offset + n] = mx.where(
                    cur_slice >= 0,
                    cur_slice + idx_off,
                    cur_slice,
                )
                offset += n

            new_token_pooling[ex, :num_pooled] = cur
            pooled_offset += num_pooled
            image_idx += c
            patch_offset += num_pooled

        return images, new_token_pooling

    def build_batched_videos(
        self,
        input_ids: mx.array,
        pixel_values_videos: mx.array,
        video_token_pooling: mx.array,
        video_grids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        end_token_id = (
            self.config.frame_end_token_id
            if self.config.use_frame_special_tokens
            else self.config.image_end_token_id
        )
        counts = mx.any(input_ids == end_token_id, axis=1).astype(mx.int32)
        batch_size = counts.shape[0]
        num_videos = int(counts.sum().item())

        if video_grids.shape[0] != num_videos:
            raise ValueError(
                f"Expected {num_videos} videos, got {video_grids.shape[0]}"
            )

        num_pooled_patches_per_video = (video_grids[:, 1] * video_grids[:, 2]).astype(
            video_token_pooling.dtype
        )

        n_frames, n_patches, pixels_per_patch = pixel_values_videos.shape

        frames_per_example = mx.zeros((batch_size,), dtype=mx.int32)
        pooled_per_example = mx.zeros((batch_size,), dtype=video_token_pooling.dtype)

        video_index = 0
        for i in range(batch_size):
            if counts[i].item() == 1:
                frames_per_example[i] = int(video_grids[video_index][0].item())
                pooled_per_example[i] = num_pooled_patches_per_video[video_index]
                video_index += 1

        max_frames = int(frames_per_example.max().item()) if num_videos else 0
        videos = mx.full(
            (batch_size, max_frames, n_patches, pixels_per_patch),
            vals=-1,
            dtype=pixel_values_videos.dtype,
        )

        offset = 0
        for i in range(batch_size):
            num = int(frames_per_example[i].item())
            if num > 0:
                videos[i, :num] = pixel_values_videos[offset : offset + num]
                offset += num

        max_pooled = int(pooled_per_example.max().item()) if num_videos else 0
        token_dim = video_token_pooling.shape[1]
        new_token_pooling = mx.full(
            (batch_size, max_pooled, token_dim),
            vals=-1,
            dtype=video_token_pooling.dtype,
        )

        pooled_offset = 0
        for i in range(batch_size):
            num = int(pooled_per_example[i].item())
            if num > 0:
                new_token_pooling[i, :num] = video_token_pooling[
                    pooled_offset : pooled_offset + num
                ]
                pooled_offset += num

        if offset != n_frames:
            raise ValueError(f"Expected {n_frames} frames, got {offset}")
        if pooled_offset != video_token_pooling.shape[0]:
            raise ValueError(
                f"Expected {video_token_pooling.shape[0]} pooled tokens, got {pooled_offset}"
            )

        return videos, new_token_pooling

    def merge_visual_inputs(
        self,
        *,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_token_pooling: Optional[mx.array] = None,
        image_grids: Optional[mx.array] = None,
        image_num_crops: Optional[mx.array] = None,
        video_token_pooling: Optional[mx.array] = None,
        video_grids: Optional[mx.array] = None,
    ) -> tuple[Optional[mx.array], Optional[mx.array]]:
        if pixel_values is None:
            return None, None

        if video_token_pooling is not None or video_grids is not None:
            if video_token_pooling is None or video_grids is None:
                raise ValueError(
                    "video_token_pooling and video_grids are required for videos"
                )
            return self.build_batched_videos(
                input_ids=input_ids,
                pixel_values_videos=pixel_values,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
            )

        if (
            image_token_pooling is None
            or image_grids is None
            or image_num_crops is None
        ):
            raise ValueError(
                "image_token_pooling, image_grids, and image_num_crops are required for images"
            )

        return self.build_batched_images(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
        )

    def build_input_embeddings(
        self,
        input_ids: mx.array,
        images: Optional[mx.array] = None,
        token_pooling: Optional[mx.array] = None,
    ) -> mx.array:
        input_ids = input_ids * (input_ids != -1).astype(input_ids.dtype)
        x = self.language_model.model.wte(input_ids)

        if images is not None:
            dtype = self.vision_tower.image_vit.patch_embedding.weight.dtype
            images = images.astype(dtype)
            image_features = self.vision_tower(images, token_pooling)
            is_image_patch = mx.reshape(input_ids, (-1,)) == self.config.image_patch_id
            if int(is_image_patch.sum().item()) != image_features.shape[0]:
                raise ValueError(
                    f"Expected {int(is_image_patch.sum().item())} image features, got {image_features.shape[0]}"
                )
            flat_x = mx.reshape(x, (-1, x.shape[-1]))
            positions = mx.array(np.where(np.array(is_image_patch))[0], dtype=mx.uint32)
            flat_x[positions] = flat_x[positions] + image_features
            x = flat_x.reshape(x.shape)

        return x

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        images, token_pooling = self.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=kwargs.get("image_token_pooling", None),
            image_grids=kwargs.get("image_grids", None),
            image_num_crops=kwargs.get("image_num_crops", None),
            video_token_pooling=kwargs.get("video_token_pooling", None),
            video_grids=kwargs.get("video_grids", None),
        )

        inputs_embeds = None
        if pixel_values is not None:
            inputs_embeds = self.build_input_embeddings(
                input_ids=input_ids,
                images=images,
                token_pooling=token_pooling,
            )

        return self.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
        )

    def sanitize(self, weights):
        def transform_key(key: str) -> str:
            if key.startswith("model.transformer."):
                key = key.replace("model.transformer.", "language_model.model.", 1)
            if key.startswith("model.vision_backbone."):
                key = key.replace("model.vision_backbone.", "vision_tower.", 1)
            if key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.lm_head.", 1)
            # Vision transformer uses list not named submodule
            key = key.replace(".transformer.resblocks.", ".transformer.")
            return key

        return {transform_key(k): v for k, v in weights.items()}
