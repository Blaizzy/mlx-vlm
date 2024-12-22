"""
From https://github.com/deepseek-ai/DeepSeek-VL2
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

from .conversation import get_conv_template


def select_best_resolution(image_size, candidate_resolutions):
    # used for cropping
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self.__dict__.values())[item]
        if item not in self.__dict__:
            raise KeyError(item)
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: mx.array
    target_ids: mx.array
    images: mx.array
    images_seq_mask: mx.array
    images_spatial_crop: mx.array
    num_image_tokens: List[int]

    def __len__(self):
        return len(self.input_ids)


@dataclass
class BatchCollateOutput(DictOutput):
    sft_format: List[str]
    input_ids: mx.array
    labels: mx.array
    images: mx.array
    attention_mask: mx.array
    images_seq_mask: mx.array
    images_spatial_crop: mx.array
    seq_lens: List[int]


class ImageTransform:
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def __call__(self, pil_img: Image.Image):
        # Convert PIL image to numpy array and normalize

        img = mx.array(np.array(pil_img)) / 255.0

        # Transpose from HWC to CHW format
        img = mx.transpose(img, [2, 0, 1])

        if self.normalize:
            mean = mx.array(self.mean).reshape(-1, 1, 1)
            std = mx.array(self.std).reshape(-1, 1, 1)
            img = (img - mean) / std

        return img


class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        candidate_resolutions: Tuple[Tuple[int, int]],
        patch_size: int,
        downsample_ratio: int,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        self.candidate_resolutions = candidate_resolutions
        self.image_size = candidate_resolutions[0][0]
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio

        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        # Add special tokens
        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
        print(
            f"Add pad token = ['{pad_token}'] to the tokenizer\n"
            f"{pad_token}:{tokenizer.encode(pad_token, add_special_tokens=False)[0]}"
        )

        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        print(
            f"Add image token = ['{image_token}'] to the tokenizer\n"
            f"{image_token}:{tokenizer.encode(image_token, add_special_tokens=False)[0]}"
        )

        # Add grounding-related tokens
        special_tokens = ["<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print("Added grounding-related tokens")

        # Add chat tokens
        special_tokens = ["<|User|>", "<|Assistant|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print("Added chat tokens")

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(tokenizer, **kwargs)

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        inference_mode: bool = True,
        system_prompt: str = "",
        **kwargs,
    ):
        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if apply_sft_format:
            sft_format = self.format_prompts(
                prompts=prompt, sft_format=self.sft_format, system_prompt=system_prompt
            )
        else:
            sft_format = prompt
        (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        ) = self.tokenize_with_images(
            sft_format, images, bos=True, eos=True, cropping=len(images) <= 2
        )
        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)

        input_ids = mx.array(tokenized_str)
        target_ids = mx.array(masked_tokenized_str)
        images_seq_mask = mx.array(images_seq_mask)

        # Set ignored indices
        target_ids = mx.where(
            (input_ids < 0) | (input_ids == self.image_token_id),
            self.ignore_id,
            target_ids,
        )
        input_ids = mx.where(input_ids < 0, self.pad_id, input_ids)

        if inference_mode:
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            images = mx.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = mx.zeros((1, 2))
        else:
            images = mx.stack(images_list)
            images_spatial_crop = mx.array(images_spatial_crop)

        return VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            target_ids=target_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            num_image_tokens=num_image_tokens,
        )

    def pad_sequence(self, sequences, padding_value):
        # Get max length of sequences
        max_len = max(len(seq) for seq in sequences)

        # Pad each sequence to max length
        padded_seqs = []
        for seq in sequences:
            pad_length = max_len - len(seq)
            if pad_length > 0:
                padding = mx.full((pad_length,), padding_value)
                padded_seq = mx.concatenate([seq, padding])
            else:
                padded_seq = seq
            padded_seqs.append(padded_seq)

        return mx.stack(padded_seqs)

    def tokenize_with_images(
        self,
        conversation: str,
        images: List[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        """Tokenize text with <image> tags."""
        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        images_list, images_seq_mask, images_spatial_crop = [], [], []
        num_image_tokens = []
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            """encode text_sep"""
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = select_best_resolution(
                    image.size, self.candidate_resolutions
                )
            else:
                best_width, best_height = self.image_size, self.image_size

            """process the global view"""
            global_view = ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(
                image,
                (best_width, best_height),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(
                            local_view.crop(
                                (j, i, j + self.image_size, i + self.image_size)
                            )
                        )
                    )

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = (
                best_width // self.image_size,
                best_height // self.image_size,
            )
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            # global views tokens h * (w + 1), 1 is for line seperator
            tokenized_image = [self.image_token_id] * h * (w + 1)
            # add a seperator between global and local views
            tokenized_image += [self.image_token_id]
            # local views tokens, (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_image += (
                [self.image_token_id]
                * (num_height_tiles * h)
                * (num_width_tiles * w + 1)
            )

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))

        """process the last text split"""
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos and eos tokens"""
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        )

    def batchify(
        self,
        sample_list: List[VLChatProcessorOutput],
        padding: Literal["left", "right"] = "left",
    ) -> BatchCollateOutput:
        batched_sft_format = [sample.sft_format for sample in sample_list]
        batched_input_ids = [sample.input_ids for sample in sample_list]
        batched_labels = [sample.target_ids for sample in sample_list]
        batched_images_seq_mask = [sample.images_seq_mask for sample in sample_list]
        seq_lens = [len(sample) for sample in sample_list]

        # Padding
        if padding == "left":
            # MLX implementation of padding
            max_len = max(len(ids) for ids in batched_input_ids)

            def pad_left(sequence, pad_val):
                pad_len = max_len - len(sequence)
                if pad_len > 0:
                    padding = mx.full((pad_len,), pad_val)
                    return mx.concatenate([padding, sequence])
                return sequence

            batched_input_ids = mx.stack(
                [pad_left(ids, self.pad_id) for ids in batched_input_ids]
            )
            batched_labels = mx.stack(
                [pad_left(ids, self.ignore_id) for ids in batched_labels]
            )
            batched_images_seq_mask = mx.stack(
                [pad_left(mask, False) for mask in batched_images_seq_mask]
            )
            batched_attention_mask = batched_input_ids != self.pad_id

        else:
            batched_input_ids = self.pad_sequence(batched_input_ids, self.pad_id)
            batched_labels = self.pad_sequence(batched_labels, self.ignore_id)
            batched_images_seq_mask = self.pad_sequence(batched_images_seq_mask, False)
            batched_attention_mask = batched_input_ids != self.pad_id

        # Padding images
        max_n_patches = max(sample.images.shape[0] for sample in sample_list)
        batched_images = []
        for sample in sample_list:
            images = sample.images
            n_pads = max_n_patches - images.shape[0]
            if n_pads > 0:
                pad_shape = (n_pads,) + images.shape[1:]
                pad_images = mx.zeros(pad_shape)
                images = mx.concatenate([images, pad_images])
            batched_images.append(images)
        batched_images = mx.stack(batched_images)

        # Padding spatial crop info
        max_n_images = max(
            sample.images_spatial_crop.shape[0] for sample in sample_list
        )
        batched_images_spatial_crop = []
        for sample in sample_list:
            spatial_crop = sample.images_spatial_crop
            n_pads = max_n_images - spatial_crop.shape[0]
            if n_pads > 0:
                pad_spatial = mx.zeros((n_pads, 2))
                spatial_crop = mx.concatenate([spatial_crop, pad_spatial])
            batched_images_spatial_crop.append(spatial_crop)
        batched_images_spatial_crop = mx.stack(batched_images_spatial_crop)

        return {
            "input_ids": batched_input_ids,
            "attention_mask": batched_attention_mask,
            "labels": batched_labels,
            "images": batched_images,
            "images_seq_mask": batched_images_seq_mask,
            "images_spatial_crop": batched_images_spatial_crop,
            "sft_format": batched_sft_format,
            "seq_lens": seq_lens,
        }

    def __call__(
        self,
        *,
        text: str = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        force_batchify: bool = True,
        inference_mode: bool = True,
        system_prompt: str = "",
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            force_batchify (bool): force batchify the inputs;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=text,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt,
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare
