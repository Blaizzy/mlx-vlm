"""
From https://github.com/deepseek-ai/DeepSeek-VL2
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(
    image, min_num=2, max_num=9, image_size=640, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


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


class DeepseekOCR2Processor(ProcessorMixin):
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

        # Add chat template
        self.chat_template = kwargs.pop("chat_template", self.default_chat_template)

    @property
    def default_chat_template(self):
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{% elif message['role'] == 'assistant' %}{% endif %}"
            "{{message['content']}} "
            "{% endfor %}"
            "{% if add_generation_prompt %}{% endif %}"
        )

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
        images: List[Image.Image] = None,
        inference_mode: bool = True,
        base_size: int = 1024,
        image_size: int = 768,
        cropping: bool = True,
        min_patches: int = 1,
        max_patches: int = 6,
    ):

        sft_format = prompt
        (
            tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        ) = self.tokenize_with_images(
            sft_format,
            images,
            base_size=base_size,
            image_size=image_size,
            cropping=cropping,
            min_patches=min_patches,
            max_patches=max_patches,
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

        return {
            "input_ids": input_ids[None, :],
            "attention_mask": input_ids != self.pad_id,
            "labels": target_ids,
            "images": images_list,
            "images_seq_mask": images_seq_mask[None, ...],
            "images_spatial_crop": images_spatial_crop,
            "num_image_tokens": num_image_tokens,
        }

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
        base_size: int = 1024,
        image_size: int = 768,
        cropping: bool = True,
        min_patches: int = 1,
        max_patches: int = 6,
    ):
        """Tokenize text with <image> tags.

        For DeepSeek-OCR-2 with Qwen2 encoder:
        - Global view (1024x1024): 256 tokens from Qwen2 encoder
        - Local patches (768x768): 144 tokens each from Qwen2 encoder
        - Plus 1 view_separator token

        Dynamic resolution:
        - Total tokens = (num_patches * 144) + 256 + 1
        - Default: 0-6 patches at 768x768 + 1 global at 1024x1024
        """
        # Token counts for Qwen2 encoder
        TOKENS_PER_PATCH = 144  # 12x12 SAM features for 768x768
        TOKENS_PER_GLOBAL = 256  # 16x16 SAM features for 1024x1024
        TOKENS_VIEW_SEP = 1

        assert conversation.count(self.image_token) == len(
            images
        ), f"The number of image tokens in the prompt does not match the number of images: {conversation.count(self.image_token)} != {len(images)}"

        text_splits = conversation.split(self.image_token)

        all_patches_list = []
        all_global_list = []
        images_seq_mask = []
        tokenized_str = []
        images_spatial_crop = []
        num_image_tokens_list = []

        for text_sep, image in zip(text_splits, images):
            # Tokenize the text before this image
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            # Process global view: pad to base_size x base_size (1024x1024)
            global_view = ImageOps.pad(
                image,
                (base_size, base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            global_tensor = self.image_transform(global_view).astype(mx.bfloat16)
            all_global_list.append(global_tensor)

            # Process local patches using dynamic resolution
            if cropping and min_patches > 0:
                # Use dynamic_preprocess to split image into patches
                patches, (rows, cols) = dynamic_preprocess(
                    image,
                    min_num=min_patches,
                    max_num=max_patches,
                    image_size=image_size,  # 768x768 patches
                    use_thumbnail=False,
                )
                num_patches = len(patches)

                # Transform each patch
                patch_tensors = []
                for patch in patches:
                    patch_tensor = self.image_transform(patch).astype(mx.bfloat16)
                    patch_tensors.append(patch_tensor)

                if patch_tensors:
                    patches_stacked = mx.stack(patch_tensors, axis=0)
                    all_patches_list.append(patches_stacked)

                images_spatial_crop.append([rows, cols])
            else:
                # No patches, only global view
                num_patches = 0
                images_spatial_crop.append([0, 0])

            # Calculate number of image tokens for this image
            # Order: [local_patches, global_view, view_separator]
            num_image_tokens = (
                (num_patches * TOKENS_PER_PATCH) + TOKENS_PER_GLOBAL + TOKENS_VIEW_SEP
            )
            num_image_tokens_list.append(num_image_tokens)

            # Add image tokens to sequence
            tokenized_image = [self.image_token_id] * num_image_tokens
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

        # Tokenize the text after the last image
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # Add the bos token
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        images_seq_mask = mx.array(images_seq_mask)

        # Stack global images
        if len(all_global_list) == 0:
            images_ori = mx.zeros((1, 3, base_size, base_size))
            images_spatial_crop = mx.zeros((1, 2))
        else:
            images_ori = mx.stack(all_global_list, axis=0)
            images_spatial_crop = mx.array(images_spatial_crop)

        # Stack patches (or zeros if no patches)
        if all_patches_list:
            # Concatenate all patches from all images
            images_patches = mx.concatenate(all_patches_list, axis=0)
        else:
            images_patches = mx.zeros((1, 3, image_size, image_size))

        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to images_seq_mask's length {len(images_seq_mask)}"

        return (
            tokenized_str,
            [images_patches, images_ori],
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens_list[0] if num_image_tokens_list else 257,
        )

    def __call__(
        self,
        *,
        text: str = None,
        images: List[Image.Image] = None,
        inference_mode: bool = True,
        image_size: int = 768,
        base_size: int = 1024,
        cropping: bool = True,
        min_patches: int = 1,
        max_patches: int = 6,
        padding: bool = True,
        return_tensors: Literal["np", "mx", "pt"] = "mx",
        **kwargs,
    ):
        """Process text and images for DeepSeek-OCR-2.

        Args:
            text (str or List[str]): the formatted prompt(s)
            images (List[ImageType]): the list of images (one per prompt for batched inputs)
            inference_mode (bool): if True, remove the last eos token
            image_size (int): size of local patches (default 768)
            base_size (int): size of global view (default 1024)
            cropping (bool): whether to use dynamic resolution with local patches
            min_patches (int): minimum number of patches (default 1)
            max_patches (int): maximum number of patches (default 6)

        Returns:
            outputs (dict): the output of the processor,
                - input_ids (mx.array): [batch_size, N + image tokens]
                - images (List[mx.array]): [patches, global_images]
                - images_seq_mask (mx.array): mask for image token positions
                - images_spatial_crop (mx.array): patch grid dimensions
        """

        # Handle batched inputs (list of prompts with list of images)
        if isinstance(text, list):
            if images is None:
                images = [None] * len(text)

            batch_results = []
            for i, prompt in enumerate(text):
                # Each prompt has one image
                img = [images[i]] if images[i] is not None else None
                result = self.process_one(
                    prompt=prompt,
                    images=img,
                    inference_mode=inference_mode,
                    image_size=image_size,
                    base_size=base_size,
                    cropping=cropping,
                    min_patches=min_patches,
                    max_patches=max_patches,
                )
                batch_results.append(result)

            # Collate batch results
            return self._collate_batch(batch_results, padding=padding)

        # Single input case
        prepare = self.process_one(
            prompt=text,
            images=images,
            inference_mode=inference_mode,
            image_size=image_size,
            base_size=base_size,
            cropping=cropping,
            min_patches=min_patches,
            max_patches=max_patches,
        )

        return prepare

    def _collate_batch(self, batch_results: List[Dict], padding: bool = True) -> Dict:
        """Collate multiple processed results into a batch."""
        if not batch_results:
            return {}

        batch_size = len(batch_results)

        # Get max sequence length for padding
        max_seq_len = max(r["input_ids"].shape[1] for r in batch_results)

        # Pad and stack input_ids
        padded_input_ids = []
        padded_images_seq_mask = []
        for r in batch_results:
            seq_len = r["input_ids"].shape[1]
            pad_len = max_seq_len - seq_len

            if pad_len > 0:
                # Pad input_ids on the left
                input_ids = mx.concatenate(
                    [
                        mx.full((1, pad_len), self.pad_id, dtype=r["input_ids"].dtype),
                        r["input_ids"],
                    ],
                    axis=1,
                )
                # Pad images_seq_mask on the left with False
                seq_mask = mx.concatenate(
                    [mx.zeros((1, pad_len), dtype=mx.bool_), r["images_seq_mask"]],
                    axis=1,
                )
            else:
                input_ids = r["input_ids"]
                seq_mask = r["images_seq_mask"]

            padded_input_ids.append(input_ids)
            padded_images_seq_mask.append(seq_mask)

        # Stack into batch
        input_ids = mx.concatenate(padded_input_ids, axis=0)
        images_seq_mask = mx.concatenate(padded_images_seq_mask, axis=0)

        # Combine images: [patches, global_images]
        all_patches = []
        all_global_images = []
        all_spatial_crops = []

        for r in batch_results:
            patches, global_img = r["images"]
            # Only add non-zero patches
            if mx.sum(patches).item() != 0:
                all_patches.append(patches)
            all_global_images.append(global_img)
            all_spatial_crops.append(r["images_spatial_crop"])

        # Stack patches and global images
        if all_patches:
            combined_patches = mx.concatenate(all_patches, axis=0)
        else:
            combined_patches = mx.zeros((1, 3, 1024, 1024))

        combined_global_images = mx.concatenate(all_global_images, axis=0)
        combined_spatial_crops = mx.concatenate(all_spatial_crops, axis=0)

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.pad_id,
            "images": [combined_patches, combined_global_images],
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": combined_spatial_crops,
        }


# Install a composable AutoProcessor patch for DeepSeek-OCR-2
from ..base import install_auto_processor_patch

install_auto_processor_patch("deepseekocr_2", DeepseekOCR2Processor)
